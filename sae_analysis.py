import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from safetensors.torch import load_file
from sae.sae import Sae, ForwardOutput, EncoderOutput
import argparse
from torch.distributed.elastic.multiprocessing.errors import record
import logging
import pandas as pd
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@record
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process SAEs with distributed GPUs")
    parser.add_argument('--sae_directory', type=str, required=True,
                        help='Path to the SAE checkpoints directory')
    parser.add_argument('--output_directory', type=str, default='output',
                        help='Directory to save the outputs')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-2-7B',
                        help='Name or path of the model to use')
    parser.add_argument('--dataset_name', type=str, default='togethercomputer/RedPajama-Data-1T-Sample',
                        help='Name of the dataset to use')
    parser.add_argument('--dataset_rows', type=int, default=None,
                        help='Number of rows from the dataset to use (default: all)')
    parser.add_argument('--max_token_length', type=int, default=2048,
                        help='Maximum token length for tokenizer (default: 2048)')
    parser.add_argument('--context_window', type=int, default=4,
                        help='Number of tokens around the activating token to include as context (default: 4)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for DataLoader (default: 2)')
    args = parser.parse_args()

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Define the path to SAEs from arguments
    sae_directory = args.sae_directory

    # Set the device for each process
    device = torch.device('cuda', rank) if torch.cuda.is_available() else 'cpu'
    logger.info(f"Rank {rank}/{world_size}, using device: {device}")

    # Load the tokenizer and model from arguments
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Function to load an SAE
    def load_sae(layer_name):
        layer_path = os.path.join(sae_directory, layer_name)
        sae_model = Sae.load_from_disk(layer_path, device=device)
        return sae_model, sae_model.cfg 

    # Tokenizer function using args.max_token_length
    def tokenize_function(examples):
        tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=args.max_token_length
        )
        tokenized['text'] = examples['text']  # Retain the 'text' field
        return tokenized

    # Get list of SAEs and distribute among processes
    sae_layer_names = [d for d in os.listdir(sae_directory) if os.path.isdir(os.path.join(sae_directory, d))]
    sae_layer_names.sort()  # Ensure consistent order

    # Determine the number of SAEs per rank
    num_saes = len(sae_layer_names)
    saes_per_rank = (num_saes + world_size - 1) // world_size  # Ceiling division

    # Pad the SAE list so it's divisible by world_size
    pad_size = saes_per_rank * world_size - num_saes
    sae_layer_names.extend([None] * pad_size)

    # Create the distribution plan
    sae_assignment = {}
    for r in range(world_size):
        start_idx = r * saes_per_rank
        end_idx = start_idx + saes_per_rank
        assigned_saes = sae_layer_names[start_idx:end_idx]
        sae_assignment[r] = assigned_saes

    # Print the distribution plan (only on rank 0)
    if rank == 0:
        logger.info("SAE distribution plan:")
        for r in range(world_size):
            logger.info(f"Rank {r}: {sae_assignment[r]}")

    # Each rank retrieves its assigned SAEs
    sae_layer_names_per_rank = sae_assignment[rank]

    # Load and tokenize the dataset from arguments
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, split='train')

    if args.dataset_rows is not None:
        dataset = dataset.select(range(args.dataset_rows))

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define custom Dataset class
    class TokenizedDataset(Dataset):
        def __init__(self, tokenized_data, include_text=False):
            self.tokenized_data = tokenized_data
            self.include_text = include_text

        def __len__(self):
            return len(self.tokenized_data)

        def __getitem__(self, idx):
            data_point = self.tokenized_data[idx]
            item = {key: torch.tensor(data_point[key]) for key in ['input_ids', 'attention_mask'] if key in data_point}
            if self.include_text:
                item['text'] = data_point['text']  # Include raw text
            return item

    # Create PyTorch dataset with text included
    torch_dataset = TokenizedDataset(tokenized_dataset, include_text=True)

    # Create DistributedSampler and DataLoader
    sampler = DistributedSampler(torch_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=False)
    sampler.set_epoch(0)
    batch_size = args.batch_size  # Use batch_size from arguments
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, sampler=sampler)

    # Log DataLoader length
    logger.info(f"Rank {rank}: DataLoader length: {len(data_loader)}")

    # Initialize variables to accumulate results
    total_tokens_rank = 0
    all_activation_counts = []
    all_neuron_activation_texts = defaultdict(list)
    all_token_context_maps = {}

    # Function to process data loader
    def process_data_loader(data_loader, model, sae_model, tokenizer, layer_to_analyze, latent_dim, d_in, expansion_factor, device, args):
        activation_counts = torch.zeros(latent_dim, dtype=torch.int32, device=device)
        logger.info(f"activation_counts.shape: {activation_counts.shape}")
        total_tokens = 0
        neuron_activation_texts = defaultdict(list)
        token_context_map = {}
        token_context_counter = 0

        # Extract layer index (assuming layer_to_analyze is in the format 'layer.X')
        layer_idx = int(layer_to_analyze.split('.')[-1])

        # Initialize progress bar with position=rank
        data_loader_length = len(data_loader)
        logger.info(f"Rank {rank}: Starting data loader processing. DataLoader length: {data_loader_length}")

        progress_bar = tqdm(
            enumerate(data_loader),
            total=data_loader_length,
            desc=f"Rank {rank} Processing {layer_to_analyze}",
            position=rank,
            leave=False  # Adjust as needed
        )

        for batch_idx_in_loader, batch in progress_bar:
            try:
                # Move inputs to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                texts = batch['text']  # List of raw texts in the batch

                batch_size_, seq_length = input_ids.size()
                logger.info(f"batch_size_: {batch_size_}") # 2?
                logger.info(f"seq_length: {seq_length}") # 2048?

                # Extract residuals
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                residuals = hidden_states[layer_idx + 1]  # Shape: [batch_size, seq_length, hidden_size]
                logger.info(f"residuals.shape: {residuals.shape}")

                # Reshape residuals to [batch_size * seq_length, hidden_size]
                residuals = residuals.view(batch_size_ * seq_length, -1)
                logger.info(f"Reshaped residuals to [batch_size * seq_length, hidden_size]: residuals.shape: {residuals.shape}")
                token_batch_size = residuals.size(0)
                total_tokens += token_batch_size

                # Flatten token IDs and move to CPU for tokenization
                token_ids = input_ids.view(-1).cpu().numpy()  # Shape: [batch_size * seq_length]
                tokens = tokenizer.convert_ids_to_tokens(token_ids)

                # Precompute full token sequences for each example in the batch
                tokens_full_list = []
                for batch_idx in range(batch_size_):
                    input_ids_batch = input_ids[batch_idx].cpu().numpy()
                    tokens_full = tokenizer.convert_ids_to_tokens(input_ids_batch)
                    tokens_full_list.append(tokens_full)

                # Forward pass through SAE model
                with torch.no_grad():
                    forward_output = sae_model(residuals)
                    sae_out = forward_output.sae_out
                    latent_acts = forward_output.latent_acts  # Shape: [batch_size * seq_length, latent_dim]
                    latent_indices = forward_output.latent_indices.view(token_batch_size, -1)  # Shape: [batch_size * seq_length, k]

                logger.info(f"latent_acts.shape: {latent_acts.shape}")
                logger.info(f"latent_indices.shape: {latent_indices.shape}")
                logger.info(f"latent_dim: {latent_dim}")

                # Create the activation mask
                activation_mask = torch.zeros(token_batch_size, latent_dim, dtype=torch.bool, device=device)
                logger.info(f"activation_mask.shape: {activation_mask.shape}")
                activation_mask.scatter_(1, latent_indices, 1)
                logger.info(f"Scattered activation_mask.shape: {activation_mask.shape}")
                logger.info(f"Shape of Sum on dimension 0 of activation_mask: {activation_mask.sum(dim=0).shape}")
                # Proper summing over the latent dimension
                activation_counts += activation_mask.sum(dim=0).cpu().numpy()

                # Get indices of active neurons
                active_token_indices, active_neuron_indices = torch.nonzero(activation_mask, as_tuple=True)

                logger.info(f"active_token_indices.shape: {active_token_indices.shape}")
                logger.info(f"active_neuron_indices.shape: {active_neuron_indices.shape}")
                logger.info(f"Max token index: {active_token_indices.max()}, Max neuron index: {active_neuron_indices.max()}")

                # Extract activation values safely
                activation_values = latent_acts[active_token_indices, active_neuron_indices].cpu().numpy()

                # Move indices to CPU for further processing
                active_token_indices = active_token_indices.cpu().numpy()
                active_neuron_indices = active_neuron_indices.cpu().numpy()

                # Process activations
                for idx in range(len(active_token_indices)):
                    token_index_in_batch = active_token_indices[idx]
                    neuron_idx = active_neuron_indices[idx]
                    activation_value = activation_values[idx]
                    token = tokens[token_index_in_batch]

                    batch_idx = token_index_in_batch // seq_length
                    seq_idx = token_index_in_batch % seq_length
                    tokens_full = tokens_full_list[batch_idx]

                    # Get context window around the token using args.context_window
                    context_window = args.context_window
                    start_idx = max(0, seq_idx - context_window)
                    end_idx = min(len(tokens_full), seq_idx + context_window + 1)
                    context_tokens = tokens_full[start_idx:end_idx]
                    context_text = ' '.join(context_tokens)

                    context_key = (token, context_text)
                    if context_key not in token_context_map:
                        token_context_map[context_key] = token_context_counter
                        token_context_counter += 1

                    neuron_activation_texts[neuron_idx].append((activation_value, token_context_map[context_key]))

                # Update progress bar
                progress_bar.set_postfix({'Total Tokens': total_tokens})

            except Exception as e:
                logger.error(f"Rank {rank}: Error in batch {batch_idx_in_loader}: {e}")
                raise

        # Close progress bar to prevent overlapping
        progress_bar.close()

        return activation_counts, total_tokens, neuron_activation_texts, token_context_map

    # Process each SAE assigned to this rank
    for layer_to_analyze in sae_layer_names_per_rank:
        if layer_to_analyze is None:
            continue  # Skip padding entries

        try:
            sae_model, sae_cfg = load_sae(layer_to_analyze)
            sae_model.to(device)
            sae_model.eval()

            logger.info(f"Rank {rank}: Processing {layer_to_analyze}")
            expansion_factor = 0
            latent_dim = 0

            # Extract expansion_factor from sae_cfg
            if hasattr(sae_cfg, 'expansion_factor'):
                expansion_factor = sae_cfg.expansion_factor
                logger.info(f"Rank {rank}: expansion_factor for {layer_to_analyze}: {expansion_factor}")
            else:
                logger.error(f"Rank {rank}: SAE config for {layer_to_analyze} lacks 'expansion_factor'")
                raise AttributeError(f"Sae config for {layer_to_analyze} lacks 'expansion_factor'")
            
            # Extract latent_dim from sae_cfg
            if hasattr(sae_cfg, 'k'):
                latent_dim = sae_cfg.k
                logger.info(f"Rank {rank}: latent_dim for {layer_to_analyze}: {latent_dim}")
            else:
                logger.error(f"Rank {rank}: SAE config for {layer_to_analyze} lacks 'k'")
                raise AttributeError(f"Sae config for {layer_to_analyze} lacks 'k'")
            
            # Extract d_in from sae_model
            if hasattr(sae_model, 'd_in'):
                d_in = sae_model.d_in
                logger.info(f"Rank {rank}: d_in for {layer_to_analyze}: {d_in}")
            else:
                logger.error(f"Rank {rank}: SAE model for {layer_to_analyze} lacks 'd_in'")
                raise AttributeError(f"Sae model for {layer_to_analyze} lacks 'd_in'")


            # Call the processing function
            activation_counts, total_tokens, neuron_activation_texts, token_context_map = process_data_loader(
                data_loader, model, sae_model, tokenizer, layer_to_analyze, latent_dim, d_in, expansion_factor, device, args
            )

            # Accumulate total_tokens
            total_tokens_rank += total_tokens

            # Accumulate the results
            all_activation_counts.append(activation_counts)
            for neuron_idx, activations in neuron_activation_texts.items():
                all_neuron_activation_texts[neuron_idx].extend(activations)
            all_token_context_maps.update(token_context_map)

            logger.info(f"Rank {rank}: Finished processing {layer_to_analyze}.")

        except Exception as e:
            logger.error(f"Rank {rank}: Error processing {layer_to_analyze}: {e}")
            raise

    # Save all results together at the end
    output_dir = args.output_directory
    os.makedirs(output_dir, exist_ok=True)

    # Save activation counts as memory-mapped arrays
    if all_activation_counts:
        np.save(os.path.join(output_dir, f'all_activation_counts_{rank}.npy'), np.concatenate(all_activation_counts))
    else:
        logger.warning(f"Rank {rank}: No activation counts to save.")

    # Save neuron activations as Parquet for fast reading
    if all_neuron_activation_texts:
        neuron_activation_df = pd.DataFrame([
            {'neuron_idx': k, 'activation_value': a, 'context_index': c}
            for k, v in all_neuron_activation_texts.items() for a, c in v
        ])
        output_file_texts = os.path.join(output_dir, f'neuron_activation_texts_{rank}.parquet')
        neuron_activation_df.to_parquet(output_file_texts)
    else:
        logger.warning(f"Rank {rank}: No neuron activations to save.")

    # Save token context map as Parquet
    if all_token_context_maps:
        token_context_df = pd.DataFrame(list(all_token_context_maps.items()), columns=['context_key', 'context_index'])
        output_file_context_map = os.path.join(output_dir, f'context_map_{rank}.parquet')
        token_context_df.to_parquet(output_file_context_map)
    else:
        logger.warning(f"Rank {rank}: No token context maps to save.")

    logger.info(f"Rank {rank}: All results saved.")

    logger.info(f"Rank {rank}: Finished processing SAEs. Syncing to barrier.")
    try:
        dist.barrier()
    except Exception as e:
        logger.error(f"Rank {rank}: Error with barrier: {e}")
        raise
    logger.info(f"Rank {rank}: After barrier")

    # Aggregate total_tokens across all ranks
    total_tokens_tensor = torch.tensor([total_tokens_rank], device=device)
    logger.info(f"Rank {rank}: Performing all_reduce on total_tokens_tensor.")
    dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
    logger.info(f"Rank {rank}: Completed all_reduce.")
    total_tokens_global = total_tokens_tensor.item()

    if rank == 0:
        logger.info(f"Total tokens processed across all ranks: {total_tokens_global}")
        # Save total_tokens_global to a file
        output_file_global_tokens = os.path.join(output_dir, 'total_tokens_global.txt')
        with open(output_file_global_tokens, 'w') as f:
            f.write(str(total_tokens_global))

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
