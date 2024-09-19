import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from sae.sae import Sae, ForwardOutput, EncoderOutput
import argparse
from torch.distributed.elastic.multiprocessing.errors import record
import logging
import pandas as pd
from collections import defaultdict
import wandb  # Import wandb

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
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name for wandb logging')
    args = parser.parse_args()

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get the local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set the device for each process using local_rank
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else 'cpu'
    logger.info(f"Global Rank {rank}/{world_size}, Local Rank {local_rank}, using device: {device}")

    # Initialize wandb only on rank 0
    if rank == 0:
        if args.run_name:
            run_name = args.run_name
        else:
            run_name = None  # Let wandb generate a run name

        wandb.init(
            name=run_name,
            project="sae-analysis",
            config=vars(args),
            save_code=True,
        )
        logger.info(f"Rank {rank}: wandb run initialized.")

    # Load the tokenizer and model from arguments
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = '</s>'
        tokenizer.pad_token_id = 2  # For LLaMA models

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = '</s>'
        tokenizer.eos_token_id = 2

    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Define the path to SAEs from arguments
    sae_directory = args.sae_directory

    # Function to load an SAE
    def load_sae(layer_name):
        layer_path = os.path.join(sae_directory, layer_name)
        sae_model = Sae.load_from_disk(layer_path, device=device)
        return sae_model, sae_model.cfg

    # Tokenizer function using args.max_token_length
    def tokenize_function(examples):
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
    def process_data_loader(data_loader, model, sae_model, tokenizer, layer_to_analyze, k, d_in, expansion_factor, device, args):
        num_latents = d_in * expansion_factor
        activation_counts = torch.zeros(num_latents, dtype=torch.int64, device=device)
        total_tokens = 0

        layer_idx = int(layer_to_analyze.split('.')[-1])

        # Progress bar
        data_loader_length = len(data_loader)
        logger.info(f"Rank {rank}: Starting data loader processing. DataLoader length: {data_loader_length}")
        progress_bar = tqdm(enumerate(data_loader), total=data_loader_length, desc=f"Rank {rank} Processing {layer_to_analyze}", position=rank, leave=False)

        for batch_idx_in_loader, batch in progress_bar:
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)           # Shape: [batch_size, seq_length]
                attention_mask = batch['attention_mask'].to(device) # Shape: [batch_size, seq_length]
                texts = batch['text']

                batch_size_, seq_length = input_ids.size()

                # Extract residuals using mixed precision for faster computation
                with autocast():
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        hidden_states = outputs.hidden_states

                residuals = hidden_states[layer_idx + 1].view(batch_size_ * seq_length, -1)  # Shape: [batch_size * seq_length, hidden_dim]

                total_tokens += residuals.size(0)

                # Get latent activations and indices
                forward_output = sae_model(residuals)
                sae_out = forward_output.sae_out
                latent_acts = forward_output.latent_acts  # Shape: [batch_size * seq_length, num_latents]
                latent_indices = forward_output.latent_indices.view(residuals.size(0), -1)  # Shape: [batch_size * seq_length, num_latents]

                # Activation mask and values using scatter in one shot
                activation_mask = torch.zeros(residuals.size(0), num_latents, dtype=torch.bool, device=device)
                activation_values = torch.zeros(residuals.size(0), num_latents, dtype=torch.float32, device=device)
                activation_mask.scatter_(1, latent_indices, 1)
                activation_values.scatter_(1, latent_indices, latent_acts)
                activation_counts += activation_mask.sum(dim=0).to(torch.int64)

                progress_bar.set_postfix({'Total Tokens': total_tokens})

            except Exception as e:
                logger.error(f"Rank {rank}: Error in batch {batch_idx_in_loader}: {e}")
                raise

        progress_bar.close()

        return activation_counts, total_tokens

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
            k = 0

            # Extract expansion_factor from sae_cfg
            if hasattr(sae_cfg, 'expansion_factor'):
                expansion_factor = sae_cfg.expansion_factor
            else:
                logger.error(f"Rank {rank}: SAE config for {layer_to_analyze} lacks 'expansion_factor'")
                raise AttributeError(f"Sae config for {layer_to_analyze} lacks 'expansion_factor'")

            # Extract k from sae_cfg
            if hasattr(sae_cfg, 'k'):
                k = sae_cfg.k
            else:
                logger.error(f"Rank {rank}: SAE config for {layer_to_analyze} lacks 'k'")
                raise AttributeError(f"Sae config for {layer_to_analyze} lacks 'k'")

            # Extract d_in from sae_model
            if hasattr(sae_model, 'd_in'):
                d_in = sae_model.d_in
            else:
                logger.error(f"Rank {rank}: SAE model for {layer_to_analyze} lacks 'd_in'")
                raise AttributeError(f"Sae model for {layer_to_analyze} lacks 'd_in'")

            # Call the processing function
            activation_counts_local, total_tokens_local = process_data_loader(
                data_loader, model, sae_model, tokenizer, layer_to_analyze, k, d_in, expansion_factor, device, args
            )

            # Accumulate total_tokens
            total_tokens_rank += total_tokens_local

            # Aggregate activation counts across all ranks
            activation_counts_global = activation_counts_local.clone()
            dist.all_reduce(activation_counts_global, op=dist.ReduceOp.SUM)

            if rank == 0:
                # Log aggregated data to wandb
                wandb.log({
                    f"Total_Tokens_{layer_to_analyze}": total_tokens_rank,
                    f"Activation_Counts_{layer_to_analyze}": wandb.Histogram(activation_counts_global.cpu().numpy())
                })

            logger.info(f"Rank {rank}: Finished processing {layer_to_analyze}.")

        except Exception as e:
            logger.error(f"Rank {rank}: Error processing {layer_to_analyze}: {e}")
            raise

    logger.info(f"Rank {rank}: All results processed.")

    # Synchronize before final aggregation
    dist.barrier()

    # Aggregate total_tokens across all ranks
    total_tokens_tensor = torch.tensor([total_tokens_rank], device=device)
    dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
    total_tokens_global = total_tokens_tensor.item()

    if rank == 0:
        logger.info(f"Total tokens processed across all ranks: {total_tokens_global}")
        # Save total_tokens_global to a file
        output_dir = args.output_directory
        os.makedirs(output_dir, exist_ok=True)
        output_file_global_tokens = os.path.join(output_dir, 'total_tokens_global.txt')
        with open(output_file_global_tokens, 'w') as f:
            f.write(str(total_tokens_global))
        wandb.log({"Total_Tokens_Global": total_tokens_global})

    # Clean up
    dist.destroy_process_group()
    if rank == 0:
        wandb.finish()  # Close the wandb run

if __name__ == '__main__':
    main()
