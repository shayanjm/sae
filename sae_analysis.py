import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
from sae.sae import Sae
import argparse
from torch.distributed.elastic.multiprocessing.errors import record
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  # Import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@record
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process SAEs with distributed GPUs")
    parser.add_argument(
        "--sae_directory",
        type=str,
        required=True,
        help="Path to the SAE checkpoints directory",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="output",
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-2-7B",
        help="Name or path of the model to use",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="togethercomputer/RedPajama-Data-1T-Sample",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--dataset_rows",
        type=int,
        default=1000,
        help="Number of rows from the dataset to use (default: all)",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=2048,
        help="Maximum token length for tokenizer (default: 2048)",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=4,
        help="Number of tokens around the activating token to include as context (default: 4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for DataLoader (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling (default: 42)",
    )

    args = parser.parse_args()
    
    # Load the tokenizer and model from arguments
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

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
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_token_length,
            return_tensors="pt",
        )
        return tokenized
    
    # Load and tokenize the dataset from arguments
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(args.dataset_rows))

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", init_method="env://"
                            )
    rank = dist.get_rank()
    world_size = dist.get_world_size()    
    
    # Get the local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the device for each process using local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Global Rank {rank}/{world_size}, Local Rank {local_rank}, using device: {device}"
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Get list of SAEs and distribute among processes
    sae_layer_names = [
        d
        for d in os.listdir(sae_directory)
        if os.path.isdir(os.path.join(sae_directory, d))
    ]
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

   
    # Define custom Dataset class
    class TokenizedDataset(Dataset):
        def __init__(self, tokenized_data):
            self.tokenized_data = tokenized_data

        def __len__(self):
            return len(self.tokenized_data)

        def __getitem__(self, idx):
            data_point = self.tokenized_data[idx]
            item = {
                key: torch.tensor(data_point[key])
                for key in ["input_ids", "attention_mask"]
                if key in data_point
            }
            item["text"] = data_point["text"]
            return item

    # Create PyTorch dataset
    torch_dataset = TokenizedDataset(tokenized_dataset)

    # Create DistributedSampler and DataLoader
    sampler = DistributedSampler(
        torch_dataset, num_replicas=world_size, rank=rank,
    )
    batch_size = args.batch_size
    data_loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0,  # Set num_workers to 0 to reduce memory usage
    )

    # Log DataLoader length
    logger.info(f"Rank {rank}: DataLoader length: {len(data_loader)}")

    # Function to process data loader
    def process_data_loader(
        data_loader,
        model,
        sae_model,
        layer_to_analyze,
        d_in,
        expansion_factor,
        device,
        args,
        tokenizer,
        rank,
    ):
        num_latents = d_in * expansion_factor
        activation_counts = torch.zeros(num_latents, dtype=torch.int64, device=device)
        total_tokens = 0

        layer_idx = int(layer_to_analyze.split(".")[-1])

        # Create output file path
        output_dir = os.path.join(args.output_directory, f"rank_{rank}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{layer_to_analyze}.parquet")

        # Define the schema
        schema = pa.schema([
            ('layer_index', pa.int32()),
            ('sample_id', pa.int32()),
            ('latent_index', pa.int32()),
            ('activation', pa.float32()),
            ('reconstruction_error', pa.float32()),
            ('trigger_token', pa.string()),
            ('context', pa.list_(pa.string())),
            ('token_position', pa.int32()),
        ])

        # Create a ParquetWriter
        writer = pq.ParquetWriter(output_file, schema, compression='snappy')

        # Start processing
        logger.info(
            f"Rank {rank}: Starting data loader processing for {layer_to_analyze}"
        )

        # Create tqdm progress bar
        data_loader_length = len(data_loader)
        progress_bar = tqdm(
            data_loader,
            desc=f"Rank {rank}, Layer {layer_to_analyze}",
            position=rank,
            leave=True,
            total=data_loader_length,
            ncols=80,
        )

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass through the model
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = (
                    outputs.hidden_states
                )  # Tuple of hidden states at each layer

            # Get the residuals (activations) at the specified layer
            residuals = hidden_states[
                layer_idx
            ]  # Shape: [batch_size, seq_length, hidden_size]

            # Flatten batch and sequence dimensions
            batch_size, seq_length, hidden_size = residuals.size()
            residuals_flat = residuals.view(
                -1, hidden_size
            )  # Shape: [batch_size * seq_length, hidden_size]

            # Pass through SAE to get outputs
            forward_output = sae_model(residuals_flat)

            # Collect activation counts
            latent_indices = forward_output.latent_indices  # Shape: [total_tokens, k]
            latent_acts = forward_output.latent_acts  # Shape: [total_tokens, k]

            # Flatten indices and count activations
            indices_flat = latent_indices.view(-1)
            activation_counts += torch.bincount(indices_flat, minlength=num_latents).to(
                torch.int64
            )

            # Total tokens processed
            total_tokens += residuals_flat.size(0)

            # Reconstruction error per token
            reconstruction_errors = torch.mean(
                (residuals_flat - forward_output.sae_out) ** 2, dim=1
            )  # Shape: [total_tokens]

            # Prepare context and token data
            # Initialize lists to store data for this batch
            data_batch = {
                "layer_index": [],
                "sample_id": [],
                "latent_index": [],
                "activation": [],
                "reconstruction_error": [],
                "trigger_token": [],
                "context": [],
                "token_position": [],
            }

            for i in range(batch_size):
                input_id_sequence = input_ids[i]
                tokens = tokenizer.convert_ids_to_tokens(input_id_sequence)
                text_length = len(tokens)

                for j in range(seq_length):
                    idx_in_batch = i * seq_length + j

                    # Get the reconstruction error, token, context, position
                    reconstruction_error = reconstruction_errors[idx_in_batch].item()
                    trigger_token = tokens[j]
                    token_position = j

                    # Get context window
                    start = max(0, j - args.context_window)
                    end = min(text_length, j + args.context_window + 1)
                    context_tokens = tokens[start:end]

                    # Get the activations and latent indices for this token
                    activations = latent_acts[idx_in_batch].detach().cpu().numpy()
                    latent_indices_token = (
                        latent_indices[idx_in_batch].detach().cpu().numpy()
                    )

                    # For each latent activated for this token
                    for k in range(len(activations)):
                        activation_value = activations[k]
                        latent_index = latent_indices_token[k]

                        # Append to batch data
                        data_batch["layer_index"].append(layer_idx)
                        data_batch["sample_id"].append(batch_idx * batch_size + i)
                        data_batch["latent_index"].append(latent_index)
                        data_batch["activation"].append(activation_value)
                        data_batch["reconstruction_error"].append(reconstruction_error)
                        data_batch["trigger_token"].append(trigger_token)
                        data_batch["context"].append(context_tokens)
                        data_batch["token_position"].append(token_position)

            # Convert data_batch to PyArrow Table
            batch_table = pa.Table.from_pydict({
                "layer_index": pa.array(data_batch["layer_index"], type=pa.int32()),
                "sample_id": pa.array(data_batch["sample_id"], type=pa.int32()),
                "latent_index": pa.array(data_batch["latent_index"], type=pa.int32()),
                "activation": pa.array(data_batch["activation"], type=pa.float32()),
                "reconstruction_error": pa.array(
                    data_batch["reconstruction_error"], type=pa.float32()
                ),
                "trigger_token": pa.array(data_batch["trigger_token"], type=pa.string()),
                "context": pa.array(data_batch["context"], type=pa.list_(pa.string())),
                "token_position": pa.array(data_batch["token_position"], type=pa.int32()),
            })

            # Write batch_table to Parquet file
            writer.write_table(batch_table)

            # Clear data_batch to free up memory
            del data_batch
            del batch_table

            # Clear variables to free up memory
            del input_ids
            del attention_mask
            del outputs
            del hidden_states
            del residuals
            del residuals_flat
            del forward_output
            del latent_indices
            del latent_acts
            del reconstruction_errors
            torch.cuda.empty_cache()

            # Update progress bar
            progress_bar.set_postfix(batch=batch_idx)

        # Close the progress bar
        progress_bar.close()

        # Close the ParquetWriter
        writer.close()

        logger.info(f"Rank {rank}: Finished processing {layer_to_analyze}")

    # Process each SAE assigned to this rank
    for layer_to_analyze in sae_layer_names_per_rank:
        if layer_to_analyze is None:
            continue  # Skip padding entries

        try:
            sae_model, sae_cfg = load_sae(layer_to_analyze)
            sae_model.to(device)
            sae_model.eval()

            logger.info(f"Rank {rank}: Processing {layer_to_analyze}")

            # Extract expansion_factor from sae_cfg
            expansion_factor = getattr(sae_cfg, "expansion_factor", None)
            if expansion_factor is None:
                logger.error(
                    f"Rank {rank}: SAE config for {layer_to_analyze} lacks 'expansion_factor'"
                )
                raise AttributeError(
                    f"SAE config for {layer_to_analyze} lacks 'expansion_factor'"
                )

            # Extract d_in from sae_model
            d_in = getattr(sae_model, "d_in", None)
            if d_in is None:
                logger.error(
                    f"Rank {rank}: SAE model for {layer_to_analyze} lacks 'd_in'"
                )
                raise AttributeError(f"SAE model for {layer_to_analyze} lacks 'd_in'")

            # Call the processing function
            process_data_loader(
                data_loader,
                model,
                sae_model,
                layer_to_analyze,
                d_in,
                expansion_factor,
                device,
                args,
                tokenizer,
                rank,
            )

        except Exception as e:
            logger.error(f"Rank {rank}: Error processing {layer_to_analyze}: {e}")
            raise

    logger.info(f"Rank {rank}: All results processed.")

    # Synchronize before final aggregation
    dist.barrier()

    # Aggregate across ranks and save as needed here
    if rank == 0:
        logger.info("Rank 0: Starting aggregation of results.")

        # Collect Parquet files from all ranks
        all_files = []
        for r in range(world_size):
            rank_output_dir = os.path.join(args.output_directory, f"rank_{r}")
            if os.path.exists(rank_output_dir):
                for fname in os.listdir(rank_output_dir):
                    if fname.endswith(".parquet"):
                        all_files.append(os.path.join(rank_output_dir, fname))

        # Read all Parquet files and combine them
        combined_tables = []
        logger.info("Rank 0: Reading Parquet files from all ranks.")
        for file in tqdm(all_files, desc="Aggregating Parquet files", ncols=80):
            table = pq.read_table(file)
            combined_tables.append(table)

        if combined_tables:
            # Concatenate all tables
            logger.info("Rank 0: Concatenating tables.")
            final_table = pa.concat_tables(combined_tables, promote=True)

            # Save the combined table
            logger.info("Rank 0: Writing combined Parquet file...")
            final_output_dir = os.path.join(args.output_directory, "combined")
            os.makedirs(final_output_dir, exist_ok=True)
            final_output_file = os.path.join(final_output_dir, "all_layers.parquet")

            pq.write_table(final_table, final_output_file, compression="snappy")

            logger.info(f"Rank {rank}: Aggregated results saved to {final_output_file}")
        else:
            logger.info("Rank 0: No data to aggregate.")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
