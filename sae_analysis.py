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
from tqdm import tqdm
import numpy as np

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
        help="Number of rows from the dataset to use (default: 1000)",
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
        sae_model = Sae.load_from_disk(layer_path)
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
    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(args.dataset_rows))

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get the local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the device for each process using local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Global Rank {rank}/{world_size}, Local Rank {local_rank}, using device: {device}"
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
    )
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
        torch_dataset,
        num_replicas=world_size,
        rank=rank,
    )
    batch_size = args.batch_size
    data_loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,  # Increased num_workers for faster data loading
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

        # Create output directories
        output_data_dir = os.path.join(args.output_directory, f"rank_{rank}", "data")
        os.makedirs(output_data_dir, exist_ok=True)

        output_summary_dir = os.path.join(
            args.output_directory, f"rank_{rank}", "summary"
        )
        os.makedirs(output_summary_dir, exist_ok=True)

        # Define the schema
        schema = pa.schema(
            [
                ("layer_index", pa.int32()),
                ("sample_id", pa.int32()),
                ("latent_index", pa.int32()),
                ("latent_bin", pa.int32()),  # New column
                ("activation", pa.float32()),
                ("reconstruction_error", pa.float32()),
                ("trigger_token", pa.string()),
                ("context", pa.list_(pa.string())),
                ("token_position", pa.int32()),
            ]
        )

        # Initialize per-neuron statistics using tensors
        per_neuron_counts = torch.zeros(num_latents, dtype=torch.int64, device=device)
        per_neuron_sum_activation = torch.zeros(
            num_latents, dtype=torch.float32, device=device
        )
        per_neuron_sum_activation_sq = torch.zeros(
            num_latents, dtype=torch.float32, device=device
        )

        # Initialize data buffer for batch writing
        data_buffer = {
            "layer_index": [],
            "sample_id": [],
            "latent_index": [],
            "latent_bin": [],
            "activation": [],
            "reconstruction_error": [],
            "trigger_token": [],
            "context": [],
            "token_position": [],
        }
        batch_write_size = 100000  # Adjust as needed

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
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Forward pass through the model
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states

            # Get the residuals (activations) at the specified layer
            residuals = hidden_states[
                layer_idx
            ]  # Shape: [batch_size, seq_length, hidden_size]
            batch_size, seq_length, hidden_size = residuals.size()
            residuals_flat = residuals.view(
                -1, hidden_size
            )  # Shape: [total_tokens, hidden_size]

            # Pass through SAE to get outputs
            with torch.no_grad():
                forward_output = sae_model(residuals_flat.to(sae_model.device))

            # Collect activation counts and per-neuron statistics
            latent_indices = forward_output.latent_indices  # Shape: [total_tokens, k]
            latent_acts = forward_output.latent_acts  # Shape: [total_tokens, k]
            latent_indices_flat = latent_indices.view(-1)
            latent_acts_flat = latent_acts.view(-1)

            # Update activation counts
            activation_counts += torch.bincount(
                latent_indices_flat, minlength=num_latents
            ).to(torch.int64)
            total_tokens += residuals_flat.size(0)

            # Update per-neuron statistics
            counts = torch.bincount(latent_indices_flat, minlength=num_latents)
            per_neuron_counts += counts
            per_neuron_sum_activation.scatter_add_(
                0, latent_indices_flat, latent_acts_flat
            )
            per_neuron_sum_activation_sq.scatter_add_(
                0, latent_indices_flat, latent_acts_flat**2
            )

            # Reconstruction error per token
            reconstruction_errors = torch.mean(
                (residuals_flat - forward_output.sae_out.to(device)) ** 2, dim=1
            )  # Shape: [total_tokens]

            # Prepare data for saving
            k = latent_indices.size(1)
            total_activations = latent_indices_flat.size(0)

            # Generate sample_ids and token_positions
            sample_ids = batch_idx * batch_size + torch.arange(
                batch_size, device=device
            ).unsqueeze(1).repeat(1, seq_length).view(
                -1
            )  # [total_tokens]
            sample_ids_expanded = sample_ids.repeat_interleave(k).cpu().numpy()

            token_positions = (
                torch.arange(seq_length, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .view(-1)
            )  # [total_tokens]
            token_positions_expanded = (
                token_positions.repeat_interleave(k).cpu().numpy()
            )

            reconstruction_errors_expanded = (
                reconstruction_errors.repeat_interleave(k).cpu().numpy()
            )

            latent_bins = (latent_indices_flat % 1024).cpu().numpy()

            # Convert input_ids to tokens
            input_ids_cpu = input_ids.cpu().numpy()
            tokens_batch = [
                tokenizer.convert_ids_to_tokens(ids) for ids in input_ids_cpu
            ]
            tokens_flat = [token for tokens_seq in tokens_batch for token in tokens_seq]
            tokens_flat = np.array(tokens_flat)
            trigger_tokens_expanded = np.repeat(tokens_flat, k)

            # Prepare context
            context_tokens_expanded = []
            for tokens_seq in tokens_batch:
                text_length = len(tokens_seq)
                for j in range(seq_length):
                    start = max(0, j - args.context_window)
                    end = min(text_length, j + args.context_window + 1)
                    context_tokens = tokens_seq[start:end]
                    context_tokens_expanded.extend([context_tokens] * k)

            # Append to data buffer
            data_buffer["layer_index"].extend([layer_idx] * total_activations)
            data_buffer["sample_id"].extend(sample_ids_expanded.tolist())
            data_buffer["latent_index"].extend(latent_indices_flat.cpu().tolist())
            data_buffer["latent_bin"].extend(latent_bins.tolist())
            data_buffer["activation"].extend(latent_acts_flat.cpu().tolist())
            data_buffer["reconstruction_error"].extend(
                reconstruction_errors_expanded.tolist()
            )
            data_buffer["trigger_token"].extend(trigger_tokens_expanded.tolist())
            data_buffer["context"].extend(context_tokens_expanded)
            data_buffer["token_position"].extend(token_positions_expanded.tolist())

            # Write to Parquet if buffer is large enough
            if len(data_buffer["layer_index"]) >= batch_write_size:
                batch_table = pa.Table.from_pydict(data_buffer, schema=schema)
                pq.write_to_dataset(
                    table=batch_table,
                    root_path=output_data_dir,
                    partition_cols=["latent_bin"],
                    use_legacy_dataset=False,
                    max_partitions=1024,
                )
                # Clear the data buffer
                for key in data_buffer:
                    data_buffer[key].clear()

            # Update progress bar
            progress_bar.set_postfix(batch=batch_idx)

        # Close the progress bar
        progress_bar.close()

        # Write any remaining data in the buffer
        if len(data_buffer["layer_index"]) > 0:
            batch_table = pa.Table.from_pydict(data_buffer, schema=schema)
            pq.write_to_dataset(
                table=batch_table,
                root_path=output_data_dir,
                partition_cols=["latent_bin"],
                use_legacy_dataset=False,
                max_partitions=1024,
            )

        logger.info(f"Rank {rank}: Finished processing {layer_to_analyze}")

        # After processing all batches, compute per-neuron statistics
        per_neuron_counts_cpu = per_neuron_counts.cpu()
        per_neuron_sum_activation_cpu = per_neuron_sum_activation.cpu()
        per_neuron_sum_activation_sq_cpu = per_neuron_sum_activation_sq.cpu()

        # Avoid division by zero
        counts = per_neuron_counts_cpu.clamp_min(1)
        mean_activation = per_neuron_sum_activation_cpu / counts
        variance = (per_neuron_sum_activation_sq_cpu / counts) - (mean_activation**2)
        std_activation = torch.sqrt(variance.clamp_min(0))

        # Create PyArrow table from per-neuron statistics
        per_neuron_table = pa.Table.from_pydict(
            {
                "layer_index": pa.array([layer_idx] * num_latents, type=pa.int32()),
                "latent_index": pa.array(range(num_latents), type=pa.int32()),
                "count": pa.array(per_neuron_counts_cpu.numpy(), type=pa.int64()),
                "mean_activation": pa.array(mean_activation.numpy(), type=pa.float32()),
                "std_activation": pa.array(std_activation.numpy(), type=pa.float32()),
            }
        )

        # Write per-neuron summary to Parquet file
        summary_file = os.path.join(
            output_summary_dir, f"{layer_to_analyze}_summary.parquet"
        )
        pq.write_table(per_neuron_table, summary_file, compression="snappy")

        logger.info(f"Rank {rank}: Summary statistics saved to {summary_file}")

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

    # Synchronize
    dist.barrier()

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
