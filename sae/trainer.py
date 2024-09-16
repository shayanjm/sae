from collections import defaultdict
from dataclasses import asdict
from typing import Sized

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from fnmatch import fnmatchcase
from natsort import natsorted
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import MemmapDataset
from .sae import Sae
from .utils import geometric_median, get_layer_list, resolve_widths


class SaeTrainer:
    def __init__(
        self, cfg: TrainConfig, dataset: HfDataset | MemmapDataset, model: PreTrainedModel
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N, cfg.layer_stride))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        self.cfg = cfg
        self.dataset = dataset
        self.distribute_modules()

        N = len(cfg.hookpoints)
        assert isinstance(dataset, Sized)
        num_examples = len(dataset)

        device = model.device
        input_widths = resolve_widths(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        self.model = model
        self.saes = {
            hook: Sae(input_widths[hook], cfg.sae, device)
            for hook in self.local_hookpoints()
        }

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.optimizer = Adam(pgs)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, num_examples // cfg.batch_size
        )

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m")

        lr_state = torch.load(f"{path}/lr_scheduler.pt", map_location=device, weights_only=True)
        opt_state = torch.load(f"{path}/optimizer.pt", map_location=device, weights_only=True)
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        num_batches = len(self.dataset) // self.cfg.batch_size
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"

            n = self.global_step * self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = self.model.device

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            total_samples = len(ds)
            samples_per_rank = total_samples // world_size

            start_index = rank * samples_per_rank
            # Ensure the last rank takes any leftover samples
            end_index = start_index + samples_per_rank if rank != world_size - 1 else total_samples

            indices = list(range(start_index, end_index))
            subset_ds = torch.utils.data.Subset(ds, indices)
        else:
            subset_ds = ds
        
        dl = DataLoader(
            subset_ds, # type: ignore
            batch_size=self.cfg.batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
            drop_last=True,
        )
        num_batches = len(dl)
        print(f"Rank {dist.get_rank()} has {num_batches} batches.")

        if dist.is_initialized():
            total_samples = len(dl.dataset)
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Calculate the number of samples per process
            samples_per_rank = total_samples // world_size
            # Handle any remaining samples in the last rank
            if rank == world_size - 1:
                samples_per_rank += total_samples % world_size
            num_batches = samples_per_rank // self.cfg.batch_size
        else:
            num_batches = len(dl)

        # Initialize the progress bar using len(dl)
        if rank_zero:
            pbar = tqdm(
                desc="Training",
                total=len(dl),
                initial=0,
            )

        # Check if DataLoader is empty
        if len(dl) == 0:
            # Even if no data, need to synchronize with other ranks
            if dist.is_initialized():
                dist.barrier()
        
        else:
            did_fire = {
                name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
                for name, sae in self.saes.items()
            }
            num_tokens_in_step = 0

            # For logging purposes
            avg_auxk_loss = defaultdict(float)
            avg_fvu = defaultdict(float)
            avg_multi_topk_fvu = defaultdict(float)

            hidden_dict: dict[str, Tensor] = {}
            name_to_module = {
                name: self.model.get_submodule(name) for name in self.cfg.hookpoints
            }
            maybe_wrapped: dict[str, DDP] | dict[str, Sae] = {}
            module_to_name = {v: k for k, v in name_to_module.items()}

            def hook(module: nn.Module, _, outputs):
                # Maybe unpack tuple outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                name = module_to_name[module]
                hidden_dict[name] = outputs.flatten(0, 1)

            batch_num = 0
            for batch in dl:
                # Synchronize at the start of each batch processing to ensure all processes are in sync
                if dist.is_initialized():
                    dist.barrier()
                hidden_dict.clear()
                print(f"Synchronized at beginning of batch {batch_num} and cleared hidden_dict")

                # Bookkeeping for dead feature detection
                num_tokens_in_step += batch["input_ids"].numel()

                # Forward pass on the model to get the next batch of activations
                handles = [
                    mod.register_forward_hook(hook) for mod in name_to_module.values()
                ]
                try:
                    with torch.no_grad():
                        self.model(batch["input_ids"].to(device))
                finally:
                    for handle in handles:
                        handle.remove()

                # Scatter hiddens if distributing modules
                if self.cfg.distribute_modules:
                    hidden_dict = self.scatter_hiddens(hidden_dict)

                for name, hiddens in hidden_dict.items():
                    raw = self.saes[name]  # 'raw' never has a DDP wrapper

                    # On the first iteration, initialize the decoder bias
                    if self.global_step == 0:
                        median = geometric_median(self.maybe_all_cat(hiddens))
                        raw.b_dec.data = median.to(raw.dtype)

                    if not maybe_wrapped:
                        # Wrap the SAEs with Distributed Data Parallel
                        maybe_wrapped = (
                            {
                                name: DDP(sae, device_ids=[device], output_device=device)
                                for name, sae in self.saes.items()
                            }
                            if ddp
                            else self.saes
                        )

                    # Ensure the decoder weights are unit-norm
                    if raw.cfg.normalize_decoder:
                        raw.set_decoder_norm_to_unit_norm()

                    acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                    denom = acc_steps * self.cfg.wandb_log_frequency
                    wrapped = maybe_wrapped[name]

                    # Save memory by chunking the activations
                    for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                        out = wrapped(
                            chunk,
                            dead_mask=(
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                                if self.cfg.auxk_alpha > 0
                                else None
                            ),
                        )

                        avg_fvu[name] += float(
                            self.maybe_all_reduce(out.fvu.detach()) / denom
                        )
                        if self.cfg.auxk_alpha > 0:
                            avg_auxk_loss[name] += float(
                                self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                            )
                        if self.cfg.sae.multi_topk:
                            avg_multi_topk_fvu[name] += float(
                                self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                            )

                        loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                        if self.cfg.sae.multi_topk:
                            loss += out.multi_topk_fvu / 8
                        loss.div(acc_steps).backward()

                        # Update the did_fire mask
                        did_fire[name][out.latent_indices.flatten()] = True
                        self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                    # Clip gradient norm independently for each SAE
                    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

                # Perform optimizer step if necessary
                step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
                if substep == 0:
                    if self.cfg.sae.normalize_decoder:
                        for sae in self.saes.values():
                            sae.remove_gradient_parallel_to_decoder_directions()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    with torch.no_grad():
                        # Update the dead feature mask
                        for name, counts in self.num_tokens_since_fired.items():
                            counts += num_tokens_in_step
                            counts[did_fire[name]] = 0

                        # Reset stats for this step
                        num_tokens_in_step = 0
                        for mask in did_fire.values():
                            mask.zero_()

                    if (
                        self.cfg.log_to_wandb
                        and (step + 1) % self.cfg.wandb_log_frequency == 0
                    ):
                        info = {}

                        for name in self.saes:
                            mask = (
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                            )

                            info.update(
                                {
                                    f"fvu/{name}": avg_fvu[name],
                                    f"dead_pct/{name}": mask.mean(
                                        dtype=torch.float32
                                    ).item(),
                                }
                            )
                            if self.cfg.auxk_alpha > 0:
                                info[f"auxk/{name}"] = avg_auxk_loss[name]
                            if self.cfg.sae.multi_topk:
                                info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]

                        avg_auxk_loss.clear()
                        avg_fvu.clear()
                        avg_multi_topk_fvu.clear()

                        if self.cfg.distribute_modules:
                            outputs = [{} for _ in range(dist.get_world_size())]
                            dist.gather_object(info, outputs if rank_zero else None)
                            info.update({k: v for out in outputs for k, v in out.items()})

                        if rank_zero:
                            wandb.log(info, step=step)

                    if (step + 1) % self.cfg.save_every == 0:
                        self.save()

                self.global_step += 1

                # Update the progress bar only in rank 0
                if rank_zero:
                    pbar.update(1)

                # Synchronize at the end of each batch processing
                if dist.is_initialized():
                    dist.barrier()
                print(f"Synchronized at end of {batch_num}")
                batch_num += 1

        # Close the progress bar
        if rank_zero:
            pbar.close()

        # Save at the end
        if dist.is_initialized():
            print("Dist is initialized. Passing barrier")
            dist.barrier()
            print("Past barrier. Saving...")
            self.save()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        path = self.cfg.run_name or "sae-ckpts"
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Log the start of the save process
        print(f"Rank {rank} entering save()")

        if rank_zero or self.cfg.distribute_modules:
            print(f"Rank {rank} is saving checkpoints.")

            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)
                print(f"Rank {rank} is saving SAE for hook {hook}.")
                try:
                    sae.save_to_disk(f"{path}/{hook}")
                    print(f"Rank {rank} successfully saved SAE for hook {hook}.")
                except Exception as e:
                    print(f"Rank {rank} encountered an error while saving SAE for hook {hook}: {e}")
                    raise e  # Re-raise to not mask the exception

        if rank_zero:
            print(f"Rank {rank} is saving optimizer and scheduler.")
            try:
                torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
                torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
                torch.save({
                    "global_step": self.global_step,
                    "num_tokens_since_fired": self.num_tokens_since_fired,
                }, f"{path}/state.pt")
                self.cfg.save_json(f"{path}/config.json")
                print(f"Rank {rank} successfully saved optimizer, scheduler, and state.")
            except Exception as e:
                print(f"Rank {rank} encountered an error while saving state: {e}")
                raise e

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            print(f"Rank {rank} reached final barrier.")
            dist.barrier()
            print(f"Rank {rank} passed final barrier.") 
