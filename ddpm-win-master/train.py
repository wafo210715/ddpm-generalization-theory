"""
Enhanced training script with config-based setup, dual GPU support, and resume.
Usage:
    python train.py                                        # Use default config
    python train.py --config configs/hcmc_vitality_v3.yaml  # Custom config
    python train.py --config configs/hcmc_vitality_v3.yaml --resume model_ckpts/hcmc_vitality_v3/xxx_step1000.pth  # Resume
"""

import torch
from torch import nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import argparse
import os
import csv
import glob
from tqdm.auto import tqdm

from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
from datasets.dataset import get_dataset
from config_loader import TrainingConfig


def save_model(ddpm: DDPM, optimizer: torch.optim.Optimizer, step: int,
               path: str, was_dual_gpu: bool):
    """Save model + optimizer + step for resumable training."""
    if was_dual_gpu and isinstance(ddpm.eps_model, DataParallel):
        eps_state = ddpm.eps_model.module.state_dict()
    else:
        eps_state = ddpm.eps_model.state_dict()
    torch.save({
        "step": step,
        "eps_model": eps_state,
        "optimizer": optimizer.state_dict(),
    }, path)


def load_model(ddpm: DDPM, optimizer: torch.optim.Optimizer, path: str,
               use_dual_gpu: bool) -> int:
    """Load model + optimizer from checkpoint. Returns the step number.

    Supports three checkpoint formats:
      1. New format: dict with "eps_model", "optimizer", "step" keys
      2. Legacy full DDPM: dict with "beta", "alpha", "eps_model" keys
      3. Old raw state_dict: OrderedDict of bare U-Net weights (no wrapper)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "eps_model" in ckpt:
        # Format 1 or 2: wrapped in a dict with eps_model key
        eps_state = ckpt["eps_model"]
        if use_dual_gpu and isinstance(ddpm.eps_model, DataParallel):
            ddpm.eps_model.module.load_state_dict(eps_state)
        else:
            ddpm.eps_model.load_state_dict(eps_state)
    elif isinstance(ckpt, dict) and "beta" in ckpt:
        # Format 2: legacy full DDPM state_dict
        eps_state = {k.replace("eps_model.", ""): v for k, v in ckpt.items()
                     if k.startswith("eps_model.")}
        if use_dual_gpu and isinstance(ddpm.eps_model, DataParallel):
            eps_state = {f"module.{k}": v for k, v in eps_state.items()}
            ddpm.eps_model.load_state_dict(eps_state)
        else:
            ddpm.eps_model.load_state_dict(eps_state)
    elif isinstance(ckpt, dict) and not ckpt:
        raise ValueError(f"Empty checkpoint: {path}")
    else:
        # Format 3: raw OrderedDict — bare U-Net state_dict
        if use_dual_gpu and isinstance(ddpm.eps_model, DataParallel):
            ddpm.eps_model.module.load_state_dict(ckpt)
        else:
            ddpm.eps_model.load_state_dict(ckpt)

    step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
    # Fallback: extract step from filename (e.g. "model_step500.pth" -> 500)
    if step == 0:
        import re
        match = re.search(r"_step(\d+)\.pth$", path)
        if match:
            step = int(match.group(1))
            print(f"[RESUME] Inferred step={step} from filename")
    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return step


def train_with_config(config: TrainingConfig, resume_from: str = None):
    """Train DDPM model with given configuration."""

    print("=" * 60)
    print("DDPM TRAINING WITH CONFIG")
    print("=" * 60)
    config.print_summary()

    # Device setup
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")

    # Check dual GPU availability
    if config.use_dual_gpu:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[WARNING] use_dual_gpu=True but only {n_gpus} GPU(s) available.")
            print(f"[WARNING] Falling back to single GPU.")
            config.use_dual_gpu = False
        else:
            print(f"Enabled dual GPU training on: GPU 0 + GPU 1")

    # Load dataset
    print(f"\nLoading {config.dataset_name} dataset...")
    # Resolve data_root relative to this script's directory
    data_dir = None
    if hasattr(config, "data_root") and config.data_root:
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), config.data_root)
        )
    dataset = get_dataset(
        config.dataset_name,
        img_size=config.img_size,
        data_dir=data_dir,
        center=config.center,
        normalize=config.normalize,
        channels=config.channels,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Validation split
    val_dataset = get_dataset(
        config.dataset_name,
        img_size=config.img_size,
        data_dir=data_dir,
        split="val",
        center=config.center,
        normalize=config.normalize,
        channels=config.channels,
    )
    if len(val_dataset) == 0:
        print("[WARNING] Validation dataset is empty — skipping validation tracking")
        val_dataloader = None
    else:
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        print(f"Validation dataset loaded: {len(val_dataset)} samples")

    # Create diffusion config
    print("\nCreating diffusion configuration...")

    # Only pass parameters relevant to the specific schedule type
    schedule_kwargs = {
        "n_steps": config.timesteps,
        "schedule_type": config.schedule_type,
        "device": config.device,
        "shift_schedule": config.shift_schedule,
        "img_size": config.img_size,
        "ref_img_size": config.ref_img_size,
    }

    if config.schedule_type == "linear":
        schedule_kwargs["b0"] = config.b0
        schedule_kwargs["bT"] = config.bT
    elif config.schedule_type == "fixed":
        schedule_kwargs["beta"] = config.beta
    elif config.schedule_type == "vp":
        schedule_kwargs["beta_d"] = config.beta_d
        schedule_kwargs["beta_min"] = config.beta_min
    elif config.schedule_type == "ve":
        schedule_kwargs["sigma_min"] = config.sigma_min
        schedule_kwargs["sigma_max"] = config.sigma_max

    diffusion_config = DiffusionConfig(**schedule_kwargs)
    diffusion_config.print_summary()

    # Create model
    print("Creating DDPM model...")
    ddpm_unet = DDPMUnet(config.img_channels, config.model_channels)
    ddpm = DDPM(config.timesteps, ddpm_unet, diffusion_config=diffusion_config)

    # Wrap with DataParallel if dual GPU enabled
    if config.use_dual_gpu:
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            ddpm.eps_model = DataParallel(
                ddpm.eps_model, device_ids=list(range(n_gpus))
            )
            print(f"Model wrapped with DataParallel ({n_gpus} GPUs)")
        else:
            print(
                f"Warning: use_dual_gpu=true but only {n_gpus} GPU(s) found. Using single GPU."
            )

    ddpm = ddpm.to(device)
    print("Model moved to device")

    # Setup training
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, f"{config.run_name}.pth")

    # --- Training log setup ---
    log_dir = os.path.join(config.log_dir, config.run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_csv_path = os.path.join(log_dir, "training_log.csv")
    log_file = open(log_csv_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "val_loss"])
    log_file.flush()
    print(f"Training logs: {log_dir}")

    # Resume from checkpoint if provided
    start_step = 0
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        print(f"\n[RESUME] Loading checkpoint: {resume_from}")
        start_step = load_model(ddpm, optimizer, resume_from, config.use_dual_gpu)
        print(f"[RESUME] Resuming from step {start_step}")
        # Skip already-saved checkpoints
        print(f"[RESUME] Will train from step {start_step + 1} to {config.n_steps}")

    # Training loop (step-based, not epoch-based)
    remaining = config.n_steps - start_step
    print(f"\nTraining for {remaining} steps (step {start_step} -> {config.n_steps})...")
    ddpm.train()

    step_count = 0
    losses = []
    checkpoint_steps = config.get_all_checkpoints()
    next_checkpoint_idx = 0
    # Skip checkpoints already saved at or before start_step
    while (next_checkpoint_idx < len(checkpoint_steps)
           and checkpoint_steps[next_checkpoint_idx] <= start_step):
        next_checkpoint_idx += 1
    if next_checkpoint_idx > 0:
        print(f"[RESUME] Skipping {next_checkpoint_idx} already-saved checkpoints")

    data_iter = iter(dataloader)

    try:
        pbar = tqdm(range(start_step + 1, config.n_steps + 1), desc="Training",
                    initial=start_step, total=config.n_steps)
        for step in pbar:
            # Get batch (handle dataloader exhaustion)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Handle dataset that returns tuples (MNIST) vs single tensors vs dicts
            x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x_0, dict):
                # "tensor" for multi-channel cell stores, "image" for the PNG datasets
                x_0 = x_0["tensor"] if "tensor" in x_0 else x_0["image"]
            x_0 = x_0.to(device)

            # Sample timestep and noise
            t = torch.randint(0, ddpm.timesteps, (x_0.shape[0],), device=device)
            eps = torch.randn_like(x_0, device=device)

            # Forward pass
            x_t = ddpm.q_sample(x_0, t, eps)
            eps_theta = ddpm.eps_model(x_t, t)
            loss = loss_fn(eps_theta, eps)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\n[WARNING] NaN loss detected at step {step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            losses.append(loss.item())
            step_count += 1

            # Update progress bar
            if step % config.log_every == 0:
                avg_loss = sum(losses[-config.log_every :]) / min(
                    len(losses), config.log_every
                )
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
                log_writer.writerow([step, f"{avg_loss:.6f}", ""])
                log_file.flush()

            # Validation loss (every val_every steps, no gradients)
            if val_dataloader is not None and step % config.val_every == 0:
                ddpm.eval()
                val_losses = []
                n_val_batches = min(config.val_steps, len(val_dataloader))
                with torch.no_grad():
                    val_iter = iter(val_dataloader)
                    for _ in range(n_val_batches):
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            break
                        val_x0 = (
                            val_batch[0]
                            if isinstance(val_batch, (list, tuple))
                            else val_batch
                        )
                        if isinstance(val_x0, dict):
                            val_x0 = (
                                val_x0["tensor"] if "tensor" in val_x0 else val_x0["image"]
                            )
                        val_x0 = val_x0.to(device)
                        val_t = torch.randint(
                            0, ddpm.timesteps, (val_x0.shape[0],), device=device
                        )
                        val_eps = torch.randn_like(val_x0, device=device)
                        val_x_t = ddpm.q_sample(val_x0, val_t, val_eps)
                        val_eps_theta = ddpm.eps_model(val_x_t, val_t)
                        val_losses.append(loss_fn(val_eps_theta, val_eps).item())
                ddpm.train()

                if val_losses:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    print(
                        f"[VAL] step {step}: val_loss={avg_val_loss:.6f}"
                    )
                    recent_loss = sum(losses[-config.log_every :]) / min(
                        len(losses), config.log_every
                    )
                    pbar.set_postfix(
                        {"loss": f"{recent_loss:.6f}", "val_loss": f"{avg_val_loss:.6f}"}
                    )
                    log_writer.writerow([step, f"{recent_loss:.6f}", f"{avg_val_loss:.6f}"])
                    log_file.flush()

            # Save checkpoints
            if (
                next_checkpoint_idx < len(checkpoint_steps)
                and step == checkpoint_steps[next_checkpoint_idx]
            ):
                checkpoint_file = checkpoint_path.replace(".pth", f"_step{step}.pth")
                save_model(ddpm, optimizer, step, checkpoint_file, config.use_dual_gpu)
                print(f"\n[CHECKPOINT] Saved at step {step}: {checkpoint_file}")
                next_checkpoint_idx += 1

        # Final summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total steps: {step_count}")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Average loss: {sum(losses) / len(losses):.6f}")

        # Save final model
        save_model(ddpm, optimizer, config.n_steps, checkpoint_path, config.use_dual_gpu)
        print(f"\nModel saved: {checkpoint_path}")

        # GPU memory info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} memory allocated: {mem:.2f} GB")
    finally:
        log_file.close()

    # Write training summary
    summary_path = os.path.join(log_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Training Summary: {config.run_name}\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Config: dataset={config.dataset_name}, img_size={config.img_size}, "
                f"model_channels={config.model_channels}, lr={config.lr}\n")
        f.write(f"Total steps: {step_count}\n")
        f.write(f"Final loss: {losses[-1]:.6f}\n")
        f.write(f"Average loss: {sum(losses) / len(losses):.6f}\n")
        f.write(f"Min loss: {min(losses):.6f}\n")
        f.write(f"Max loss: {max(losses):.6f}\n")
        f.write(f"Checkpoints saved: {next_checkpoint_idx}\n")
    print(f"Training summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DDPM with config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth to resume training from",
    )
    args = parser.parse_args()

    try:
        # Load config from YAML
        config = TrainingConfig.from_yaml(args.config)
        train_with_config(config, resume_from=args.resume)

    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
