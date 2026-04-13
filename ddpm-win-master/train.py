"""
Enhanced training script with config-based setup and dual GPU support.
Usage:
    python train.py                           # Use default config
    python train.py --config configs/mnist.yaml
    python train.py --config configs/geospatial.yaml
"""

import torch
from torch import nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import argparse
import os
from tqdm.auto import tqdm

from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
from datasets.dataset import get_dataset
from config_loader import TrainingConfig


def train_with_config(config: TrainingConfig):
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
        config.dataset_name, img_size=config.img_size, data_dir=data_dir
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create diffusion config
    print("\nCreating diffusion configuration...")

    # Only pass parameters relevant to the specific schedule type
    schedule_kwargs = {
        "n_steps": config.timesteps,
        "schedule_type": config.schedule_type,
        "device": config.device,
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

    # Training loop (step-based, not epoch-based)
    print(f"\nTraining for {config.n_steps} steps...")
    ddpm.train()

    step_count = 0
    losses = []
    checkpoint_steps = config.get_all_checkpoints()
    next_checkpoint_idx = 0

    data_iter = iter(dataloader)

    pbar = tqdm(range(1, config.n_steps + 1), desc="Training")
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
            x_0 = x_0["image"]
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

        # Save checkpoints
        if (
            next_checkpoint_idx < len(checkpoint_steps)
            and step == checkpoint_steps[next_checkpoint_idx]
        ):
            checkpoint_file = checkpoint_path.replace(".pth", f"_step{step}.pth")
            save_model(ddpm, checkpoint_file, config.use_dual_gpu)
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
    save_model(ddpm, checkpoint_path, config.use_dual_gpu)
    print(f"\nModel saved: {checkpoint_path}")

    # GPU memory info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i} memory allocated: {mem:.2f} GB")


def save_model(ddpm: DDPM, path: str, was_dual_gpu: bool):
    """Save model, unwrapping DataParallel if needed."""
    # Unwrap DataParallel before saving
    if was_dual_gpu and isinstance(ddpm.eps_model, DataParallel):
        # Save only the underlying module
        torch.save(ddpm.eps_model.module.state_dict(), path)
    else:
        torch.save(ddpm.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description="Train DDPM with config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    try:
        # Load config from YAML
        config = TrainingConfig.from_yaml(args.config)
        train_with_config(config)

    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
