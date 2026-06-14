"""
VRAM test for DDPMUnet with ch=128 on dual RTX 3090.

Tests peak VRAM usage at batch_size=32 (16 per GPU) with a realistic
forward + backward pass to confirm it fits within 24GB.

Usage:
    python test_vram.py
"""

import torch
from torch.nn.parallel import DataParallel
from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
from config_loader import TrainingConfig
import argparse
import os


def test_vram(model_channels: int = 128, batch_size: int = 32, img_size: int = 128):
    """Run a single forward+backward pass and report peak VRAM per GPU."""

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        print("[ERROR] No CUDA GPUs detected.")
        return

    use_dual = n_gpus >= 2

    print(f"{'='*60}")
    print(f"VRAM TEST")
    print(f"{'='*60}")
    print(f"  model_channels: {model_channels}")
    print(f"  batch_size:     {batch_size} ({'per GPU' if use_dual else 'total'})")
    print(f"  img_size:       {img_size}x{img_size}x3")
    print(f"  GPUs:           {n_gpus} ({'dual GPU' if use_dual else 'single GPU'})")
    print(f"  per_gpu_bs:     {batch_size // n_gpus if use_dual else batch_size}")
    print(f"{'='*60}")

    # Reset peak stats (each device needs an initial alloc to be valid)
    for i in range(n_gpus):
        with torch.cuda.device(i):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    device = torch.device("cuda:0")

    # Create model — exactly like train.py
    print("\n[1/5] Creating DDPMUnet...")
    ddpm_unet = DDPMUnet(img_channels=3, model_channels=model_channels)
    n_params = sum(p.numel() for p in ddpm_unet.parameters())
    print(f"      Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    print("[2/5] Creating DDPM wrapper + DiffusionConfig...")
    diffusion_config = DiffusionConfig(
        n_steps=1000,
        schedule_type="linear",
        device=str(device),
        b0=0.0001,
        bT=0.02,
    )

    ddpm = DDPM(1000, ddpm_unet, diffusion_config=diffusion_config)

    print("[3/5] Wrapping with DataParallel...")
    if use_dual:
        ddpm.eps_model = DataParallel(ddpm.eps_model, device_ids=list(range(n_gpus)))
        print(f"      DataParallel on {n_gpus} GPUs")

    ddpm = ddpm.to(device)

    # Report memory after model load
    print("\n[4/5] Model loaded. Memory after model:")
    for i in range(n_gpus):
        with torch.cuda.device(i):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"      GPU {i}: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

    # Forward + backward pass — exactly like train.py
    print(f"\n[5/5] Running forward + backward pass (bs={batch_size})...")
    ddpm.train()

    x_0 = torch.randn(batch_size, 3, img_size, img_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    eps = torch.randn_like(x_0)

    # Forward
    x_t = ddpm.q_sample(x_0, t, eps)
    eps_theta = ddpm.eps_model(x_t, t)
    loss = torch.nn.functional.mse_loss(eps_theta, eps)

    # Backward
    loss.backward()

    # Results
    print(f"\n{'='*60}")
    print(f"VRAM TEST RESULTS")
    print(f"{'='*60}")

    oom = False
    for i in range(n_gpus):
        with torch.cuda.device(i):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            peak = torch.cuda.max_memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
        status = "OK" if peak < 23.5 else "WARNING (tight)"
        if peak >= 24.0:
            status = "OOM RISK"
            oom = True
        print(f"  GPU {i}:")
        print(f"    Allocated:  {alloc:.2f} GB")
        print(f"    Peak:       {peak:.2f} GB  <- {status}")
        print(f"    Reserved:   {reserved:.2f} GB")

    print(f"{'='*60}")
    if oom:
        print("  >> RESULT: OOM RISK — reduce batch_size or model_channels")
    else:
        print("  >> RESULT: FITS — safe to proceed with this config")
    print(f"{'='*60}")

    # Cleanup
    del ddpm, x_0, t, eps, x_t, eps_theta, loss
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="VRAM test for DDPM training")
    parser.add_argument(
        "--ch", type=int, default=128, help="Model channels (default: 128)"
    )
    parser.add_argument(
        "--bs", type=int, default=32, help="Total batch size (default: 32)"
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="Image size (default: 128)"
    )
    args = parser.parse_args()

    test_vram(model_channels=args.ch, batch_size=args.bs, img_size=args.img_size)


if __name__ == "__main__":
    main()
