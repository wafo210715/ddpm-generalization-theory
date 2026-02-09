"""
Test dual GPU memory usage with DataParallel (like actual training).
"""

import torch
import torch.nn as nn
import gc


def test_dual_gpu_memory(
    img_size=32,
    img_channels=1,
    channels=64,
    batch_size=128,
    timesteps=1000
):
    """Test memory usage with DataParallel like actual training."""

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")

    for gpu_id in range(num_gpus):
        torch.cuda.empty_cache()

    from model.unet import DDPMUnet
    from model.ddpm import DDPM

    device = torch.device("cuda:0")

    print(f"\nCreating model (channels={channels}, batch_size={batch_size})...")

    # Create model
    ddpm_unet = DDPMUnet(img_channels, channels)

    # Wrap with DataParallel (like actual training)
    if num_gpus > 1:
        print(f"Wrapping with DataParallel (GPUs: {[0, 1]})...")
        ddpm_unet = nn.DataParallel(ddpm_unet, device_ids=[0, 1])

    ddpm = DDPM(timesteps, ddpm_unet)
    ddpm = ddpm.to(device)

    # Get memory after model creation
    print(f"After model creation:")
    for gpu_id in range(num_gpus):
        mem = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        print(f"  GPU {gpu_id}: {mem:.3f} GB")

    # Create dummy batch
    x_0 = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
    t = torch.randint(0, timesteps, (batch_size,), device=device)
    eps = torch.randn_like(x_0, device=device)

    # Forward pass
    print(f"\nAfter forward pass:")
    x_t = ddpm.q_sample(x_0, t, eps)
    eps_theta = ddpm.eps_model(x_t, t)
    loss = torch.nn.functional.mse_loss(eps_theta, eps)

    for gpu_id in range(num_gpus):
        mem = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        print(f"  GPU {gpu_id}: {mem:.3f} GB")

    # Backward pass
    print(f"\nAfter backward pass:")
    loss.backward()

    for gpu_id in range(num_gpus):
        mem = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        print(f"  GPU {gpu_id}: {mem:.3f} GB")

    # Optimizer step
    print(f"\nAfter optimizer step:")
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-4)
    optimizer.step()

    for gpu_id in range(num_gpus):
        mem = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        print(f"  GPU {gpu_id}: {mem:.3f} GB")

    # Peak memory
    print(f"\nPeak allocated:")
    for gpu_id in range(num_gpus):
        peak = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
        print(f"  GPU {gpu_id}: {peak:.3f} GB")

    # Reserved memory
    print(f"\nReserved (pytorch cache):")
    for gpu_id in range(num_gpus):
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
        print(f"  GPU {gpu_id}: {reserved:.3f} GB")


if __name__ == "__main__":
    print("=" * 70)
    print("DUAL GPU MEMORY TEST (DataParallel)")
    print("=" * 70)

    # Test with default config
    test_dual_gpu_memory(
        img_size=32,
        img_channels=1,
        channels=64,
        batch_size=128
    )
