"""
GPU Memory Estimation for DDPM Training.
Actually measures real GPU memory usage by creating the model and running a forward pass.

Usage:
    python estimate_memory_usage.py
    python estimate_memory_usage.py --channels 128 --batch-size 256 --img-size 64
    python estimate_memory_usage.py --config configs/geospatial.yaml
"""

import torch
import argparse
import gc
import sys


def measure_real_memory(
    img_size=32,
    img_channels=1,
    channels=64,
    batch_size=128,
    num_res_blocks=2,
    timesteps=1000,
    device="cuda"
):
    """
    Measure actual GPU memory usage by creating model and running forward pass.

    Args:
        img_size: Image resolution (assumes square images)
        img_channels: Number of input/output channels
        channels: Base channel dimension for U-Net
        batch_size: Training batch size
        num_res_blocks: Number of residual blocks
        timesteps: Number of diffusion timesteps
        device: GPU device to use

    Returns:
        dict with memory measurements for each GPU
    """
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available!")
        return None

    num_gpus = torch.cuda.device_count()
    print(f"\nFound {num_gpus} GPU(s)")

    results = {}

    for gpu_id in range(num_gpus):
        torch.cuda.empty_cache()
        gc.collect()

        current_device = torch.device(f"cuda:{gpu_id}")

        try:
            # Import models
            from model.unet import DDPMUnet
            from model.ddpm import DDPM

            print(f"\nTesting on GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})...")

            # Create model
            ddpm_unet = DDPMUnet(img_channels, channels)
            ddpm = DDPM(timesteps, ddpm_unet)
            ddpm = ddpm.to(current_device)

            # Get memory after model creation
            after_model = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)

            # Create dummy batch
            x_0 = torch.randn(batch_size, img_channels, img_size, img_size, device=current_device)
            t = torch.randint(0, timesteps, (batch_size,), device=current_device)
            eps = torch.randn_like(x_0, device=current_device)

            # Forward pass
            x_t = ddpm.q_sample(x_0, t, eps)
            eps_theta = ddpm.eps_model(x_t, t)

            # Create loss outside no_grad to ensure gradients are computed
            loss = torch.nn.functional.mse_loss(eps_theta, eps)

            # Get memory after forward pass
            after_forward = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)

            # Backward pass (to get gradient memory)
            loss.backward()

            # Get memory after backward pass
            after_backward = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)

            # Optimizer (to get optimizer state memory)
            optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-4)
            optimizer.step()

            # Get memory after optimizer step
            after_optimizer = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)

            # Peak memory during the process
            peak_memory = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
            reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)

            results[gpu_id] = {
                'name': torch.cuda.get_device_name(gpu_id),
                'after_model_gb': after_model,
                'after_forward_gb': after_forward,
                'after_backward_gb': after_backward,
                'after_optimizer_gb': after_optimizer,
                'peak_allocated_gb': peak_memory,
                'reserved_gb': reserved_memory
            }

            # Clean up
            del ddpm, ddpm_unet, x_0, t, eps, x_t, eps_theta, loss, optimizer
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  [OOM] GPU {gpu_id} ran out of memory!")
                results[gpu_id] = {'error': 'OOM', 'name': torch.cuda.get_device_name(gpu_id)}
            else:
                raise

    return results


def print_memory_results(results, available_gb=24):
    """Print measured memory results."""
    print("\n" + "=" * 80)
    print("REAL GPU MEMORY MEASUREMENT")
    print("=" * 80)

    total_used = 0

    for gpu_id, result in results.items():
        if 'error' in result:
            print(f"\nGPU {gpu_id} ({result['name']}): {result['error']}")
            continue

        print(f"\nGPU {gpu_id}: {result['name']}")
        print("-" * 80)
        print(f"  After model creation:    {result['after_model_gb']:.3f} GB")
        print(f"  After forward pass:      {result['after_forward_gb']:.3f} GB")
        print(f"  After backward pass:     {result['after_backward_gb']:.3f} GB")
        print(f"  After optimizer step:    {result['after_optimizer_gb']:.3f} GB")
        print(f"  Peak allocated:          {result['peak_allocated_gb']:.3f} GB")
        print(f"  Reserved (pytorch):      {result['reserved_gb']:.3f} GB")

        total_used += result['peak_allocated_gb']

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total peak allocated:     {total_used:.3f} GB")
    print(f"Available per GPU:        {available_gb:.1f} GB")

    num_gpus = len(results)
    usage_pct = (total_used / (available_gb * num_gpus)) * 100
    print(f"Usage percentage:         {usage_pct:.1f}%")

    if usage_pct < 50:
        print(f"\nStatus: [OK] Plenty of memory headroom")
    elif usage_pct < 75:
        print(f"\nStatus: [GOOD] Memory usage is reasonable")
    elif usage_pct < 90:
        print(f"\nStatus: [WARNING] Memory usage is getting high")
    else:
        print(f"\nStatus: [ALERT] Approaching GPU memory limit!")


def main():
    parser = argparse.ArgumentParser(description="Measure actual GPU memory usage for DDPM training")
    parser.add_argument("--img-size", type=int, default=32, help="Image resolution (default: 32)")
    parser.add_argument("--img-channels", type=int, default=1, help="Image channels (1=grayscale, 3=RGB)")
    parser.add_argument("--channels", type=int, default=64, help="Base model channels (default: 64)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128)")
    parser.add_argument("--gpu-memory", type=float, default=24, help="Available GPU memory in GB per GPU (default: 24)")
    parser.add_argument("--config", type=str, help="Load settings from config YAML file")
    parser.add_argument("--single-gpu", action="store_true", help="Test only on single GPU (GPU 0)")

    args = parser.parse_args()

    # If config is provided, load it
    if args.config:
        try:
            from config_loader import TrainingConfig
            config = TrainingConfig.from_yaml(args.config)

            # Override args with config values
            args.img_size = config.img_size
            args.img_channels = config.img_channels
            args.batch_size = config.batch_size

            print(f"Loaded config from: {args.config}")
            print(f"  Dataset: {config.dataset_name}")
            print(f"  Image size: {args.img_size}x{args.img_size}")
            print(f"  Channels: {args.img_channels}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Dual GPU: {config.use_dual_gpu}")
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using command-line arguments instead.")

    print("\n" + "=" * 80)
    print("DDPM GPU MEMORY MEASUREMENT TOOL")
    print("=" * 80)
    print("This will create the model and run a forward/backward pass to measure real GPU memory.")

    # Determine which GPUs to test
    if args.single_gpu:
        print("\nTesting on GPU 0 only (--single-gpu)")
        # Temporarily set CUDA to only see GPU 0
        original_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Measure memory
    results = measure_real_memory(
        img_size=args.img_size,
        img_channels=args.img_channels,
        channels=args.channels,
        batch_size=args.batch_size
    )

    if args.single_gpu and original_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_visible

    # Print results
    if results:
        print_memory_results(results, available_gb=args.gpu_memory)


if __name__ == "__main__":
    import os
    main()
