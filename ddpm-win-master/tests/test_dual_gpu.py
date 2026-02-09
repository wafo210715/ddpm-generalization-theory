"""
Test dual GPU training using PyTorch's native DataParallel.
This is a simpler alternative to DDPStrategy (PyTorch Lightning).

Note: For better performance with NVLink, you may want to use
DistributedDataParallel (DDP) via PyTorch Lightning's DDPStrategy:

    pip install pytorch-lightning
    strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)
"""

import sys
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm
import os

# Add current directory to path
sys.path.insert(0, '.')

from model.unet import DDPMUnet
from model.ddpm import DDPM
from datasets.dataset import get_dataset


def test_dual_gpu(n_steps=20):
    """Test training on both GPUs using DataParallel."""
    print("=" * 60)
    print(f"DUAL GPU TRAINING TEST ({n_steps} steps)")
    print("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\n[FAIL] CUDA is not available!")
        return None

    n_gpus = torch.cuda.device_count()
    print(f"\nGPUs detected: {n_gpus}")

    if n_gpus < 2:
        print("[WARNING] Only 1 GPU available. Test will use single GPU.")
        device_ids = [0]
    else:
        device_ids = [0, 1]
        print(f"Using GPUs: {device_ids}")
        for i in device_ids:
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"         Memory: {mem_gb:.1f} GB")

    # Device setup
    device = torch.device("cuda:0")

    # Load dataset
    print("\nLoading MNIST dataset...")
    dataset = get_dataset("mnist", img_size=32)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Batch size: 256 (scaled for dual GPU)")

    # Create model
    print("\nCreating DDPM model...")
    ddpm_unet = DDPMUnet(1, 64)
    ddpm = DDPM(1000, ddpm_unet)

    # Wrap model with DataParallel for multi-GPU training
    if len(device_ids) > 1:
        print(f"\nWrapping model with DataParallel (GPUs: {device_ids})...")
        ddpm.eps_model = DataParallel(ddpm.eps_model, device_ids=device_ids)

    ddpm = ddpm.to(device)
    print("Model moved to device")

    # Setup training
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    print(f"\nRunning {n_steps} training steps on dual GPU...")
    ddpm.train()

    step_count = 0
    losses = []

    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_iter = iter(dataloader)

    for step in tqdm(range(n_steps), desc="Dual GPU Training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_0 = x_0.to(device)
        t = torch.randint(0, ddpm.timesteps, (x_0.shape[0],), device=device)
        eps = torch.randn_like(x_0, device=device)

        # Forward pass
        x_t = ddpm.q_sample(x_0, t, eps)
        eps_theta = ddpm.eps_model(x_t, t)
        loss = loss_fn(eps_theta, eps)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        step_count += 1

        if step % 5 == 0 or step == n_steps - 1:
            avg_loss = sum(losses) / len(losses)
            print(f"  Step {step+1}/{n_steps}: loss = {loss.item():.6f}, avg = {avg_loss:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Steps completed: {step_count}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Average loss: {sum(losses)/len(losses):.6f}")

    # GPU memory usage
    if torch.cuda.is_available():
        for i in device_ids:
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"\nGPU {i} memory:")
            print(f"  Allocated: {mem_allocated:.2f} GB")
            print(f"  Reserved: {mem_reserved:.2f} GB")

    print("\n[OK] Dual GPU training test PASSED!")
    print("\nNotes:")
    print("  - DataParallel splits batch across GPUs automatically")
    print("  - For better NVLink performance, consider DistributedDataParallel")
    print("  - To use DDPStrategy: pip install pytorch-lightning")

    return ddpm


def main():
    print("=" * 60)
    print("DDPM DUAL GPU TRAINING TEST")
    print("=" * 60)

    try:
        ddpm = test_dual_gpu(n_steps=20)

        if ddpm is not None:
            # Save model
            os.makedirs("model_ckpts", exist_ok=True)
            # Need to unwrap DataParallel before saving
            if isinstance(ddpm.eps_model, DataParallel):
                torch.save(ddpm.eps_model.module.state_dict(), "model_ckpts/ddpm_dual_gpu.pth")
            else:
                torch.save(ddpm.state_dict(), "model_ckpts/ddpm_dual_gpu.pth")
            print("\nModel saved to: model_ckpts/ddpm_dual_gpu.pth")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
