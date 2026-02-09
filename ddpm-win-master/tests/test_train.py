"""
Quick test for train.py - runs for just 10 steps to verify everything works.
This is much faster than running the full training.
"""

import sys
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import from the existing codebase
sys.path.insert(0, '.')
from model.unet import DDPMUnet
from model.ddpm import DDPM
from datasets.dataset import get_dataset


def quick_train_test(n_steps=10):
    """Test training loop for n_steps."""
    print("=" * 60)
    print(f"QUICK TRAIN TEST ({n_steps} steps)")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load dataset
    print("\nLoading MNIST dataset...")
    dataset = get_dataset("mnist", img_size=32)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create model
    print("\nCreating DDPM model...")
    ddpm_unet = DDPMUnet(1, 64)
    ddpm = DDPM(1000, ddpm_unet)
    ddpm.to(device)
    print("Model created and moved to device")

    # Setup training
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Training loop (quick test)
    print(f"\nRunning {n_steps} training steps...")
    ddpm.train()

    step_count = 0
    losses = []

    data_iter = iter(dataloader)

    for step in range(n_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # MNIST returns (images, labels) - we only need images
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

    # Test GPU memory
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU memory allocated: {mem_allocated:.2f} GB")
        print(f"GPU memory reserved: {mem_reserved:.2f} GB")

    print("\n[OK] Training test PASSED!")
    print("\nNext: You can run full training with:")
    print("  python train.py")
    print("\nOr generate samples with:")
    print("  python sample.py")


if __name__ == "__main__":
    try:
        quick_train_test(n_steps=10)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
