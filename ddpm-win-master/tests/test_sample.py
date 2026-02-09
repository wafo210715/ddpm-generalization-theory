"""
Test training + sampling pipeline.
Trains for 50 steps, saves model, then generates samples.
"""

import sys
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

# Add current directory to path
sys.path.insert(0, '.')

from model.unet import DDPMUnet
from model.ddpm import DDPM
from datasets.dataset import get_dataset


def quick_train(n_steps=50):
    """Train model for n_steps and save checkpoint."""
    print("=" * 60)
    print(f"TRAINING ({n_steps} steps)")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading MNIST dataset...")
    dataset = get_dataset("mnist", img_size=32)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create model
    print("\nCreating model...")
    ddpm_unet = DDPMUnet(1, 64)
    ddpm = DDPM(1000, ddpm_unet)
    ddpm.to(device)

    # Training
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining for {n_steps} steps...")
    ddpm.train()

    data_iter = iter(dataloader)
    losses = []

    for step in tqdm(range(n_steps), desc="Training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_0 = x_0.to(device)
        t = torch.randint(0, ddpm.timesteps, (x_0.shape[0],), device=device)
        eps = torch.randn_like(x_0, device=device)

        x_t = ddpm.q_sample(x_0, t, eps)
        eps_theta = ddpm.eps_model(x_t, t)
        loss = loss_fn(eps_theta, eps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"\nTraining complete! Average loss: {avg_loss:.6f}")

    # Save model
    os.makedirs("model_ckpts", exist_ok=True)
    torch.save(ddpm.state_dict(), "model_ckpts/ddpm.pth")
    print("Model saved to: model_ckpts/ddpm.pth")

    return ddpm


def sample_images(ddpm, num_samples=9, device="cuda"):
    """Generate samples from trained model."""
    print("\n" + "=" * 60)
    print(f"SAMPLING ({num_samples} samples)")
    print("=" * 60)

    ddpm.eval()
    with torch.no_grad():
        x_t = torch.randn((num_samples, 1, 32, 32), device=device)

        print("\nRunning diffusion reverse process...")
        for t in tqdm(range(ddpm.timesteps - 1, -1, -1), desc="Denoising"):
            if t > 0:
                z = torch.randn_like(x_t, device=device)
            else:
                z = torch.zeros_like(x_t, device=device)

            t_tensor = torch.full((num_samples,), t, device=device)
            x_t = ddpm.p_sample(x_t, t_tensor, z)

        x_t = x_t.clamp(0, 1)

    print("\nSamples generated!")
    return x_t


def main():
    print("=" * 60)
    print("DDPM TRAINING + SAMPLING TEST")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Step 1: Quick training
    ddpm = quick_train(n_steps=50)

    # Step 2: Sample from trained model
    samples = sample_images(ddpm, num_samples=9, device=device)

    # Step 3: Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nSample range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"Sample mean: {samples.mean():.3f}")
    print(f"Sample shape: {samples.shape}")

    # Save a few sample images for visual inspection
    try:
        from torchvision import transforms

        transform = transforms.ToPILImage()
        os.makedirs("test_samples", exist_ok=True)

        print("\nSaving sample images to test_samples/...")
        for i in range(min(4, samples.shape[0])):
            img = transform(samples[i].cpu())
            img.save(f"test_samples/sample_{i}.png")
            print(f"  Saved: test_samples/sample_{i}.png")

        print("\n[SUCCESS] All tests passed!")
        print("\nGenerated images saved to: test_samples/")
        print("Check these images to verify generation quality.")

    except Exception as e:
        print(f"\n[WARNING] Could not save images: {e}")
        print("But sampling still worked!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
