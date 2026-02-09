"""
Visual comparison of schedule comparison experiments.

Generates side-by-side sample grids and plots loss curves from training logs.
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import os
from tqdm.auto import tqdm

from model.unet import DDPMUnet
from model.ddpm import DDPM
from util.img_utils import show_images, save_individual_images


def generate_samples(ddpm: DDPM, num_samples: int, device: torch.device) -> list[Image.Image]:
    """Generate samples from a trained DDPM model."""
    ddpm.eval()
    with torch.no_grad():
        x_t = torch.randn((num_samples, 1, 32, 32), device=device)

        for t in tqdm(range(ddpm.timesteps - 1, -1, -1), desc="Sampling"):
            if t > 0:
                z = torch.randn_like(x_t, device=device)
            else:
                z = torch.zeros_like(x_t, device=device)

            t_tensor = torch.full((num_samples,), t, device=device)
            x_t = ddpm.p_sample(x_t, t_tensor, z)

        x_t = x_t.clamp(0, 1)
        transform = transforms.ToPILImage()
        img_list = []
        for i in range(num_samples):
            img = transform(x_t[i].cpu())
            img_list.append(img)
        return img_list


def compare_schedules_visually(checkpoint1: str, checkpoint2: str,
                              name1: str, name2: str, device: torch.device):
    """
    Generate side-by-side comparison of two trained schedules.

    Creates a 3x6 grid:
    - Row 1: Schedule 1 samples (9 images)
    - Row 2: Schedule 2 samples (9 images)
    - Row 3: Labels
    """
    print(f"\n{'=' * 60}")
    print(f"VISUAL COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 60}")

    # Load both models
    print(f"\nLoading {name1}...")
    ckpt1 = torch.load(checkpoint1, map_location=device)
    unet1 = DDPMUnet(1, 64)
    ddpm1 = DDPM(1000, unet1)

    if "eps_model" in ckpt1 or "beta" in ckpt1:
        ddpm1.load_state_dict(ckpt1)
    else:
        ddpm1.eps_model.load_state_dict(ckpt1)
    ddpm1 = ddpm1.to(device)

    print(f"Loading {name2}...")
    ckpt2 = torch.load(checkpoint2, map_location=device)
    unet2 = DDPMUnet(1, 64)
    ddpm2 = DDPM(1000, unet2)

    if "eps_model" in ckpt2 or "beta" in ckpt2:
        ddpm2.load_state_dict(ckpt2)
    else:
        ddpm2.eps_model.load_state_dict(ckpt2)
    ddpm2 = ddpm2.to(device)

    # Generate samples
    print(f"\nGenerating samples for {name1}...")
    samples1 = generate_samples(ddpm1, 9, device)

    print(f"Generating samples for {name2}...")
    samples2 = generate_samples(ddpm2, 9, device)

    # Create comparison grid
    print(f"\nCreating comparison grid...")
    fig, axes = plt.subplots(3, 9, figsize=(18, 6))
    fig.suptitle(f'Schedule Comparison: {name1} vs {name2}', fontsize=16)

    # Row 1: Schedule 1
    for i, img in enumerate(samples1):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
    axes[0, 4].set_title(f'{name1}', fontsize=14, fontweight='bold')

    # Row 2: Schedule 2
    for i, img in enumerate(samples2):
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
    axes[1, 4].set_title(f'{name2}', fontsize=14, fontweight='bold')

    # Row 3: Labels
    for i in range(9):
        axes[2, i].text(0.5, 0.5, f'Sample {i+1}',
                       ha='center', va='center', fontsize=12)
        axes[2, i].axis('off')

    plt.tight_layout()

    # Save comparison (use script directory as base)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "model_ckpts", "schedule_comparison", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name1}_vs_{name2}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Visual comparison: {output_path}")

    # Also save individual sample grids
    save_individual_images(samples1, os.path.join(output_dir, f"{name1}_samples"))
    save_individual_images(samples2, os.path.join(output_dir, f"{name2}_samples"))

    plt.close()

    return output_path


def plot_beta_schedules_comparison():
    """
    Plot and compare all noise schedule types.

    Visualizes how noise level progresses for each schedule type.
    """
    print(f"\n{'=' * 60}")
    print("BETA SCHEDULE COMPARISON")
    print(f"{'=' * 60}")

    from model.diffusion_config import DiffusionConfig

    schedules = ["linear", "cosine", "fixed", "vp", "ve"]
    configs = {}

    for sched in schedules:
        configs[sched] = DiffusionConfig(n_steps=1000, schedule_type=sched, device="cpu")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Noise Schedule Comparison', fontsize=16)

    # Plot 1: Beta values
    for sched in schedules:
        cfg = configs[sched]
        axes[0].plot(cfg.beta, label=sched, linewidth=2)
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Beta (β)')
    axes[0].set_title('Beta Schedule')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Alpha cumulative (signal preservation)
    for sched in schedules:
        cfg = configs[sched]
        axes[1].plot(cfg.alpha_cumulative, label=sched, linewidth=2)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Alpha Bar (ᾱ)')
    axes[1].set_title('Signal Preservation (Higher = More Signal)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save (use script directory as base)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "model_ckpts", "schedule_comparison", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "beta_schedule_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Beta schedule comparison: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nSchedule Statistics (at key checkpoints):")
    print(f"{'Schedule':<12} {'β_start':<12} {'β_end':<12} {'ᾱ_1000':<12}")
    print("-" * 50)
    for sched in schedules:
        cfg = configs[sched]
        print(f"{sched:<12} {cfg.beta[0]:<12.6f} {cfg.beta[-1]:<12.6f} {cfg.alpha_cumulative[-1]:<12.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare trained schedules")
    parser.add_argument("--checkpoint1", type=str,
                       default="model_ckpts/schedule_comparison/linear/mnist_linear_steps10000.pth",
                       help="First schedule checkpoint")
    parser.add_argument("--checkpoint2", type=str,
                       default="model_ckpts/schedule_comparison/cosine/mnist_cosine_steps10000.pth",
                       help="Second schedule checkpoint")
    parser.add_argument("--name1", type=str, default="Linear", help="First schedule name")
    parser.add_argument("--name2", type=str, default="Cosine", help="Second schedule name")
    parser.add_argument("--plot-schedules", action="store_true",
                       help="Plot beta schedule comparison (no checkpoints needed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Plot schedule comparison (doesn't require checkpoints)
    if args.plot_schedules:
        plot_beta_schedules_comparison()
        return

    # Verify checkpoints exist
    if not os.path.exists(args.checkpoint1):
        print(f"[ERROR] Checkpoint 1 not found: {args.checkpoint1}")
        print(f"[INFO] Run parallel training first:")
        print(f"  python train_parallel_schedules.py --schedule1 linear --schedule2 cosine --steps 10000")
        sys.exit(1)

    if not os.path.exists(args.checkpoint2):
        print(f"[ERROR] Checkpoint 2 not found: {args.checkpoint2}")
        print(f"[INFO] Run parallel training first:")
        print(f"  python train_parallel_schedules.py --schedule1 linear --schedule2 cosine --steps 10000")
        sys.exit(1)

    # Generate visual comparison
    compare_schedules_visually(args.checkpoint1, args.checkpoint2,
                               args.name1, args.name2, device)

    print(f"\n[COMPLETE] Comparison visualization complete!")
    print(f"\nNext steps:")
    print(f"1. View the comparison grid in: model_ckpts/schedule_comparison/visualizations/")
    print(f"2. Compare sample quality between schedules")
    print(f"3. When implemented: Run memorization metrics comparison")


if __name__ == "__main__":
    main()
