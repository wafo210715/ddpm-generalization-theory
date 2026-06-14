"""Publication-quality sample grid generator for DDPM checkpoints.

Generates grids in three modes:
  - grid: standard sample grid
  - comparison: real vs generated side-by-side
  - channels: per-channel decomposition (B, G, R)

Usage:
    python scripts/generate_grids.py --checkpoints 5000,50000,500000 --grid_size 8 --mode grid
    python scripts/generate_grids.py --checkpoints 50000 --mode comparison
    python scripts/generate_grids.py --checkpoints 50000 --mode channels
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid
from tqdm.auto import tqdm

# Ensure project root is on sys.path so we can import model/sampling/datasets
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig

# Channel semantics for HCMC vitality data
# B=0 (density), G=1 (diversity), R=2 (accessibility)
_CHANNEL_NAMES = {0: "B_density", 1: "G_diversity", 2: "R_accessibility"}

IMG_CHANNELS = 3
IMG_SIZE = 128
MODEL_CHANNELS = 128
TIMESTEPS = 1000


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_checkpoint(step: int, ckpt_dir: str) -> str:
    """Find checkpoint file for a given step number."""
    pattern = f"*_step{step}.pth"
    matches = list(Path(ckpt_dir).glob(pattern))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No checkpoint for step {step} in {ckpt_dir}")


def load_ddpm_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: str,
) -> DDPM:
    """Load DDPM model from a checkpoint file, using config for schedule."""
    img_channels = config.get("img_channels", IMG_CHANNELS)
    model_channels = config.get("model_channels", MODEL_CHANNELS)
    timesteps = config.get("timesteps", TIMESTEPS)
    schedule_type = config.get("schedule_type", "linear")

    diff_cfg = DiffusionConfig(
        n_steps=timesteps,
        schedule_type=schedule_type,
        b0=config.get("b0", 0.0001),
        bT=config.get("bT", 0.02),
        device=device,
    )

    model = DDPMUnet(img_channels, model_channels)
    ddpm = DDPM(timesteps, model, diffusion_config=diff_cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "eps_model" in ckpt:
        ddpm.eps_model.load_state_dict(ckpt["eps_model"])
        loaded_step = ckpt.get("step", 0)
    elif isinstance(ckpt, dict) and "beta" in ckpt:
        eps_state = {
            k.replace("eps_model.", ""): v
            for k, v in ckpt.items()
            if k.startswith("eps_model.")
        }
        ddpm.eps_model.load_state_dict(eps_state)
        loaded_step = 0
    else:
        ddpm.eps_model.load_state_dict(ckpt)
        loaded_step = 0

    ddpm.to(device)
    ddpm.eval()
    return ddpm, loaded_step, diff_cfg


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    ddpm: DDPM,
    diff_cfg: DiffusionConfig,
    n_samples: int,
    device: str,
    sampling_method: str = "ddim",
) -> torch.Tensor:
    """Generate samples. Returns tensor in [-1, 1]."""
    if sampling_method == "ddim":
        from sampling.ddim import sample_ddim
        x, _ = sample_ddim(
            ddpm, n_images=n_samples, config=None,
            diffusion_config=diff_cfg, device=device,
        )
    else:
        from sampling.ddpm_sample import sample_ddpm
        x, _ = sample_ddpm(
            ddpm, n_images=n_samples, config=None,
            diffusion_config=diff_cfg, device=device,
        )
    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert [-1, 1] tensor (B, C, H, W) to uint8 numpy (B, H, W, C)."""
    x = ((x + 1) / 2).clamp(0, 1)
    x = (x * 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def tensors_to_grid_image(
    tensors: list[torch.Tensor],
    nrow: int,
    padding: int = 2,
) -> np.ndarray:
    """Stack a list of (C, H, W) float [0,1] tensors into a grid HWC uint8 array."""
    batch = torch.stack(tensors)
    grid = make_grid(batch, nrow=nrow, padding=padding)
    return (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_grid_png(arr: np.ndarray, path: str, dpi: int = 300):
    """Save a uint8 HWC array as a PNG at the given DPI."""
    Image.fromarray(arr, mode="RGB").save(path, dpi=(dpi, dpi))


# ---------------------------------------------------------------------------
# Mode: grid
# ---------------------------------------------------------------------------

def mode_grid(
    ddpm: DDPM,
    diff_cfg: DiffusionConfig,
    step_label: str,
    args: argparse.Namespace,
    device: str,
):
    """Standard sample grid."""
    n_samples = min(args.n_samples, args.grid_size ** 2)
    print(f"  Generating {n_samples} samples ...")
    x = generate_samples(ddpm, diff_cfg, n_samples, device, args.sampling_method)

    # Convert to [0,1] tensors
    imgs = [((x[i] + 1) / 2).clamp(0, 1) for i in range(n_samples)]

    grid_arr = tensors_to_grid_image(imgs, nrow=args.grid_size)
    out_path = os.path.join(args.output, f"grid_step{step_label}_{args.grid_size}x{args.grid_size}.png")
    save_grid_png(grid_arr, out_path, dpi=args.dpi)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Mode: comparison
# ---------------------------------------------------------------------------

def mode_comparison(
    ddpm: DDPM,
    diff_cfg: DiffusionConfig,
    step_label: str,
    config: dict,
    args: argparse.Namespace,
    device: str,
):
    """Real vs generated side-by-side."""
    n_samples = args.n_samples

    # Generate synthetic images
    print(f"  Generating {n_samples} samples ...")
    x_gen = generate_samples(ddpm, diff_cfg, n_samples, device, args.sampling_method)
    gen_imgs = [((x_gen[i] + 1) / 2).clamp(0, 1) for i in range(n_samples)]

    # Load real training images
    data_root = config.get("data_root", "../data-processing/data/hcmc/processed/cells")
    dataset_name = config.get("dataset_name", "hcmc_vitality_v2")
    img_size = config.get("img_size", 128)

    print(f"  Loading real images from {data_root} ...")
    from datasets.dataset import get_dataset
    dataset = get_dataset(
        dataset_name,
        img_size=img_size,
        data_dir=data_root,
        split="train",
        center=True,
        normalize=True,
    )

    # Pick random real images
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    real_imgs = []
    for idx in indices:
        sample = dataset[idx]
        # Handle both dict and tensor returns
        if isinstance(sample, dict):
            t = sample["image"]
        else:
            t = sample
        # Already in [-1, 1], convert to [0, 1]
        real_imgs.append(((t + 1) / 2).clamp(0, 1))

    # Interleave: [real_1, gen_1, real_2, gen_2, ...]
    interleaved = []
    for r, g in zip(real_imgs, gen_imgs):
        interleaved.append(r)
        interleaved.append(g)

    nrow = min(args.grid_size, n_samples)
    grid_arr = tensors_to_grid_image(interleaved, nrow=nrow)
    out_path = os.path.join(args.output, f"comparison_step{step_label}.png")
    save_grid_png(grid_arr, out_path, dpi=args.dpi)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Mode: channels
# ---------------------------------------------------------------------------

def mode_channels(
    ddpm: DDPM,
    diff_cfg: DiffusionConfig,
    step_label: str,
    args: argparse.Namespace,
    device: str,
):
    """Per-channel decomposition: original | B | G | R."""
    # Use a number of samples divisible by channels for clean layout
    n_samples = max(3, (args.n_samples // 3) * 3)
    n_images_per_row = min(args.grid_size, n_samples)
    print(f"  Generating {n_samples} samples ...")
    x = generate_samples(ddpm, diff_cfg, n_samples, device, args.sampling_method)

    # For each image, produce 4 panels: original | B | G | R
    rows = []
    for i in range(n_samples):
        img_uint8 = tensor_to_uint8(x[i : i + 1])[0]  # (H, W, 3)

        # Build 4-channel-wide row: [original, B_only, G_only, R_only]
        panels = [img_uint8]
        for ch_idx in range(3):
            gray = np.full_like(img_uint8, 128)  # neutral gray background
            gray[:, :, ch_idx] = img_uint8[:, :, ch_idx]
            panels.append(gray)
        # Concatenate horizontally: (H, 4*W, 3)
        row = np.concatenate(panels, axis=1)
        rows.append(row)

    # Stack rows vertically
    full_grid = np.concatenate(rows, axis=0)  # (n_samples*H, 4*W, 3)

    # Add channel labels using PIL
    out_path = os.path.join(args.output, f"channels_step{step_label}.png")
    pil_img = Image.fromarray(full_grid, mode="RGB")
    pil_img.save(out_path, dpi=(args.dpi, args.dpi))
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality sample grid generator for DDPM checkpoints"
    )
    parser.add_argument(
        "--checkpoints", type=str, required=True,
        help="Comma-separated step numbers, e.g. '5000,50000,500000'",
    )
    parser.add_argument(
        "--grid_size", type=int, default=8,
        help="Grid columns (default: 8)",
    )
    parser.add_argument(
        "--mode", type=str, default="grid",
        choices=["grid", "comparison", "channels"],
        help="Output mode (default: grid)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="model_ckpts/hcmc_vitality_v3",
        help="Checkpoint directory (default: model_ckpts/hcmc_vitality_v3)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/hcmc_vitality_v3_5090.yaml",
        help="Config YAML path (default: configs/hcmc_vitality_v3_5090.yaml)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=16,
        help="Samples per checkpoint (default: 16)",
    )
    parser.add_argument(
        "--output", type=str, default="publication_grids",
        help="Output directory (default: publication_grids)",
    )
    parser.add_argument(
        "--sampling_method", type=str, default="ddim",
        choices=["ddim", "ddpm"],
        help="Sampling method (default: ddim)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    config = load_config(args.config)

    # Parse checkpoint steps
    steps = [int(s.strip()) for s in args.checkpoints.split(",")]

    device = args.device

    for step in steps:
        print(f"\n{'=' * 60}")
        print(f"  Checkpoint step {step}  |  mode={args.mode}")
        print(f"{'=' * 60}")

        ckpt_path = find_checkpoint(step, args.checkpoint_dir)
        print(f"  Loading: {ckpt_path}")

        ddpm, loaded_step, diff_cfg = load_ddpm_from_checkpoint(ckpt_path, config, device)
        n_params = sum(p.numel() for p in ddpm.parameters()) / 1e6
        print(f"  Loaded step={loaded_step}, params={n_params:.1f}M")

        if args.mode == "grid":
            mode_grid(ddpm, diff_cfg, step, args, device)
        elif args.mode == "comparison":
            mode_comparison(ddpm, diff_cfg, step, config, args, device)
        elif args.mode == "channels":
            mode_channels(ddpm, diff_cfg, step, args, device)

        del ddpm
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
