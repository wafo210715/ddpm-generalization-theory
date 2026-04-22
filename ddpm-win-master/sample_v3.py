"""Sample from v3 HCMC vitality checkpoints.

Usage:
    python sample_v3.py                                          # default: step 500K, 16 samples
    python sample_v3.py --checkpoint model_ckpts/hcmc_vitality_v3/xxx_step5000.pth
    python sample_v3.py --step 50000 --num 32
    python sample_v3.py --step 5000,10000,50000,100000           # compare multiple checkpoints
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig


CKPT_DIR = "model_ckpts/hcmc_vitality_v3"
IMG_CHANNELS = 3
IMG_SIZE = 128
MODEL_CHANNELS = 128
TIMESTEPS = 1000


def load_ddpm_from_checkpoint(checkpoint_path: str, device: str) -> DDPM:
    """Load DDPM model from a v3 checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    diff_cfg = DiffusionConfig(
        n_steps=TIMESTEPS, schedule_type="linear",
        b0=0.0001, bT=0.02, device=device,
    )

    model = DDPMUnet(IMG_CHANNELS, MODEL_CHANNELS)
    ddpm = DDPM(TIMESTEPS, model, diffusion_config=diff_cfg)

    # Handle checkpoint formats
    if isinstance(ckpt, dict) and "eps_model" in ckpt:
        ddpm.eps_model.load_state_dict(ckpt["eps_model"])
        step = ckpt.get("step", 0)
    elif isinstance(ckpt, dict) and "beta" in ckpt:
        eps_state = {k.replace("eps_model.", ""): v for k, v in ckpt.items()
                     if k.startswith("eps_model.")}
        ddpm.eps_model.load_state_dict(eps_state)
        step = 0
    else:
        ddpm.eps_model.load_state_dict(ckpt)
        step = 0

    ddpm.to(device)
    ddpm.eval()
    return ddpm, step, diff_cfg


@torch.no_grad()
def sample_images(ddpm: DDPM, num_samples: int, device: str) -> torch.Tensor:
    """Sample images via DDPM reverse process. Returns tensor in [-1, 1]."""
    x_t = torch.randn((num_samples, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=device)

    for t in tqdm(range(ddpm.timesteps - 1, -1, -1), desc="Sampling"):
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        t_tensor = torch.full((num_samples,), t, device=device)
        x_t = ddpm.p_sample(x_t, t_tensor, z)

    return x_t.clamp(-1, 1)


def tensor_to_images(x: torch.Tensor) -> list[Image.Image]:
    """Convert [-1, 1] tensor batch to PIL Images [0, 255]."""
    # [-1, 1] -> [0, 1] -> [0, 255] uint8
    x = ((x + 1) / 2).clamp(0, 1)
    x = (x * 255).to(torch.uint8)
    # CHW -> HWC
    images = []
    for i in range(x.shape[0]):
        img = x[i].permute(1, 2, 0).cpu().numpy()
        images.append(Image.fromarray(img, mode="RGB"))
    return images


def save_grid(images: list[Image.Image], path: str, nrow: int = 4):
    """Save images as a grid."""
    # Convert PIL -> tensor for make_grid
    tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
               for img in images]
    batch = torch.stack(tensors)
    grid = make_grid(batch, nrow=nrow, padding=2)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(path)
    print(f"Saved grid: {path}")


def save_individual(images: list[Image.Image], out_dir: str):
    """Save each image individually."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(out_dir, f"sample_{i:04d}.png"))
    print(f"Saved {len(images)} images to {out_dir}/")


def find_checkpoint(step: int, ckpt_dir: str) -> str:
    """Find checkpoint file for a given step."""
    pattern = f"*_step{step}.pth"
    matches = list(Path(ckpt_dir).glob(pattern))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No checkpoint for step {step} in {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample from v3 HCMC checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint .pth")
    parser.add_argument("--step", type=str, default="500000",
                        help="Step number(s), comma-separated (default: 500000)")
    parser.add_argument("--num", type=int, default=16, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nrow", type=int, default=4, help="Grid columns")
    parser.add_argument("--save_individual", action="store_true",
                        help="Also save individual images")
    parser.add_argument("--out_dir", type=str, default="samples/v3",
                        help="Output directory")
    parser.add_argument("--sampling_method", type=str, choices=["ddim", "ddpm"],
                        default="ddim", help="Sampling method (default: ddim)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.checkpoint:
        checkpoints = [args.checkpoint]
        steps = ["custom"]
    else:
        steps = [int(s.strip()) for s in args.step.split(",")]
        checkpoints = [find_checkpoint(s, CKPT_DIR) for s in steps]

    device = args.device

    for ckpt_path, step_label in zip(checkpoints, steps):
        print(f"\n{'='*60}")
        print(f"Sampling from step {step_label}: {ckpt_path}")
        print(f"{'='*60}")

        ddpm, loaded_step, diff_cfg = load_ddpm_from_checkpoint(ckpt_path, device)
        print(f"Loaded step={loaded_step}, params={sum(p.numel() for p in ddpm.parameters())/1e6:.1f}M")

        if args.sampling_method == "ddpm":
            from sampling.ddpm_sample import sample_ddpm
            x, _ = sample_ddpm(ddpm, n_images=args.num, config=None,
                               diffusion_config=diff_cfg, device=device)
        else:
            from sampling.ddim import sample_ddim
            x, _ = sample_ddim(ddpm, n_images=args.num, config=None,
                               diffusion_config=diff_cfg, device=device)
        images = tensor_to_images(x)

        # Save grid
        grid_name = f"grid_step{step_label}.png"
        save_grid(images, os.path.join(args.out_dir, grid_name), nrow=args.nrow)

        # Optionally save individual
        if args.save_individual:
            ind_dir = os.path.join(args.out_dir, f"step{step_label}")
            save_individual(images, ind_dir)

        del ddpm, x
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
