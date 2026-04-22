"""Generate 100 samples per checkpoint and compute F_mem with gap ratio distribution.

Outputs:
  - analysis/section_2_4/fmem_curve_100samples.csv
  - Console comparison with previous 50-sample results (if fmem_curve.csv exists)

Usage:
    .venv\Scripts\python.exe scripts/compute_fmem_100samples.py
"""

import sys
import os
import re
import csv
import glob
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.memorization import compute_memorization_fraction
from datasets.dataset import get_dataset
from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
from sampling.ddim import sample_ddim
from config_loader import TrainingConfig


def discover_checkpoints(ckpt_dir: str):
    """Find all .pth files with _step{N}.pth suffix, return sorted (step, path) pairs."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    checkpoints = []
    for f in files:
        basename = os.path.basename(f)
        m = re.search(r"_step(\d+)\.pth$", basename)
        if m:
            checkpoints.append((int(m.group(1)), f))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config & diffusion schedule
    config = TrainingConfig.from_yaml("configs/hcmc_vitality_v3_5090.yaml")
    diff_cfg = DiffusionConfig(
        n_steps=1000, schedule_type="linear",
        b0=0.0001, bT=0.02, device=device,
    )

    # Load training data (all at once, ~3259 images x 128x128x3 x float32 ≈ 600 MB on CPU)
    print("Loading training data (split=train)...")
    train_ds = get_dataset(
        "hcmc_vitality_v2", img_size=128,
        data_dir="../data-processing/data/hcmc/processed/cells",
        split="train",
        center=config.center,
        normalize=config.normalize,
    )
    loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    batch = next(iter(loader))
    train_imgs = batch["image"] if isinstance(batch, dict) else batch
    print(f"Training images: {train_imgs.shape} ({train_imgs.shape[0]} samples)")

    # Discover checkpoints
    ckpt_dir = "model_ckpts/hcmc_vitality_v3"
    checkpoints = discover_checkpoints(ckpt_dir)
    print(f"Checkpoints found: {len(checkpoints)}")
    for step, path in checkpoints:
        print(f"  step={step:>7d}  {os.path.basename(path)}")

    # Build model once, reload weights per checkpoint
    unet = DDPMUnet(3, 128)
    ddpm = DDPM(1000, unet, diffusion_config=diff_cfg)
    ddpm = ddpm.to(device)

    # Process each checkpoint
    N_SAMPLES = 100
    SEED = 42
    results = []
    total_start = time.time()

    for i, (step, ckpt_path) in enumerate(checkpoints):
        t0 = time.time()
        print(f"\n[{i + 1}/{len(checkpoints)}] step={step:>7d}  {os.path.basename(ckpt_path)}")

        # Load weights
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "eps_model" in ckpt:
            state_dict = ckpt["eps_model"]
        elif "optimizer" in ckpt:
            state_dict = ckpt["eps_model"]
        else:
            state_dict = ckpt

        # Strip DataParallel 'module.' prefix if present
        first_key = next(iter(state_dict), "")
        if first_key.startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        ddpm.eps_model.load_state_dict(state_dict)

        # Generate 100 samples via DDIM (50 steps, eta=0 deterministic)
        torch.manual_seed(SEED)
        gen_imgs, _ = sample_ddim(
            ddpm, n_images=N_SAMPLES, config=config,
            diffusion_config=diff_cfg, device=device,
        )
        gen_imgs = gen_imgs.cpu()  # move to CPU for k-NN

        # Compute F_mem (k-NN on CPU — train_imgs is on CPU)
        result = compute_memorization_fraction(
            train_imgs, gen_imgs,
            gap_threshold=1.0 / 3.0,
            device="cpu",
        )
        ratios = result["gap_ratios"]

        row = {
            "step": step,
            "f_mem": result["fraction_mem"],
            "n_memorized": result["n_memorized"],
            "n_total": result["n_total"],
            "min_gap_ratio": float(ratios.min()),
            "median_gap_ratio": float(np.median(ratios)),
            "mean_gap_ratio": float(ratios.mean()),
            "std": result["std"],
            "lower_ci": result["lower_ci"],
            "upper_ci": result["upper_ci"],
        }
        results.append(row)

        elapsed = time.time() - t0
        print(f"  F_mem={result['fraction_mem']:.4f}  "
              f"memorized={result['n_memorized']}/{N_SAMPLES}  "
              f"gap=[{ratios.min():.4f}, {np.median(ratios):.4f}, {ratios.mean():.4f}]  "
              f"{elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # ---- Save CSV ----
    out_dir = "analysis/section_2_4"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fmem_curve_100samples.csv")

    fieldnames = [
        "step", "f_mem", "n_memorized", "n_total",
        "min_gap_ratio", "median_gap_ratio", "mean_gap_ratio",
        "std", "lower_ci", "upper_ci",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_path}")

    # ---- Comparison with 50-sample results ----
    prev_path = os.path.join(out_dir, "fmem_curve.csv")
    if os.path.exists(prev_path):
        print("\n" + "=" * 70)
        print("COMPARISON: 50 samples vs 100 samples")
        print("=" * 70)

        prev_data = {}
        with open(prev_path, "r") as f:
            for row in csv.DictReader(f):
                prev_data[int(row["step"])] = float(row["f_mem"])

        print(f"{'Step':>8}  {'F_mem (50)':>10}  {'F_mem (100)':>11}  {'Delta':>8}")
        print("-" * 48)

        for row in results:
            step = row["step"]
            f100 = row["f_mem"]
            f50 = prev_data.get(step)
            if f50 is not None:
                delta = f100 - f50
                marker = " ***" if delta != 0 else ""
                print(f"{step:>8}  {f50:>10.4f}  {f100:>11.4f}  {delta:>+8.4f}{marker}")
            else:
                print(f"{step:>8}  {'N/A':>10}  {f100:>11.4f}")

        # Summary stats
        common = [r for r in results if r["step"] in prev_data]
        if common:
            max_50 = max(prev_data.get(r["step"], 0) for r in common)
            max_100 = max(r["f_mem"] for r in common)
            any_mem_100 = sum(1 for r in results if r["n_memorized"] > 0)
            any_mem_50 = sum(1 for s, f in prev_data.items() if f > 0)
            print(f"\nSummary:")
            print(f"  Checkpoints with any memorization: 50-sample={any_mem_50}, 100-sample={any_mem_100}")
            print(f"  Peak F_mem: 50-sample={max_50:.4f}, 100-sample={max_100:.4f}")
            print(f"  Overall: {'F_mem unchanged at 0 for most checkpoints' if max_100 < 0.05 else 'F_mem detected at later checkpoints'}")


if __name__ == "__main__":
    main()
