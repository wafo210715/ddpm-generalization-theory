"""Generate 1000 samples per checkpoint and compute F_mem with gap ratio distribution.

Outputs (CSV only, no visualization):
  - analysis/section_2_4/fmem_curve_1000samples.csv   (summary per checkpoint)
  - analysis/section_2_4/gap_ratios_1000samples.csv    (raw per-sample gap ratios)

Generation is batched (100 images at a time) to avoid GPU OOM.

Usage:
    .venv\\Scripts\\python.exe scripts/compute_fmem_1000samples.py
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

    # Fixed seed at START so all 1000 samples are deterministic across checkpoints
    SEED = 42
    torch.manual_seed(SEED)
    print(f"Global seed: {SEED}")

    N_SAMPLES = 1000
    GEN_BATCH = 100  # generate 100 at a time to manage GPU memory

    # Config & diffusion schedule
    config = TrainingConfig.from_yaml("configs/hcmc_vitality_v3_5090.yaml")
    diff_cfg = DiffusionConfig(
        n_steps=1000, schedule_type="linear",
        b0=0.0001, bT=0.02, device=device,
    )

    # Load training data once (all at once on CPU)
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

    # Output directory
    out_dir = "analysis/section_2_4"
    os.makedirs(out_dir, exist_ok=True)

    # Summary results
    summary_fieldnames = [
        "step", "f_mem", "n_memorized", "n_total",
        "min_gap_ratio", "median_gap_ratio", "mean_gap_ratio",
        "p10_gap_ratio", "p25_gap_ratio", "p75_gap_ratio", "p90_gap_ratio",
        "std", "lower_ci", "upper_ci", "n_samples",
    ]
    summary_rows = []

    # Raw gap ratios: accumulate all rows, write at end
    raw_rows = []
    raw_fieldnames = ["step", "sample_idx", "gap_ratio", "is_memorized"]

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

        # Generate 1000 samples in batches of 100
        # Seed is set once at start; each call to sample_ddim advances the RNG state
        gen_batches = []
        n_generated = 0
        batch_idx = 0
        while n_generated < N_SAMPLES:
            n_this = min(GEN_BATCH, N_SAMPLES - n_generated)
            print(f"  Generating batch {batch_idx + 1}: {n_this} images ...", end=" ", flush=True)
            bt = time.time()
            gen_imgs, _ = sample_ddim(
                ddpm, n_images=n_this, config=config,
                diffusion_config=diff_cfg, device=device,
            )
            gen_batches.append(gen_imgs.cpu())
            n_generated += n_this
            batch_idx += 1
            print(f"{time.time() - bt:.1f}s")

        all_gen = torch.cat(gen_batches, dim=0)
        assert all_gen.shape[0] == N_SAMPLES, f"Expected {N_SAMPLES}, got {all_gen.shape[0]}"
        print(f"  All {N_SAMPLES} samples collected: {all_gen.shape}")

        # Compute F_mem (k-NN on CPU)
        result = compute_memorization_fraction(
            train_imgs, all_gen,
            gap_threshold=1.0 / 3.0,
            device="cpu",
        )
        ratios = result["gap_ratios"]
        assert len(ratios) == N_SAMPLES

        # Percentiles
        p10 = float(np.percentile(ratios, 10))
        p25 = float(np.percentile(ratios, 25))
        p75 = float(np.percentile(ratios, 75))
        p90 = float(np.percentile(ratios, 90))

        summary_rows.append({
            "step": step,
            "f_mem": result["fraction_mem"],
            "n_memorized": result["n_memorized"],
            "n_total": result["n_total"],
            "min_gap_ratio": float(ratios.min()),
            "median_gap_ratio": float(np.median(ratios)),
            "mean_gap_ratio": float(ratios.mean()),
            "p10_gap_ratio": p10,
            "p25_gap_ratio": p25,
            "p75_gap_ratio": p75,
            "p90_gap_ratio": p90,
            "std": result["std"],
            "lower_ci": result["lower_ci"],
            "upper_ci": result["upper_ci"],
            "n_samples": N_SAMPLES,
        })

        # Accumulate raw gap ratios for this checkpoint
        for idx in range(N_SAMPLES):
            raw_rows.append({
                "step": step,
                "sample_idx": idx,
                "gap_ratio": float(ratios[idx]),
                "is_memorized": int(ratios[idx] < 1.0 / 3.0),
            })

        elapsed = time.time() - t0
        print(f"  F_mem={result['fraction_mem']:.4f}  "
              f"memorized={result['n_memorized']}/{N_SAMPLES}  "
              f"gap=[{ratios.min():.4f}, {np.median(ratios):.4f}, {ratios.mean():.4f}]  "
              f"{elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # ---- Save summary CSV ----
    summary_path = os.path.join(out_dir, "fmem_curve_1000samples.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary: {summary_path}")

    # ---- Save raw gap ratios CSV ----
    raw_path = os.path.join(out_dir, "gap_ratios_1000samples.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Saved raw data: {raw_path}")


if __name__ == "__main__":
    main()
