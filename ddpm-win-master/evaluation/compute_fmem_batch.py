"""
Batch computation of F_mem (memorization fraction) across training checkpoints.

For each .pth checkpoint in a directory, generates samples via DDIM, computes
the k-NN gap-ratio memorization metric, and produces:
  1. CSV table  (fmem_curve.csv)
  2. Plot       (fmem_curve.png)

Usage:
    python evaluation/compute_fmem_batch.py \
        --checkpoint_dir model_ckpts \
        --config configs/default.yaml

Reference: Bonnaire et al., "Why Diffusion Models Don't Memorize" (arXiv:2505.17638)
"""

import argparse
import csv
import os
import re
import sys

# ---------------------------------------------------------------------------
# Make ddpm-win-master importable when running as a standalone script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import torch

from config_loader import TrainingConfig
from datasets.dataset import get_dataset
from evaluation.memorization import evaluate_checkpoint_memorization


# ======================================================================
# Checkpoint discovery
# ======================================================================

_STEP_RE = re.compile(r"_step(\d+)\.pth$", re.IGNORECASE)


def discover_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """Find all ``*_step{N}.pth`` files and return ``(step, path)`` pairs sorted by step."""
    results: list[tuple[int, str]] = []
    for fname in os.listdir(checkpoint_dir):
        m = _STEP_RE.search(fname)
        if m:
            step = int(m.group(1))
            results.append((step, os.path.join(checkpoint_dir, fname)))
    results.sort(key=lambda t: t[0])
    return results


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-compute F_mem across DDPM training checkpoints.",
    )
    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing checkpoint .pth files.",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file used during training.",
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="Override data root directory (default: from config).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split for training images (default: train).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of images to generate per checkpoint (default: 200).",
    )
    p.add_argument(
        "--gap_threshold",
        type=float,
        default=1 / 3,
        help="k-NN gap-ratio threshold for memorization (default: 0.333).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of nearest neighbors (default: 2).",
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for confidence interval (default: 1000).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: checkpoint_dir/../fmem_results).",
    )
    return p.parse_args()


# ======================================================================
# Plotting
# ======================================================================


def plot_fmem_curve(rows: list[dict], output_path: str) -> None:
    """Save F_mem vs. training step with 95 % CI error bars."""
    steps = [r["checkpoint_step"] for r in rows]
    fmem = [r["f_mem"] for r in rows]
    lower = [r["lower_ci"] for r in rows]
    upper = [r["upper_ci"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr_lower = [f - lo for f, lo in zip(fmem, lower)]
    yerr_upper = [hi - f for hi, f in zip(upper, fmem)]
    ax.errorbar(
        steps,
        fmem,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        capsize=4,
        markersize=5,
        linewidth=1.5,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("F_mem")
    ax.set_title("Memorization Fraction vs. Training Step")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    args = parse_args()

    # --- Resolve device ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = "cpu"

    # --- Discover checkpoints ---
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  step {step:>7d}  {os.path.basename(path)}")

    # --- Load config ---
    config = TrainingConfig.from_yaml(args.config)

    # --- Load training dataset ---
    dataset = get_dataset(
        name=config.dataset_name,
        img_size=config.img_size,
        data_dir=args.data_dir or config.data_root,
        split=args.split,
        center=config.center,
        normalize=config.normalize,
    )
    print(
        f"Training dataset: {len(dataset)} images ({config.dataset_name}, split={args.split})"
    )

    # --- Output directory ---
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint_dir.rstrip("/\\")), "fmem_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "fmem_curve.csv")
    png_path = os.path.join(output_dir, "fmem_curve.png")

    # --- Process each checkpoint ---
    rows: list[dict] = []
    for step, ckpt_path in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Checkpoint step {step}: {os.path.basename(ckpt_path)}")
        print(f"{'=' * 60}")

        result = evaluate_checkpoint_memorization(
            checkpoint_path=ckpt_path,
            train_dataset=dataset,
            config=config,
            n_samples=args.n_samples,
            gap_threshold=args.gap_threshold,
            device=device,
        )

        row = {
            "checkpoint_step": step,
            "f_mem": result["fraction_mem"],
            "std": result["std"],
            "lower_ci": result["lower_ci"],
            "upper_ci": result["upper_ci"],
            "n_memorized": result["n_memorized"],
            "n_total": result["n_total"],
        }
        rows.append(row)

        print(
            f"  F_mem = {row['f_mem']:.4f}  "
            f"95% CI [{row['lower_ci']:.4f}, {row['upper_ci']:.4f}]  "
            f"({row['n_memorized']}/{row['n_total']} memorized)"
        )

    # --- Sort by step ---
    rows.sort(key=lambda r: r["checkpoint_step"])

    # --- Write CSV ---
    fieldnames = [
        "checkpoint_step",
        "f_mem",
        "std",
        "lower_ci",
        "upper_ci",
        "n_memorized",
        "n_total",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to {csv_path}")

    # --- Plot ---
    plot_fmem_curve(rows, png_path)
    print(f"Plot  saved to {png_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for row in rows:
        print(
            f"  Step {row['checkpoint_step']:>7d}: "
            f"F_mem = {row['f_mem']:.4f} "
            f"[{row['lower_ci']:.4f}, {row['upper_ci']:.4f}]"
        )


if __name__ == "__main__":
    main()
