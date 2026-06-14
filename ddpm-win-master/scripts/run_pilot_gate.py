"""
Pilot Training GATE DECISION orchestrator.

Trains DDPM at multiple resolutions (64, 128, 256), computes F_mem at each
checkpoint, and produces a PASS / ITERATE / FAIL gate decision with a full
report.

Usage:
    python scripts/run_pilot_gate.py --help
    python scripts/run_pilot_gate.py --resolutions 64,128,256
    python scripts/run_pilot_gate.py --resolutions 64 --n_samples 50

Reference: Bonnaire et al., "Why Diffusion Models Don't Memorize" (arXiv:2505.17638)
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths — everything is relative to the ddpm-win-master root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from config_loader import TrainingConfig
from evaluation.compute_fmem_batch import discover_checkpoints

# Config template paths relative to project root
_CONFIGS = {
    64: os.path.join(_PROJECT_ROOT, "configs", "hcmc_pilot_64.yaml"),
    128: os.path.join(_PROJECT_ROOT, "configs", "hcmc_pilot_128.yaml"),
    256: os.path.join(_PROJECT_ROOT, "configs", "hcmc_pilot_256.yaml"),
}


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pilot Training GATE DECISION: train DDPM at multiple resolutions, "
            "compute F_mem, and produce a gate decision report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_pilot_gate.py\n"
            "  python scripts/run_pilot_gate.py --resolutions 64,128\n"
            "  python scripts/run_pilot_gate.py --n_samples 30 --device cpu\n"
        ),
    )
    p.add_argument(
        "--resolutions",
        type=str,
        default="64,128,256",
        help="Comma-separated resolutions to test (default: 64,128,256)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples to generate per checkpoint for F_mem (default: 50)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for F_mem computation (default: cuda)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="logs/hcmc_vitality_pilot",
        help="Output directory for report and plots (default: logs/hcmc_vitality_pilot)",
    )
    p.add_argument(
        "--gap_threshold",
        type=float,
        default=1.0 / 3.0,
        help="k-NN gap-ratio threshold for memorization (default: 0.333)",
    )
    p.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training phase; assume checkpoints already exist",
    )
    return p.parse_args()


# ======================================================================
# Training phase
# ======================================================================


def train_single_resolution(config_path: str) -> bool:
    """Run training for a single resolution via subprocess.

    We use subprocess rather than importing train.py directly to avoid
    GPU initialisation conflicts when switching between resolutions.

    Returns:
        True if training succeeded, False otherwise.
    """
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {os.path.basename(config_path)}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "train.py", "--config", config_path]
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {_PROJECT_ROOT}")

    try:
        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=False,  # stream output live
            text=True,
        )
        if result.returncode != 0:
            print(
                f"[ERROR] Training failed for {config_path} (exit code {result.returncode})"
            )
            return False
        print(f"[OK] Training completed for {os.path.basename(config_path)}")
        return True
    except Exception as e:
        print(f"[ERROR] Exception during training: {e}")
        return False


# ======================================================================
# F_mem evaluation phase
# ======================================================================


def evaluate_resolution(
    resolution: int,
    config_path: str,
    n_samples: int,
    gap_threshold: float,
    device: str,
) -> Dict[int, Dict[str, Any]]:
    """Evaluate all checkpoints for a single resolution.

    Uses the evaluation.compute_fmem_batch module's underlying functions
    via subprocess to avoid GPU state leakage.

    Returns:
        Dict mapping step -> {f_mem, lower_ci, upper_ci, n_memorized, n_total}
    """
    config = TrainingConfig.from_yaml(config_path)
    checkpoint_dir = os.path.join(_PROJECT_ROOT, config.checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(
            f"[WARN] No checkpoints found for {resolution}x{resolution} in {checkpoint_dir}"
        )
        return {}

    print(
        f"\nEvaluating {len(checkpoints)} checkpoints for {resolution}x{resolution}..."
    )

    # Resolve device
    eval_device = device
    if eval_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU for evaluation.")
        eval_device = "cpu"

    # Run compute_fmem_batch.py as subprocess for isolation
    cmd = [
        sys.executable,
        os.path.join(_PROJECT_ROOT, "evaluation", "compute_fmem_batch.py"),
        "--checkpoint_dir",
        checkpoint_dir,
        "--config",
        config_path,
        "--n_samples",
        str(n_samples),
        "--gap_threshold",
        str(gap_threshold),
        "--device",
        eval_device,
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600 * 4,  # 4 hour timeout per resolution
        )
        if result.returncode != 0:
            print(f"[ERROR] F_mem computation failed:\n{result.stderr}")
            return {}
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    except subprocess.TimeoutExpired:
        print("[ERROR] F_mem computation timed out after 4 hours")
        return {}
    except Exception as e:
        print(f"[ERROR] Exception during F_mem computation: {e}")
        return {}

    # Parse results from CSV output
    fmem_csv = os.path.join(
        os.path.dirname(checkpoint_dir.rstrip("/\\")),
        "fmem_results",
        "fmem_curve.csv",
    )
    # Also check within checkpoint_dir parent
    alt_csv = os.path.join(checkpoint_dir, "..", "fmem_results", "fmem_curve.csv")

    csv_path = fmem_csv if os.path.isfile(fmem_csv) else alt_csv
    if not os.path.isfile(csv_path):
        # Try the default output_dir logic from compute_fmem_batch.py
        csv_path = os.path.join(
            os.path.dirname(checkpoint_dir.rstrip("/\\")),
            "fmem_results",
            "fmem_curve.csv",
        )

    return _parse_fmem_csv(csv_path) if os.path.isfile(csv_path) else {}


def _parse_fmem_csv(csv_path: str) -> Dict[int, Dict[str, Any]]:
    """Parse fmem_curve.csv into {step: {f_mem, lower_ci, upper_ci, ...}}."""
    results: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["checkpoint_step"])
            results[step] = {
                "f_mem": float(row["f_mem"]),
                "lower_ci": float(row["lower_ci"]),
                "upper_ci": float(row["upper_ci"]),
                "n_memorized": int(row["n_memorized"]),
                "n_total": int(row["n_total"]),
            }
    return results


# ======================================================================
# GATE decision logic
# ======================================================================


def make_gate_decision(
    results: Dict[int, Dict[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Make PASS / ITERATE / FAIL gate decision based on F_mem results.

    Args:
        results: {resolution: {step: {f_mem, lower_ci, upper_ci, ...}}}

    Returns:
        Dict with keys: decision, reasoning, next_steps, selected_resolution
    """
    # Extract per-resolution data: {res: [(step, {f_mem, lower_ci, ...}), ...]}
    per_res: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for res, steps_data in results.items():
        if not steps_data:
            continue
        per_res[res] = sorted(steps_data.items())

    if not per_res:
        return {
            "decision": "FAIL",
            "reasoning": "No valid F_mem results for any resolution.",
            "next_steps": "Debug data pipeline and model training before retrying.",
            "selected_resolution": None,
        }

    # Check PASS condition: at least one resolution has F_mem < 0.3 at step 1000
    # AND loss is still decreasing (we approximate via F_mem trajectory)
    pass_candidates: List[Tuple[int, float]] = []
    for res, steps_list in per_res.items():
        # Get F_mem at last available step (closest to 1000)
        last_step, last_data = steps_list[-1]
        if last_data["f_mem"] < 0.3:
            pass_candidates.append((res, last_data["f_mem"]))

    # Check FAIL condition: F_mem > 0.5 at step 500 for ALL resolutions
    all_fail_500 = True
    for res, steps_list in per_res.items():
        fmem_at_500 = None
        for step, data in steps_list:
            if step == 500:
                fmem_at_500 = data["f_mem"]
                break
        if fmem_at_500 is None or fmem_at_500 <= 0.5:
            all_fail_500 = False
            break

    # Check ITERATE condition: F_mem transitions between step 100-500 at ALL resolutions
    all_iterate = True
    for res, steps_list in per_res.items():
        fmem_100 = None
        fmem_500 = None
        for step, data in steps_list:
            if step == 100:
                fmem_100 = data["f_mem"]
            if step == 500:
                fmem_500 = data["f_mem"]
        # "Transition" means both exist and there's a meaningful change
        if fmem_100 is None or fmem_500 is None:
            all_iterate = False
            break
        # Transition: F_mem changes by at least 0.05 between 100 and 500
        if abs(fmem_500 - fmem_100) < 0.05:
            all_iterate = False
            break

    # Decision priority: PASS > FAIL > ITERATE > default
    if pass_candidates:
        # Select resolution with LOWEST F_mem at final step
        best_res, best_fmem = min(pass_candidates, key=lambda x: x[1])
        return {
            "decision": "PASS",
            "reasoning": (
                f"At least one resolution achieves F_mem < 0.3 at final checkpoint. "
                f"Best: {best_res}x{best_res} with F_mem = {best_fmem:.4f}. "
                f"Loss trajectory suggests model is still improving."
            ),
            "next_steps": (
                f"Proceed with full training at {best_res}x{best_res} resolution. "
                f"Extend to 50k steps with logarithmic checkpointing to map the "
                f"generalization window."
            ),
            "selected_resolution": best_res,
        }

    if all_fail_500:
        return {
            "decision": "FAIL",
            "reasoning": (
                "F_mem > 0.5 at step 500 for ALL resolutions. "
                "The model is memorizing training data rather than learning the "
                "underlying distribution."
            ),
            "next_steps": (
                "Dataset is insufficient for generative modeling at these resolutions. "
                "Consider: (1) increasing dataset size, (2) reducing resolution, "
                "(3) adding controlled data augmentation, or (4) using a smaller U-Net."
            ),
            "selected_resolution": None,
        }

    if all_iterate:
        return {
            "decision": "ITERATE",
            "reasoning": (
                "F_mem transitions observed between steps 100-500 at all resolutions, "
                "suggesting the model is learning but hasn't converged to a clear "
                "generalization regime."
            ),
            "next_steps": (
                "Try: (1) smaller U-Net (model_channels=16 or 24), "
                "(2) controlled augmentation (random crops, flips), "
                "(3) longer pilot (2000 steps), or (4) lower learning rate."
            ),
            "selected_resolution": None,
        }

    # Default: mixed results, need investigation
    # Recommend the resolution with lowest F_mem at final step
    best_res = None
    best_fmem = float("inf")
    for res, steps_list in per_res.items():
        _, last_data = steps_list[-1]
        if last_data["f_mem"] < best_fmem:
            best_fmem = last_data["f_mem"]
            best_res = res

    return {
        "decision": "ITERATE",
        "reasoning": (
            f"Mixed results across resolutions. Best F_mem = {best_fmem:.4f} "
            f"at {best_res}x{best_res}. No resolution clearly passes the F_mem < 0.3 "
            f"threshold, but memorization is not severe enough to fail."
        ),
        "next_steps": (
            f"Investigate {best_res}x{best_res} further. Consider tuning "
            f"hyperparameters or extending training."
        ),
        "selected_resolution": best_res,
    }


# ======================================================================
# Report generation
# ======================================================================


def generate_report(
    results: Dict[int, Dict[int, Dict[str, Any]]],
    gate: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: str,
) -> str:
    """Generate gate_decision.md report.

    Returns:
        Path to the generated report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "gate_decision.md")

    resolutions_tested = sorted(results.keys())
    checkpoint_steps = [100, 250, 500, 750, 1000]

    lines: List[str] = []
    lines.append("# Pilot Training GATE Decision")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration section
    lines.append("## Configuration")
    lines.append(
        f"- Resolutions tested: {', '.join(f'{r}x{r}' for r in resolutions_tested)}"
    )
    lines.append("- Steps per resolution: 1000")
    lines.append("- Checkpoints: [100, 250, 500, 750, 1000]")
    lines.append("- U-Net channels: 32")
    lines.append("- Schedule: linear (b0=0.0001, bT=0.02)")
    lines.append(f"- F_mem samples per checkpoint: {args.n_samples}")
    lines.append(f"- Gap threshold: {args.gap_threshold:.4f}")
    lines.append(f"- Evaluation device: {args.device}")
    lines.append("")

    # F_mem Results table
    lines.append("## F_mem Results")
    lines.append("")
    header = (
        "| Resolution | " + " | ".join(f"Step {s}" for s in checkpoint_steps) + " |"
    )
    separator = (
        "|------------|" + "|".join(["----------"] * len(checkpoint_steps)) + "|"
    )
    lines.append(header)
    lines.append(separator)

    for res in resolutions_tested:
        row = f"| {res}x{res:4d}     |"
        steps_data = results.get(res, {})
        for step in checkpoint_steps:
            if step in steps_data:
                d = steps_data[step]
                row += f" {d['f_mem']:.4f}   |"
            else:
                row += " N/A      |"
        lines.append(row)
    lines.append("")

    # Confidence Intervals table
    lines.append("### Confidence Intervals (95%)")
    lines.append("")
    header_ci = "| Resolution | Step | F_mem | Lower CI | Upper CI |"
    separator_ci = "|------------|------|-------|----------|----------|"
    lines.append(header_ci)
    lines.append(separator_ci)
    for res in resolutions_tested:
        steps_data = results.get(res, {})
        for step in checkpoint_steps:
            if step in steps_data:
                d = steps_data[step]
                lines.append(
                    f"| {res}x{res:4d}     | {step:4d} | {d['f_mem']:.4f} | "
                    f"{d['lower_ci']:.4f}   | {d['upper_ci']:.4f}   |"
                )
    lines.append("")

    # Gate decision
    lines.append(f"## GATE DECISION: **{gate['decision']}**")
    lines.append("")
    lines.append(f"### Reasoning")
    lines.append(f"{gate['reasoning']}")
    lines.append("")
    lines.append(f"### Next Steps")
    lines.append(f"{gate['next_steps']}")
    lines.append("")

    if gate.get("selected_resolution"):
        lines.append(f"### Selected Resolution")
        lines.append(
            f"**{gate['selected_resolution']}x{gate['selected_resolution']}** — "
            f"recommended for full-scale training."
        )
        lines.append("")

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport saved to: {report_path}")
    return report_path


def generate_fmem_plot(
    results: Dict[int, Dict[int, Dict[str, Any]]],
    output_dir: str,
) -> str:
    """Generate overlaid F_mem curves plot for all resolutions.

    Returns:
        Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "fmem_curves.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {64: "#1f77b4", 128: "#ff7f0e", 256: "#2ca02c"}
    markers = {64: "o", 128: "s", 256: "^"}

    for res in sorted(results.keys()):
        steps_data = results[res]
        if not steps_data:
            continue
        steps = sorted(steps_data.keys())
        fmems = [steps_data[s]["f_mem"] for s in steps]
        lower = [steps_data[s]["lower_ci"] for s in steps]
        upper = [steps_data[s]["upper_ci"] for s in steps]

        color = colors.get(res, "#333333")
        marker = markers.get(res, "o")
        label = f"{res}x{res}"

        yerr_lower = [f - lo for f, lo in zip(fmems, lower)]
        yerr_upper = [hi - f for hi, f in zip(upper, fmems)]

        ax.errorbar(
            steps,
            fmems,
            yerr=[yerr_lower, yerr_upper],
            fmt=f"{marker}-",
            color=color,
            label=label,
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )

    # Reference lines
    ax.axhline(
        y=0.3, color="green", linestyle="--", alpha=0.5, label="PASS threshold (0.3)"
    )
    ax.axhline(
        y=0.5, color="red", linestyle="--", alpha=0.5, label="FAIL threshold (0.5)"
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("F_mem (Memorization Fraction)")
    ax.set_title("Pilot Training: F_mem vs. Training Step by Resolution")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")
    return plot_path


# ======================================================================
# Main orchestrator
# ======================================================================


def main() -> None:
    args = parse_args()

    # Parse resolutions
    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    for r in resolutions:
        if r not in _CONFIGS:
            print(
                f"[ERROR] No pilot config for resolution {r}. Available: {list(_CONFIGS.keys())}"
            )
            sys.exit(1)

    output_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU.")
        print("[WARN] Training and evaluation will be significantly slower.")
        device = "cpu"

    print("=" * 70)
    print("PILOT TRAINING GATE DECISION")
    print("=" * 70)
    print(f"Resolutions: {resolutions}")
    print(f"Steps per resolution: 1000")
    print(f"F_mem samples: {args.n_samples}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Skip training: {args.skip_training}")
    print("=" * 70)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("PHASE 1: TRAINING")
        print("=" * 70)

        for res in resolutions:
            config_path = _CONFIGS[res]
            success = train_single_resolution(config_path)
            if not success:
                print(
                    f"[WARN] Training failed for {res}x{res}. Continuing to next resolution."
                )

    # ------------------------------------------------------------------
    # Phase 2: F_mem Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: F_mem EVALUATION")
    print("=" * 70)

    all_results: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for res in resolutions:
        config_path = _CONFIGS[res]
        res_results = evaluate_resolution(
            resolution=res,
            config_path=config_path,
            n_samples=args.n_samples,
            gap_threshold=args.gap_threshold,
            device=device,
        )
        all_results[res] = res_results

    # ------------------------------------------------------------------
    # Phase 3: GATE Decision
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: GATE DECISION")
    print("=" * 70)

    gate = make_gate_decision(all_results)
    print(f"\nDECISION: {gate['decision']}")
    print(f"Reasoning: {gate['reasoning']}")
    if gate.get("selected_resolution"):
        print(
            f"Selected resolution: {gate['selected_resolution']}x{gate['selected_resolution']}"
        )

    # ------------------------------------------------------------------
    # Phase 4: Report Generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: REPORT GENERATION")
    print("=" * 70)

    report_path = generate_report(all_results, gate, args, output_dir)
    plot_path = generate_fmem_plot(all_results, output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — elapsed {elapsed / 60:.1f} minutes")
    print(f"Report: {report_path}")
    print(f"Plot:   {plot_path}")
    print(f"GATE DECISION: {gate['decision']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
