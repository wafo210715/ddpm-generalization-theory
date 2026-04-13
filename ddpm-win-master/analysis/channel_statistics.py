"""
Section 2.4 "Minimum Viable" analysis: per-channel statistics across DDPM training.

Processes DDPM checkpoints from full training, generates samples, and computes
per-channel (B/G/R) statistics to detect the generalization window and identify
"irreducible characteristics."

Usage:
    python analysis/channel_statistics.py \
        --checkpoint_dir model_ckpts/hcmc_vitality_v2 \
        --config configs/hcmc_vitality_v2.yaml \
        --n_samples 50 \
        --output_dir analysis/section_2_4
"""

import argparse
import csv
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Make ddpm-win-master importable when running as a standalone script
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
from scipy.stats import wasserstein_distance

from config_loader import TrainingConfig
from datasets.dataset import get_dataset
from evaluation.memorization import (
    _collect_images,
    evaluate_checkpoint_memorization,
)
from evaluation.compute_fmem_batch import discover_checkpoints

# Channel labels matching tensor index order (B=0, G=1, R=2)
CHANNEL_NAMES = {0: "B (density)", 1: "G (diversity)", 2: "R (accessibility)"}
CHANNEL_KEYS = {0: "b", 1: "g", 2: "r"}
CHANNEL_COLORS = {0: "blue", 1: "green", 2: "red"}


# ======================================================================
# Core analysis functions
# ======================================================================


def detect_generalization_window(
    fmem_data: dict,
    window_lower: float = 0.05,
    window_threshold: float = 0.3,
) -> tuple:
    """Detect the generalization window from F_mem values.

    The window starts where F_mem first exceeds *window_lower* (transitioning
    from memorization to generalization) and ends where F_mem exceeds
    *window_threshold* (onset of memorization).

    Args:
        fmem_data: Mapping {step: f_mem_value}.
        window_lower: F_mem threshold for window start (default 0.05).
        window_threshold: F_mem threshold for window end (default 0.3).

    Returns:
        (window_start_step, window_end_step).  Either may be *None* if
        the corresponding boundary is not found.
    """
    sorted_steps = sorted(fmem_data.keys())

    window_start = None
    window_end = None

    for step in sorted_steps:
        val = fmem_data[step]
        if window_start is None and val >= window_lower:
            window_start = step
        if window_start is not None and val >= window_threshold:
            window_end = step
            break

    if window_start is None:
        warnings.warn(
            f"No generalization window detected: F_mem never exceeded "
            f"window_lower={window_lower:.3f}. "
            f"Max F_mem = {max(fmem_data.values()):.4f}"
        )
    elif window_end is None:
        warnings.warn(
            f"Generalization window has no upper bound: F_mem never reached "
            f"window_threshold={window_threshold:.3f}. "
            f"Window starts at step {window_start}."
        )

    return (window_start, window_end)


def compute_channel_stats(images: torch.Tensor) -> dict:
    """Compute per-channel statistics for a batch of images.

    Args:
        images: Float tensor (N, 3, H, W) in [-1, 1] range.

    Returns:
        Dict with keys: {b,g,r}_mean, {b,g,r}_std,
        {b,g,r}_per_sample_means, {b,g,r}_histogram.
        Histograms are (counts, bin_edges) tuples with 50 bins over [0, 1].
    """
    # Convert from [-1, 1] to [0, 1]
    imgs_01 = (images + 1.0) / 2.0
    imgs_01 = imgs_01.clamp(0.0, 1.0)

    result = {}
    for ch_idx in range(3):
        key = CHANNEL_KEYS[ch_idx]
        channel = imgs_01[:, ch_idx, :, :]  # (N, H, W)

        # Global mean and std across all pixels and samples
        result[f"{key}_mean"] = float(channel.mean())
        result[f"{key}_std"] = float(channel.std())

        # Per-sample mean for distribution comparison
        per_sample = channel.mean(dim=(1, 2))  # (N,)
        result[f"{key}_per_sample_means"] = per_sample.cpu().numpy()

        # Histogram (50 bins over [0, 1])
        flat = channel.reshape(-1).cpu().numpy()
        counts, bin_edges = np.histogram(flat, bins=50, range=(0.0, 1.0))
        result[f"{key}_histogram"] = (counts, bin_edges)

    return result


def compute_wasserstein_per_channel(
    samples: torch.Tensor,
    training: torch.Tensor,
) -> dict:
    """Compute 1D Wasserstein distance per channel between samples and training.

    Args:
        samples: Generated images (N_gen, 3, H, W) in [-1, 1].
        training: Training images (N_train, 3, H, W) in [-1, 1].

    Returns:
        Dict with keys: b_wasserstein, g_wasserstein, r_wasserstein.
    """
    samples_01 = (samples + 1.0) / 2.0
    training_01 = (training + 1.0) / 2.0

    result = {}
    for ch_idx in range(3):
        key = CHANNEL_KEYS[ch_idx]
        s_flat = samples_01[:, ch_idx, :, :].reshape(-1).cpu().numpy()
        t_flat = training_01[:, ch_idx, :, :].reshape(-1).cpu().numpy()
        result[f"{key}_wasserstein"] = float(wasserstein_distance(t_flat, s_flat))

    return result


def compute_morans_i(channel_2d: np.ndarray) -> float:
    """Compute Moran's I for spatial autocorrelation of a 2D channel.

    Uses rook contiguity (4-neighbor adjacency: up, down, left, right) weights.

    Args:
        channel_2d: 2D numpy array (H, W) with channel values.

    Returns:
        Moran's I value.  Range approximately [-1, 1].
        Positive = spatial clustering, Negative = dispersion.
    """
    h, w = channel_2d.shape
    n = h * w
    if n < 2:
        return 0.0

    flat = channel_2d.flatten().astype(np.float64)
    mean_val = flat.mean()
    deviations = flat - mean_val

    # Sum of squared deviations (denominator)
    ss = np.sum(deviations**2)
    if ss == 0:
        return 0.0

    # Build 4-neighbor adjacency weights and compute spatial lag
    # Using roll for efficiency (rook contiguity = 4 neighbors: up, down, left, right)
    dev_2d = deviations.reshape(h, w)

    # Neighbors: up, down, left, right
    neighbors = np.zeros_like(dev_2d)
    neighbors[1:, :] += dev_2d[:-1, :]  # from above
    neighbors[:-1, :] += dev_2d[1:, :]  # from below
    neighbors[:, 1:] += dev_2d[:, :-1]  # from left
    neighbors[:, :-1] += dev_2d[:, 1:]  # from right

    # Count of neighbors per cell
    count = np.full((h, w), 4.0)
    count[0, :] -= 1
    count[-1, :] -= 1
    count[:, 0] -= 1
    count[:, -1] -= 1

    # Weighted spatial lag: sum of (w_ij * (x_j - mean)) for each i
    # With 4-neighbor, w_ij = 1/count_i for each neighbor
    weighted_lag = neighbors / count

    # Moran's I = (n / S0) * sum_i(z_i * sum_j(w_ij * z_j)) / sum_i(z_i^2)
    # where S0 = sum of all weights = n (each cell has 4 neighbors with w=1/4)
    # Actually: S0 = sum_ij w_ij. With 4-neighbor and equal weights 1/count_i:
    # Each interior edge is counted once, so S0 = sum of count / avg_weight...
    # Simpler: S0 = total number of neighbor pairs.
    S0 = float(np.sum(count))  # each neighbor pair counted once per direction

    numerator = np.sum(dev_2d * weighted_lag)
    I = (n / S0) * (numerator / ss)

    return float(I)


def identify_irreducible_characteristics(stats_by_step: dict) -> dict:
    """Identify stable vs evolving channel characteristics across checkpoints.

    A characteristic is "stable" (irreducible) if its coefficient of variation
    (CV = std/mean) across checkpoints is below 0.05.

    Args:
        stats_by_step: Dict {step: {channel_stats_dict}} across window
            checkpoints.

    Returns:
        Dict mapping each metric name to {"status": "stable"|"evolving",
        "cv": float, "mean": float, "std": float}.
    """
    if not stats_by_step:
        return {}

    # Collect all metric keys from the first step
    sample_step = next(iter(stats_by_step))
    scalar_keys = [
        k
        for k in stats_by_step[sample_step]
        if isinstance(stats_by_step[sample_step][k], (int, float))
    ]

    result = {}
    for key in scalar_keys:
        values = [stats_by_step[step][key] for step in sorted(stats_by_step)]
        arr = np.array(values, dtype=np.float64)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        cv = float(std / mean) if abs(mean) > 1e-12 else float("inf")
        result[key] = {
            "status": "stable" if cv < 0.05 else "evolving",
            "cv": cv,
            "mean": float(mean),
            "std": float(std),
        }

    return result


# ======================================================================
# Sample generation (reuses evaluate_checkpoint_memorization internals)
# ======================================================================


def generate_samples_for_checkpoint(
    checkpoint_path: str,
    config: "TrainingConfig",
    n_samples: int,
    device: str,
    diffusion_config=None,
) -> torch.Tensor:
    """Generate images from a checkpoint using DDIM sampling.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        config: TrainingConfig instance.
        n_samples: Number of images to generate.
        device: Torch device.
        diffusion_config: Optional pre-built DiffusionConfig.

    Returns:
        Generated images tensor (N, C, H, W) in [-1, 1].
    """
    from model.unet import DDPMUnet
    from model.ddpm import DDPM
    from model.diffusion_config import DiffusionConfig
    from sampling.ddim import sample_ddim

    if diffusion_config is None:
        schedule_kwargs = {
            "n_steps": config.timesteps,
            "schedule_type": config.schedule_type,
            "device": device,
        }
        if config.schedule_type == "linear":
            schedule_kwargs["b0"] = config.b0
            schedule_kwargs["bT"] = config.bT
        elif config.schedule_type == "fixed":
            schedule_kwargs["beta"] = config.beta
        elif config.schedule_type == "vp":
            schedule_kwargs["beta_d"] = config.beta_d
            schedule_kwargs["beta_min"] = config.beta_min
        elif config.schedule_type == "ve":
            schedule_kwargs["sigma_min"] = config.sigma_min
            schedule_kwargs["sigma_max"] = config.sigma_max
        diffusion_config = DiffusionConfig(**schedule_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ddpm_unet = DDPMUnet(config.img_channels, config.model_channels)
    ddpm = DDPM(config.timesteps, ddpm_unet, diffusion_config=diffusion_config)

    if "eps_model" in checkpoint or "beta" in checkpoint:
        ddpm.load_state_dict(checkpoint)
    else:
        ddpm.eps_model.load_state_dict(checkpoint)

    ddpm = ddpm.to(device)
    result = sample_ddim(
        ddpm,
        n_images=n_samples,
        config=config,
        diffusion_config=diffusion_config,
        device=device,
    )
    # sample_ddim returns (generated, noise) or (generated, noise, trajectory)
    generated = result[0]
    return generated


# ======================================================================
# Plotting functions
# ======================================================================


def plot_fmem_curve_with_window(
    fmem_rows: list,
    window_start,
    window_end,
    output_path: str,
    window_lower: float = 0.05,
    window_threshold: float = 0.3,
) -> None:
    """F_mem vs step with shaded generalization window region."""
    steps = [r["step"] for r in fmem_rows]
    fmem = [r["f_mem"] for r in fmem_rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade generalization window
    if window_start is not None and window_end is not None:
        ax.axvspan(
            window_start,
            window_end,
            alpha=0.2,
            color="green",
            label=f"Generalization window ({window_start}-{window_end})",
        )
    elif window_start is not None:
        ax.axvspan(
            window_start,
            max(steps),
            alpha=0.2,
            color="green",
            label=f"Window start at {window_start} (no upper bound)",
        )

    # Threshold lines
    ax.axhline(
        y=window_lower,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Window lower ({window_lower:.2f})",
    )
    ax.axhline(
        y=window_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Window threshold ({window_threshold:.2f})",
    )

    # Error bars
    lower_ci = [r.get("lower_ci", r["f_mem"]) for r in fmem_rows]
    upper_ci = [r.get("upper_ci", r["f_mem"]) for r in fmem_rows]
    yerr_lower = [f - lo for f, lo in zip(fmem, lower_ci)]
    yerr_upper = [hi - f for hi, f in zip(upper_ci, fmem)]

    ax.errorbar(
        steps,
        fmem,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        capsize=4,
        markersize=5,
        linewidth=1.5,
        color="black",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("F_mem")
    ax.set_title("Memorization Fraction vs. Training Step (Generalization Window)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_channel_evolution(
    stats_by_step: dict,
    metric: str,
    output_path: str,
    title: str,
) -> None:
    """Plot B/G/R metric values vs training step within window.

    Args:
        stats_by_step: {step: {b_mean, g_mean, ...}}
        metric: One of 'mean' or 'std'.
        output_path: Where to save the plot.
        title: Plot title.
    """
    sorted_steps = sorted(stats_by_step.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    for ch_idx in range(3):
        key = CHANNEL_KEYS[ch_idx]
        col = f"{key}_{metric}"
        values = [stats_by_step[s][col] for s in sorted_steps]
        ax.plot(
            sorted_steps,
            values,
            "o-",
            color=CHANNEL_COLORS[ch_idx],
            label=CHANNEL_NAMES[ch_idx],
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(f"Channel {metric}")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_wasserstein_evolution(
    wasserstein_by_step: dict,
    output_path: str,
) -> None:
    """Wasserstein distance per channel vs training step."""
    sorted_steps = sorted(wasserstein_by_step.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    for ch_idx in range(3):
        key = CHANNEL_KEYS[ch_idx]
        col = f"{key}_wasserstein"
        values = [wasserstein_by_step[s][col] for s in sorted_steps]
        ax.plot(
            sorted_steps,
            values,
            "o-",
            color=CHANNEL_COLORS[ch_idx],
            label=CHANNEL_NAMES[ch_idx],
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Wasserstein Distance (1D EMD)")
    ax.set_title("Distribution Distance vs. Training Step (Per Channel)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_channel_histograms(
    train_stats: dict,
    sample_stats: dict,
    step: int,
    output_path: str,
) -> None:
    """Per-channel histograms comparing training vs samples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ch_idx in range(3):
        ax = axes[ch_idx]
        key = CHANNEL_KEYS[ch_idx]

        t_counts, t_edges = train_stats[f"{key}_histogram"]
        s_counts, s_edges = sample_stats[f"{key}_histogram"]

        # Normalize to density
        t_density = t_counts / (t_counts.sum() * (t_edges[1] - t_edges[0]))
        s_density = s_counts / (s_counts.sum() * (s_edges[1] - s_edges[0]))
        centers = (t_edges[:-1] + t_edges[1:]) / 2.0
        width = (t_edges[1] - t_edges[0]) * 0.8

        ax.bar(
            centers - width * 0.25,
            t_density,
            width=width * 0.45,
            alpha=0.6,
            color=CHANNEL_COLORS[ch_idx],
            label="Training",
        )
        ax.bar(
            centers + width * 0.25,
            s_density,
            width=width * 0.45,
            alpha=0.6,
            color="gray",
            label="Generated",
        )

        ax.set_title(CHANNEL_NAMES[ch_idx])
        ax.set_xlabel("Pixel value [0, 1]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(f"Channel Histograms at Step {step}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ======================================================================
# Report generation
# ======================================================================


def generate_report(
    fmem_rows: list,
    window_start,
    window_end,
    stats_by_step: dict,
    wasserstein_by_step: dict,
    irreducible: dict,
    morans_i_by_step: dict,
    output_path: str,
) -> None:
    """Generate the minimum_viable_report.md."""
    lines = []
    lines.append("# Section 2.4: Minimum Viable Analysis Report")
    lines.append("")
    lines.append("## 1. F_mem Curve and Generalization Window")
    lines.append("")

    if window_start is not None and window_end is not None:
        lines.append(
            f"The generalization window spans **steps {window_start} to "
            f"{window_end}**, where F_mem transitions from near-zero to above "
            f"the memorization threshold."
        )
    elif window_start is not None:
        lines.append(
            f"The generalization window starts at **step {window_start}** "
            f"but no upper bound was detected (F_mem never reached the "
            f"memorization threshold)."
        )
    else:
        lines.append(
            "**No generalization window was detected.** F_mem remained near "
            "zero throughout training, or the data does not exhibit the "
            "expected transition pattern."
        )
    lines.append("")

    # F_mem table
    lines.append("### F_mem Values at Each Checkpoint")
    lines.append("")
    lines.append("| Step | F_mem | 95% CI | Window |")
    lines.append("|------|-------|--------|--------|")
    for row in fmem_rows:
        step = row["step"]
        f_mem = row["f_mem"]
        lo = row.get("lower_ci", 0.0)
        hi = row.get("upper_ci", 0.0)
        in_window = ""
        if window_start is not None and window_end is not None:
            if window_start <= step <= window_end:
                in_window = "**window**"
            elif step < window_start:
                in_window = "pre-window"
            else:
                in_window = "post-window"
        elif window_start is not None and step >= window_start:
            in_window = "window (open)"
        lines.append(f"| {step} | {f_mem:.4f} | [{lo:.4f}, {hi:.4f}] | {in_window} |")
    lines.append("")

    # Per-channel statistics within window
    window_steps = sorted(stats_by_step.keys())
    if window_steps:
        lines.append("## 2. Per-Channel Statistics (Window Checkpoints)")
        lines.append("")
        lines.append("| Step | B mean | B std | G mean | G std | R mean | R std |")
        lines.append("|------|--------|-------|--------|-------|--------|-------|")
        for step in window_steps:
            s = stats_by_step[step]
            lines.append(
                f"| {step} | {s['b_mean']:.4f} | {s['b_std']:.4f} "
                f"| {s['g_mean']:.4f} | {s['g_std']:.4f} "
                f"| {s['r_mean']:.4f} | {s['r_std']:.4f} |"
            )
        lines.append("")

        # Wasserstein distances
        lines.append("## 3. Wasserstein Distance per Channel (Window Checkpoints)")
        lines.append("")
        lines.append("| Step | B EMD | G EMD | R EMD |")
        lines.append("|------|-------|-------|-------|")
        for step in window_steps:
            w = wasserstein_by_step.get(step, {})
            lines.append(
                f"| {step} | {w.get('b_wasserstein', float('nan')):.4f} "
                f"| {w.get('g_wasserstein', float('nan')):.4f} "
                f"| {w.get('r_wasserstein', float('nan')):.4f} |"
            )
        lines.append("")

        # Moran's I
        if morans_i_by_step:
            lines.append("### Spatial Autocorrelation (Moran's I) of Generated Samples")
            lines.append("")
            lines.append("| Step | B Moran's I | G Moran's I | R Moran's I |")
            lines.append("|------|-------------|-------------|-------------|")
            for step in window_steps:
                mi = morans_i_by_step.get(step, {})
                lines.append(
                    f"| {step} | {mi.get('b', float('nan')):.4f} "
                    f"| {mi.get('g', float('nan')):.4f} "
                    f"| {mi.get('r', float('nan')):.4f} |"
                )
            lines.append("")

    # Irreducible characteristics
    lines.append("## 4. Irreducible Characteristics")
    lines.append("")
    if irreducible:
        stable = [k for k, v in irreducible.items() if v["status"] == "stable"]
        evolving = [k for k, v in irreducible.items() if v["status"] == "evolving"]

        if stable:
            lines.append("### Stable (Irreducible) Characteristics")
            lines.append("")
            lines.append(
                "These statistics remain constant across the generalization "
                "window — the model has captured them as irreducible properties."
            )
            lines.append("")
            lines.append("| Metric | Mean | CV |")
            lines.append("|--------|------|----|")
            for k in sorted(stable):
                v = irreducible[k]
                lines.append(f"| {k} | {v['mean']:.4f} | {v['cv']:.4f} |")
            lines.append("")

        if evolving:
            lines.append("### Evolving Characteristics")
            lines.append("")
            lines.append(
                "These statistics are still changing — the model has not "
                "converged on these aspects."
            )
            lines.append("")
            lines.append("| Metric | Mean | CV |")
            lines.append("|--------|------|----|")
            for k in sorted(evolving):
                v = irreducible[k]
                lines.append(f"| {k} | {v['mean']:.4f} | {v['cv']:.4f} |")
            lines.append("")
    else:
        lines.append("No window checkpoints available for characteristic analysis.")
        lines.append("")

    # Summary
    lines.append("## 5. Summary")
    lines.append("")
    lines.append("### What the Model Learned (Within Window)")
    if window_steps and irreducible:
        stable = [k for k, v in irreducible.items() if v["status"] == "stable"]
        if stable:
            for k in sorted(stable):
                lines.append(f"- **{k}**: stable at {irreducible[k]['mean']:.4f}")
        else:
            lines.append(
                "- No stable characteristics identified (all metrics evolving)"
            )
    else:
        lines.append("- Not enough data to determine")
    lines.append("")

    lines.append("### What Remains Noise")
    if window_steps and irreducible:
        evolving = [k for k, v in irreducible.items() if v["status"] == "evolving"]
        if evolving:
            for k in sorted(evolving):
                lines.append(
                    f"- **{k}**: still evolving (CV={irreducible[k]['cv']:.4f})"
                )
        else:
            lines.append("- All characteristics are stable within the window")
    else:
        lines.append("- Not enough data to determine")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Section 2.4 'Minimum Viable' analysis: per-channel statistics "
            "across DDPM training checkpoints to detect the generalization "
            "window and identify irreducible characteristics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analysis/channel_statistics.py \\\n"
            "      --checkpoint_dir model_ckpts/hcmc_vitality_v2 \\\n"
            "      --config configs/hcmc_vitality_v2.yaml \\\n"
            "      --n_samples 50 \\\n"
            "      --output_dir analysis/section_2_4\n"
        ),
    )
    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing *_step{N}.pth checkpoint files.",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML config file.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of images to generate per checkpoint (default: 50).",
    )
    p.add_argument(
        "--gap_threshold",
        type=float,
        default=1.0 / 3.0,
        help="k-NN gap threshold for F_mem computation (default: 0.333).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda, falls back to cpu if unavailable).",
    )
    p.add_argument(
        "--output_dir",
        default="analysis/section_2_4",
        help="Where to save analysis outputs (default: analysis/section_2_4).",
    )
    p.add_argument(
        "--window_threshold",
        type=float,
        default=0.3,
        help="F_mem threshold for generalization window upper bound (default: 0.3).",
    )
    p.add_argument(
        "--window_lower",
        type=float,
        default=0.05,
        help="F_mem threshold for generalization window lower bound (default: 0.05).",
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
    return p.parse_args()


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

    # --- Load config ---
    config = TrainingConfig.from_yaml(args.config)

    # --- Discover checkpoints ---
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  step {step:>7d}  {os.path.basename(path)}")

    # --- Create output directory ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Load training dataset ---
    dataset = get_dataset(
        name=config.dataset_name,
        img_size=config.img_size,
        data_dir=args.data_dir or config.data_root,
        split=args.split,
        center=config.center,
        normalize=config.normalize,
    )
    n_dataset: int = len(dataset)  # type: ignore[arg-type]
    print(
        f"Training dataset: {n_dataset} images ({config.dataset_name}, split={args.split})"
    )

    # Collect training images once
    train_images = _collect_images(dataset)
    # Also compute training channel stats once
    train_channel_stats = compute_channel_stats(train_images)
    print(
        f"Training channel stats: "
        f"B mean={train_channel_stats['b_mean']:.4f}, "
        f"G mean={train_channel_stats['g_mean']:.4f}, "
        f"R mean={train_channel_stats['r_mean']:.4f}"
    )

    # ==================================================================
    # Phase 1: Compute F_mem for all checkpoints
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("Phase 1: Computing F_mem for all checkpoints")
    print(f"{'=' * 70}")

    fmem_rows = []
    for step, ckpt_path in checkpoints:
        print(f"\n--- Step {step}: {os.path.basename(ckpt_path)} ---")

        result = evaluate_checkpoint_memorization(
            checkpoint_path=ckpt_path,
            train_dataset=train_images,
            config=config,
            n_samples=args.n_samples,
            gap_threshold=args.gap_threshold,
            device=device,
        )

        fmem_rows.append(
            {
                "step": step,
                "f_mem": result["fraction_mem"],
                "std": result["std"],
                "lower_ci": result["lower_ci"],
                "upper_ci": result["upper_ci"],
                "n_memorized": result["n_memorized"],
                "n_total": result["n_total"],
            }
        )

        print(
            f"  F_mem = {result['fraction_mem']:.4f} "
            f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}] "
            f"({result['n_memorized']}/{result['n_total']} memorized)"
        )

    # Sort by step
    fmem_rows.sort(key=lambda r: r["step"])

    # ==================================================================
    # Detect generalization window
    # ==================================================================
    fmem_data = {r["step"]: r["f_mem"] for r in fmem_rows}
    window_start, window_end = detect_generalization_window(
        fmem_data,
        window_lower=args.window_lower,
        window_threshold=args.window_threshold,
    )
    print(f"\nGeneralization window: start={window_start}, end={window_end}")

    # Plot F_mem curve with window
    plot_fmem_curve_with_window(
        fmem_rows,
        window_start,
        window_end,
        os.path.join(output_dir, "fmem_curve_window.png"),
        window_lower=args.window_lower,
        window_threshold=args.window_threshold,
    )
    print(f"  Saved: fmem_curve_window.png")

    # ==================================================================
    # Phase 2: Per-channel statistics for window checkpoints
    # ==================================================================
    # Determine window steps (use all checkpoints if no window detected)
    if window_start is not None:
        if window_end is not None:
            window_steps = [
                s for s, _ in checkpoints if window_start <= s <= window_end
            ]
        else:
            window_steps = [s for s, _ in checkpoints if s >= window_start]
    else:
        # No window — analyze all checkpoints
        window_steps = [s for s, _ in checkpoints]

    # If no steps qualify, use all
    if not window_steps:
        window_steps = [s for s, _ in checkpoints]

    step_to_path = {s: p for s, p in checkpoints}

    print(f"\n{'=' * 70}")
    print(f"Phase 2: Per-channel analysis for {len(window_steps)} checkpoints")
    print(f"{'=' * 70}")

    stats_by_step = {}
    wasserstein_by_step = {}
    morans_i_by_step = {}

    for step in window_steps:
        ckpt_path = step_to_path[step]
        print(f"\n--- Generating samples for step {step} ---")

        generated = generate_samples_for_checkpoint(
            ckpt_path,
            config,
            args.n_samples,
            device,
        )

        # Channel statistics
        ch_stats = compute_channel_stats(generated)
        stats_by_step[step] = ch_stats

        # Wasserstein distance
        w_dist = compute_wasserstein_per_channel(generated, train_images)
        wasserstein_by_step[step] = w_dist

        # Moran's I (compute from first generated sample per channel)
        morans = {}
        imgs_01 = (generated + 1.0) / 2.0
        for ch_idx in range(3):
            key = CHANNEL_KEYS[ch_idx]
            ch_2d = imgs_01[0, ch_idx, :, :].cpu().numpy()
            morans[key] = compute_morans_i(ch_2d)
        morans_i_by_step[step] = morans

        print(
            f"  B: mean={ch_stats['b_mean']:.4f} std={ch_stats['b_std']:.4f} "
            f"EMD={w_dist['b_wasserstein']:.4f} Moran={morans['b']:.4f}"
        )
        print(
            f"  G: mean={ch_stats['g_mean']:.4f} std={ch_stats['g_std']:.4f} "
            f"EMD={w_dist['g_wasserstein']:.4f} Moran={morans['g']:.4f}"
        )
        print(
            f"  R: mean={ch_stats['r_mean']:.4f} std={ch_stats['r_std']:.4f} "
            f"EMD={w_dist['r_wasserstein']:.4f} Moran={morans['r']:.4f}"
        )

        # Channel histogram for this step
        plot_channel_histograms(
            train_channel_stats,
            ch_stats,
            step,
            os.path.join(output_dir, f"channel_histograms_{step}.png"),
        )

    # ==================================================================
    # Phase 3: Identify irreducible characteristics
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Identifying irreducible characteristics")
    print(f"{'=' * 70}")

    # Build scalar-only stats dict for irreducible analysis
    scalar_stats = {}
    for step, stats in stats_by_step.items():
        scalar_stats[step] = {
            k: v for k, v in stats.items() if isinstance(v, (int, float))
        }

    irreducible = identify_irreducible_characteristics(scalar_stats)

    for metric, info in sorted(irreducible.items()):
        print(
            f"  {metric}: {info['status']} (CV={info['cv']:.4f}, "
            f"mean={info['mean']:.4f})"
        )

    # ==================================================================
    # Phase 4: Generate all plots and report
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: Generating plots and report")
    print(f"{'=' * 70}")

    if stats_by_step:
        plot_channel_evolution(
            stats_by_step,
            "mean",
            os.path.join(output_dir, "channel_mean_evolution.png"),
            "Channel Mean Evolution Within Generalization Window",
        )
        plot_channel_evolution(
            stats_by_step,
            "std",
            os.path.join(output_dir, "channel_std_evolution.png"),
            "Channel Std Evolution Within Generalization Window",
        )

    if wasserstein_by_step:
        plot_wasserstein_evolution(
            wasserstein_by_step,
            os.path.join(output_dir, "wasserstein_evolution.png"),
        )

    # Save F_mem CSV
    csv_path = os.path.join(output_dir, "fmem_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "f_mem",
                "std",
                "lower_ci",
                "upper_ci",
                "n_memorized",
                "n_total",
            ],
        )
        writer.writeheader()
        writer.writerows(fmem_rows)

    # Generate report
    report_path = os.path.join(output_dir, "minimum_viable_report.md")
    generate_report(
        fmem_rows=fmem_rows,
        window_start=window_start,
        window_end=window_end,
        stats_by_step=stats_by_step,
        wasserstein_by_step=wasserstein_by_step,
        irreducible=irreducible,
        morans_i_by_step=morans_i_by_step,
        output_path=report_path,
    )

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  Report:  minimum_viable_report.md")
    print(f"  F_mem:   fmem_curve.csv, fmem_curve_window.png")
    print(f"  Plots:   channel_mean_evolution.png")
    print(f"           channel_std_evolution.png")
    print(f"           wasserstein_evolution.png")
    for step in window_steps:
        print(f"           channel_histograms_{step}.png")


if __name__ == "__main__":
    main()
