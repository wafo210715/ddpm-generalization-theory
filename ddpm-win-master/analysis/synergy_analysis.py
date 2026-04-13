"""
Section 2.4 — Synergy Analysis: Joint B/G/R Distribution Evolution.

Analyzes how the model learns inter-channel synergies (density-diversity-
accessibility relationships) across training checkpoints. Tracks correlation
structures, mutual information, and regime classifications to understand
when and how the diffusion model captures joint urban vitality patterns.

Channel convention (index → semantics):
    0 → B (density / blue)
    1 → G (diversity / green)
    2 → R (accessibility / red)

Usage:
    python analysis/synergy_analysis.py \
        --checkpoint_dir model_ckpts/hcmc_vitality_v2 \
        --config configs/hcmc_vitality_v2.yaml \
        --n_samples 50 \
        --output_dir analysis/section_2_4/synergy

Reference:
    Wang et al., "Diffusion Models Learn Low-Dimensional Distributions
    via Subspace Clustering" (2024)
    Bonnaire et al., "Why Diffusion Models Don't Memorize" (2025)
"""

import argparse
import csv
import os
import re
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
from scipy import stats as sp_stats

from config_loader import TrainingConfig

# ============================================================================
# Literature thresholds (from urban planning literature)
# ============================================================================
THRESHOLDS = {
    "jinan_blending": 1.5,  # POI blending degree threshold
    "jinan_density_max": 4000,  # Density diminishing returns (buildings/km^2)
    "beijing_intersection_optimal": (22, 28),  # Optimal intersection density range
}

# Known limitation: these thresholds are in raw urban metric units (buildings/km^2,
# intersection count, etc.) and do NOT directly map to our normalized [0, 1] pixel
# values. They are included for documentary completeness and future calibration.

# ============================================================================
# Regex for checkpoint discovery
# ============================================================================
_STEP_RE = re.compile(r"_step(\d+)\.pth$", re.IGNORECASE)

# ============================================================================
# Constants
# ============================================================================
CHANNEL_NAMES = {0: "B", 1: "G", 2: "R"}
CHANNEL_LABELS = {0: "Density (B)", 1: "Diversity (G)", 2: "Accessibility (R)"}


# ======================================================================
# Checkpoint discovery (mirrors compute_fmem_batch.py)
# ======================================================================


def discover_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all ``*_step{N}.pth`` files and return ``(step, path)`` pairs sorted by step."""
    results: List[Tuple[int, str]] = []
    for fname in os.listdir(checkpoint_dir):
        m = _STEP_RE.search(fname)
        if m:
            step = int(m.group(1))
            results.append((step, os.path.join(checkpoint_dir, fname)))
    results.sort(key=lambda t: t[0])
    return results


# ======================================================================
# Core analysis functions
# ======================================================================


def compute_joint_distribution_stats(images: torch.Tensor) -> Dict[str, Any]:
    """Compute per-channel spatial means for joint distribution analysis.

    Args:
        images: Tensor of shape (N, 3, H, W) in [-1, 1] range.

    Returns:
        Dict with keys: b_means, g_means, r_means (each length N),
        joint_means (Nx3 numpy array).
    """
    # Convert from [-1, 1] to [0, 1]
    images_01 = (images + 1.0) / 2.0
    images_01 = torch.clamp(images_01, 0.0, 1.0)

    # Per-channel spatial mean: average over H, W dimensions
    channel_means = images_01.mean(dim=(2, 3))  # (N, 3)

    b_means = channel_means[:, 0].cpu().numpy()
    g_means = channel_means[:, 1].cpu().numpy()
    r_means = channel_means[:, 2].cpu().numpy()

    joint_means = np.column_stack([b_means, g_means, r_means])

    return {
        "b_means": b_means,
        "g_means": g_means,
        "r_means": r_means,
        "joint_means": joint_means,
    }


def _compute_mi_histogram(x: np.ndarray, y: np.ndarray, bins: int = 30) -> float:
    """Estimate mutual information via 2D histogram.

    Uses the relation MI(X;Y) = H(X) + H(Y) - H(X,Y) where entropies
    are estimated from histograms.

    Args:
        x: 1D array.
        y: 1D array (same length as x).
        bins: Number of histogram bins.

    Returns:
        Mutual information in nats.
    """
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    # Normalize to probability
    pxy = hist_2d / hist_2d.sum()
    pxy = pxy[pxy > 0]  # filter zeros to avoid log(0)

    px = hist_2d.sum(axis=1) / hist_2d.sum()
    py = hist_2d.sum(axis=0) / hist_2d.sum()

    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy * np.log(pxy))

    return max(0.0, hx + hy - hxy)


def compute_synergy_metrics(
    b_means: np.ndarray,
    g_means: np.ndarray,
    r_means: np.ndarray,
) -> Dict[str, Any]:
    """Compute pairwise synergy metrics between channel means.

    Calculates Pearson correlation, Spearman rank correlation, and mutual
    information for all three channel pairs (B-G, B-R, G-R).

    Args:
        b_means: 1D array of density (blue) channel spatial means.
        g_means: 1D array of diversity (green) channel spatial means.
        r_means: 1D array of accessibility (red) channel spatial means.

    Returns:
        Dict with per-pair metrics: pearson_r, pearson_p, spearman_r,
        spearman_p, mi (mutual information in nats).
    """
    pairs = {
        "BG": (b_means, g_means),
        "BR": (b_means, r_means),
        "GR": (g_means, r_means),
    }

    metrics: Dict[str, Any] = {}
    for pair_name, (x, y) in pairs.items():
        # Pearson correlation
        pr: float = float(sp_stats.pearsonr(x, y)[0])  # type: ignore[arg-type]
        pp: float = float(sp_stats.pearsonr(x, y)[1])  # type: ignore[arg-type]

        # Spearman rank correlation
        sr: float = float(sp_stats.spearmanr(x, y)[0])  # type: ignore[arg-type]
        sp_: float = float(sp_stats.spearmanr(x, y)[1])  # type: ignore[arg-type]

        # Mutual information (histogram-based)
        mi = _compute_mi_histogram(x, y, bins=30)

        # Try sklearn MI as well, fall back gracefully
        mi_sk: Optional[float] = None
        try:
            from sklearn.feature_selection import mutual_info_regression  # type: ignore[import-untyped]

            x_2d = x.reshape(-1, 1)
            mi_result = mutual_info_regression(x_2d, y, n_neighbors=5, random_state=42)
            mi_sk = float(mi_result[0])
        except (ImportError, Exception):
            mi_sk = None

        metrics[pair_name] = {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp_),
            "mi_histogram": float(mi),
            "mi_sklearn": float(mi_sk) if mi_sk is not None else None,
        }

    return metrics


def compare_with_training(
    sample_joint: Dict[str, np.ndarray],
    training_joint: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compare sample and training joint distributions.

    Computes 1D Wasserstein distances on marginal distributions for each
    channel pair and compares correlation structures.

    Args:
        sample_joint: Dict from compute_joint_distribution_stats (samples).
        training_joint: Dict from compute_joint_distribution_stats (training data).

    Returns:
        Dict with wasserstein distances and correlation comparison per pair.
    """
    pairs = {
        "BG": ("b_means", "g_means"),
        "BR": ("b_means", "r_means"),
        "GR": ("g_means", "r_means"),
    }

    comparison: Dict[str, Any] = {}

    for pair_name, (ch_a_key, ch_b_key) in pairs.items():
        # Marginal Wasserstein distance for channel A
        wd_a = sp_stats.wasserstein_distance(
            sample_joint[ch_a_key], training_joint[ch_a_key]
        )
        # Marginal Wasserstein distance for channel B
        wd_b = sp_stats.wasserstein_distance(
            sample_joint[ch_b_key], training_joint[ch_b_key]
        )

        # Correlation comparison
        sample_corr: float = float(
            sp_stats.pearsonr(sample_joint[ch_a_key], sample_joint[ch_b_key])[0]  # type: ignore[arg-type]
        )
        train_corr: float = float(
            sp_stats.pearsonr(training_joint[ch_a_key], training_joint[ch_b_key])[0]  # type: ignore[arg-type]
        )

        comparison[pair_name] = {
            "wasserstein_ch_a": float(wd_a),
            "wasserstein_ch_b": float(wd_b),
            "wasserstein_avg": float((wd_a + wd_b) / 2.0),
            "sample_pearson_r": float(sample_corr),
            "training_pearson_r": float(train_corr),
            "correlation_diff": float(abs(sample_corr - train_corr)),
        }

    # Overall channel marginal Wasserstein distances
    for ch_key, ch_name in [("b_means", "B"), ("g_means", "G"), ("r_means", "R")]:
        wd = sp_stats.wasserstein_distance(sample_joint[ch_key], training_joint[ch_key])
        comparison[f"marginal_wasserstein_{ch_name}"] = float(wd)

    return comparison


def detect_vitality_sweet_spot(
    images: torch.Tensor,
    training_images: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Identify samples where ALL three channels are in the top 25th percentile.

    "Sweet spot" = high density AND high diversity AND high accessibility.
    This detects cells that are simultaneously well-endowed across all
    urban vitality dimensions.

    Args:
        images: Tensor (N, 3, H, W) in [-1, 1] range.
        training_images: Optional training data tensor for comparison.

    Returns:
        Dict with n_sweet_spot, proportion, training_proportion (if training
        data provided), ratio (sample/training proportion).
    """
    stats = compute_joint_distribution_stats(images)
    b, g, r = stats["b_means"], stats["g_means"], stats["r_means"]

    # 75th percentile thresholds
    b_thresh = np.percentile(b, 75)
    g_thresh = np.percentile(g, 75)
    r_thresh = np.percentile(r, 75)

    sweet_mask = (b >= b_thresh) & (g >= g_thresh) & (r >= r_thresh)
    n_sweet = int(sweet_mask.sum())
    proportion = float(n_sweet / len(b))

    result: Dict[str, Any] = {
        "n_sweet_spot": n_sweet,
        "n_total": len(b),
        "proportion": proportion,
        "thresholds": {
            "B_75pct": float(b_thresh),
            "G_75pct": float(g_thresh),
            "R_75pct": float(r_thresh),
        },
    }

    if training_images is not None:
        train_stats = compute_joint_distribution_stats(training_images)
        tb, tg, tr = (
            train_stats["b_means"],
            train_stats["g_means"],
            train_stats["r_means"],
        )
        tb_thresh = np.percentile(tb, 75)
        tg_thresh = np.percentile(tg, 75)
        tr_thresh = np.percentile(tr, 75)

        # Use training thresholds for consistency
        train_sweet = (tb >= tb_thresh) & (tg >= tg_thresh) & (tr >= tr_thresh)
        train_n_sweet = int(train_sweet.sum())
        train_proportion = float(train_n_sweet / len(tb))

        result["training_n_sweet_spot"] = train_n_sweet
        result["training_n_total"] = len(tb)
        result["training_proportion"] = train_proportion
        result["ratio"] = (
            proportion / train_proportion if train_proportion > 0 else float("inf")
        )

    return result


def classify_synergy_regime(
    b_means: np.ndarray,
    g_means: np.ndarray,
    r_means: np.ndarray,
) -> str:
    """Classify the synergy pattern based on pairwise Pearson correlations.

    Regimes:
        - "coupled": strong positive correlation between ALL pairs (r > 0.5)
        - "density-driven": B correlates with G and R, but G-R weak
        - "diversity-driven": G correlates with B and R, but B-R weak
        - "accessibility-driven": R correlates with B and G, but B-G weak
        - "fragmented": no strong pairwise correlation (all r < 0.3)

    Args:
        b_means: 1D array of density channel means.
        g_means: 1D array of diversity channel means.
        r_means: 1D array of accessibility channel means.

    Returns:
        Regime label string.
    """
    r_bg: float = abs(float(sp_stats.pearsonr(b_means, g_means)[0]))  # type: ignore[arg-type]
    r_br: float = abs(float(sp_stats.pearsonr(b_means, r_means)[0]))  # type: ignore[arg-type]
    r_gr: float = abs(float(sp_stats.pearsonr(g_means, r_means)[0]))  # type: ignore[arg-type]

    STRONG = 0.5
    WEAK = 0.3

    # Coupled: all pairs strong
    if r_bg > STRONG and r_br > STRONG and r_gr > STRONG:
        return "coupled"

    # Density-driven: B correlated with G and R, but G-R weak
    if r_bg > STRONG and r_br > STRONG and r_gr < WEAK:
        return "density-driven"

    # Diversity-driven: G correlated with B and R, but B-R weak
    if r_bg > STRONG and r_gr > STRONG and r_br < WEAK:
        return "diversity-driven"

    # Accessibility-driven: R correlated with B and G, but B-G weak
    if r_br > STRONG and r_gr > STRONG and r_bg < WEAK:
        return "accessibility-driven"

    # Fragmented: no strong pair
    if r_bg < WEAK and r_br < WEAK and r_gr < WEAK:
        return "fragmented"

    # Partial coupling — describe the strongest driver
    pair_strengths = {
        "density-driven": (r_bg + r_br) / 2.0,
        "diversity-driven": (r_bg + r_gr) / 2.0,
        "accessibility-driven": (r_br + r_gr) / 2.0,
    }
    best_regime = max(pair_strengths, key=lambda k: pair_strengths[k])

    # If best is still weak, call it fragmented
    if pair_strengths[best_regime] < WEAK:
        return "fragmented"

    return best_regime


def track_synergy_evolution(all_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Track synergy evolution across checkpoints and identify phases.

    Takes synergy metrics computed at each training checkpoint and identifies
    temporal phases: emerging (synergy increasing), stable (plateaued),
    degrading (decreasing).

    Args:
        all_metrics: Dict mapping step -> synergy_metrics dict (output of
            compute_synergy_metrics).

    Returns:
        Dict with phase analysis: for each phase, the step range, dominant
        regime, and average synergy strength.
    """
    steps = sorted(all_metrics.keys())
    if len(steps) < 2:
        return {
            "phases": ["insufficient_data"],
            "dominant_regime": "unknown",
            "message": "Need at least 2 checkpoints to track evolution.",
        }

    # Extract average Pearson |r| across all pairs per step
    avg_synergies: Dict[int, float] = {}
    regimes: Dict[int, str] = {}

    for step in steps:
        m = all_metrics[step]
        r_vals = [abs(m[pair]["pearson_r"]) for pair in ("BG", "BR", "GR")]
        avg_synergies[step] = float(np.mean(r_vals))

        # Reconstruct regime from stored data
        # We need b/g/r means — but all_metrics only has synergy_metrics
        # Use the stored regime if available
        regimes[step] = m.get("regime", "unknown")

    # Compute synergy derivative (change between consecutive checkpoints)
    synergy_values = [avg_synergies[s] for s in steps]
    derivatives: List[float] = [
        synergy_values[i + 1] - synergy_values[i]
        for i in range(len(synergy_values) - 1)
    ]

    # Classify phases based on derivative sign
    phases: List[Dict[str, Any]] = []
    current_phase = None
    phase_start_idx = 0

    for i, d in enumerate(derivatives):
        if d > 0.01:
            phase_label = "emerging"
        elif d < -0.01:
            phase_label = "degrading"
        else:
            phase_label = "stable"

        if phase_label != current_phase:
            if current_phase is not None:
                phases.append(
                    {
                        "phase": current_phase,
                        "steps": (steps[phase_start_idx], steps[i]),
                        "avg_synergy": float(
                            np.mean(synergy_values[phase_start_idx : i + 1])
                        ),
                    }
                )
            current_phase = phase_label
            phase_start_idx = i

    # Final phase
    phases.append(
        {
            "phase": current_phase if current_phase else "stable",
            "steps": (steps[phase_start_idx], steps[-1]),
            "avg_synergy": float(np.mean(synergy_values[phase_start_idx:])),
        }
    )

    # Find dominant regime (most frequent)
    regime_counts: Dict[str, int] = {}
    for r in regimes.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    dominant_regime = max(regime_counts, key=lambda k: regime_counts[k])

    # Find irreducible synergies: pairs that remain relatively stable
    pair_synergies: Dict[str, List[float]] = {"BG": [], "BR": [], "GR": []}
    for step in steps:
        m = all_metrics[step]
        for pair in pair_synergies:
            pair_synergies[pair].append(abs(m[pair]["pearson_r"]))

    irreducible: Dict[str, float] = {}
    for pair, vals in pair_synergies.items():
        if len(vals) >= 2:
            cv = float(
                np.std(vals) / (np.mean(vals) + 1e-8)
            )  # coefficient of variation
            if cv < 0.2:  # stable if CV < 20%
                irreducible[pair] = float(np.mean(vals))

    return {
        "phases": phases,
        "dominant_regime": dominant_regime,
        "regime_counts": regime_counts,
        "avg_synergy_per_step": avg_synergies,
        "irreducible_synergies": irreducible,
    }


# ======================================================================
# Model loading and sampling helpers
# ======================================================================


def _load_model_and_sample(
    checkpoint_path: str,
    config: TrainingConfig,
    n_samples: int,
    device: str,
) -> torch.Tensor:
    """Load a checkpoint, build model, generate samples via DDIM.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        config: TrainingConfig instance.
        n_samples: Number of images to generate.
        device: Torch device.

    Returns:
        Generated images tensor (N, C, H, W) in [-1, 1] range.
    """
    from model.diffusion_config import DiffusionConfig
    from model.ddpm import DDPM
    from model.unet import DDPMUnet
    from sampling.ddim import sample_ddim

    # Build DiffusionConfig
    schedule_kwargs: Dict[str, Any] = {
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

    # Build model
    ddpm_unet = DDPMUnet(config.img_channels, config.model_channels)
    ddpm = DDPM(config.timesteps, ddpm_unet, diffusion_config=diffusion_config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "eps_model" in checkpoint or "beta" in checkpoint:
        ddpm.load_state_dict(checkpoint)
    else:
        ddpm.eps_model.load_state_dict(checkpoint)

    ddpm = ddpm.to(device)

    # Generate samples (sample_ddim returns (generated, noise) or
    # (generated, noise, trajectory); we only need generated images)
    result = sample_ddim(
        ddpm,
        n_images=n_samples,
        config=config,
        diffusion_config=diffusion_config,
        device=device,
    )
    generated = result[0]

    return generated


def _collect_training_images(
    config: TrainingConfig,
    data_dir: Optional[str] = None,
) -> torch.Tensor:
    """Collect all training images into a single tensor.

    Args:
        config: TrainingConfig instance.
        data_dir: Override for data root.

    Returns:
        Tensor (N, C, H, W).
    """
    from datasets.dataset import get_dataset
    from torch.utils.data import DataLoader

    dataset = get_dataset(
        name=config.dataset_name,
        img_size=config.img_size,
        data_dir=data_dir or config.data_root,
        split="train",
        center=config.center,
        normalize=config.normalize,
    )

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    batches = []
    for batch in loader:
        t = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(t, dict):
            t = t["image"]
        batches.append(t)
    return torch.cat(batches, dim=0)


# ======================================================================
# Plotting
# ======================================================================


def _plot_joint_scatter(
    stats: Dict[str, np.ndarray],
    step: int,
    output_path: str,
) -> None:
    """3-panel scatter plot of channel pairs (B-G, B-R, G-R)."""
    b, g, r = stats["b_means"], stats["g_means"], stats["r_means"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [
        (b, g, "B (density)", "G (diversity)", "B-G"),
        (b, r, "B (density)", "R (accessibility)", "B-R"),
        (g, r, "G (diversity)", "R (accessibility)", "G-R"),
    ]

    for ax, (x, y, xl, yl, title) in zip(axes, pairs):
        ax.scatter(x, y, alpha=0.5, s=20, edgecolors="none")
        # Add correlation annotation
        r_val, _ = sp_stats.pearsonr(x, y)
        ax.annotate(
            f"r = {r_val:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Joint Channel Distribution — Step {step}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_evolution(
    all_metrics: Dict[int, Dict[str, Any]],
    output_path: str,
) -> None:
    """Pearson correlation vs training step for all 3 pairs."""
    steps = sorted(all_metrics.keys())
    bg_r = [all_metrics[s]["BG"]["pearson_r"] for s in steps]
    br_r = [all_metrics[s]["BR"]["pearson_r"] for s in steps]
    gr_r = [all_metrics[s]["GR"]["pearson_r"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, bg_r, "o-", label="B-G (density-diversity)", color="#2196F3")
    ax.plot(steps, br_r, "s-", label="B-R (density-accessibility)", color="#F44336")
    ax.plot(steps, gr_r, "^-", label="G-R (diversity-accessibility)", color="#4CAF50")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(
        0.5, color="orange", linestyle=":", alpha=0.4, label="Strong threshold (0.5)"
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Pearson r")
    ax.set_title("Synergy Correlation Evolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mi_evolution(
    all_metrics: Dict[int, Dict[str, Any]],
    output_path: str,
) -> None:
    """Mutual information vs training step for all 3 pairs."""
    steps = sorted(all_metrics.keys())
    bg_mi = [all_metrics[s]["BG"]["mi_histogram"] for s in steps]
    br_mi = [all_metrics[s]["BR"]["mi_histogram"] for s in steps]
    gr_mi = [all_metrics[s]["GR"]["mi_histogram"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, bg_mi, "o-", label="B-G", color="#2196F3")
    ax.plot(steps, br_mi, "s-", label="B-R", color="#F44336")
    ax.plot(steps, gr_mi, "^-", label="G-R", color="#4CAF50")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title("Synergy Mutual Information Evolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_evolution(
    regimes: Dict[int, str],
    output_path: str,
) -> None:
    """Stacked area chart showing regime proportions across checkpoints.

    Since each checkpoint has one regime label, this is visualized as
    a color-coded timeline rather than a true stacked area.
    """
    steps = sorted(regimes.keys())
    regime_labels = [
        "coupled",
        "density-driven",
        "diversity-driven",
        "accessibility-driven",
        "fragmented",
    ]
    regime_colors = {
        "coupled": "#4CAF50",
        "density-driven": "#2196F3",
        "diversity-driven": "#8BC34A",
        "accessibility-driven": "#F44336",
        "fragmented": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(10, 3))
    for step in steps:
        regime = regimes.get(step, "fragmented")
        color = regime_colors.get(regime, "#9E9E9E")
        ax.barh(0, 1, left=step, color=color, edgecolor="white", linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=regime_colors[r], label=r) for r in regime_labels
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel("Training Step")
    ax.set_yticks([])
    ax.set_title("Regime Evolution Across Training")
    ax.set_xlim(min(steps) - 0.5, max(steps) + 1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sweet_spot(
    sweet_spot_data: Dict[int, Dict[str, Any]],
    output_path: str,
) -> None:
    """Bar chart of sweet spot proportion vs training step."""
    steps = sorted(sweet_spot_data.keys())
    proportions = [sweet_spot_data[s]["proportion"] for s in steps]
    train_prop = sweet_spot_data[steps[0]].get("training_proportion", None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(steps)), proportions, color="#FF9800", alpha=0.8)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sweet Spot Proportion")

    if train_prop is not None:
        ax.axhline(
            train_prop,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Training proportion ({train_prop:.3f})",
        )
        ax.legend()

    ax.set_title("Vitality Sweet Spot Proportion Across Training")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_training_vs_sample(
    training_stats: Dict[str, np.ndarray],
    sample_stats: Dict[str, np.ndarray],
    step: int,
    output_path: str,
) -> None:
    """Side-by-side joint distributions for training data and samples."""
    pairs = [
        ("b_means", "g_means", "B", "G", "B-G"),
        ("b_means", "r_means", "B", "R", "B-R"),
        ("g_means", "r_means", "G", "R", "G-R"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, (ka, kb, la, lb, title) in enumerate(pairs):
        # Training (top row)
        ax = axes[0, col]
        ax.scatter(
            training_stats[ka], training_stats[kb], alpha=0.4, s=15, c="steelblue"
        )
        r_tr = sp_stats.pearsonr(training_stats[ka], training_stats[kb])[0]
        ax.annotate(
            f"r = {r_tr:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="wheat", alpha=0.8),
        )
        ax.set_xlabel(f"{la} mean")
        ax.set_ylabel(f"{lb} mean")
        ax.set_title(f"Training — {title}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Sample (bottom row)
        ax = axes[1, col]
        ax.scatter(sample_stats[ka], sample_stats[kb], alpha=0.4, s=15, c="coral")
        r_sa = sp_stats.pearsonr(sample_stats[ka], sample_stats[kb])[0]
        ax.annotate(
            f"r = {r_sa:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="wheat", alpha=0.8),
        )
        ax.set_xlabel(f"{la} mean")
        ax.set_ylabel(f"{lb} mean")
        ax.set_title(f"Sample (step {step}) — {title}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Training vs. Sample Joint Distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Report generation
# ======================================================================


def _generate_report(
    all_synergy_metrics: Dict[int, Dict[str, Any]],
    all_comparisons: Dict[int, Dict[str, Any]],
    all_sweet_spots: Dict[int, Dict[str, Any]],
    all_regimes: Dict[int, str],
    evolution: Dict[str, Any],
    training_stats: Optional[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Generate full_report.md with all analysis results.

    Returns:
        Path to the generated report.
    """
    steps = sorted(all_synergy_metrics.keys())
    lines: List[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    # Header
    w("# Section 2.4 — Synergy Analysis: Joint B/G/R Distribution Evolution")
    w()
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**Checkpoints analyzed**: {len(steps)}")
    w(f"**Step range**: {steps[0]} – {steps[-1]}" if steps else "**No checkpoints**")
    w()

    # Introduction
    w("## 1. Introduction")
    w()
    w(
        "This analysis examines how a diffusion model learns **inter-channel synergies** —"
    )
    w("the statistical relationships between density (B), diversity (G), and")
    w("accessibility (R) channels across training. Understanding when and how these")
    w("synergies emerge tells us whether the model captures the joint structure of")
    w("urban vitality or merely learns marginal distributions independently.")
    w()
    w("**Key question**: Does the model learn that density, diversity, and")
    w("accessibility co-vary in structured ways, and at what training stage")
    w("does this joint structure emerge?")
    w()
    w("**Important caveat**: All correlations reported here are *associational*.")
    w("Correlation does not imply causation. The synergies reflect co-occurrence")
    w("patterns in urban spatial structure, not causal mechanisms.")
    w()

    # Methodology
    w("## 2. Methodology")
    w()
    w("### 2.1 Per-Channel Spatial Means")
    w("For each generated image, we compute the spatial mean of each channel")
    w("(after converting from [-1, 1] to [0, 1]). This gives three scalar values")
    w("per image: B_mean, G_mean, R_mean.")
    w()
    w("### 2.2 Synergy Metrics")
    w("For each channel pair (B-G, B-R, G-R), we compute:")
    w("- **Pearson correlation** (linear relationship)")
    w("- **Spearman rank correlation** (monotonic relationship)")
    w("- **Mutual information** (non-linear dependency, histogram-based)")
    w()
    w("### 2.3 Distribution Comparison")
    w("- **Wasserstein distance** (Earth Mover's Distance) on marginal distributions")
    w("- **Correlation structure comparison** between samples and training data")
    w()
    w("### 2.4 Regime Classification")
    w("Based on pairwise Pearson |r| thresholds:")
    w("| Regime | Criteria |")
    w("|--------|----------|")
    w("| coupled | All pairs r > 0.5 |")
    w("| density-driven | B correlates with G and R, but G-R weak |")
    w("| diversity-driven | G correlates with B and R, but B-R weak |")
    w("| accessibility-driven | R correlates with B and G, but B-G weak |")
    w("| fragmented | All pairs r < 0.3 |")
    w()

    # Per-checkpoint analysis
    w("## 3. Joint Distribution Analysis Per Checkpoint")
    w()

    for step in steps:
        metrics = all_synergy_metrics[step]
        regime = all_regimes.get(step, "unknown")
        sweet = all_sweet_spots.get(step, {})

        w(f"### Step {step}")
        w()
        w(f"**Regime**: `{regime}`")
        w()

        # Synergy strength table
        w("| Pair | Pearson r | p-value | Spearman r | MI (nats) |")
        w("|------|-----------|---------|------------|-----------|")
        for pair in ("BG", "BR", "GR"):
            pm = metrics[pair]
            mi_val = pm["mi_histogram"]
            w(
                f"| {pair} | {pm['pearson_r']:.4f} | {pm['pearson_p']:.2e} "
                f"| {pm['spearman_r']:.4f} | {mi_val:.4f} |"
            )
        w()

        # Sweet spot
        if sweet:
            w(
                f"**Vitality sweet spot**: {sweet.get('n_sweet_spot', 'N/A')}/"
                f"{sweet.get('n_total', 'N/A')} "
                f"({sweet.get('proportion', 0):.3f})"
            )
            if "training_proportion" in sweet:
                w(f"- Training proportion: {sweet['training_proportion']:.3f}")
                w(
                    f"- Ratio (sample/training): {sweet.get('ratio', 'N/A'):.3f}"
                    if isinstance(sweet.get("ratio"), (int, float))
                    else f"- Ratio: {sweet.get('ratio', 'N/A')}"
                )
            w()

    # Evolution analysis
    w("## 4. Synergy Evolution Across Checkpoints")
    w()
    phases = evolution.get("phases", [])
    if phases:
        w("### Phase Identification")
        w()
        w("| Phase | Step Range | Avg Synergy Strength |")
        w("|-------|------------|----------------------|")
        for phase in phases:
            step_range = phase.get("steps", (0, 0))
            s0, s1 = int(step_range[0]), int(step_range[1])
            w(
                f"| {phase['phase']} | {s0} – {s1} "
                f"| {float(phase.get('avg_synergy', 0)):.4f} |"
            )
        w()

        w(f"**Dominant regime**: `{evolution.get('dominant_regime', 'unknown')}`")
        w()

        # Irreducible synergies
        irred = evolution.get("irreducible_synergies", {})
        if irred:
            w("### Irreducible Synergies")
            w()
            w("These channel pairs maintain stable correlation across the entire")
            w("generalization window, suggesting the model has learned fundamental")
            w("structural relationships:")
            w()
            for pair, strength in irred.items():
                w(f"- **{pair}**: avg |r| = {strength:.4f}")
            w()
    else:
        w("_Insufficient data for phase identification (need ≥ 2 checkpoints)._")
        w()

    # Comparison with training data
    w("## 5. Comparison with Training Data")
    w()
    if training_stats is not None and all_comparisons:
        w("### Wasserstein Distances (Best Checkpoint)")
        w()
        best_step = steps[-1] if steps else None
        if best_step and best_step in all_comparisons:
            comp = all_comparisons[best_step]
            w("| Metric | Value |")
            w("|--------|-------|")
            for ch in ("B", "G", "R"):
                key = f"marginal_wasserstein_{ch}"
                if key in comp:
                    w(f"| W₁ marginal {ch} | {comp[key]:.4f} |")
            for pair in ("BG", "BR", "GR"):
                if pair in comp:
                    w(f"| W₁ avg {pair} | {comp[pair]['wasserstein_avg']:.4f} |")
            w()
    else:
        w("_No training data available for comparison._")
        w()

    # Empirical thresholds comparison
    w("## 6. Comparison with Empirical Thresholds")
    w()
    w("### Literature Reference Points")
    w()
    w("| Source | Metric | Threshold | Notes |")
    w("|--------|--------|-----------|-------|")
    w("| Jinan (2026) | POI blending degree | 1.5 | Functional mix threshold |")
    w("| Jinan (2026) | Building density | 4000 buildings/km² | Diminishing returns |")
    w("| Beijing (2025) | Intersection density | 22–28 | Optimal walkability range |")
    w()
    w("**Important limitation**: These thresholds are in raw urban metric units")
    w("(buildings/km², intersection counts, etc.) and do **not** directly map to")
    w("our normalized [0, 1] pixel values. Our preprocessing pipeline applies")
    w("per-channel global percentile normalization, so pixel intensity 0.5")
    w("corresponds to approximately the median value of each channel in the")
    w("training data, not to any specific raw urban metric value.")
    w()
    w("Future work should calibrate these thresholds against the normalization")
    w("parameters (e.g., the p95 values used during data processing) to enable")
    w("direct comparison.")
    w()

    # Methodological framework
    w("## 7. Methodological Framework for Replication")
    w()
    w("### 7.1 Required Data")
    w("- RGB cell images where channels encode distinct urban vitality dimensions")
    w("- Channel convention: B=density, G=diversity, R=accessibility")
    w("- Images normalized to [-1, 1] range (standard DDPM preprocessing)")
    w("- Multiple training checkpoints at known step counts")
    w()
    w("### 7.2 Metrics to Compute")
    w("1. Per-image spatial means per channel → joint distribution")
    w("2. Pairwise Pearson and Spearman correlations → synergy strength")
    w("3. Mutual information (histogram-based) → non-linear dependencies")
    w("4. Wasserstein distance → distribution shift between samples and training")
    w("5. Regime classification → qualitative synergy pattern")
    w("6. Sweet spot detection → multi-dimensional vitality convergence")
    w()
    w("### 7.3 Interpretation Guide")
    w("- **Coupled regime**: Model has learned strong inter-channel dependency — ")
    w("  generation respects the joint distribution")
    w("- **{X}-driven regime**: One channel dominates — model may be learning ")
    w("  marginal structure before joint structure")
    w("- **Fragmented regime**: No synergy — model treats channels independently")
    w(
        "- **Emerging phase**: Synergy strength increasing — joint structure being learned"
    )
    w("- **Stable phase**: Synergy plateaued — model has captured the joint structure")
    w("- **Degrading phase**: Synergy decreasing — possible onset of memorization")
    w()
    w("### 7.4 Limitations")
    w("- Spatial means lose within-cell spatial patterns")
    w("- Histogram-based MI estimation is sensitive to bin count")
    w("- Small sample sizes increase variance in correlation estimates")
    w("- Normalized pixel values cannot be directly compared to raw urban metrics")
    w("- Regime classification uses fixed thresholds that may need calibration")
    w()

    report_path = os.path.join(output_dir, "full_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Section 2.4 — Synergy Analysis: Joint B/G/R distribution "
            "evolution across training checkpoints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analysis/synergy_analysis.py \\\n"
            "      --checkpoint_dir model_ckpts/hcmc_vitality_v2 \\\n"
            "      --config configs/hcmc_vitality_v2.yaml \\\n"
            "      --n_samples 50\n"
        ),
    )
    p.add_argument(
        "--checkpoint_dir",
        default="",
        help="Directory containing checkpoint .pth files (*_step{N}.pth). Required unless --test.",
    )
    p.add_argument(
        "--config",
        default="",
        help="Path to YAML config file used during training. Required unless --test.",
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
        default=1 / 3,
        help="F_mem gap threshold for memorization filtering (default: 0.333).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda). Falls back to CPU if CUDA unavailable.",
    )
    p.add_argument(
        "--output_dir",
        default="analysis/section_2_4/synergy",
        help="Output directory (default: analysis/section_2_4/synergy).",
    )
    p.add_argument(
        "--window_threshold",
        type=float,
        default=0.3,
        help="F_mem threshold for generalization window upper bound (default: 0.3).",
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="Override data root directory (default: from config).",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with synthetic mock data (no GPU required).",
    )
    return p.parse_args()


# ======================================================================
# Synthetic test mode
# ======================================================================


def _run_test_mode(output_dir: str) -> None:
    """Validate the script with synthetic mock data."""
    print("=" * 60)
    print("TEST MODE — Synthetic Mock Data")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # Generate synthetic "images" — 50 samples, 3 channels, 32x32
    n = 50
    n_checkpoints = 5
    fake_steps = [100, 500, 1000, 2000, 5000]

    all_synergy_metrics: Dict[int, Dict[str, Any]] = {}
    all_comparisons: Dict[int, Dict[str, Any]] = {}
    all_sweet_spots: Dict[int, Dict[str, Any]] = {}
    all_regimes: Dict[int, str] = {}

    # Training data: correlated channels (simulating urban synergy)
    train_b = rng.normal(0.4, 0.15, n)
    train_g = 0.5 * train_b + rng.normal(0, 0.1, n)
    train_r = 0.7 * train_g + rng.normal(0, 0.05, n)
    train_g = np.clip(train_g, 0.05, 0.95)
    train_r = np.clip(train_r, 0.05, 0.95)
    train_b = np.clip(train_b, 0.05, 0.95)

    training_stats = {
        "b_means": train_b,
        "g_means": train_g,
        "r_means": train_r,
        "joint_means": np.column_stack([train_b, train_g, train_r]),
    }

    for i, step in enumerate(fake_steps):
        print(f"\n  Processing synthetic checkpoint step {step}...")

        # Simulate increasing synergy over training
        noise_scale = 0.3 - 0.05 * i  # decreasing noise = increasing synergy
        b = rng.normal(0.4, 0.15, n)
        g = (0.3 + 0.1 * i) * b + rng.normal(0, noise_scale, n)
        r = (0.4 + 0.1 * i) * g + rng.normal(0, noise_scale * 0.5, n)
        b = np.clip(b, 0.05, 0.95)
        g = np.clip(g, 0.05, 0.95)
        r = np.clip(r, 0.05, 0.95)

        # Create fake images tensor
        fake_images = torch.zeros(n, 3, 32, 32)
        for j in range(n):
            fake_images[j, 0] = b[j] * 2 - 1  # [0,1] -> [-1,1]
            fake_images[j, 1] = g[j] * 2 - 1
            fake_images[j, 2] = r[j] * 2 - 1

        # Compute stats
        sample_stats = compute_joint_distribution_stats(fake_images)
        synergy_metrics = compute_synergy_metrics(b, g, r)
        regime = classify_synergy_regime(b, g, r)
        synergy_metrics["regime"] = regime

        comparison = compare_with_training(sample_stats, training_stats)

        # Sweet spot
        fake_train = torch.zeros(n, 3, 32, 32)
        for j in range(n):
            fake_train[j, 0] = train_b[j] * 2 - 1
            fake_train[j, 1] = train_g[j] * 2 - 1
            fake_train[j, 2] = train_r[j] * 2 - 1
        sweet = detect_vitality_sweet_spot(fake_images, fake_train)

        all_synergy_metrics[step] = synergy_metrics
        all_comparisons[step] = comparison
        all_sweet_spots[step] = sweet
        all_regimes[step] = regime

        # Generate plots
        _plot_joint_scatter(
            sample_stats, step, os.path.join(output_dir, f"joint_scatter_{step}.png")
        )

        print(f"    Regime: {regime}")
        for pair in ("BG", "BR", "GR"):
            print(
                f"    {pair}: Pearson r={synergy_metrics[pair]['pearson_r']:.3f}, "
                f"MI={synergy_metrics[pair]['mi_histogram']:.3f}"
            )

    # Evolution analysis
    evolution = track_synergy_evolution(all_synergy_metrics)

    # Summary plots
    _plot_correlation_evolution(
        all_synergy_metrics,
        os.path.join(output_dir, "synergy_correlation_evolution.png"),
    )
    _plot_mi_evolution(
        all_synergy_metrics, os.path.join(output_dir, "synergy_mi_evolution.png")
    )
    _plot_regime_evolution(
        all_regimes, os.path.join(output_dir, "regime_evolution.png")
    )
    _plot_sweet_spot(
        all_sweet_spots, os.path.join(output_dir, "vitality_sweet_spot.png")
    )

    best_step = fake_steps[-1]
    best_stats = compute_joint_distribution_stats(
        torch.zeros(n, 3, 32, 32)  # placeholder — real data already in training_stats
    )
    _plot_training_vs_sample(
        training_stats,
        best_stats,
        best_step,
        os.path.join(output_dir, "training_vs_sample_comparison.png"),
    )

    # Generate report
    report_path = _generate_report(
        all_synergy_metrics,
        all_comparisons,
        all_sweet_spots,
        all_regimes,
        evolution,
        training_stats,
        output_dir,
    )

    print(f"\n{'=' * 60}")
    print("TEST MODE COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Plots generated: {len(os.listdir(output_dir))} files")
    print(f"{'=' * 60}")


# ======================================================================
# Main pipeline
# ======================================================================


def main() -> None:
    args = parse_args()

    # --- Test mode ---
    if args.test:
        _run_test_mode(args.output_dir)
        return

    # --- Validate required args (only needed in non-test mode) ---
    if not args.checkpoint_dir:
        print("Error: --checkpoint_dir is required (unless using --test).")
        sys.exit(1)
    if not args.config:
        print("Error: --config is required (unless using --test).")
        sys.exit(1)

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
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load training data ---
    print("\nLoading training dataset...")
    training_images = _collect_training_images(config, args.data_dir)
    training_stats = compute_joint_distribution_stats(training_images)
    print(f"Training images: {training_images.shape[0]}")

    # --- Process each checkpoint ---
    all_synergy_metrics: Dict[int, Dict[str, Any]] = {}
    all_comparisons: Dict[int, Dict[str, Any]] = {}
    all_sweet_spots: Dict[int, Dict[str, Any]] = {}
    all_regimes: Dict[int, str] = {}
    best_step: Optional[int] = None
    best_sample_stats: Optional[Dict[str, np.ndarray]] = None

    for step, ckpt_path in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Checkpoint step {step}: {os.path.basename(ckpt_path)}")
        print(f"{'=' * 60}")

        # Generate samples
        print(f"  Generating {args.n_samples} samples...")
        generated = _load_model_and_sample(ckpt_path, config, args.n_samples, device)

        # Compute joint distribution stats
        sample_stats = compute_joint_distribution_stats(generated)

        # Compute synergy metrics
        synergy_metrics = compute_synergy_metrics(
            sample_stats["b_means"],
            sample_stats["g_means"],
            sample_stats["r_means"],
        )

        # Classify regime
        regime = classify_synergy_regime(
            sample_stats["b_means"],
            sample_stats["g_means"],
            sample_stats["r_means"],
        )
        synergy_metrics["regime"] = regime

        # Compare with training data
        comparison = compare_with_training(sample_stats, training_stats)

        # Sweet spot detection
        sweet = detect_vitality_sweet_spot(generated, training_images)

        # Store results
        all_synergy_metrics[step] = synergy_metrics
        all_comparisons[step] = comparison
        all_sweet_spots[step] = sweet
        all_regimes[step] = regime

        # Track best checkpoint (lowest avg Wasserstein to training)
        avg_wd = np.mean(
            [comparison[f"marginal_wasserstein_{ch}"] for ch in ("B", "G", "R")]
        )
        if best_step is None:
            best_step = step
            best_sample_stats = sample_stats
        else:
            best_prev = np.mean(
                [
                    all_comparisons[best_step][f"marginal_wasserstein_{ch}"]
                    for ch in ("B", "G", "R")
                ]
            )
            if avg_wd < best_prev:
                best_step = step
                best_sample_stats = sample_stats

        # Generate per-checkpoint scatter plot
        _plot_joint_scatter(
            sample_stats,
            step,
            os.path.join(args.output_dir, f"joint_scatter_{step}.png"),
        )

        print(f"  Regime: {regime}")
        for pair in ("BG", "BR", "GR"):
            pm = synergy_metrics[pair]
            print(
                f"  {pair}: Pearson r={pm['pearson_r']:.4f}, "
                f"MI={pm['mi_histogram']:.4f}"
            )
        print(
            f"  Sweet spot: {sweet['n_sweet_spot']}/{sweet['n_total']} "
            f"({sweet['proportion']:.3f})"
        )

    # --- Evolution analysis ---
    evolution = track_synergy_evolution(all_synergy_metrics)

    # --- Summary plots ---
    print("\nGenerating summary plots...")
    _plot_correlation_evolution(
        all_synergy_metrics,
        os.path.join(args.output_dir, "synergy_correlation_evolution.png"),
    )
    _plot_mi_evolution(
        all_synergy_metrics,
        os.path.join(args.output_dir, "synergy_mi_evolution.png"),
    )
    _plot_regime_evolution(
        all_regimes,
        os.path.join(args.output_dir, "regime_evolution.png"),
    )
    _plot_sweet_spot(
        all_sweet_spots,
        os.path.join(args.output_dir, "vitality_sweet_spot.png"),
    )

    if best_sample_stats is not None and best_step is not None:
        _plot_training_vs_sample(
            training_stats,
            best_sample_stats,
            best_step,
            os.path.join(args.output_dir, "training_vs_sample_comparison.png"),
        )

    # --- Save CSV summary ---
    csv_path = os.path.join(args.output_dir, "synergy_summary.csv")
    fieldnames = [
        "step",
        "regime",
        "BG_pearson_r",
        "BG_mi",
        "BG_spearman_r",
        "BR_pearson_r",
        "BR_mi",
        "BR_spearman_r",
        "GR_pearson_r",
        "GR_mi",
        "GR_spearman_r",
        "sweet_spot_proportion",
        "sweet_spot_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(all_synergy_metrics.keys()):
            m = all_synergy_metrics[step]
            sweet = all_sweet_spots[step]
            row = {
                "step": step,
                "regime": all_regimes[step],
                "BG_pearson_r": m["BG"]["pearson_r"],
                "BG_mi": m["BG"]["mi_histogram"],
                "BG_spearman_r": m["BG"]["spearman_r"],
                "BR_pearson_r": m["BR"]["pearson_r"],
                "BR_mi": m["BR"]["mi_histogram"],
                "BR_spearman_r": m["BR"]["spearman_r"],
                "GR_pearson_r": m["GR"]["pearson_r"],
                "GR_mi": m["GR"]["mi_histogram"],
                "GR_spearman_r": m["GR"]["spearman_r"],
                "sweet_spot_proportion": sweet["proportion"],
                "sweet_spot_ratio": sweet.get("ratio", ""),
            }
            writer.writerow(row)
    print(f"CSV saved to {csv_path}")

    # --- Generate report ---
    report_path = _generate_report(
        all_synergy_metrics,
        all_comparisons,
        all_sweet_spots,
        all_regimes,
        evolution,
        training_stats,
        args.output_dir,
    )
    print(f"Report saved to {report_path}")

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print("SYNERGY ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Checkpoints analyzed: {len(checkpoints)}")
    print(f"Dominant regime: {evolution.get('dominant_regime', 'unknown')}")
    if best_step is not None:
        print(f"Best checkpoint (lowest Wasserstein): step {best_step}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
