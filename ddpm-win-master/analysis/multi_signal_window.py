"""Multi-signal generalization window visualization.

Reads F_mem and synergy CSV outputs and produces a combined 2x2 figure:
  (1) F_mem curve with generalization window thresholds
  (2) Wasserstein distance per channel (parsed from minimum_viable_report.md)
  (3) Synergy regime evolution timeline
  (4) Combined normalized signals overlay

Usage:
    python analysis/multi_signal_window.py \
        --fmem_csv analysis/section_2_4/fmem_curve.csv \
        --synergy_csv analysis/section_2_4/synergy/synergy_metrics.csv \
        --output analysis/section_2_4/multi_signal_window.png
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ======================================================================
# Data loading
# ======================================================================


def load_fmem_csv(path: str) -> Dict[str, List[float]]:
    """Load F_mem CSV into columnar lists.

    Returns dict with keys: step, f_mem, std, lower_ci, upper_ci, n_memorized, n_total.
    """
    data: Dict[str, List[float]] = {
        "step": [],
        "f_mem": [],
        "std": [],
        "lower_ci": [],
        "upper_ci": [],
        "n_memorized": [],
        "n_total": [],
    }
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data


def load_synergy_csv(path: str) -> List[Dict[str, float]]:
    """Load synergy metrics CSV as list of row dicts."""
    rows: List[Dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step": float(row["step"]),
                "pair": row["pair"],
                "pearson_r": float(row["pearson_r"]),
                "pearson_p": float(row["pearson_p"]),
                "spearman_r": float(row["spearman_r"]),
                "spearman_p": float(row["spearman_p"]),
                "mi_histogram": float(row["mi_histogram"]),
                "regime": row["regime"],
            })
    return rows


def load_wasserstein_csv(path: str) -> Dict[str, List[float]]:
    """Load Wasserstein CSV if available.

    Expected columns: step, b_emd, g_emd, r_emd.
    """
    data: Dict[str, List[float]] = {"step": [], "b_emd": [], "g_emd": [], "r_emd": []}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(float(row["step"]))
            data["b_emd"].append(float(row["b_emd"]))
            data["g_emd"].append(float(row["g_emd"]))
            data["r_emd"].append(float(row["r_emd"]))
    return data


def parse_wasserstein_from_report(path: str) -> Optional[Dict[str, List[float]]]:
    """Parse Wasserstein data from minimum_viable_report.md Section 3 table.

    Looks for the table with columns: Step, B EMD, G EMD, R EMD.

    Returns dict with keys: step, b_emd, g_emd, r_emd, or None if not found.
    """
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Find Section 3 Wasserstein table
    # Pattern: lines after "## 3. Wasserstein" containing "| Step | B EMD | G EMD | R EMD |"
    in_section = False
    rows: List[List[str]] = []

    for line in content.splitlines():
        line = line.strip()
        if "## 3. Wasserstein" in line:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break  # next section
        if in_section and line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.split("|")]
            # Remove empty strings from leading/trailing |
            cells = [c for c in cells if c]
            if len(cells) >= 4:
                # Skip header row
                try:
                    float(cells[0])  # first cell should be a number
                    rows.append(cells)
                except ValueError:
                    continue

    if not rows:
        return None

    data: Dict[str, List[float]] = {"step": [], "b_emd": [], "g_emd": [], "r_emd": []}
    for row in rows:
        data["step"].append(float(row[0]))
        data["b_emd"].append(float(row[1]))
        data["g_emd"].append(float(row[2]))
        data["r_emd"].append(float(row[3]))

    return data


# ======================================================================
# Plotting
# ======================================================================

# Regime color mapping
REGIME_COLORS = {
    "coupled": "#4CAF50",
    "density-driven": "#2196F3",
    "diversity-driven": "#8BC34A",
    "accessibility-driven": "#F44336",
    "fragmented": "#9E9E9E",
}

# Channel colors
CHANNEL_COLORS = {
    "B": "#2196F3",
    "G": "#4CAF50",
    "R": "#F44336",
}

# Pair colors
PAIR_COLORS = {
    "BG": "#2196F3",
    "BR": "#F44336",
    "GR": "#4CAF50",
}


def _setup_log_xaxis(ax: plt.Axes, steps: List[float]) -> None:
    """Configure log-scale x-axis with nice tick labels."""
    ax.set_xscale("log")
    ax.set_xlim(steps[0] * 0.5, steps[-1] * 2)

    # Custom tick positions
    tick_vals = [100, 1000, 10000, 100000, 500000]
    tick_vals = [t for t in tick_vals if steps[0] * 0.5 <= t <= steps[-1] * 2]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{t:,}" for t in tick_vals], fontsize=9)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())


def plot_fmem_panel(
    ax: plt.Axes,
    fmem: Dict[str, List[float]],
) -> None:
    """Panel 1: F_mem curve with generalization window thresholds."""
    steps = fmem["step"]
    f_vals = fmem["f_mem"]
    lower = fmem["lower_ci"]
    upper = fmem["upper_ci"]

    # Main F_mem line
    ax.plot(steps, f_vals, "o-", color="#1A237E", linewidth=1.8, markersize=4,
            label="$F_{mem}$", zorder=3)

    # Confidence interval shading
    ax.fill_between(steps, lower, upper, alpha=0.15, color="#1A237E", zorder=2)

    # Threshold lines
    ax.axhline(0.05, color="#FF9800", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Window lower (0.05)")
    ax.axhline(0.3, color="#F44336", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Window upper (0.30)")

    # Shade generalization window if F_mem is between thresholds
    for i in range(len(steps)):
        if 0.05 <= f_vals[i] <= 0.3:
            ax.axvspan(steps[i] * 0.8, steps[i] * 1.2, alpha=0.08, color="#4CAF50",
                       zorder=0)

    _setup_log_xaxis(ax, steps)
    ax.set_ylabel("$F_{mem}$", fontsize=10)
    ax.set_title("(a) Memorization Fraction", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.01, max(0.35, max(f_vals) * 1.2))


def plot_wasserstein_panel(
    ax: plt.Axes,
    wass: Dict[str, List[float]],
    fmem_steps: List[float],
) -> None:
    """Panel 2: Wasserstein distance per channel."""
    steps = wass["step"]

    for ch, color in CHANNEL_COLORS.items():
        key = f"{ch.lower()}_emd"
        ax.plot(steps, wass[key], "o-", color=color, linewidth=1.5, markersize=3.5,
                label=f"Channel {ch}", zorder=3)

    _setup_log_xaxis(ax, fmem_steps)
    ax.set_ylabel("Wasserstein Distance", fontsize=10)
    ax.set_title("(b) Distribution Shift (per Channel)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")


def plot_regime_panel(
    ax: plt.Axes,
    synergy: List[Dict],
    fmem_steps: List[float],
) -> None:
    """Panel 3: Synergy regime evolution with pearson_r curves.

    Background shaded by dominant regime per step. Lines show pearson_r per pair.
    """
    # Organize data by step and pair
    steps_set = sorted(set(row["step"] for row in synergy))
    pair_data: Dict[str, Dict[float, float]] = {"BG": {}, "BR": {}, "GR": {}}
    step_regime: Dict[float, str] = {}

    for row in synergy:
        step = row["step"]
        pair = row["pair"]
        if pair in pair_data:
            pair_data[pair][step] = row["pearson_r"]
        step_regime[step] = row["regime"]

    # Background regime shading
    sorted_steps = sorted(step_regime.keys())
    for i in range(len(sorted_steps) - 1):
        s0 = sorted_steps[i]
        s1 = sorted_steps[i + 1]
        regime = step_regime[s0]
        color = REGIME_COLORS.get(regime, "#9E9E9E")
        # Shade from midpoint to midpoint
        mid0 = np.sqrt(s0 * s1) if i > 0 else s0 * 0.5
        mid1 = np.sqrt(s1 * sorted_steps[i + 2]) if i + 2 < len(sorted_steps) else s1 * 2
        ax.axvspan(mid0, mid1, alpha=0.12, color=color, zorder=0)

    # Last step shading
    if sorted_steps:
        s_last = sorted_steps[-1]
        regime = step_regime[s_last]
        color = REGIME_COLORS.get(regime, "#9E9E9E")
        s_prev = sorted_steps[-2] if len(sorted_steps) >= 2 else s_last * 0.5
        ax.axvspan(np.sqrt(s_prev * s_last), s_last * 2, alpha=0.12, color=color, zorder=0)

    # Pair correlation lines
    for pair, color in PAIR_COLORS.items():
        s_vals = sorted(pair_data[pair].keys())
        r_vals = [pair_data[pair][s] for s in s_vals]
        marker = "o" if pair == "BG" else ("s" if pair == "BR" else "^")
        ax.plot(s_vals, r_vals, f"{marker}-", color=color, linewidth=1.5, markersize=3.5,
                label=f"{pair}", zorder=3)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.4)
    ax.axhline(-0.5, color="orange", linestyle=":", alpha=0.4)

    _setup_log_xaxis(ax, fmem_steps)
    ax.set_ylabel("Pearson $r$", fontsize=10)
    ax.set_title("(c) Synergy Regime Evolution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # Regime legend (via text)
    regime_counts: Dict[str, int] = {}
    for r in step_regime.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    # Show top 3 regimes in subtitle
    top_regimes = sorted(regime_counts.items(), key=lambda x: -x[1])[:3]
    regime_text = "  ".join(f"{r}" for r, _ in top_regimes)
    ax.text(0.98, 0.02, regime_text, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", style="italic", color="#555555")

    ax.grid(True, alpha=0.3, which="both")


def plot_combined_panel(
    ax: plt.Axes,
    fmem: Dict[str, List[float]],
    wass: Dict[str, List[float]],
    synergy: List[Dict],
    fmem_steps: List[float],
) -> None:
    """Panel 4: Combined normalized signals overlay.

    - F_mem normalized to [0, 1] by dividing by max(F_mem, 0.01)
    - Wasserstein per channel normalized by max per channel
    - Synergy: average |pearson_r| across pairs (already in [-1, 1])
    """
    # Normalize F_mem
    fmem_max = max(max(fmem["f_mem"]), 0.01)
    f_steps = fmem["step"]
    f_norm = [v / fmem_max for v in fmem["f_mem"]]

    # Normalize Wasserstein per channel
    w_steps = wass["step"]
    w_norms: Dict[str, List[float]] = {}
    for ch in ("B", "G", "R"):
        key = f"{ch.lower()}_emd"
        vals = wass[key]
        w_max = max(max(vals), 1e-8)
        w_norms[ch] = [v / w_max for v in vals]

    # Synergy: average |pearson_r| across pairs per step
    step_pairs: Dict[float, List[float]] = {}
    for row in synergy:
        step_pairs.setdefault(row["step"], []).append(abs(row["pearson_r"]))

    syn_steps = sorted(step_pairs.keys())
    syn_norm = [np.mean(step_pairs[s]) for s in syn_steps]

    # Plot all signals
    ax.plot(f_steps, f_norm, "o-", color="#1A237E", linewidth=2.0, markersize=4,
            label="$F_{mem}$ (norm)", zorder=4)

    for ch, color in CHANNEL_COLORS.items():
        marker = "D" if ch == "B" else ("s" if ch == "G" else "^")
        ax.plot(w_steps, w_norms[ch], f"{marker}-", color=color, linewidth=1.3,
                markersize=3, label=f"W₁ {ch} (norm)", zorder=3, alpha=0.85)

    ax.plot(syn_steps, syn_norm, "v-", color="#FF9800", linewidth=1.5, markersize=3.5,
            label="Avg |r| synergy", zorder=4)

    _setup_log_xaxis(ax, fmem_steps)
    ax.set_ylabel("Normalized Value", fontsize=10)
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_title("(d) Combined Normalized Signals", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3, which="both")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-signal generalization window visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fmem_csv", required=True,
                        help="Path to F_mem CSV (analysis/section_2_4/fmem_curve.csv)")
    parser.add_argument("--synergy_csv", required=True,
                        help="Path to synergy metrics CSV")
    parser.add_argument("--output", required=True,
                        help="Output PNG path")
    parser.add_argument("--wass_csv", default=None,
                        help="Optional: Wasserstein CSV (step,b_emd,g_emd,r_emd)")
    parser.add_argument("--report_md", default=None,
                        help="Optional: path to minimum_viable_report.md for Wasserstein parsing")
    args = parser.parse_args()

    # Load F_mem data
    if not os.path.exists(args.fmem_csv):
        print(f"Error: F_mem CSV not found: {args.fmem_csv}", file=sys.stderr)
        sys.exit(1)
    fmem = load_fmem_csv(args.fmem_csv)
    print(f"Loaded F_mem: {len(fmem['step'])} checkpoints")

    # Load synergy data
    if not os.path.exists(args.synergy_csv):
        print(f"Error: Synergy CSV not found: {args.synergy_csv}", file=sys.stderr)
        sys.exit(1)
    synergy = load_synergy_csv(args.synergy_csv)
    n_steps = len(set(row["step"] for row in synergy))
    print(f"Loaded synergy: {len(synergy)} rows ({n_steps} checkpoints)")

    # Load Wasserstein data (from CSV or report)
    wass = None
    if args.wass_csv and os.path.exists(args.wass_csv):
        wass = load_wasserstein_csv(args.wass_csv)
        print(f"Loaded Wasserstein from CSV: {len(wass['step'])} checkpoints")
    else:
        # Try to find minimum_viable_report.md
        report_path = args.report_md
        if report_path is None:
            # Infer from fmem_csv path
            fmem_dir = os.path.dirname(args.fmem_csv)
            report_path = os.path.join(fmem_dir, "minimum_viable_report.md")

        wass = parse_wasserstein_from_report(report_path)
        if wass:
            print(f"Parsed Wasserstein from report: {len(wass['step'])} checkpoints")
        else:
            print("Warning: No Wasserstein data found. Panel (b) will be empty.", file=sys.stderr)

    # Use F_mem steps as the shared x-axis reference
    fmem_steps = fmem["step"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle("Multi-Signal Generalization Window Analysis", fontsize=14, fontweight="bold",
                 y=0.98)

    # Panel 1: F_mem
    plot_fmem_panel(axes[0, 0], fmem)

    # Panel 2: Wasserstein
    if wass:
        plot_wasserstein_panel(axes[0, 1], wass, fmem_steps)
    else:
        axes[0, 1].text(0.5, 0.5, "No Wasserstein data available",
                        transform=axes[0, 1].transAxes, ha="center", va="center",
                        fontsize=12, color="gray")
        axes[0, 1].set_title("(b) Distribution Shift (per Channel)", fontsize=11, fontweight="bold")

    # Panel 3: Synergy regime
    plot_regime_panel(axes[1, 0], synergy, fmem_steps)

    # Panel 4: Combined normalized
    if wass:
        plot_combined_panel(axes[1, 1], fmem, wass, synergy, fmem_steps)
    else:
        # Fallback: only F_mem + synergy
        plot_combined_panel(
            axes[1, 1], fmem,
            {"step": fmem_steps, "b_emd": [0] * len(fmem_steps),
             "g_emd": [0] * len(fmem_steps), "r_emd": [0] * len(fmem_steps)},
            synergy, fmem_steps,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.subplots_adjust(hspace=0.30, wspace=0.28)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
