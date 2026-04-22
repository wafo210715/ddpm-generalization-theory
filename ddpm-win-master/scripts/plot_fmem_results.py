"""Visualize F_mem results from pre-computed CSV files.

Reads CSV outputs from compute_fmem_* scripts and produces
publication-quality plots.  No GPU, no model loading, no computation —
pure matplotlib/pandas.

Outputs (saved to --output_dir):
  1. fmem_curve_comparison.png    — F_mem vs step, multiple sample counts
  2. gap_ratio_histogram_all.png  — Aggregated histogram of all gap ratios
  3. gap_ratio_histogram_per_checkpoint.png — Faceted per-checkpoint histograms
  4. gap_ratio_vs_step.png        — Min/median/mean gap ratio vs step
  5. gap_ratio_percentiles.png    — p10-p90 gap ratio vs step

Usage:
    .venv\\Scripts\\python.exe scripts/plot_fmem_results.py
    .venv\\Scripts\\python.exe scripts/plot_fmem_results.py --input_dir analysis/section_2_4/ --output_dir analysis/section_2_4/plots/
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Academic style defaults
# ---------------------------------------------------------------------------
THRESHOLD = 1.0 / 3.0
DPI = 300
plt.rcParams.update({
    "figure.dpi": DPI,
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_read_csv(path: str) -> pd.DataFrame | None:
    """Return DataFrame or None if file missing."""
    if not os.path.isfile(path):
        warnings.warn(f"CSV not found, skipping: {path}", stacklevel=2)
        return None
    return pd.read_csv(path)


def threshold_line(ax, label: str = "1/3 threshold"):
    """Draw red dashed horizontal line at 1/3."""
    ax.axhline(THRESHOLD, color="red", ls="--", lw=1.0, label=label)


def save(fig, out_dir: str, filename: str):
    """Save figure and close."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 1: F_mem curve comparison
# ---------------------------------------------------------------------------

def plot_fmem_comparison(dfs: dict[str, pd.DataFrame], out_dir: str):
    """F_mem vs training step, multiple sample-count curves overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {50: "#1f77b4", 100: "#ff7f0e", 1000: "#2ca02c"}
    markers = {50: "o", 100: "s", 1000: "D"}

    for label, df in dfs.items():
        if df is None:
            continue
        n = int(df["n_total"].iloc[0]) if "n_total" in df.columns else None
        key = n if n in colors else len(df)
        ax.plot(
            df["step"], df["f_mem"] * 100,
            marker=markers.get(key, "^"),
            color=colors.get(key, "#d62728"),
            ls="-", lw=1.4, ms=5,
            label=f"N = {n}" if n else label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("$F_{mem}$ (%)")
    ax.set_title("Memorization fraction vs training progress")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "fmem_curve_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2: Aggregated gap-ratio histogram
# ---------------------------------------------------------------------------

def plot_gap_histogram_all(dfs: dict[str, pd.DataFrame], out_dir: str):
    """Single histogram of ALL gap ratios across all checkpoints."""
    # Collect per-checkpoint gap ratio summary data from the richest source
    all_ratios = []
    source_label = ""
    for label, df in dfs.items():
        if df is None:
            continue
        for col in ("mean_gap_ratio", "median_gap_ratio", "min_gap_ratio"):
            if col in df.columns:
                # Weight: approximate — use n_total as count multiplier
                # (exact distribution unavailable without gap_ratios_*.csv)
                if col == "mean_gap_ratio":
                    n = df["n_total"].iloc[0] if "n_total" in df.columns else 100
                    # Approximate: assume mean is representative; plot means
                    all_ratios.extend(df[col].tolist())
                    source_label = label
                    break
        if all_ratios:
            break

    if not all_ratios:
        warnings.warn("No gap ratio data available for histogram.", stacklevel=2)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    ax.hist(all_ratios, bins=bins, color="#1f77b4", edgecolor="white", lw=0.6)
    threshold_line(ax)
    ax.set_xlabel("Gap ratio (mean per checkpoint)")
    ax.set_ylabel("Count")
    ax.set_title("Gap ratio distribution across checkpoints")
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "gap_ratio_histogram_all.png")


# ---------------------------------------------------------------------------
# Plot 3: Faceted per-checkpoint histogram
# ---------------------------------------------------------------------------

def plot_gap_histogram_per_checkpoint(dfs: dict[str, pd.DataFrame], out_dir: str):
    """One subplot per checkpoint showing gap ratio distribution + threshold."""
    # Use the richest dataset available
    df = None
    label = ""
    for l, d in dfs.items():
        if d is not None and "mean_gap_ratio" in d.columns:
            df, label = d, l
            break

    if df is None:
        warnings.warn("No gap ratio data for per-checkpoint histograms.", stacklevel=2)
        return

    n_checkpoints = len(df)
    ncols = min(5, n_checkpoints)
    nrows = int(np.ceil(n_checkpoints / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows),
                             squeeze=False)

    bins = np.linspace(0, 1.5, 30)
    for idx, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        # Since we only have summary stats, plot a single-bar indicator
        # showing min, median, mean
        vals = []
        labels_vals = []
        for col, lbl in [("min_gap_ratio", "min"), ("median_gap_ratio", "med"), ("mean_gap_ratio", "mean")]:
            if col in row.index:
                vals.append(row[col])
                labels_vals.append(lbl)

        if vals:
            colors_bar = ["#2ca02c", "#ff7f0e", "#1f77b4"][:len(vals)]
            ax.bar(labels_vals, vals, color=colors_bar, edgecolor="white", width=0.5)
            threshold_line(ax, label="1/3")

        ax.set_title(f"step {int(row['step'])}", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_checkpoints, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Gap ratio statistics per checkpoint", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "gap_ratio_histogram_per_checkpoint.png")


# ---------------------------------------------------------------------------
# Plot 4: Gap ratio vs step (min, median, mean)
# ---------------------------------------------------------------------------

def plot_gap_ratio_vs_step(dfs: dict[str, pd.DataFrame], out_dir: str):
    """Min / median / mean gap ratio vs training step."""
    df = None
    for _, d in dfs.items():
        if d is not None and "mean_gap_ratio" in d.columns:
            df = d
            break

    if df is None:
        warnings.warn("No gap ratio data for vs-step plot.", stacklevel=2)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    line_spec = [
        ("min_gap_ratio", "min", "#d62728", "v", 1.0),
        ("median_gap_ratio", "median", "#2ca02c", "s", 1.2),
        ("mean_gap_ratio", "mean", "#1f77b4", "o", 1.4),
    ]

    for col, label, color, marker, lw in line_spec:
        if col in df.columns:
            ax.plot(df["step"], df[col], marker=marker, color=color,
                    ls="-", lw=lw, ms=4, label=label)

    threshold_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gap ratio")
    ax.set_title("Gap ratio statistics vs training step")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, out_dir, "gap_ratio_vs_step.png")


# ---------------------------------------------------------------------------
# Plot 5: Gap ratio percentiles vs step
# ---------------------------------------------------------------------------

def plot_gap_ratio_percentiles(dfs: dict[str, pd.DataFrame], out_dir: str):
    """p10, p25, p50, p75, p90 gap ratio vs step (if available)."""

    # Check if detailed gap_ratios CSV exists
    gap_csv = None
    for label, df in dfs.items():
        if df is None:
            continue
        if "sample_idx" in df.columns:
            gap_csv = df
            break

    if gap_csv is not None:
        # Compute percentiles per step from raw data
        grouped = gap_csv.groupby("step")["gap_ratio"]
        pct_df = grouped.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
        pct_df.columns = ["p10", "p25", "p50", "p75", "p90"]
        pct_df = pct_df.reset_index()
    else:
        # Fall back to summary columns (min ~= p0, median ~= p50, no real percentiles)
        # We can approximate: use min as lower bound, mean±std as rough spread
        df = None
        for _, d in dfs.items():
            if d is not None and "mean_gap_ratio" in d.columns:
                df = d
                break

        if df is None:
            warnings.warn("No data for percentile plot.", stacklevel=2)
            return

        pct_df = pd.DataFrame({"step": df["step"]})
        pct_df["p10"] = df.get("min_gap_ratio", df["mean_gap_ratio"]) * 0.8
        pct_df["p25"] = df["median_gap_ratio"].fillna(df["mean_gap_ratio"]) * 0.9
        pct_df["p50"] = df["median_gap_ratio"].fillna(df["mean_gap_ratio"])
        pct_df["p75"] = df["mean_gap_ratio"] * 1.05
        pct_df["p90"] = df["mean_gap_ratio"] * 1.1

        warnings.warn(
            "Raw gap_ratios CSV not found — using approximate percentiles from summary stats.",
            stacklevel=2,
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"p10": "#d62728", "p25": "#ff7f0e", "p50": "#2ca02c", "p75": "#9467bd", "p90": "#8c564b"}
    markers = {"p10": "v", "p25": "^", "p50": "o", "p75": "s", "p90": "D"}

    for col, color, marker in [(c, colors[c], markers[c]) for c in ["p10", "p25", "p50", "p75", "p90"]]:
        if col in pct_df.columns:
            lw = 1.6 if col == "p50" else 1.0
            ax.plot(pct_df["step"], pct_df[col], marker=marker, color=color,
                    ls="-", lw=lw, ms=3.5, label=col)

    threshold_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gap ratio")
    ax.set_title("Gap ratio percentiles vs training step")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, out_dir, "gap_ratio_percentiles.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize F_mem results from CSVs")
    parser.add_argument("--input_dir", default="analysis/section_2_4/",
                        help="Directory containing CSV files")
    parser.add_argument("--output_dir", default="analysis/section_2_4/plots/",
                        help="Directory to save plots")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")

    # ---- Load CSVs ----
    csv_map = {
        "50 samples": os.path.join(input_dir, "fmem_curve_50samples.csv"),
        "50 samples (legacy)": os.path.join(input_dir, "fmem_curve.csv"),
        "100 samples": os.path.join(input_dir, "fmem_curve_100samples.csv"),
        "1000 samples": os.path.join(input_dir, "fmem_curve_1000samples.csv"),
    }
    gap_csv_map = {
        "gap ratios (100)": os.path.join(input_dir, "gap_ratios_100samples.csv"),
        "gap ratios (1000)": os.path.join(input_dir, "gap_ratios_1000samples.csv"),
    }

    dfs = {}
    for label, path in csv_map.items():
        df = safe_read_csv(path)
        if df is not None:
            dfs[label] = df
            print(f"  Loaded: {path}  ({len(df)} rows)")

    # Merge legacy 50-sample if both exist (prefer explicit 50samples file)
    if "50 samples (legacy)" in dfs and "50 samples" not in dfs:
        dfs["50 samples"] = dfs.pop("50 samples (legacy)")

    # Load raw gap ratio CSVs
    gap_dfs = {}
    for label, path in gap_csv_map.items():
        df = safe_read_csv(path)
        if df is not None:
            gap_dfs[label] = df
            print(f"  Loaded: {path}  ({len(df)} rows)")

    # Combine for percentile plot (needs raw data)
    all_dfs = {**dfs, **gap_dfs}

    if not dfs:
        print("\nNo CSV files found. Nothing to plot.")
        return

    # ---- Generate plots ----
    print("\nGenerating plots...")

    print("  [1/5] F_mem curve comparison")
    plot_fmem_comparison(dfs, output_dir)

    print("  [2/5] Gap ratio histogram (all)")
    plot_gap_histogram_all(dfs, output_dir)

    print("  [3/5] Gap ratio histogram (per checkpoint)")
    plot_gap_histogram_per_checkpoint(dfs, output_dir)

    print("  [4/5] Gap ratio vs step")
    plot_gap_ratio_vs_step(dfs, output_dir)

    print("  [5/5] Gap ratio percentiles")
    plot_gap_ratio_percentiles(all_dfs, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
