#!/usr/bin/env python
"""Pre-encoding PCA analysis and typology classification for urban cell images.

Analyses:
  1. PCA intrinsic dimension estimation (scree plot + cumulative variance)
  2. Typology classification via per-channel statistics + k-means
  3. Per-channel distribution analysis (histograms + scatter plots)

Usage:
  # Test with synthetic random images
  python pre_encoding_analysis.py --test --output data-processing/data/hcmc/reports/pre_encoding/

  # Real data
  python pre_encoding_analysis.py --input path/to/cells/ --output data-processing/data/hcmc/reports/pre_encoding/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_images(input_dir, target_size=None):
    """Load all PNG files from input_dir, resize to common shape, return array + filenames.

    Returns:
        images: ndarray (N, H*W*3) flattened float64
        filenames: list of str
        shape: (H, W, 3) original shape before flattening
    """
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {input_dir}")

    loaded = []
    filenames = []
    ref_size = None

    for p in png_files:
        img = Image.open(p).convert("RGB")
        if ref_size is None:
            ref_size = img.size  # (W, H)
        if target_size is not None:
            img = img.resize(target_size, Image.BILINEAR)
        elif img.size != ref_size:
            img = img.resize(ref_size, Image.BILINEAR)
        loaded.append(np.array(img, dtype=np.float64))
        filenames.append(p.name)

    # Stack and flatten: (N, H, W, 3) -> (N, H*W*3)
    arr = np.stack(loaded, axis=0)
    shape = arr.shape[1:]  # (H, W, 3)
    flat = arr.reshape(arr.shape[0], -1)
    return flat, filenames, shape


def run_pca(input_dir, output_dir):
    """Load PNGs, flatten, PCA, scree plot, cumulative variance, intrinsic dim."""
    from sklearn.decomposition import PCA

    print("[PCA] Loading images...")
    X, filenames, shape = _load_images(input_dir)

    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features, 200)  # cap for speed
    print(f"[PCA] Data matrix: {n_samples} samples x {n_features} features")
    print(f"[PCA] Fitting PCA with {n_components} components...")

    pca = PCA(n_components=n_components)
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)

    # Intrinsic dimension: first k where cumulative >= 0.95
    k95 = int(np.searchsorted(cumulative, 0.95) + 1)
    k99 = int(np.searchsorted(cumulative, 0.99) + 1)
    print(f"[PCA] Intrinsic dim (95% variance): {k95}")
    print(f"[PCA] Intrinsic dim (99% variance): {k99}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Scree plot (8x4, 150 dpi)
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=150)
    components = np.arange(1, len(eigenvalues) + 1)
    ax1.bar(components, eigenvalues, color="steelblue", alpha=0.7, label="Eigenvalue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        components,
        cumulative,
        color="crimson",
        linewidth=1.5,
        label="Cumulative Variance",
    )
    ax2.axhline(
        y=0.95, color="gray", linestyle="--", linewidth=0.8, label="95% threshold"
    )
    ax2.set_ylabel("Cumulative Variance", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title(f"PCA Scree Plot (intrinsic dim @95%: {k95})")
    plt.tight_layout()
    out_path = Path(output_dir) / "pca_scree.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PCA] Saved scree plot to {out_path}")


def classify_typology(input_dir, output_dir, k_range):
    """Extract per-channel stats, k-means, silhouette, save CSV."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    print("[Typology] Loading images...")
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {input_dir}")

    rows = []
    for p in png_files:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float64)
        # Channel order: R=accessibility (0), G=diversity (1), B=density (2)
        R_mean, G_mean, B_mean = (
            img[:, :, 0].mean(),
            img[:, :, 1].mean(),
            img[:, :, 2].mean(),
        )
        R_std, G_std, B_std = img[:, :, 0].std(), img[:, :, 1].std(), img[:, :, 2].std()
        rows.append(
            {
                "filename": p.name,
                "B_mean": B_mean,
                "G_mean": G_mean,
                "R_mean": R_mean,
                "B_std": B_std,
                "G_std": G_std,
                "R_std": R_std,
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    feature_cols = ["B_mean", "G_mean", "R_mean", "B_std", "G_std", "R_std"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = k_range[0]
    best_score = -1.0
    scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if k > 1 and len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1.0
        scores[k] = score
        print(f"[Typology] k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"[Typology] Best k={best_k} (silhouette={best_score:.4f})")

    # Refit with best k
    best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = best_km.fit_predict(X_scaled)

    csv_path = Path(output_dir) / "typology_labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Typology] Saved {csv_path} ({len(df)} rows, {best_k} clusters)")


def analyze_distributions(input_dir, output_dir):
    """Per-channel histograms, cross-channel scatter plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[Distributions] Loading images...")
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {input_dir}")

    # Collect per-pixel channel values (sample if too many)
    all_r, all_g, all_b = [], [], []
    pixel_cap = 500_000  # subsample cap per channel for plotting speed

    for p in png_files:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float64)
        flat = img.reshape(-1, 3)
        all_r.append(flat[:, 0])
        all_g.append(flat[:, 1])
        all_b.append(flat[:, 2])

    r = np.concatenate(all_r).ravel()
    g = np.concatenate(all_g).ravel()
    b = np.concatenate(all_b).ravel()

    # Subsample for scatter plots if too large
    rng = np.random.default_rng(42)
    if len(r) > pixel_cap:
        idx = rng.choice(len(r), size=pixel_cap, replace=False)
        r_s, g_s, b_s = r[idx], g[idx], b[idx]
    else:
        r_s, g_s, b_s = r, g, b

    # --- Marginal histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    channels = [
        (b, "B (density)", "Blues"),
        (g, "G (diversity)", "Greens"),
        (r, "R (accessibility)", "Reds"),
    ]
    for ax, (data, label, cmap_name) in zip(axes, channels):
        cmap = plt.get_cmap(cmap_name)
        ax.hist(data, bins=50, color=cmap(0.6), edgecolor="white", linewidth=0.3)
        ax.set_title(label)
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Count")
    plt.tight_layout()
    hist_path = Path(output_dir) / "channel_histograms.png"
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"[Distributions] Saved {hist_path}")

    # --- Scatter plots (6x6, 150 dpi) ---
    fig, axes = plt.subplots(1, 3, figsize=(6, 6), dpi=150)
    scatter_cfg = [
        (b_s, g_s, "B (density)", "G (diversity)", "steelblue", "forestgreen"),
        (b_s, r_s, "B (density)", "R (accessibility)", "steelblue", "firebrick"),
        (g_s, r_s, "G (diversity)", "R (accessibility)", "forestgreen", "firebrick"),
    ]
    for ax, (x, y, xlabel, ylabel, xcol, ycol) in zip(axes, scatter_cfg):
        ax.scatter(x, y, s=0.3, alpha=0.3, c=xcol, edgecolors="none")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    scatter_path = Path(output_dir) / "channel_scatter.png"
    fig.savefig(scatter_path)
    plt.close(fig)
    print(f"[Distributions] Saved {scatter_path}")


def _generate_test_images(output_dir, n=50, size=(128, 128)):
    """Generate synthetic test images for --test mode."""
    test_dir = Path(output_dir) / "_test_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(n):
        # Create 3-channel images with some structure
        r = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
        img = Image.fromarray(r)
        img.save(test_dir / f"{i:04d}.png")

    print(f"[Test] Generated {n} test images ({size[0]}x{size[1]}) in {test_dir}")
    return str(test_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-encoding PCA analysis and typology classification for urban cell images."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing PNG cell images (required if not --test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate 50 random 128x128x3 test images and run analysis on them",
    )
    parser.add_argument(
        "--k-range",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
        help="k values for k-means typology classification (default: 3 4 5 6 7)",
    )
    args = parser.parse_args()

    if not args.test and args.input is None:
        parser.error("--input is required when not using --test mode")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test:
        input_dir = _generate_test_images(output_dir)
    else:
        input_dir = args.input

    print(f"{'=' * 60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"K-range: {args.k_range}")
    print(f"{'=' * 60}")

    run_pca(input_dir, str(output_dir))
    classify_typology(input_dir, str(output_dir), args.k_range)
    analyze_distributions(input_dir, str(output_dir))

    # Clean up test images
    if args.test:
        import shutil

        test_dir = Path(output_dir) / "_test_images"
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"[Test] Cleaned up {test_dir}")

    print(f"\n{'=' * 60}")
    print("All analyses complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
