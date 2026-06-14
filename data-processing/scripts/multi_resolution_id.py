"""
multi_resolution_id.py — Measure intrinsic dimension at multiple resolutions

Purpose: Test the literature prediction that ID is approximately resolution-
invariant for the same underlying data (Haldar et al. AISTATS 2024, Pope et al.
ICLR 2021). Compares PCA k95/k99 and MLE intrinsic dimension across resolutions.

Method:
- Load 4,183 HCMC cell PNGs from data-processing/data/hcmc/processed/cells/
- For each target resolution: resize with PIL.BILINEAR, flatten to 1D vector
- PCA via numpy SVD (no sklearn required)
- MLE via Levina-Bickel 2004 estimator

Output:
- data-processing/data/hcmc/reports/multi_resolution_id/id_vs_resolution.csv
- data-processing/data/hcmc/reports/multi_resolution_id/id_vs_resolution.png
- Per-resolution reports (scree, intrinsic_dim)

Usage:
    python multi_resolution_id.py
    python multi_resolution_id.py --resolutions 128 256 512
    python multi_resolution_id.py --input /path/to/cells --output /path/to/output

References:
- Haldar et al. AISTATS 2024 — "ID unchanged even with varying resolution"
- Pope et al. ICLR 2021 — Cross-dataset ID measurements
- Levina & Bickel 2004 — MLE intrinsic dimension estimator
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import KDTree


def load_images_at_resolution(input_dir: Path, target_size: int):
    """Load all PNG files, resize to (target_size, target_size), return flat matrix.

    Returns:
        X: (n_samples, target_size * target_size * 3) float64
        filenames: list of str
    """
    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {input_dir}")

    print(f"[LOAD] Found {len(png_files)} PNGs in {input_dir}")
    print(f"[LOAD] Resizing to ({target_size}, {target_size})...")

    loaded = []
    filenames = []
    t0 = time.time()
    for p in png_files:
        img = Image.open(p).convert("RGB")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.BILINEAR)
        loaded.append(np.asarray(img, dtype=np.float64).flatten())
        filenames.append(p.name)

    X = np.stack(loaded, axis=0)
    elapsed = time.time() - t0
    print(f"[LOAD] Loaded {X.shape[0]} samples x {X.shape[1]} features in {elapsed:.1f}s")
    return X, filenames


def pca_intrinsic_dim(X: np.ndarray):
    """Compute PCA k95, k99 from cumulative variance.

    Uses numpy SVD. Caps n_components at min(n_samples, n_features, 200) for speed.
    Note: PIL.BILINEAR-bilinear-upscaled 128x128 to 256/512/1024 contains no new
    information, so k95 should be approximately resolution-invariant.
    """
    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features, 200)
    print(f"[PCA] Data: {n_samples} x {n_features}, computing {n_components} components...")

    t0 = time.time()
    # Center data
    X_centered = X - X.mean(axis=0)
    # SVD on centered data; explained_variance = singular_values^2 / (n-1)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance = (S ** 2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_ratio = explained_variance / total_var
    cumulative = np.cumsum(explained_ratio)
    elapsed = time.time() - t0

    # Find k where cumulative >= 0.95 / 0.99
    k95 = int(np.searchsorted(cumulative, 0.95) + 1)
    k99 = int(np.searchsorted(cumulative, 0.99) + 1)
    print(f"[PCA] Computed in {elapsed:.1f}s. k95={k95}, k99={k99}")
    return k95, k99, cumulative, explained_ratio


def mle_intrinsic_dim(X: np.ndarray, k: int = 10):
    """Levina-Bickel 2004 MLE intrinsic dimension estimator.

    For each point x_i, find k nearest neighbors with distances d_ij (j=1..k).
    Estimate local dimension:
        m_k(x_i) = [(1/(k-2)) * sum_{j=1}^{k-1} log(d_ik / d_ij)]^(-1)
    where d_ik is the k-th neighbor distance (max in the k-NN).

    Final: average m_k over all points.

    Note: computationally expensive for large n. For 4183 samples x k=10
    neighbors, KDTree query is O(n log n).
    """
    n_samples = X.shape[0]
    print(f"[MLE] Computing with k={k} neighbors on {n_samples} samples...")

    t0 = time.time()
    tree = KDTree(X)
    # Query k+1 neighbors (includes self at distance 0)
    distances, _ = tree.query(X, k=k + 1)
    # Drop self (first column, distance 0)
    distances = distances[:, 1:]
    elapsed = time.time() - t0
    print(f"[MLE] KDTree built and queried in {elapsed:.1f}s")

    # Compute m_k for each point
    # m_k = (1 / (k-2) * sum_{j=1}^{k-1} log(d_k / d_j))^(-1)
    d_k = distances[:, -1]  # k-th neighbor (max in k-NN)
    log_ratios = np.zeros(n_samples)
    for j in range(k - 1):
        log_ratios += np.log(d_k / distances[:, j])
    m_k = (k - 2) / log_ratios  # Inverse: (1/k-2) * sum -> (k-2)/sum

    # Average over all points
    d_mle = float(np.mean(m_k))
    print(f"[MLE] Intrinsic dimension: {d_mle:.2f}")
    return d_mle


def plot_id_vs_resolution(results, output_path):
    """Plot ID (PCA k95, k99, MLE) vs pixel resolution."""
    resolutions = [r["resolution"] for r in results]
    k95_list = [r["k95"] for r in results]
    k99_list = [r["k99"] for r in results]
    mle_list = [r["mle"] for r in results]
    n_features_list = [r["n_features"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: ID vs resolution
    ax1 = axes[0]
    ax1.plot(resolutions, k95_list, "o-", label="PCA k95 (95% variance)", color="steelblue", linewidth=2, markersize=8)
    ax1.plot(resolutions, k99_list, "s-", label="PCA k99 (99% variance)", color="darkorange", linewidth=2, markersize=8)
    ax1.plot(resolutions, mle_list, "^-", label="MLE (Levina-Bickel)", color="crimson", linewidth=2, markersize=8)
    ax1.set_xlabel("Resolution (px)", fontsize=12)
    ax1.set_ylabel("Intrinsic Dimension", fontsize=12)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(resolutions)
    ax1.set_xticklabels([str(r) for r in resolutions])
    ax1.set_title("ID vs Resolution (PIL.BILINEAR upsampling of 128x128 source)", fontsize=12)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Right: Compression ratio (n_features / k95) vs resolution
    ax2 = axes[1]
    compression_95 = [n / k for n, k in zip(n_features_list, k95_list)]
    compression_99 = [n / k for n, k in zip(n_features_list, k99_list)]
    ax2.plot(resolutions, compression_95, "o-", label="Compression @ k95 (n_features / k95)", color="steelblue", linewidth=2, markersize=8)
    ax2.plot(resolutions, compression_99, "s-", label="Compression @ k99 (n_features / k99)", color="darkorange", linewidth=2, markersize=8)
    ax2.set_xlabel("Resolution (px)", fontsize=12)
    ax2.set_ylabel("Compression Ratio (n_features / ID)", fontsize=12)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(resolutions)
    ax2.set_xticklabels([str(r) for r in resolutions])
    ax2.set_yscale("log")
    ax2.set_title("Compression Ratio vs Resolution", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"[PLOT] Saved ID vs resolution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure intrinsic dimension at multiple resolutions (PCA + MLE)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Directory containing PNG cell images (default: data-processing/data/hcmc/processed/cells/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data-processing/data/hcmc/reports/multi_resolution_id/)",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="List of resolutions to test (default: 64 128 256 512 1024)",
    )
    parser.add_argument(
        "--mle-k",
        type=int,
        default=10,
        help="Number of neighbors for MLE estimator (default: 10)",
    )
    parser.add_argument(
        "--skip-mle",
        action="store_true",
        help="Skip MLE (faster, only PCA)",
    )
    args = parser.parse_args()

    # Default paths
    repo_root = Path(__file__).parent.parent.parent
    if args.input is None:
        args.input = str(repo_root / "data-processing" / "data" / "hcmc" / "processed" / "cells")
    if args.output is None:
        args.output = str(repo_root / "data-processing" / "data" / "hcmc" / "reports" / "multi_resolution_id")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"Multi-Resolution Intrinsic Dimension Measurement")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Resolutions: {args.resolutions}")
    print(f"  MLE k: {args.mle_k} (skip: {args.skip_mle})")
    print(f"{'=' * 70}")

    results = []
    for res in args.resolutions:
        print(f"\n{'─' * 70}")
        print(f"Resolution: {res}x{res}")
        print(f"{'─' * 70}")
        t_start = time.time()

        X, filenames = load_images_at_resolution(input_dir, res)
        n_features = X.shape[1]
        k95, k99, cumulative, explained_ratio = pca_intrinsic_dim(X)

        if args.skip_mle:
            d_mle = float("nan")
        else:
            d_mle = mle_intrinsic_dim(X, k=args.mle_k)

        elapsed = time.time() - t_start
        result = {
            "resolution": res,
            "n_features": n_features,
            "k95": k95,
            "k99": k99,
            "mle": d_mle,
            "elapsed_sec": elapsed,
        }
        results.append(result)
        print(f"[DONE] {res}x{res}: k95={k95}, k99={k99}, MLE={d_mle:.2f} ({elapsed:.1f}s)")

    # Write CSV
    csv_path = output_dir / "id_vs_resolution.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("resolution,n_features,k95,k99,mle,elapsed_sec\n")
        for r in results:
            f.write(f"{r['resolution']},{r['n_features']},{r['k95']},{r['k99']},{r['mle']:.4f},{r['elapsed_sec']:.1f}\n")
    print(f"\n[CSV] Saved to {csv_path}")

    # Write summary
    summary_path = output_dir / "id_vs_resolution_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Resolution Intrinsic Dimension Summary\n\n")
        f.write(f"Input: `{input_dir}`\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results Table\n\n")
        f.write("| Resolution | n_features | PCA k95 | PCA k99 | MLE | Time (s) |\n")
        f.write("|------------|------------|---------|---------|-----|----------|\n")
        for r in results:
            mle_str = f"{r['mle']:.2f}" if not np.isnan(r["mle"]) else "skipped"
            f.write(f"| {r['resolution']} | {r['n_features']} | {r['k95']} | {r['k99']} | {mle_str} | {r['elapsed_sec']:.1f} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("**Haldar et al. AISTATS 2024** explicitly showed ID is unchanged even with varying resolution.\n")
        f.write("**Pope et al. ICLR 2021** showed CIFAR-10 (3K px, ID=26) vs ImageNet (150K px, ID=43) — only 1.65x ID growth for 49x pixels.\n\n")
        f.write("**Expected for this data**: ID should grow logarithmically at most. PIL.BILINEAR upsampling of 128x128 source\n")
        f.write("to higher resolutions adds no new information, so ID should remain approximately 33 across all resolutions.\n")
        f.write("If ID grows significantly (>50 at 512), this suggests the 128x128 representation is lossy at the source level,\n")
        f.write("which would justify a higher-resolution data pipeline (SUBDIV > 20).\n")
    print(f"[SUMMARY] Saved to {summary_path}")

    # Plot
    if not args.skip_mle:
        plot_path = output_dir / "id_vs_resolution.png"
        plot_id_vs_resolution(results, plot_path)

    print(f"\n{'=' * 70}")
    print("All resolutions complete.")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
