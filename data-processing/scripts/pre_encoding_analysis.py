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

    # Write intrinsic dimension estimates
    d_mle = estimate_mle_intrinsic_dim(X, k=10)
    dim_path = Path(output_dir) / "intrinsic_dimension.txt"
    with open(dim_path, "w", encoding="utf-8") as f:
        f.write(f"PCA k95: {k95}\n")
        f.write(f"PCA k99: {k99}\n")
        f.write(f"MLE intrinsic dimension: {d_mle:.2f}\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Total features: {n_features}\n")
    print(f"[PCA] Saved intrinsic dimension estimates to {dim_path}")


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


def estimate_mle_intrinsic_dim(X, k=10):
    """Estimate intrinsic dimension using Levina-Bickel (2004) MLE.

    Uses the correct formula: for each point i, compute
        m_k(i) = (k-2) / sum_{j=1}^{k-1} log(d_k(i) / d_j(i))
    then average over all points.

    Args:
        X: ndarray (N, D) data matrix
        k: number of nearest neighbors to use (default 10)

    Returns:
        float: estimated intrinsic dimension
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:, 0] = self-distance (0), distances[:, 1] = nearest, ..., distances[:, k] = k-th
    # Drop self (column 0)
    distances = distances[:, 1:]  # now columns 0..k-1 = 1st..k-th neighbors

    # d_k = k-th neighbor distance (farthest in k-NN)
    d_k = distances[:, -1]  # column k-1

    # Sum log(d_k / d_j) for j = 1..k-1 (columns 0..k-2)
    eps = 1e-10
    log_ratios = np.zeros(len(X))
    for j in range(k - 1):
        log_ratios += np.log(d_k / (distances[:, j] + eps) + eps)

    # Per-point estimate: m_k = (k-2) / sum_of_log_ratios
    m_k = (k - 2) / (log_ratios + eps)

    # Average over all points
    d_mle = float(np.mean(m_k))
    # Clamp to reasonable range
    d_mle = max(1.0, min(d_mle, X.shape[1]))
    print(f"[MLE] Estimated intrinsic dimension (k={k}): {d_mle:.2f}")
    return d_mle


def _extract_features(input_dir):
    """Extract 6 per-channel features from all PNGs in input_dir.

    Returns:
        pd.DataFrame with columns: filename, B_mean, G_mean, R_mean, B_std, G_std, R_std
        np.ndarray (N, 6) standardized feature matrix
        StandardScaler fitted on features
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

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

    df = pd.DataFrame(rows)
    feature_cols = ["B_mean", "G_mean", "R_mean", "B_std", "G_std", "R_std"]
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, scaler, feature_cols


def plot_cluster_stability(input_dir, output_dir, k_range):
    """Run k-means for each k, compute silhouette scores, plot stability."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print("[ClusterStability] Extracting features...")
    df, X_scaled, _, _ = _extract_features(input_dir)

    k_values = list(range(min(k_range), max(k_range) + 1))
    scores = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if k > 1 and len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1.0
        scores[k] = score
        print(f"[ClusterStability] k={k}  silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(k_values, [scores[k] for k in k_values], color="steelblue", alpha=0.8)
    ax.plot(k_values, [scores[k] for k in k_values], "o-", color="crimson", linewidth=2, markersize=8)
    # Annotate best
    ax.annotate(
        f"Best k={best_k}\nscore={best_score:.4f}",
        xy=(best_k, best_score),
        xytext=(best_k + 0.5, best_score + 0.02),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        fontsize=9,
        color="crimson",
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Cluster Stability: Silhouette Score vs k")
    ax.set_xticks(k_values)
    plt.tight_layout()
    out_path = Path(output_dir) / "cluster_stability.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ClusterStability] Saved {out_path}")


def run_umap(input_dir, output_dir, best_k, cluster_labels=None):
    """UMAP 2D projection of per-channel features, colored by cluster."""
    try:
        from umap import UMAP
    except ImportError:
        print("[UMAP] umap-learn not installed. Skipping UMAP projection.")
        return None

    print("[UMAP] Extracting features...")
    df, X_scaled, _, _ = _extract_features(input_dir)

    print("[UMAP] Projecting to 2D...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)

    # If cluster_labels provided (list matching df rows), use them
    if cluster_labels is not None and len(cluster_labels) == len(df):
        labels = cluster_labels
    else:
        labels = None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    if labels is not None:
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("Set2", len(unique_labels))
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap(lbl)], label=f"Cluster {lbl}", s=8, alpha=0.6, edgecolors="none",
            )
        ax.legend(markerscale=3, fontsize=8)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=0.6, edgecolors="none")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP Typology Projection (k={best_k})")
    plt.tight_layout()
    out_path = Path(output_dir) / "umap_typology.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[UMAP] Saved {out_path}")


def compute_mi_heatmap(input_dir, output_dir):
    """Compute pairwise mutual information between 6 feature columns, plot heatmap."""
    print("[MI] Extracting features...")
    df, _, _, feature_cols = _extract_features(input_dir)
    X = df[feature_cols].values
    n_features = X.shape[1]

    # Histogram-based MI estimation: MI(X,Y) = H(X) + H(Y) - H(X,Y)
    def _entropy_1d(data, bins=20):
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist + 1e-10))

    def _mi_xy(x, y, bins=20):
        """Compute MI between two continuous variables via histogram discretization."""
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        hist_2d = hist_2d / hist_2d.sum()  # normalize to probability
        # Marginal entropies
        px = hist_2d.sum(axis=1)
        py = hist_2d.sum(axis=0)
        px = px[px > 0]
        py = py[py > 0]
        h_x = -np.sum(px * np.log(px + 1e-10))
        h_y = -np.sum(py * np.log(py + 1e-10))
        # Joint entropy
        pxy = hist_2d[hist_2d > 0]
        h_xy = -np.sum(pxy * np.log(pxy + 1e-10))
        return h_x + h_y - h_xy

    mi_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            mi = _mi_xy(X[:, i], X[:, j], bins=20)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(mi_matrix, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_cols, fontsize=9)

    # Annotate cells
    for i in range(n_features):
        for j in range(n_features):
            val = mi_matrix[i, j]
            color = "white" if val > mi_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Mutual Information (nats)")
    ax.set_title("Pairwise Mutual Information Between Feature Columns")
    plt.tight_layout()
    out_path = Path(output_dir) / "mi_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[MI] Saved {out_path}")


def plot_geographic_clusters(input_dir, output_dir, typology_csv_path):
    """Plot HCMC map with cells colored by cluster assignment."""
    try:
        import geopandas as gpd
    except ImportError:
        print("[GeoCluster] geopandas not installed. Skipping geographic cluster plot.")
        return

    import pandas as pd

    # Load typology labels
    csv_path = Path(typology_csv_path)
    if not csv_path.exists():
        print(f"[GeoCluster] Typology CSV not found at {csv_path}. Skipping.")
        return

    df_labels = pd.read_csv(csv_path)
    # Create cell_id from filename: "000_000.png" -> "000_000"
    df_labels["cell_id"] = df_labels["filename"].str.replace(".png", "", regex=False)
    label_map = dict(zip(df_labels["cell_id"], df_labels["cluster"]))

    # Load GeoPackage
    gpk_path = Path(input_dir).parent.parent / "raw" / "hcmc_grid_1km.gpkg"
    if not gpk_path.exists():
        print(f"[GeoCluster] GeoPackage not found at {gpk_path}. Skipping.")
        return

    gdf = gpd.read_file(gpk_path)
    gdf["cluster"] = gdf["cell_id"].map(label_map)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 12), dpi=150)
    # Plot cells without cluster as light gray
    gdf_nolabel = gdf[gdf["cluster"].isna()]
    gdf_labeled = gdf[gdf["cluster"].notna()].copy()
    gdf_labeled["cluster"] = gdf_labeled["cluster"].astype(int)

    if not gdf_nolabel.empty:
        gdf_nolabel.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.2)
    if not gdf_labeled.empty:
        n_clusters = gdf_labeled["cluster"].nunique()
        gdf_labeled.plot(
            ax=ax, column="cluster", cmap="Set2", legend=True,
            categorical=True, edgecolor="white", linewidth=0.2,
            legend_kwds={"title": "Cluster", "loc": "lower left"},
        )

    ax.set_title("HCMC Urban Typology Clusters (1km Grid)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    out_path = Path(output_dir) / "geographic_clusters.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[GeoCluster] Saved {out_path}")
    n_labeled = len(gdf_labeled)
    n_total = len(gdf)
    print(f"[GeoCluster] {n_labeled}/{n_total} cells labeled on map")


def save_representative_examples(input_dir, output_dir, typology_csv_path, n_per_cluster=3):
    """For each cluster, copy n_per_cluster PNGs closest to centroid."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import shutil

    csv_path = Path(typology_csv_path)
    if not csv_path.exists():
        print(f"[Examples] Typology CSV not found at {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    feature_cols = ["B_mean", "G_mean", "R_mean", "B_std", "G_std", "R_std"]
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_path = Path(input_dir)
    out_base = Path(output_dir) / "representative_examples"

    unique_clusters = sorted(df["cluster"].unique())
    print(f"[Examples] Selecting {n_per_cluster} examples per cluster ({len(unique_clusters)} clusters)...")

    for cluster_id in unique_clusters:
        mask = df["cluster"] == cluster_id
        cluster_X = X_scaled[mask]
        cluster_files = df.loc[mask, "filename"].values

        # Centroid
        centroid = cluster_X.mean(axis=0)
        # Distance to centroid
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        # Pick closest
        n = min(n_per_cluster, len(cluster_files))
        closest_idx = np.argsort(dists)[:n]

        out_dir = out_base / f"cluster_{cluster_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in closest_idx:
            src = input_path / cluster_files[idx]
            if src.exists():
                shutil.copy2(src, out_dir / cluster_files[idx])

        selected = [cluster_files[i] for i in closest_idx]
        print(f"[Examples] Cluster {cluster_id}: {selected}")


def write_curation_guide(output_dir, typology_csv_path):
    """Write CURATION_GUIDE.md with cluster info, code examples, per-cluster lists."""
    import pandas as pd

    csv_path = Path(typology_csv_path)
    if not csv_path.exists():
        print(f"[CurationGuide] Typology CSV not found at {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    feature_cols = ["B_mean", "G_mean", "R_mean", "B_std", "G_std", "R_std"]
    unique_clusters = sorted(df["cluster"].unique())

    lines = []
    lines.append("# Pre-Encoding Curation Guide")
    lines.append("")
    lines.append(f"Generated from {typology_csv_path}")
    lines.append(f"Total cells: {len(df)}")
    lines.append(f"Clusters: {len(unique_clusters)}")
    lines.append("")

    # Cluster size distribution
    lines.append("## Cluster Size Distribution")
    lines.append("")
    lines.append("| Cluster | Count | Percentage |")
    lines.append("|---------|-------|------------|")
    for c in unique_clusters:
        n = (df["cluster"] == c).sum()
        pct = 100 * n / len(df)
        lines.append(f"| {c} | {n} | {pct:.1f}% |")
    lines.append("")

    # Mean feature values per cluster
    lines.append("## Mean Feature Values per Cluster")
    lines.append("")
    header = "| Cluster | " + " | ".join(feature_cols) + " |"
    sep = "|---------|" + "|".join(["-------" for _ in feature_cols]) + "|"
    lines.append(header)
    lines.append(sep)
    for c in unique_clusters:
        subset = df[df["cluster"] == c]
        vals = " | ".join(f"{subset[col].mean():.2f}" for col in feature_cols)
        lines.append(f"| {c} | {vals} |")
    lines.append("")

    # Recommended min samples
    min_size = (df["cluster"].value_counts().min())
    lines.append(f"## Recommended Minimum Samples per Cluster")
    lines.append("")
    lines.append(f"Smallest cluster has **{min_size}** cells.")
    lines.append(f"Recommended minimum for diffusion training: at least 50-100 cells per cluster.")
    lines.append("")

    # How to load labels
    lines.append("## How to Load Labels")
    lines.append("")
    lines.append("```python")
    lines.append("import pandas as pd")
    lines.append("df = pd.read_csv('typology_labels.csv')")
    lines.append("# Filter to a specific cluster")
    lines.append("cluster_0 = df[df['cluster'] == 0]")
    lines.append("print(f'Cluster 0: {len(cluster_0)} cells')")
    lines.append("```")
    lines.append("")

    # How to create controlled subsets
    lines.append("## How to Create Controlled Subsets")
    lines.append("")
    lines.append("```python")
    lines.append("import pandas as pd")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("df = pd.read_csv('typology_labels.csv')")
    lines.append("input_dir = Path('path/to/cells/')")
    lines.append("")
    lines.append("# Create a balanced subset (equal samples per cluster)")
    lines.append("n_per_cluster = 50")
    lines.append("subset = df.groupby('cluster').sample(n=min(n_per_cluster, ...), random_state=42)")
    lines.append("print(f'Balanced subset: {len(subset)} cells')")
    lines.append("")
    lines.append("# Create a high-density subset")
    lines.append("high_density = df[df['B_mean'] > df['B_mean'].quantile(0.75)]")
    lines.append("print(f'High density: {len(high_density)} cells')")
    lines.append("```")
    lines.append("")

    # Per-cluster cell lists
    lines.append("## Per-Cluster Cell Lists")
    lines.append("")
    for c in unique_clusters:
        cells = sorted(df[df["cluster"] == c]["filename"].values)
        lines.append(f"### Cluster {c} ({len(cells)} cells)")
        lines.append("")
        lines.append("```\n" + "\n".join(cells) + "\n```")
        lines.append("")

    guide_path = Path(output_dir) / "CURATION_GUIDE.md"
    guide_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[CurationGuide] Saved {guide_path}")


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

    # --- New analyses ---
    typology_csv = Path(output_dir) / "typology_labels.csv"

    # Load cluster labels from typology CSV for reuse
    cluster_labels = None
    best_k = args.k_range[0]
    if typology_csv.exists():
        import pandas as pd
        df = pd.read_csv(typology_csv)
        cluster_labels = df["cluster"].values
        best_k = df["cluster"].nunique()

    # 1. Cluster stability (silhouette vs k)
    plot_cluster_stability(input_dir, str(output_dir), args.k_range)

    # 2. UMAP projection
    run_umap(input_dir, str(output_dir), best_k, cluster_labels)

    # 3. Mutual information heatmap
    compute_mi_heatmap(input_dir, str(output_dir))

    # 4. Geographic cluster plot
    plot_geographic_clusters(input_dir, str(output_dir), str(typology_csv))

    # 5. Representative examples
    save_representative_examples(input_dir, str(output_dir), str(typology_csv), n_per_cluster=3)

    # 6. Curation guide
    write_curation_guide(str(output_dir), str(typology_csv))

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
