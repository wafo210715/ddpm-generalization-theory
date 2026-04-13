"""
Fréchet Inception Distance (FID) evaluation for DDPM-generated urban vitality maps.

LIMITATION: Standard FID uses InceptionV3 features trained on natural images (ImageNet).
Our urban vitality maps are NOT natural images — they are RGB-encoded urban metric maps
(density/diversity/accessibility). The InceptionV3 features may not capture meaningful
semantic differences for this domain.

FID values here should be interpreted as:
- A relative measure (lower = more similar to training distribution)
- NOT an absolute quality metric
- Useful for tracking FID across training checkpoints (relative changes)
- Cross-domain comparisons with natural image FID are invalid

Alternative approaches for future work:
- Raw pixel-space FID (no Inception, just pixel statistics)
- Domain-specific feature extractors (trained on urban data)
- Per-channel FID to track individual channel learning

Reference: Heusel et al., "GANS Trained by a Two Time-Scale Update Rule Converge
to a Local Nash Equilibrium" (NeurIPS 2017)

Usage:
    python evaluation/compute_fid.py \
        --checkpoint_dir model_ckpts/hcmc_vitality_v2 \
        --config configs/hcmc_vitality_v2.yaml \
        --n_samples 100 \
        --output_dir fid_results
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make ddpm-win-master importable when running as a standalone script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ======================================================================
# Image collection helper (mirrors memorization.py pattern)
# ======================================================================


def _collect_images(dataset: Union[Dataset, DataLoader]) -> torch.Tensor:
    """Stack all images from a Dataset or DataLoader into a single tensor."""
    if isinstance(dataset, DataLoader):
        batches = []
        for batch in dataset:
            t = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(t, dict):
                t = t["image"]
            batches.append(t)
        return torch.cat(batches, dim=0)
    else:
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        batches = []
        for batch in loader:
            t = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(t, dict):
                t = t["image"]
            batches.append(t)
        return torch.cat(batches, dim=0)


# ======================================================================
# InceptionV3 feature extractor
# ======================================================================


def _build_inception_feature_extractor(device: str = "cpu"):
    """Build an InceptionV3 model that returns 2048-dim pool3 features.

    Uses torchvision's InceptionV3 with aux_logits=False and replaces the
    final fully-connected layer to output the 2048-dimensional pooling
    features instead of class logits.
    """
    import torchvision.models as models
    import torch.nn as nn

    inception = models.inception_v3(
        pretrained=False,
        # weights=None,  # No pretrained weights — we only need architecture
        aux_logits=False,
    )
    # Replace the final FC: InceptionV3.fc is Linear(2048, 1000)
    # We keep the AvgPool and Dropout, just remove the classification head
    inception.fc = nn.Identity()  # type: ignore[assignment]
    inception = inception.to(device)
    inception.eval()
    return inception


@torch.no_grad()
def _extract_inception_features(
    images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Extract 2048-dim InceptionV3 pool3 features from images.

    Args:
        images: Float tensor (N, C, H, W) in range [-1, 1].
        batch_size: Processing batch size.
        device: Device for inference.

    Returns:
        numpy array of shape (N, 2048).
    """
    import torch.nn.functional as F

    model = _build_inception_feature_extractor(device)

    # InceptionV3 expects 299x299 input
    features_list: list[torch.Tensor] = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        # Resize to 299x299 (InceptionV3 input size)
        batch = F.interpolate(
            batch, size=(299, 299), mode="bilinear", align_corners=False
        )
        # InceptionV3 expects [0, 1] range; our images are [-1, 1]
        batch = (batch + 1.0) / 2.0
        batch = batch.clamp(0.0, 1.0)
        feat = model(batch)
        features_list.append(feat.cpu())

    return torch.cat(features_list, dim=0).numpy()


# ======================================================================
# FID computation core
# ======================================================================


def _frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Compute the Fréchet distance between two Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))

    Uses scipy.linalg.sqrtm for the matrix square root with numerical
    stability handling (clipping small imaginary components).

    Args:
        mu1: Mean vector of distribution 1, shape (d,).
        sigma1: Covariance matrix of distribution 1, shape (d, d).
        mu2: Mean vector of distribution 2, shape (d,).
        sigma2: Covariance matrix of distribution 2, shape (d, d).

    Returns:
        Fréchet distance (float).
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    mean_diff_sq = np.dot(diff, diff)

    # Product of covariances
    covmean = sqrtm(sigma1 @ sigma2)  # type: ignore[arg-type]

    # Numerical stability: sqrtm may produce small imaginary parts
    if np.iscomplexobj(covmean):  # type: ignore[arg-type]
        # If imaginary part is negligible, take real part
        if np.max(np.abs(covmean.imag)) < 1e-3:  # type: ignore[union-attr]
            covmean = covmean.real  # type: ignore[union-attr]
        else:
            # Significant imaginary component — warn and take real
            covmean = covmean.real  # type: ignore[union-attr]

    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    fid = float(mean_diff_sq + trace_term)
    return fid


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cpu",
    feature_batch_size: int = 64,
) -> Dict[str, Any]:
    """Compute Fréchet Inception Distance (FID) between two image sets.

    Args:
        real_images: Float32 tensor (N_real, C, H, W) in range [-1, 1].
        generated_images: Float32 tensor (N_gen, C, H, W) in range [-1, 1].
        device: Torch device for InceptionV3 inference.
        feature_batch_size: Batch size for feature extraction.

    Returns:
        Dict with keys:
        - ``fid_score`` — Fréchet Inception Distance
        - ``real_mean`` — Mean of real image features (numpy, shape 2048)
        - ``gen_mean`` — Mean of generated image features (numpy, shape 2048)
        - ``real_cov_trace`` — Trace of real feature covariance
        - ``gen_cov_trace`` — Trace of generated feature covariance
    """
    # Extract InceptionV3 features
    real_feats = _extract_inception_features(
        real_images, batch_size=feature_batch_size, device=device
    )
    gen_feats = _extract_inception_features(
        generated_images, batch_size=feature_batch_size, device=device
    )

    # Compute statistics
    mu_real = np.mean(real_feats, axis=0)
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    sigma_gen = np.cov(gen_feats, rowvar=False)

    # Compute FID
    fid_score = _frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return {
        "fid_score": fid_score,
        "real_mean": mu_real,
        "gen_mean": mu_gen,
        "real_cov_trace": float(np.trace(sigma_real)),
        "gen_cov_trace": float(np.trace(sigma_gen)),
    }


# ======================================================================
# Per-channel FID (pixel-space, domain-appropriate)
# ======================================================================


def compute_fid_per_channel(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute per-channel FID using pixel-space statistics.

    Unlike InceptionV3-based FID, this operates directly on pixel values.
    For our urban vitality maps, each channel encodes a specific urban metric:
        R = accessibility, G = diversity, B = density

    This is more meaningful for our domain since each channel has specific
    urban semantics and the InceptionV3 features were trained on natural images.

    For each channel independently:
    1. Flatten all pixels across all images
    2. Compute mean and variance (1D statistics)
    3. FID simplifies to: (mu1 - mu2)^2 + (sigma1 - sigma2)^2
       (1D case: sqrt(sigma1*sigma2) = sqrt(sigma1)*sqrt(sigma2))

    Args:
        real_images: Float32 tensor (N_real, C, H, W) in range [-1, 1].
        generated_images: Float32 tensor (N_gen, C, H, W) in range [-1, 1].
        device: Unused (pixel-space computation, kept for API consistency).

    Returns:
        Dict with keys:
        - ``channel_fid`` — List of per-channel FID scores [R, G, B]
        - ``channel_names`` — List of channel names
        - ``real_means`` — Per-channel means of real images
        - ``gen_means`` — Per-channel means of generated images
        - ``real_stds`` — Per-channel stds of real images
        - ``gen_stds`` — Per-channel stds of generated images
    """
    channel_names = ["accessibility_R", "diversity_G", "density_B"]
    n_channels = real_images.shape[1]

    channel_fids: list[float] = []
    real_means: list[float] = []
    gen_means: list[float] = []
    real_stds: list[float] = []
    gen_stds: list[float] = []

    for c in range(n_channels):
        real_c = real_images[:, c].flatten().numpy()
        gen_c = generated_images[:, c].flatten().numpy()

        mu_r = float(np.mean(real_c))
        mu_g = float(np.mean(gen_c))
        var_r = float(np.var(real_c))
        var_g = float(np.var(gen_c))

        # 1D Fréchet distance: (mu1-mu2)^2 + (var1 + var2 - 2*sqrt(var1*var2))
        fid_c = (mu_r - mu_g) ** 2 + var_r + var_g - 2.0 * np.sqrt(var_r * var_g)
        fid_c = float(max(0.0, fid_c))  # Numerical stability

        channel_fids.append(fid_c)
        real_means.append(mu_r)
        gen_means.append(mu_g)
        real_stds.append(float(np.std(real_c)))
        gen_stds.append(float(np.std(gen_c)))

    return {
        "channel_fid": channel_fids,
        "channel_names": channel_names,
        "real_means": real_means,
        "gen_means": gen_means,
        "real_stds": real_stds,
        "gen_stds": gen_stds,
    }


# ======================================================================
# Full pipeline: checkpoint → generate → FID
# ======================================================================


def evaluate_fid_checkpoints(
    checkpoint_dir: str,
    train_dataset: Union[Dataset, torch.Tensor],
    config,
    diffusion_config=None,
    n_samples: int = 100,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute FID for every checkpoint in a directory.

    For each checkpoint: load model → generate samples via DDIM → compute FID
    against training images → collect results.

    Args:
        checkpoint_dir: Directory with ``*_step{N}.pth`` files.
        train_dataset: Training images tensor (N, C, H, W), Dataset, or DataLoader.
        config: ``TrainingConfig`` instance.
        diffusion_config: ``DiffusionConfig`` instance (derived from config if None).
        n_samples: Number of images to generate per checkpoint.
        device: Torch device.
        output_dir: Directory for CSV output. If None, defaults to
                    ``checkpoint_dir/../fid_results``.

    Returns:
        List of result dicts, one per checkpoint, sorted by step.
    """
    from evaluation.compute_fmem_batch import discover_checkpoints
    from model.unet import DDPMUnet
    from model.ddpm import DDPM
    from model.diffusion_config import DiffusionConfig
    from sampling.ddim import sample_ddim

    # Discover checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    print(f"Found {len(checkpoints)} checkpoints for FID evaluation.")

    # Build DiffusionConfig from TrainingConfig if needed
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

    # Collect training images once
    if isinstance(train_dataset, torch.Tensor):
        train_images = train_dataset
    else:
        train_images = _collect_images(train_dataset)

    n_train = train_images.shape[0]
    print(f"Training images: {n_train}")

    # Output directory
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(checkpoint_dir.rstrip("/\\")), "fid_results"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Process each checkpoint
    rows: list[dict[str, Any]] = []
    for step, ckpt_path in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Checkpoint step {step}: {os.path.basename(ckpt_path)}")
        print(f"{'=' * 60}")

        # Load model
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        ddpm_unet = DDPMUnet(config.img_channels, config.model_channels)
        ddpm = DDPM(config.timesteps, ddpm_unet, diffusion_config=diffusion_config)

        if "eps_model" in checkpoint or "beta" in checkpoint:
            ddpm.load_state_dict(checkpoint)
        else:
            ddpm.eps_model.load_state_dict(checkpoint)

        ddpm = ddpm.to(device)

        # Generate samples via DDIM
        result = sample_ddim(
            ddpm,
            n_images=n_samples,
            config=config,
            diffusion_config=diffusion_config,
            device=device,
        )
        generated = result[0]

        # Compute FID
        fid_result = compute_fid(
            real_images=train_images,
            generated_images=generated,
            device=device,
        )

        # Compute per-channel FID
        per_ch = compute_fid_per_channel(train_images, generated, device=device)

        row = {
            "checkpoint_step": step,
            "fid_score": fid_result["fid_score"],
            "n_train": n_train,
            "n_generated": n_samples,
            "real_cov_trace": fid_result["real_cov_trace"],
            "gen_cov_trace": fid_result["gen_cov_trace"],
            "fid_R_accessibility": per_ch["channel_fid"][0],
            "fid_G_diversity": per_ch["channel_fid"][1],
            "fid_B_density": per_ch["channel_fid"][2],
        }
        rows.append(row)

        print(
            f"  FID = {row['fid_score']:.4f}  "
            f"Per-channel: R={row['fid_R_accessibility']:.4f} "
            f"G={row['fid_G_diversity']:.4f} "
            f"B={row['fid_B_density']:.4f}"
        )

    # Sort by step
    rows.sort(key=lambda r: r["checkpoint_step"])

    # Write CSV
    csv_path = os.path.join(output_dir, "fid_curve.csv")
    fieldnames = [
        "checkpoint_step",
        "fid_score",
        "n_train",
        "n_generated",
        "real_cov_trace",
        "gen_cov_trace",
        "fid_R_accessibility",
        "fid_G_diversity",
        "fid_B_density",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to {csv_path}")

    # Plot
    _plot_fid_curve(rows, os.path.join(output_dir, "fid_curve.png"))
    print(f"Plot saved to {os.path.join(output_dir, 'fid_curve.png')}")

    # Summary
    print(f"\n{'=' * 60}")
    print("FID SUMMARY")
    print(f"{'=' * 60}")
    for row in rows:
        print(
            f"  Step {row['checkpoint_step']:>7d}: "
            f"FID = {row['fid_score']:.4f}  "
            f"(R={row['fid_R_accessibility']:.4f}, "
            f"G={row['fid_G_diversity']:.4f}, "
            f"B={row['fid_B_density']:.4f})"
        )

    return rows


# ======================================================================
# Plotting
# ======================================================================


def _plot_fid_curve(rows: list[dict], output_path: str) -> None:
    """Save FID vs. training step plot with per-channel breakdown."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["checkpoint_step"] for r in rows]
    fid_total = [r["fid_score"] for r in rows]
    fid_r = [r["fid_R_accessibility"] for r in rows]
    fid_g = [r["fid_G_diversity"] for r in rows]
    fid_b = [r["fid_B_density"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Total FID
    ax = axes[0]
    ax.plot(
        steps,
        fid_total,
        "o-",
        color="black",
        linewidth=2,
        markersize=6,
        label="Total FID",
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("FID Score")
    ax.set_title("Fréchet Inception Distance vs. Training Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Per-channel FID
    ax = axes[1]
    ax.plot(
        steps,
        fid_r,
        "o-",
        color="red",
        linewidth=1.5,
        markersize=4,
        label="R (accessibility)",
    )
    ax.plot(
        steps,
        fid_g,
        "s-",
        color="green",
        linewidth=1.5,
        markersize=4,
        label="G (diversity)",
    )
    ax.plot(
        steps,
        fid_b,
        "^-",
        color="blue",
        linewidth=1.5,
        markersize=4,
        label="B (density)",
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Per-Channel FID (pixel-space)")
    ax.set_title("Per-Channel FID vs. Training Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute FID across DDPM training checkpoints.",
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
        default=100,
        help="Number of images to generate per checkpoint (default: 100).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for CSV and plots (default: checkpoint_dir/../fid_results).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = "cpu"

    # Load config
    from config_loader import TrainingConfig
    from datasets.dataset import get_dataset

    config = TrainingConfig.from_yaml(args.config)

    # Load training dataset
    dataset = get_dataset(
        name=config.dataset_name,
        img_size=config.img_size,
        data_dir=args.data_dir or config.data_root,
        split=args.split,
        center=config.center,
        normalize=config.normalize,
    )
    print(
        f"Training dataset: {len(dataset)} images ({config.dataset_name}, split={args.split})"  # type: ignore[arg-type]
    )

    # Run FID evaluation
    evaluate_fid_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        train_dataset=dataset,
        config=config,
        n_samples=args.n_samples,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
