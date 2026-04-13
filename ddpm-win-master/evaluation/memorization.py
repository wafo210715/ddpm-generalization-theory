"""
Memorization evaluation via k-NN gap ratio analysis.

Ported from: Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation/compute_fmem.py
Reference: Bonnaire et al., "Why Diffusion Models Don't Memorize" (arXiv:2505.17638)

A generated sample is considered *memorized* when its nearest training-image
distance is disproportionately smaller than its second-nearest — i.e. the
**gap ratio** d₁/d₂ falls below a threshold (default 1/3).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Optional, Tuple, Union
from tqdm import tqdm


# ======================================================================
# Bootstrap confidence intervals
# ======================================================================


def bootstrap_mean_se(
    data: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Bootstrap estimate of the *fraction below threshold* with 95 % CI.

    Args:
        data: 1-D array of gap-ratio values.
        threshold: A sample is counted when its value < threshold.
        n_bootstrap: Number of resamples.
        random_state: RNG seed for reproducibility.

    Returns:
        ``(mean_estimate, standard_error, lower_ci, upper_ci)``
    """
    rng = np.random.default_rng(random_state)
    means = np.empty(n_bootstrap)
    n_data = len(data)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n_data, replace=True)
        collapsed = np.where(sample < threshold)[0]
        means[i] = len(collapsed) / len(sample)

    mean_est = means.mean()
    se_est = means.std(ddof=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return mean_est, se_est, lower, upper


# ======================================================================
# Core memorization metric
# ======================================================================


def compute_memorization_fraction(
    train_images: torch.Tensor,
    generated_images: torch.Tensor,
    gap_threshold: float = 1 / 3,
    k: int = 2,
    n_bootstrap: int = 1000,
    device: str = "cpu",
    batch_size: int = 100,
) -> Dict[str, Any]:
    """Compute memorization fraction via k-NN gap-ratio analysis.

    For every generated image the distances to the *k* nearest training
    images are computed.  The **gap ratio** is d₁/d₂ (nearest divided by
    second-nearest).  A sample whose gap ratio < *gap_threshold* (default
    1/3, following Bonnaire et al., 2025) is flagged as memorized.

    Args:
        train_images: Float tensor ``(N_train, C, H, W)``.
        generated_images: Float tensor ``(N_gen, C, H, W)``.
        gap_threshold: Threshold below which a sample is "memorized".
        k: Number of nearest neighbours (clamped to N_train).
        n_bootstrap: Bootstrap resamples for the confidence interval.
        device: Device used for distance computation.
        batch_size: Process generated images in chunks of this size
            to limit peak GPU memory.

    Returns:
        Dict with keys:
        - ``fraction_mem`` — bootstrap mean of memorized fraction
        - ``std`` — standard error of the estimate
        - ``lower_ci``, ``upper_ci`` — 95 % confidence interval
        - ``n_memorized`` — raw count below threshold
        - ``n_total`` — total number of generated images
        - ``gap_ratios`` — per-sample gap-ratio array (numpy)
    """
    N_train = train_images.shape[0]
    N_gen = generated_images.shape[0]
    D = int(np.prod(train_images.shape[1:]))

    X_train = train_images.reshape(N_train, D).float().to(device)
    k = min(k, N_train)

    # Compute k-NN distances in batches (memory-friendly)
    all_distances = torch.zeros(N_gen, k, device=device)

    for start in tqdm(range(0, N_gen, batch_size), desc="k-NN distances"):
        end = min(start + batch_size, N_gen)
        batch = generated_images[start:end].reshape(-1, D).float().to(device)

        # Pairwise L2: (batch_size, N_train)
        dist = torch.cdist(batch, X_train, p=2)
        topk = dist.topk(k, dim=1, largest=False)
        all_distances[start:end] = topk.values

    # Gap ratios
    gap_ratios = (all_distances[:, 0] / all_distances[:, 1]).cpu().numpy()

    # Bootstrap CI
    fraction_mem, std_frac, lower, upper = bootstrap_mean_se(
        gap_ratios, gap_threshold, n_bootstrap
    )

    n_memorized = int(np.sum(gap_ratios < gap_threshold))

    return {
        "fraction_mem": float(fraction_mem),
        "std": float(std_frac),
        "lower_ci": float(lower),
        "upper_ci": float(upper),
        "n_memorized": n_memorized,
        "n_total": N_gen,
        "gap_ratios": gap_ratios,
    }


# ======================================================================
# Convenience: full pipeline (checkpoint → samples → metric)
# ======================================================================


def _collect_images(dataset: Union[Dataset, DataLoader]) -> torch.Tensor:
    """Stack all images from a Dataset or DataLoader into a single tensor."""
    if isinstance(dataset, DataLoader):
        batches = []
        for batch in dataset:
            # DataLoader may yield (tensor,) or just tensor or dict
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


def evaluate_checkpoint_memorization(
    checkpoint_path: str,
    train_dataset: Union[Dataset, torch.Tensor],
    config,
    diffusion_config=None,
    n_samples: int = 100,
    gap_threshold: float = 1 / 3,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Full pipeline: load checkpoint → generate → compute memorization.

    Args:
        checkpoint_path: Path to a saved ``.pth`` checkpoint (full DDPM or
            U-Net-only state dict).
        train_dataset: Training images as a ``torch.Tensor`` of shape
            ``(N, C, H, W)``, a ``Dataset``, or a ``DataLoader``.
        config: ``TrainingConfig`` instance.
        diffusion_config: ``DiffusionConfig`` instance. Created from
            *config* if *None*.
        n_samples: Number of images to generate via DDIM.
        gap_threshold: Gap-ratio threshold (default 1/3).
        device: Torch device.

    Returns:
        Dict from :func:`compute_memorization_fraction` plus a ``"checkpoint"``
        key with the path.
    """
    # Lazy imports to avoid circular / unnecessary deps at module level
    from model.unet import DDPMUnet
    from model.ddpm import DDPM
    from model.diffusion_config import DiffusionConfig
    from sampling.ddim import sample_ddim

    # ------------------------------------------------------------------
    # Build DiffusionConfig from TrainingConfig if needed
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ddpm_unet = DDPMUnet(config.img_channels, config.model_channels)
    ddpm = DDPM(config.timesteps, ddpm_unet, diffusion_config=diffusion_config)

    if "eps_model" in checkpoint or "beta" in checkpoint:
        ddpm.load_state_dict(checkpoint)
    else:
        ddpm.eps_model.load_state_dict(checkpoint)

    ddpm = ddpm.to(device)

    # ------------------------------------------------------------------
    # Generate samples via DDIM
    # ------------------------------------------------------------------
    result = sample_ddim(
        ddpm,
        n_images=n_samples,
        config=config,
        diffusion_config=diffusion_config,
        device=device,
    )
    generated = result[0]

    # ------------------------------------------------------------------
    # Collect training images
    # ------------------------------------------------------------------
    if isinstance(train_dataset, torch.Tensor):
        train_images = train_dataset
    else:
        train_images = _collect_images(train_dataset)

    # ------------------------------------------------------------------
    # Compute memorization metric
    # ------------------------------------------------------------------
    results = compute_memorization_fraction(
        train_images=train_images,
        generated_images=generated,
        gap_threshold=gap_threshold,
        device=device,
    )
    results["checkpoint"] = checkpoint_path
    return results
