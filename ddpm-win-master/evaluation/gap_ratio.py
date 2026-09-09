"""k-NN gap ratio and F_mem on standardized tensors (E2 of the v4 map).

    Memorization: k-NN gap ratio on the standardized tensors.  Raw distance
    is dominated by the highest-variance channel.  The pilot measured
    accessibility memorization.  The descriptor is too coarse to detect
    copying of a specific cell.

That middle sentence is why this module exists next to
``evaluation/memorization.py`` rather than inside it.  The metric is the same
one as in Bonnaire et al.: for each generated cell, the distance to the nearest
training cell divided by the distance to the second nearest, and a ratio below
1/3 counts the sample as a copy.  What changed is the space it is measured in.

A six-channel cell in data space mixes footprint area, gravity scores and bits
of entropy in one Euclidean distance, so the channel with the widest numeric
range decides which cells count as near.  Measured that way the pilot's F_mem
was a statement about accessibility alone.  Standardized, every channel
contributes its own variance and the distance is over the cell.

Hence :func:`knn_gap_ratio` checks that what it was handed really is
standardized, and says so rather than returning a number that looks fine.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.memorization import bootstrap_mean_se

# Bonnaire et al.'s threshold: below this the sample sits on top of one
# training cell.
DEFAULT_GAP_THRESHOLD = 1.0 / 3.0

# How far a channel's std may sit from 1 before the standardization check
# complains.  Generous: a generated batch is not required to reproduce the
# training variance exactly, and early checkpoints will not.
STD_TOLERANCE = 0.5


def check_standardized(
    tensors: torch.Tensor | np.ndarray,
    name: str = "tensors",
    tolerance: float = STD_TOLERANCE,
) -> None:
    """Raise unless every channel has roughly unit variance.

    The failure this guards against is silent: distances computed in data space
    are perfectly well-defined numbers that answer a different question.
    """
    arr = _to_numpy(tensors)
    if arr.ndim != 4:
        raise ValueError(f"{name}: expected (N, C, H, W), got shape {arr.shape}")

    stds = arr.std(axis=(0, 2, 3))
    off = np.abs(stds - 1.0) > tolerance
    if np.any(off):
        offenders = {int(i): float(stds[i]) for i in np.flatnonzero(off)}
        raise ValueError(
            f"{name} do not look standardized: channel std {offenders} "
            f"(tolerance {tolerance}). Distances would be dominated by the "
            f"widest channel, which is what E2 exists to avoid. Apply the "
            f"store's ChannelPipeline.forward first, or pass "
            f"check_standardized=False if this is deliberate."
        )


def knn_gap_ratio(
    train_tensors: torch.Tensor | np.ndarray,
    generated_tensors: torch.Tensor | np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
    check: bool = True,
) -> np.ndarray:
    """Gap ratio d1/d2 for every generated cell.

    Args:
        train_tensors: ``(N_train, C, H, W)`` standardized training cells.
        generated_tensors: ``(N_gen, C, H, W)`` standardized generated cells.
        device: Where the distance matrix is computed.
        batch_size: Generated cells per chunk, to bound peak memory --- at
            1024 px a single cell is six million numbers.
        check: Verify both sides are standardized.

    Returns:
        ``(N_gen,)`` array of d1/d2 in [0, 1].  Small means the sample sits on
        one training cell; near 1 means its two nearest neighbours are equally
        far, which is what a genuinely new cell looks like.
    """
    train = _to_numpy(train_tensors)
    generated = _to_numpy(generated_tensors)

    if train.ndim != 4 or generated.ndim != 4:
        raise ValueError(
            f"expected (N, C, H, W) on both sides, got {train.shape} and "
            f"{generated.shape}"
        )
    if train.shape[1:] != generated.shape[1:]:
        raise ValueError(
            f"train cells are {train.shape[1:]}, generated are "
            f"{generated.shape[1:]}"
        )
    if train.shape[0] < 2:
        raise ValueError(
            "the gap ratio needs a second-nearest neighbour, so at least two "
            f"training cells; got {train.shape[0]}"
        )

    if check:
        check_standardized(train, "train_tensors")
        check_standardized(generated, "generated_tensors")

    n_train = train.shape[0]
    dim = int(np.prod(train.shape[1:]))
    train_flat = torch.from_numpy(train.reshape(n_train, dim)).float().to(device)

    ratios = np.empty(generated.shape[0], dtype=np.float64)
    for start in range(0, generated.shape[0], batch_size):
        end = min(start + batch_size, generated.shape[0])
        batch = (
            torch.from_numpy(generated[start:end].reshape(end - start, dim))
            .float()
            .to(device)
        )
        # float32 throughout: at 1024 px a cell is six million numbers and the
        # distance matrix inputs would double in float64.  The cost is that an
        # exact copy reads ~1e-4 rather than 0, which is far below the 1/3
        # threshold and so cannot move F_mem.
        distances = torch.cdist(batch, train_flat, p=2)
        nearest = distances.topk(2, dim=1, largest=False).values
        d1 = nearest[:, 0]
        d2 = nearest[:, 1]
        # d2 == 0 means two identical training cells sit on this sample.  That
        # is a duplicate in the training set, not a near-copy by the model, and
        # B1's "no overlapping windows" rule exists to keep it from happening.
        ratio = torch.where(d2 > 0, d1 / d2, torch.ones_like(d1))
        ratios[start:end] = ratio.cpu().numpy()

    return ratios


def memorization_fraction(
    gap_ratios: np.ndarray,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> dict:
    """F_mem: the share of gap ratios below the threshold, with a 95 % CI.

    Returns a dict with ``fraction_mem``, ``std``, ``lower_ci``, ``upper_ci``,
    ``n_memorized``, ``n_total``, ``gap_threshold`` and ``gap_ratios``.
    """
    ratios = np.asarray(gap_ratios, dtype=np.float64)
    if ratios.size == 0:
        raise ValueError("no gap ratios")

    mean_est, se, lower, upper = bootstrap_mean_se(
        ratios, gap_threshold, n_bootstrap, random_state
    )
    return {
        "fraction_mem": float(mean_est),
        "std": float(se),
        "lower_ci": float(lower),
        "upper_ci": float(upper),
        "n_memorized": int(np.count_nonzero(ratios < gap_threshold)),
        "n_total": int(ratios.size),
        "gap_threshold": float(gap_threshold),
        "gap_ratios": ratios,
    }


def compute_fmem(
    train_tensors: torch.Tensor | np.ndarray,
    generated_tensors: torch.Tensor | np.ndarray,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    device: str = "cpu",
    batch_size: int = 64,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    check: bool = True,
) -> dict:
    """:func:`knn_gap_ratio` then :func:`memorization_fraction`."""
    ratios = knn_gap_ratio(
        train_tensors, generated_tensors, device=device, batch_size=batch_size,
        check=check,
    )
    return memorization_fraction(ratios, gap_threshold, n_bootstrap, random_state)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)
