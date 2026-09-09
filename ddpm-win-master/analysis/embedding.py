"""PCA + UMAP map of the cell dataset (C4b and E3b of the v4 pipeline map).

C4b, before any model exists
    Map the real dataset in two dimensions, coloured by typology.  PCA to
    about 50 numbers per cell first, then UMAP or t-SNE.  One map for the
    whole stack and one per grid, fitted on real cells only.

E3b, after every checkpoint
    Put generated cells on the real dataset's 2-D map, per grid and per
    checkpoint.  Same PCA and UMAP fitted in C4b, so real and generated
    points share one map.

Fitted once, projected many times.  That is the whole interface, and it is why
UMAP rather than t-SNE: UMAP can place new points on a map fitted earlier and
t-SNE cannot.  :meth:`CellEmbedding.fit` is called once on real cells; every
checkpoint's generated cells go through :meth:`CellEmbedding.transform`.

A caution the plan states and this module cannot enforce: distances on the
resulting map are not measurements.  The picture shows where generated cells
fill, avoid or pile onto real ones.  The numbers come from the descriptor
histograms and the Wasserstein distances.

Usage
-----
    embedding = CellEmbedding.fit(real_cells)              # whole stack
    per_grid = CellEmbedding.fit(real_cells, channel=0)    # one grid
    embedding.save("maps/all_channels.pkl")

    real_xy = embedding.transform(real_cells)
    generated_xy = embedding.transform(generated_cells)    # same map
"""

from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np

# PCA target from C4b: "PCA to about 50 numbers per cell first".
DEFAULT_N_PCA = 50
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_SEED = 42


def flatten_cells(
    tensors,
    channel: int | Sequence[int] | None = None,
    pool: int | None = None,
) -> np.ndarray:
    """Flatten cells into one vector each, optionally per grid and pooled.

    Args:
        tensors: ``(N, C, H, W)`` array or an iterable of ``(C, H, W)`` cells.
        channel: A channel index, or several, to keep.  ``None`` keeps the
            whole stack.  C4b asks for "one map for the whole stack and one per
            grid"; this is how the per-grid maps are built.
        pool: Average-pool each grid by this factor before flattening.  A
            1024 px six-channel cell is 6.3 million numbers, and PCA over that
            many columns is neither necessary nor affordable --- pooling to 64
            px keeps the layout that a 2-D map can show at all.  ``None`` for
            no pooling.

    Returns:
        ``(N, D)`` float64 array.
    """
    stack = _as_stack(tensors)
    if channel is not None:
        idx = [channel] if isinstance(channel, (int, np.integer)) else list(channel)
        stack = stack[:, idx]
    if pool is not None and pool > 1:
        stack = _average_pool(stack, pool)
    return stack.reshape(stack.shape[0], -1)


class CellEmbedding:
    """A PCA-then-UMAP map fitted on real cells, able to place new points.

    Do not fit a second one for generated cells.  The point of E3b is that both
    sets land on the same map, so the only fitted object is this one, saved
    beside the checkpoints it will be used with.
    """

    def __init__(self, pca, umap_model, channel, pool, n_features: int):
        self.pca = pca
        self.umap = umap_model
        self.channel = channel
        self.pool = pool
        self.n_features = n_features

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        tensors,
        channel: int | Sequence[int] | None = None,
        pool: int | None = None,
        n_pca: int = DEFAULT_N_PCA,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        min_dist: float = DEFAULT_MIN_DIST,
        random_state: int = DEFAULT_SEED,
    ) -> "CellEmbedding":
        """Fit on real cells only.

        Args:
            tensors: ``(N, C, H, W)`` real cells, standardized.
            channel: Which grid, or ``None`` for the whole stack.
            pool: Average-pooling factor; see :func:`flatten_cells`.
            n_pca: PCA components before UMAP.  Capped at what the data
                supports, so a small pilot set does not raise.
            n_neighbors, min_dist: UMAP's shape parameters.
            random_state: Seeded so the map is the same one every time it is
                rebuilt --- a map that moves between runs cannot be compared
                across checkpoints.
        """
        from sklearn.decomposition import PCA
        from umap import UMAP

        features = flatten_cells(tensors, channel=channel, pool=pool)
        n_samples, n_features = features.shape
        if n_samples < 3:
            raise ValueError(f"need at least 3 cells to fit a map, got {n_samples}")

        n_components = min(n_pca, n_samples - 1, n_features)
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced = pca.fit_transform(features)

        umap_model = UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, n_samples - 1),
            min_dist=min_dist,
            random_state=random_state,
            # Seeded UMAP is single-threaded anyway; saying so avoids the
            # warning and makes the reproducibility trade explicit.
            n_jobs=1,
        )
        umap_model.fit(reduced)

        return cls(pca, umap_model, channel, pool, n_features)

    # -- projecting ---------------------------------------------------------

    def transform(self, tensors) -> np.ndarray:
        """Place cells on the fitted map.  Returns ``(N, 2)``.

        Works for real cells (reproducing the C4b picture) and for generated
        cells at any checkpoint (E3b), through exactly the same code.
        """
        features = flatten_cells(tensors, channel=self.channel, pool=self.pool)
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"cells give {features.shape[1]} features, the map was fitted "
                f"on {self.n_features}. Same resolution, channel selection and "
                f"pooling are required for both sides to share a map."
            )
        return np.asarray(self.umap.transform(self.pca.transform(features)))

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """How much of the spread the PCA step kept, per component.

        Its cumulative sum is the k95 that C4 reads for intrinsic dimension, so
        it is worth looking at even when the map itself is only a picture.
        """
        return self.pca.explained_variance_ratio_

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the fitted map.  UMAP models have no other serialization."""
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CellEmbedding":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_stack(tensors) -> np.ndarray:
    if isinstance(tensors, np.ndarray) and tensors.ndim == 4:
        return tensors.astype(np.float64, copy=False)
    try:
        import torch

        if isinstance(tensors, torch.Tensor):
            return tensors.detach().cpu().numpy().astype(np.float64, copy=False)
    except ImportError:  # pragma: no cover
        pass

    cells = [np.asarray(t, dtype=np.float64) for t in tensors]
    if not cells:
        raise ValueError("no cells")
    if any(c.ndim != 3 for c in cells):
        raise ValueError("expected (C, H, W) cells or an (N, C, H, W) array")
    return np.stack(cells)


def _average_pool(stack: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool ``(N, C, H, W)`` by an integer factor, cropping the remainder."""
    n, c, h, w = stack.shape
    h_out, w_out = h // factor, w // factor
    if h_out == 0 or w_out == 0:
        raise ValueError(f"pool factor {factor} is larger than the {h}x{w} grid")
    cropped = stack[:, :, : h_out * factor, : w_out * factor]
    return cropped.reshape(n, c, h_out, factor, w_out, factor).mean(axis=(3, 5))
