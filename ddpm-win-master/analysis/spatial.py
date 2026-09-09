"""Spatial statistics on cell grids.

Two things the v4 pipeline map asks for:

C2, inside the cell descriptor
    per-channel ... Moran's I ...

E3c, spatial relations between grids
    Within a cell, how the grids sit relative to each other in space:
    bivariate Moran's I between two grids ...  The pilot's synergy
    analysis compared cell-level averages, which cannot tell whether
    buildings sit on accessible units or merely share a cell with them.

The distinction E3c draws is the whole reason bivariate Moran's I is here. A
correlation between two cells' averages says dense cells tend to be accessible
cells. Bivariate Moran's I says whether, inside one cell, the building units
sit next to the accessible units. Only the second is a statement about layout.

Weights are the pixel lattice itself, row-standardized: every unit's neighbours
are its 8 (or 4) touching units, each weighted equally, and edge units are
normalized by however many neighbours they actually have. No distance decay ---
the indicator's own measurement scale already carries that, and imposing a
second one here would double-count it.
"""

from __future__ import annotations

import numpy as np

CONNECTIVITY = (4, 8)


def spatial_lag(grid: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """The row-standardized spatial lag of a grid: each unit's neighbour mean.

    Args:
        grid: ``(H, W)`` array of unit scores.
        connectivity: 4 (rook) or 8 (queen) neighbours.

    Returns:
        ``(H, W)`` array, the mean of each unit's neighbours.  Units on the
        edge average over the neighbours they have, which is what
        row-standardizing the weight matrix means at the boundary.
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"expected a (H, W) grid, got shape {g.shape}")
    if connectivity not in CONNECTIVITY:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    total = np.zeros_like(g)
    count = np.zeros_like(g)
    ones = np.ones_like(g)
    for dy, dx in offsets:
        total += _shift(g, dy, dx)
        count += _shift(ones, dy, dx)

    # count is >= 2 everywhere on any grid of at least 2x2.
    return total / np.maximum(count, 1.0)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``a`` shifted by (dy, dx) with zeros shifted in from outside."""
    out = np.zeros_like(a)
    ys_dst = slice(max(dy, 0), a.shape[0] + min(dy, 0))
    ys_src = slice(max(-dy, 0), a.shape[0] + min(-dy, 0))
    xs_dst = slice(max(dx, 0), a.shape[1] + min(dx, 0))
    xs_src = slice(max(-dx, 0), a.shape[1] + min(-dx, 0))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


def _z_scores(grid: np.ndarray) -> np.ndarray | None:
    """Standardized grid, or None when the grid is constant."""
    g = np.asarray(grid, dtype=np.float64)
    std = g.std()
    if std < 1e-12:
        return None
    return (g - g.mean()) / std


def morans_i(grid: np.ndarray, connectivity: int = 8) -> float:
    """Moran's I of one grid: spatial clumping.

    Near 1, neighbouring units look alike; near 0, there is no spatial pattern.
    A constant grid has no pattern to describe and returns 0.0 --- which is the
    honest answer, not a missing value, and it is the value the pilot's
    flattened building channel produced.

    Args:
        grid: ``(H, W)`` array of unit scores, in data space.
        connectivity: 4 (rook) or 8 (queen) neighbours.  The choice is not
            cosmetic on fine-grained patterns: a checkerboard is maximally
            dispersed under rook weights (I near -1) but nearly unstructured
            under queen weights (I near -0.05), because its diagonal
            neighbours match.  At a 1.17 m unit, alley-and-rowhouse banding is
            exactly that kind of pattern.
    """
    z = _z_scores(grid)
    if z is None:
        return 0.0
    return float(np.mean(z * spatial_lag(z, connectivity)))


def bivariate_morans_i(
    grid_x: np.ndarray, grid_y: np.ndarray, connectivity: int = 8
) -> float:
    """Bivariate Moran's I: whether high values of x sit next to high y.

    Anselin's form, with row-standardized weights:

        I_xy = (1 / n) * sum_i  z_x,i * lag(z_y)_i

    where ``z`` are the standardized grids.  Near 1, buildings cluster around
    accessible units; near 0, the two grids ignore each other spatially; near
    -1, each avoids the other's neighbourhood.

    Note that it is not symmetric --- ``I(x, y)`` asks whether x sits beside
    y's neighbourhood, ``I(y, x)`` the converse --- so callers that want one
    number per pair should say which order they took, or average the two.

    Returns 0.0 when either grid is constant.
    """
    x = np.asarray(grid_x, dtype=np.float64)
    y = np.asarray(grid_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"grids differ in shape: {x.shape} vs {y.shape}")

    zx = _z_scores(x)
    zy = _z_scores(y)
    if zx is None or zy is None:
        return 0.0
    return float(np.mean(zx * spatial_lag(zy, connectivity)))


def bivariate_morans_i_matrix(
    tensor: np.ndarray, connectivity: int = 8
) -> np.ndarray:
    """Every ordered pair of channels in one cell.

    Args:
        tensor: ``(C, H, W)`` cell tensor, in data space.

    Returns:
        ``(C, C)`` matrix ``M`` with ``M[i, j] = bivariate_morans_i(ch_i, ch_j)``.
        The diagonal is each channel's own Moran's I.
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim != 3:
        raise ValueError(f"expected a (C, H, W) tensor, got shape {t.shape}")

    n_channels = t.shape[0]
    z = [_z_scores(t[i]) for i in range(n_channels)]
    lag = [spatial_lag(zi, connectivity) if zi is not None else None for zi in z]

    out = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        if z[i] is None:
            continue
        for j in range(n_channels):
            if lag[j] is None:
                continue
            out[i, j] = float(np.mean(z[i] * lag[j]))
    return out
