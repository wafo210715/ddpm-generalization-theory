"""The cell descriptor (C2 of the v4 pipeline map).

    Write the cell descriptor: one function for real and generated cells.
    Per-channel mean, std, zero fraction, Moran's I, skewness; within-cell
    cross-channel correlation; unit-class composition.  Must be computable
    from the raster alone.  Generated cells have no vectors behind them.
    The pilot's statistics ran only on generated samples through separate
    code paths.

"One function" is the design constraint, so there is exactly one public entry
point here and both sides of every comparison in E3 go through it.

Descriptors are computed in **data space**, not standardized space: "zero
fraction" means the share of units with no building on them, and after
standardization zero is wherever the channel mean happened to land.  E1 unpacks
a generated sample first; this runs after that.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from analysis.spatial import morans_i

# The per-channel statistics, in the order the plan lists them.
CHANNEL_STATISTICS = ("mean", "std", "zero_fraction", "morans_i", "skewness")


def describe_cell(
    tensor: np.ndarray,
    channel_names: Sequence[str],
    connectivity: int = 8,
    zero_atol: float = 0.0,
    unit_labels: np.ndarray | None = None,
    unit_classes: Sequence[int] | None = None,
) -> dict[str, float]:
    """Turn one cell into a flat dict of named numbers.

    Args:
        tensor: ``(C, H, W)`` cell tensor in data space.
        channel_names: One name per channel, in channel order.  They become
            the prefix of every key, so a descriptor read years later says
            which indicator each number belongs to.
        connectivity: Neighbour rule for Moran's I --- 4 (rook) or 8 (queen).
        zero_atol: A unit counts as empty when ``|value| <= zero_atol``.  Exact
            zero by default, which is right for a rasterized footprint or a POI
            count; a smooth field that only approaches zero wants a tolerance.
        unit_labels: Optional ``(H, W)`` array of per-unit class labels from
            C3's clustering, for the unit-class composition.  Omitted when no
            typology has been fitted yet --- there is no sensible default class
            set, and inventing one would put a decision in a statistic.
        unit_classes: The full label vocabulary, so that a cell missing a class
            reports 0.0 for it rather than dropping the key and making the
            descriptors of two cells different lengths.  Required with
            ``unit_labels``.

    Returns:
        A flat ``{name: value}`` dict.  Flat rather than nested so a set of
        descriptors stacks straight into a DataFrame and the keys are the
        column names.

    Keys:
        ``{channel}_{stat}`` for each channel and each of
        :data:`CHANNEL_STATISTICS`; ``corr_{a}_{b}`` for each unordered channel
        pair; ``unitclass_{label}_fraction`` when ``unit_labels`` is given.
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.ndim != 3:
        raise ValueError(f"expected a (C, H, W) tensor, got shape {t.shape}")
    if len(channel_names) != t.shape[0]:
        raise ValueError(
            f"got {len(channel_names)} channel names for {t.shape[0]} channels"
        )
    if not np.all(np.isfinite(t)):
        raise ValueError("cell contains non-finite values")

    out: dict[str, float] = {}

    for i, name in enumerate(channel_names):
        grid = t[i]
        out[f"{name}_mean"] = float(grid.mean())
        out[f"{name}_std"] = float(grid.std())
        out[f"{name}_zero_fraction"] = float(np.mean(np.abs(grid) <= zero_atol))
        out[f"{name}_morans_i"] = morans_i(grid, connectivity)
        out[f"{name}_skewness"] = _skewness(grid)

    flat = t.reshape(t.shape[0], -1)
    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            key = f"corr_{channel_names[i]}_{channel_names[j]}"
            out[key] = _correlation(flat[i], flat[j])

    if unit_labels is not None:
        if unit_classes is None:
            raise ValueError(
                "unit_labels needs unit_classes, so that every cell's "
                "descriptor has the same keys"
            )
        out.update(_unit_class_composition(unit_labels, unit_classes, t.shape[1:]))

    return out


def describe_cells(
    tensors,
    channel_names: Sequence[str],
    cell_ids: Sequence[str] | None = None,
    connectivity: int = 8,
    zero_atol: float = 0.0,
    unit_labeller: Callable[[np.ndarray], np.ndarray] | None = None,
    unit_classes: Sequence[int] | None = None,
):
    """Descriptors for many cells, as a DataFrame indexed by cell id.

    Args:
        tensors: ``(N, C, H, W)`` array or any iterable of ``(C, H, W)`` cells.
        cell_ids: Index values.  Positional integers when omitted, which is the
            case for generated cells: they have no id because they are not
            anywhere.
        unit_labeller: Maps one ``(C, H, W)`` cell to its ``(H, W)`` unit
            labels.  This is where a fitted C3 classifier plugs in.

    Returns:
        ``pandas.DataFrame``, one row per cell, one column per descriptor key.
    """
    import pandas as pd

    rows = []
    for tensor in tensors:
        labels = unit_labeller(tensor) if unit_labeller is not None else None
        rows.append(
            describe_cell(
                tensor,
                channel_names,
                connectivity=connectivity,
                zero_atol=zero_atol,
                unit_labels=labels,
                unit_classes=unit_classes,
            )
        )
    if not rows:
        raise ValueError("no cells to describe")

    frame = pd.DataFrame(rows)
    if cell_ids is not None:
        cell_ids = list(cell_ids)
        if len(cell_ids) != len(frame):
            raise ValueError(
                f"got {len(cell_ids)} cell ids for {len(frame)} cells"
            )
        frame.index = pd.Index(cell_ids, name="cell_id")
    return frame


def descriptor_keys(
    channel_names: Sequence[str], unit_classes: Sequence[int] | None = None
) -> list[str]:
    """The keys :func:`describe_cell` will produce, without computing any.

    Useful for asserting that real and generated descriptors line up before a
    comparison, rather than after one silently compares different columns.
    """
    keys = [f"{n}_{s}" for n in channel_names for s in CHANNEL_STATISTICS]
    keys += [
        f"corr_{channel_names[i]}_{channel_names[j]}"
        for i in range(len(channel_names))
        for j in range(i + 1, len(channel_names))
    ]
    if unit_classes is not None:
        keys += [f"unitclass_{c}_fraction" for c in unit_classes]
    return keys


# ---------------------------------------------------------------------------
# Statistics that need a defined answer on degenerate grids
# ---------------------------------------------------------------------------


def _skewness(grid: np.ndarray) -> float:
    """Fisher-Pearson skewness; 0.0 for a constant grid."""
    x = np.asarray(grid, dtype=np.float64).ravel()
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - x.mean()) / std) ** 3))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either channel is constant.

    A constant channel is uncorrelated with everything in the only sense the
    descriptor can report, and nan would poison every downstream comparison.
    """
    a_std, b_std = a.std(), b.std()
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (a_std * b_std))


def _unit_class_composition(
    unit_labels: np.ndarray, unit_classes: Sequence[int], grid_shape: tuple
) -> dict[str, float]:
    labels = np.asarray(unit_labels)
    if labels.shape != grid_shape:
        raise ValueError(
            f"unit_labels shape {labels.shape} does not match the grid {grid_shape}"
        )
    total = labels.size
    return {
        f"unitclass_{c}_fraction": float(np.count_nonzero(labels == c) / total)
        for c in unit_classes
    }
