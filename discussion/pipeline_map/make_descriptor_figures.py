#!/usr/bin/env python3
"""Measure everything section 8 of the companion page shows, and cache it.

The section teaches the cell descriptor (C2) from real HCMC cells rather than
from a schematic, so the numbers in it have to come from the data.  This script
reads the rasterized store, computes them, renders the thumbnails, and writes
``descriptor_figures.json``.  ``gen.py`` then builds the page from that file
alone, so the page can be regenerated on a machine that has no raw data.

Thumbnails are alpha ramps of one mid-tone colour on a transparent background,
so they read on the light and the dark theme alike --- a fixed-background
raster would only work on one.

Usage:
    python make_descriptor_figures.py [--store PATH] [--out PATH]
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DDPM = os.path.join(REPO, "ddpm-win-master")
if DDPM not in sys.path:
    sys.path.insert(0, DDPM)

DEFAULT_STORE = os.path.join(
    REPO, "data-processing", "data", "hcmc", "processed", "cells_z15_1024"
)

THUMB = 160          # thumbnail edge, in pixels
UNIT_M = 1.1736      # ground metres per unit at 1024 px, zoom 15


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def thumbnail(grid, colour=(31, 111, 139), gamma=1.0, size=THUMB, crisp=False):
    """A grid as a base64 PNG: one colour, alpha carrying the value.

    Transparent where the value is zero, so the page's own surface shows
    through and the image works on both themes.

    Args:
        crisp: For synthetic patterns, whose point is the arrangement of
            individual units.  Scales by a whole number with nearest-neighbour
            so a chequer reads as a chequer; a fractional scale turns one into
            a moire and destroys the thing the picture is there to show.
            Real rasters take the default: area-average down, then Lanczos,
            which is what an aerial view of footprints should look like.
    """
    from PIL import Image

    g = np.asarray(grid, dtype=np.float64)
    if not crisp:
        # Pool to roughly twice the target, then resample down cleanly.
        factor = max(g.shape[0] // (2 * size), 1)
        if factor > 1:
            n = g.shape[0] // factor
            g = g[: n * factor, : n * factor].reshape(n, factor, n, factor).mean(axis=(1, 3))

    n = g.shape[0]
    top = g.max()
    alpha = (g / top) ** gamma if top > 0 else np.zeros_like(g)
    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = colour
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    if crisp:
        scale = max(round(size / n), 1)
        image = image.resize((n * scale, n * scale), Image.NEAREST)
    else:
        image = image.resize((size, size), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------------------
# the four things the section shows
# ---------------------------------------------------------------------------


def morans_strip():
    """Five patterns whose spatial arrangement is known, with their Moran's I.

    The point of the strip is that Moran's I reads *arrangement*, and that the
    rook and queen answers differ on a fine chequer --- which is the pattern
    HCMC's alleys most resemble at a 1.17 m unit.
    """
    from analysis.spatial import morans_i

    size = 32
    rng = np.random.default_rng(3)
    yy, xx = np.indices((size, size))

    patterns = [
        ("chequer", "every unit's neighbours are its opposite",
         ((yy + xx) % 2).astype(float)),
        ("noise", "no relation between a unit and its neighbours",
         rng.random((size, size))),
        ("small blobs", "clumps a few units across",
         _blobs(size, 26, 2, rng)),
        ("large blobs", "clumps tens of units across",
         _blobs(size, 5, 6, rng)),
        ("smooth ramp", "every unit almost equals its neighbours",
         xx / (size - 1.0)),
    ]

    out = []
    for name, note, grid in patterns:
        out.append({
            "name": name,
            "note": note,
            "queen": round(morans_i(grid, 8), 3),
            "rook": round(morans_i(grid, 4), 3),
            "png": thumbnail(grid, size=128, crisp=True),
        })
    return out


def _blobs(size, count, radius, rng):
    yy, xx = np.indices((size, size))
    grid = np.zeros((size, size))
    for _ in range(count):
        cy, cx = rng.integers(0, size, 2)
        grid[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = 1.0
    return grid


def shuffle_pair(store, cell_id, channel=0):
    """One real cell, and the same cell with its units shuffled.

    Shuffling is the cleanest demonstration of what the descriptor's spatial
    term is for: it preserves the multiset of unit values exactly, so mean,
    standard deviation, zero fraction and skewness cannot move, and it
    destroys every spatial relation, so Moran's I collapses.  Whatever the
    descriptor still tells apart afterwards is the part of it that is about
    layout rather than about amount.
    """
    from analysis.cell_descriptor import describe_cell

    names = store.channel_names
    cell = store.read_cell(cell_id).astype(np.float64)

    rng = np.random.default_rng(0)
    shuffled = cell.copy()
    for i in range(shuffled.shape[0]):
        flat = shuffled[i].ravel()
        rng.shuffle(flat)
        shuffled[i] = flat.reshape(cell.shape[1:])

    real = describe_cell(cell, names)
    fake = describe_cell(shuffled, names)
    keys = [f"{names[channel]}_{s}"
            for s in ("mean", "std", "zero_fraction", "skewness", "morans_i")]

    return {
        "cell_id": cell_id,
        "channel": names[channel],
        "real_png": thumbnail(cell[channel], size=200),
        "shuffled_png": thumbnail(shuffled[channel], size=200),
        "rows": [
            {
                "key": k.replace(names[channel] + "_", ""),
                "real": round(real[k], 4),
                "shuffled": round(fake[k], 4),
                "moved": abs(real[k] - fake[k]) > 1e-6,
            }
            for k in keys
        ],
    }


def real_cells(store, cell_ids):
    """Two or three real cells with their descriptors, side by side."""
    from analysis.cell_descriptor import describe_cell
    from analysis.spatial import bivariate_morans_i

    names = store.channel_names
    out = []
    for cell_id in cell_ids:
        cell = store.read_cell(cell_id).astype(np.float64)
        d = describe_cell(cell, names)
        out.append({
            "cell_id": cell_id,
            "binary_png": thumbnail(cell[0], size=180),
            "density_png": thumbnail(cell[1], colour=(183, 121, 31), size=180),
            "built_fraction": round(1.0 - d["building_binary_zero_fraction"], 4),
            "binary_morans_i": round(d["building_binary_morans_i"], 3),
            "density_mean": round(d["building_density_mean"], 4),
            "density_morans_i": round(d["building_density_morans_i"], 4),
            "cross": round(bivariate_morans_i(cell[0], cell[1]), 3),
        })
    return out


def resolution_curve(store, cell_ids):
    """How much of each channel survives being read on a coarser grid.

    Moran's I says a field is smooth; it does not say how much information the
    fine grid is carrying.  This does, directly: pool to a coarser grid,
    interpolate back, and measure the share of variance that survives.
    """
    from scipy.ndimage import zoom

    names = store.channel_names
    factors = [2, 4, 8, 16, 32, 64, 128]
    grids = [store.read_cell(c).astype(np.float64) for c in cell_ids]
    size = grids[0].shape[-1]

    series = []
    for channel in (0, 1):
        points = [{"unit_m": round(UNIT_M, 2), "grid": size, "retained": 1.0}]
        for f in factors:
            retained = []
            for cell in grids:
                g = cell[channel]
                if g.std() == 0:
                    continue
                n = g.shape[0] // f
                coarse = g[: n * f, : n * f].reshape(n, f, n, f).mean(axis=(1, 3))
                back = zoom(coarse, g.shape[0] / n, order=1,
                            grid_mode=True, mode="nearest")[: g.shape[0], : g.shape[0]]
                retained.append(1.0 - (g - back).var() / g.var())
            if retained:
                points.append({
                    "unit_m": round(UNIT_M * f, 1),
                    "grid": size // f,
                    "retained": round(float(np.mean(retained)), 4),
                })
        series.append({"channel": names[channel], "points": points})
    return series


def sparsity_map(store, cell_ids, channel_name):
    """Where a sparse channel's zeros actually are.

    A zero fraction is a number about a distribution; it says nothing about
    arrangement.  Two channels can both be 87 % zero with one of them empty in
    a coherent shape --- a river, a park, the inside of a block --- and the
    other empty at random.  Only the first is something a model can learn.
    """
    from analysis.spatial import morans_i

    names = store.channel_names
    if channel_name not in names:
        return None
    index = names.index(channel_name)

    out = []
    for cell_id in cell_ids:
        grid = store.read_cell(cell_id).astype(np.float64)[index]
        if grid.max() <= 0:
            continue
        empty = (grid == 0).astype(float)
        out.append({
            "cell_id": cell_id,
            "channel": channel_name,
            "zero_fraction": round(float(empty.mean()), 4),
            # Moran's I of the emptiness mask itself: whether the zeros are a
            # shape or a scatter.
            "zero_morans_i": round(morans_i(empty), 3),
            "png": thumbnail(empty, colour=(92, 107, 109), size=200),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--out", default=os.path.join(HERE, "descriptor_figures.json"))
    parser.add_argument("--cells", nargs="+", default=[
        "15_26097_15392", "15_26098_15391", "15_26102_15388",
    ])
    args = parser.parse_args()

    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(args.store)
    present = set(store.cell_ids())
    cells = [c for c in args.cells if c in present]
    if not cells:
        raise SystemExit(f"none of {args.cells} are in {args.store}")

    print(f"Reading {len(cells)} cells from {args.store}")
    payload = {
        "store": os.path.relpath(args.store, REPO),
        "unit_m": UNIT_M,
        "img_size": store.img_size,
        "cells": cells,
        "morans_strip": morans_strip(),
        "shuffle": shuffle_pair(store, cells[0]),
        "real_cells": real_cells(store, cells),
        "resolution": resolution_curve(store, cells),
        "sparsity": sparsity_map(store, cells, "building_binary"),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out} ({size_kb:.0f} kB)")

    print("\nMoran's I strip:")
    for row in payload["morans_strip"]:
        print(f"  {row['name']:>12}: queen {row['queen']:>7.3f}  rook {row['rook']:>7.3f}")
    print("\nShuffle test, " + payload["shuffle"]["channel"] + ":")
    for row in payload["shuffle"]["rows"]:
        mark = "MOVED" if row["moved"] else "same"
        print(f"  {row['key']:>14}: {row['real']:>9.4f} -> {row['shuffled']:>9.4f}  {mark}")


if __name__ == "__main__":
    main()
