#!/usr/bin/env python3
"""Fix cell PNG images: re-slice canvases with proper per-cell R/G normalization.

The existing cell PNGs were produced by an older pipeline version that did raw
float-to-uint8 casts on R (accessibility) and G (POI entropy) channels, yielding
values in the 0-30 range instead of the intended 0-255 per-cell-normalized range.
The canvases themselves are correct — this script only regenerates the cell PNGs.

Usage:
    cd data-processing
    python scripts/fix_cell_images.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "hcmc"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CELLS_DIR = PROC_DIR / "cells_256x256"

CELL_PX = 256
UTM_EPSG = 32648
GRID_RES_KM = 1.0


def main():
    print("=" * 64)
    print("FIX CELL IMAGES — per-cell R/G normalization")
    print("=" * 64)

    # ── Load canvases ────────────────────────────────────────────────
    b_path = PROC_DIR / "b_canvas.npy"
    g_path = PROC_DIR / "g_canvas.npy"
    r_path = PROC_DIR / "r_canvas.npy"

    print("\n[1/4] Loading canvases...")
    b_canvas = np.load(b_path)
    g_canvas = np.load(g_path)
    r_canvas = np.load(r_path)
    print(
        f"  B canvas: shape={b_canvas.shape}, dtype={b_canvas.dtype}, "
        f"range=[{b_canvas.min()}, {b_canvas.max()}]"
    )
    print(
        f"  G canvas: shape={g_canvas.shape}, dtype={g_canvas.dtype}, "
        f"range=[{g_canvas.min():.6f}, {g_canvas.max():.6f}]"
    )
    print(
        f"  R canvas: shape={r_canvas.shape}, dtype={r_canvas.dtype}, "
        f"range=[{r_canvas.min():.6f}, {r_canvas.max():.6f}]"
    )

    # ── Load grid ────────────────────────────────────────────────────
    grid_path = RAW_DIR / "hcmc_1km_grid_full.gpkg"
    print("\n[2/4] Loading grid...")
    grid_gdf = gpd.read_file(grid_path)
    print(f"  Grid cells: {len(grid_gdf)}")

    # ── Grid-to-pixel mapping ────────────────────────────────────────
    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values

    unique_x = np.unique(np.floor(cx / 1000) * 1000)
    unique_y = np.unique(np.floor(cy / 1000) * 1000)
    x_to_col = {v: i for i, v in enumerate(unique_x)}
    y_to_row = {v: i for i, v in enumerate(unique_y)}

    n_rows = len(unique_y)
    n_cols = len(unique_x)
    print(f"  Grid layout: {n_rows} rows x {n_cols} cols")
    print(f"  Expected canvas: {n_rows * CELL_PX} x {n_cols * CELL_PX}")

    # ── Re-slice and save ────────────────────────────────────────────
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/4] Re-slicing to {CELLS_DIR}...")

    saved = 0
    empty = 0
    r_stats = {"has_data": 0, "maxes": []}
    g_stats = {"has_data": 0, "maxes": []}
    b_stats = {"has_data": 0}

    for idx, (i, row) in enumerate(grid_gdf.iterrows()):
        cell_x0 = np.floor(cx[i] / 1000) * 1000
        cell_y0 = np.floor(cy[i] / 1000) * 1000
        grid_col = x_to_col.get(cell_x0)
        grid_row = y_to_row.get(cell_y0)
        if grid_col is None or grid_row is None:
            continue

        px_row = grid_row * CELL_PX
        px_col = grid_col * CELL_PX

        if px_row + CELL_PX > b_canvas.shape[0] or px_col + CELL_PX > b_canvas.shape[1]:
            continue

        b_cell = b_canvas[px_row : px_row + CELL_PX, px_col : px_col + CELL_PX]
        g_cell = g_canvas[px_row : px_row + CELL_PX, px_col : px_col + CELL_PX]
        r_cell = r_canvas[px_row : px_row + CELL_PX, px_col : px_col + CELL_PX]

        # Per-cell normalization for G channel
        g_max = g_cell.max()
        g_uint8 = (
            np.clip(g_cell / g_max * 255, 0, 255).astype(np.uint8)
            if g_max > 0
            else g_cell.astype(np.uint8)
        )

        # Per-cell normalization for R channel
        r_max = r_cell.max()
        r_uint8 = (
            np.clip(r_cell / r_max * 255, 0, 255).astype(np.uint8)
            if r_max > 0
            else r_cell.astype(np.uint8)
        )

        # B channel is already binary uint8 — no normalization
        b_out = b_cell.astype(np.uint8)

        rgb = np.stack([r_uint8, g_uint8, b_out], axis=-1)

        label = f"{saved:04d}"
        out_path = CELLS_DIR / f"{label}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path)
        saved += 1

        # Track statistics
        if r_max > 0:
            r_stats["has_data"] += 1
            r_stats["maxes"].append(r_uint8.max())
        if g_max > 0:
            g_stats["has_data"] += 1
            g_stats["maxes"].append(g_uint8.max())
        if b_cell.max() > 0:
            b_stats["has_data"] += 1
        if rgb.max() == 0:
            empty += 1

    # ── Quality report ───────────────────────────────────────────────
    print(f"\n[4/4] Quality Report")
    print("=" * 64)
    print(f"  Total cells saved:    {saved}")
    print(f"  Empty (all-black):    {empty} ({100 * empty / saved:.1f}%)")

    print(f"\n  R channel (accessibility):")
    print(
        f"    Cells with data:    {r_stats['has_data']}/{saved} "
        f"({100 * r_stats['has_data'] / saved:.1f}%)"
    )
    if r_stats["maxes"]:
        arr = np.array(r_stats["maxes"])
        print(
            f"    R_max distribution: min={arr.min()}, median={np.median(arr):.0f}, "
            f"mean={arr.mean():.1f}, max={arr.max()}"
        )
        print(
            f"    R_max == 255:       {np.sum(arr == 255)}/{len(arr)} "
            f"({100 * np.sum(arr == 255) / len(arr):.1f}%)"
        )
        print(
            f"    R_max >= 250:       {np.sum(arr >= 250)}/{len(arr)} "
            f"({100 * np.sum(arr >= 250) / len(arr):.1f}%)"
        )

    print(f"\n  G channel (POI entropy):")
    print(
        f"    Cells with data:    {g_stats['has_data']}/{saved} "
        f"({100 * g_stats['has_data'] / saved:.1f}%)"
    )
    if g_stats["maxes"]:
        arr = np.array(g_stats["maxes"])
        print(
            f"    G_max distribution: min={arr.min()}, median={np.median(arr):.0f}, "
            f"mean={arr.mean():.1f}, max={arr.max()}"
        )
        print(
            f"    G_max == 255:       {np.sum(arr == 255)}/{len(arr)} "
            f"({100 * np.sum(arr == 255) / len(arr):.1f}%)"
        )

    print(f"\n  B channel (buildings — binary):")
    print(
        f"    Cells with data:    {b_stats['has_data']}/{saved} "
        f"({100 * b_stats['has_data'] / saved:.1f}%)"
    )

    # Quick sanity: spot-check a few images
    print(f"\n  Spot-check (first 3 cells):")
    for label_idx in range(min(3, saved)):
        img_path = CELLS_DIR / f"{label_idx:04d}.png"
        img = np.array(Image.open(img_path))
        print(
            f"    {label_idx:04d}.png: R=[{img[:, :, 0].min()},{img[:, :, 0].max()}] "
            f"G=[{img[:, :, 1].min()},{img[:, :, 1].max()}] "
            f"B=[{img[:, :, 2].min()},{img[:, :, 2].max()}]"
        )

    print("\n" + "=" * 64)
    print("DONE — fixed cell images written to", CELLS_DIR)
    print("=" * 64)


if __name__ == "__main__":
    main()
