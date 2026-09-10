#!/usr/bin/env python3
"""Per-tile inspector: every layer and every channel on one shared extent.

The thing this is built to make visible is the one that is easy to get wrong
and impossible to check from a table: that every layer is cut on the *same*
fishnet, not on each source's own extent. So every panel below is drawn on
identical axis limits --- the zoom-15 tile's bounds in UTM --- and the tile
boundary is stroked on all of them. If any layer were being clipped to its own
bounding box, or to a grid offset from the tile index, the strokes would not
line up and the panels would not register.

The first panel widens to the tile's eight neighbours, which shows two further
things at once: that the tiling is a partition with no gaps or overlaps, and
how far past its own edge each tile's data was fetched, since a ring-based
indicator on the boundary reads from outside it.

Layout, per tile:

    row 1   the neighbourhood, then each raw vector layer on the tile extent
    row 2   each rasterized channel from the store, same extent
    row 3   each channel's marginal, so the picture and the histogram sit
            together --- a zero fraction means something different depending
            on whether the zeros have a shape

Usage:
    python inspect_tile.py --cell_ids 15_26097_15392
    python inspect_tile.py --busiest 4 --out_dir ../data/hcmc/processed/reports
    python inspect_tile.py --cell_ids 15_26097_15392 --channels building_binary
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "hcmc" / "raw"
TILES_DIR = RAW_DIR / "tiles"
PROC_DIR = ROOT / "data" / "hcmc" / "processed"
REPORTS_DIR = PROC_DIR / "reports"

_DDPM_ROOT = ROOT.parent / "ddpm-win-master"
if str(_DDPM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDPM_ROOT))

UTM_EPSG = 32648

VECTOR_STYLE = {
    "buildings": dict(facecolor="#1B2426", edgecolor="none", label="footprints"),
    "roads": dict(color="#1F6F8B", linewidth=0.7, label="road centrelines"),
    "pois": dict(color="#D9534F", markersize=14, label="POIs"),
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_layer(cell_id: str, layer: str, citywide=None, bounds=None, buffer_m=600.0):
    """A vector layer for one tile in UTM, or None when there is nothing."""
    path = TILES_DIR / cell_id / f"{layer}.gpkg"
    if path.exists():
        return gpd.read_file(path).to_crs(epsg=UTM_EPSG)
    if citywide is not None and bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        window = citywide.cx[
            xmin - buffer_m : xmax + buffer_m, ymin - buffer_m : ymax + buffer_m
        ]
        return window if len(window) else None
    return None


def tile_bounds_utm(tile_row):
    frame = gpd.GeoDataFrame(geometry=[tile_row.geometry], crs="EPSG:4326")
    return tuple(frame.to_crs(epsg=UTM_EPSG).total_bounds)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def _frame_axes(ax, bounds, title, boundary=True):
    """Every panel gets the same limits.  That is the whole point of the figure."""
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)
    if boundary:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, edgecolor="#B7791F", linewidth=1.6,
                               zorder=10))


def draw_neighbourhood(ax, tiles, tile_row, layers, bounds):
    """The tile and its eight neighbours, with every layer drawn over them."""
    xmin, ymin, xmax, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    wide = (xmin - width, ymin - height, xmax + width, ymax + height)

    neighbours = tiles[
        (tiles.x.between(tile_row.x - 1, tile_row.x + 1))
        & (tiles.y.between(tile_row.y - 1, tile_row.y + 1))
    ].to_crs(epsg=UTM_EPSG)
    neighbours.boundary.plot(ax=ax, color="#B7C2C3", linewidth=0.9, zorder=5)

    for name, frame in layers.items():
        if frame is None or not len(frame):
            continue
        style = dict(VECTOR_STYLE[name])
        style.pop("label", None)
        if name == "pois":
            ax.plot(frame.geometry.x, frame.geometry.y, "o",
                    color=style["color"], markersize=2.5, zorder=4)
        elif name == "roads":
            frame.plot(ax=ax, **style, zorder=3)
        else:
            frame.plot(ax=ax, **style, zorder=2)

    ax.add_patch(Rectangle((xmin, ymin), width, height, fill=False,
                           edgecolor="#B7791F", linewidth=2.2, zorder=11))
    _frame_axes(ax, wide, "the tile in its fishnet\n(gold = this cell, grey = neighbours)",
                boundary=False)


def draw_vector(ax, name, frame, bounds):
    if frame is None or not len(frame):
        _frame_axes(ax, bounds, f"{VECTOR_STYLE[name]['label']} — none fetched")
        return
    style = dict(VECTOR_STYLE[name])
    label = style.pop("label")
    if name == "pois":
        inside = frame.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        ax.plot(frame.geometry.x, frame.geometry.y, "o", color=style["color"],
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.4)
        count = f"{len(inside)} in tile, {len(frame)} with buffer"
    elif name == "roads":
        frame.plot(ax=ax, **style)
        count = f"{len(frame)} ways"
    else:
        frame.plot(ax=ax, **style)
        count = f"{len(frame)} polygons"
    _frame_axes(ax, bounds, f"{label}\n{count}")


def draw_channel(ax, grid, name, bounds, unit_m):
    from analysis.spatial import morans_i

    xmin, ymin, xmax, ymax = bounds
    # imshow wants (left, right, bottom, top), which is not the order a
    # bounding box comes in. Getting this wrong puts the raster somewhere off
    # the axes entirely, and since the axes are then clamped to the tile it
    # shows up as a stray rectangle rather than as an error.
    extent = [xmin, xmax, ymin, ymax]

    finite = grid[np.isfinite(grid)]
    if finite.size and finite.max() > finite.min():
        vmax = float(np.percentile(finite, 99.5))
        if vmax <= float(finite.min()):
            vmax = float(finite.max())
        ax.imshow(grid, cmap="magma", vmin=float(finite.min()), vmax=vmax,
                  interpolation="nearest", extent=extent, origin="upper")
        note = f"zeros {np.mean(grid == 0):.0%}  I={morans_i(grid):.2f}"
    else:
        ax.imshow(np.zeros_like(grid), cmap="magma", vmin=0, vmax=1,
                  interpolation="nearest", extent=extent, origin="upper")
        note = "constant — no data behind it"
    _frame_axes(ax, bounds, f"{name}\n{note}")


def draw_marginal(ax, grid, name):
    values = grid[np.isfinite(grid)].ravel()
    zero_share = float(np.mean(values == 0)) if values.size else 1.0
    non_zero = values[values != 0]
    if non_zero.size:
        ax.hist(non_zero, bins=60, color="#1F6F8B")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "all zero", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#5C6B6D")
    ax.set_title(f"{name}\nnon-zero only · {zero_share:.0%} zeros", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def inspect(tile_row, tiles, store, channels, citywide, out_path):
    bounds = tile_bounds_utm(tile_row)
    cell_id = tile_row.cell_id
    unit_m = float(tile_row.width_m) / store.img_size

    layers = {
        name: load_layer(cell_id, name, citywide.get(name), bounds)
        for name in ("buildings", "roads", "pois")
    }

    cell = store.read_cell(cell_id).astype(np.float64)
    names = store.channel_names
    keep = [names.index(c) for c in channels]

    n_top = 1 + len(layers)
    n_ch = len(keep)
    ncols = max(n_top, n_ch)
    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 11.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    draw_neighbourhood(axes[0][0], tiles, tile_row, layers, bounds)
    axes[0][0].axis("on")
    for i, (name, frame) in enumerate(layers.items(), start=1):
        draw_vector(axes[0][i], name, frame, bounds)
        axes[0][i].axis("on")

    for i, index in enumerate(keep):
        draw_channel(axes[1][i], cell[index], names[index], bounds, unit_m)
        axes[1][i].axis("on")
        draw_marginal(axes[2][i], cell[index], names[index])
        axes[2][i].axis("on")

    meta = store.manifest.get("metadata", {})
    radii = ", ".join(
        f"{k.replace('_m', '')} {v:.0f} m"
        for k, v in meta.items()
        if k.endswith("_radius_m") and isinstance(v, (int, float))
    )
    fig.suptitle(
        f"{cell_id}   ·   zoom 15, {tile_row.width_m:.0f} m across, "
        f"{store.img_size} px, {unit_m:.2f} m per unit\n"
        f"every panel on identical axis limits — the tile's own bounds; "
        f"gold outline is the fishnet cell" + (f"   ·   {radii}" if radii else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=105)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(RAW_DIR / "hcmc_tiles_z15.gpkg"))
    parser.add_argument("--store", default=str(PROC_DIR / "cells_z15_1024"))
    parser.add_argument("--cell_ids", nargs="+", default=None)
    parser.add_argument("--busiest", type=int, default=None,
                        help="Inspect the N tiles with the most built area.")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Channel subset; defaults to all of them.")
    parser.add_argument("--citywide_buildings", default=None)
    parser.add_argument("--citywide_pois", default=None)
    parser.add_argument("--citywide_roads", default=None)
    parser.add_argument("--out_dir", default=str(REPORTS_DIR))
    args = parser.parse_args()

    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(args.store)
    tiles = gpd.read_file(args.index)
    present = set(store.cell_ids())

    if args.cell_ids:
        selected = tiles[tiles["cell_id"].isin(args.cell_ids)]
    elif args.busiest:
        built = {
            c: float(store.read_cell(c)[0].mean()) for c in sorted(present)
        }
        top = sorted(built, key=built.get, reverse=True)[: args.busiest]
        selected = tiles[tiles["cell_id"].isin(top)]
    else:
        selected = tiles[tiles["cell_id"].isin(present)]

    missing = set(selected["cell_id"]) - present
    if missing:
        raise SystemExit(f"not in the store: {sorted(missing)}")
    if selected.empty:
        raise SystemExit("no tiles selected")

    citywide = {}
    for name in ("buildings", "pois", "roads"):
        path = getattr(args, f"citywide_{name}")
        if path and Path(path).exists():
            log.info(f"Loading city-wide {name} from {path}")
            citywide[name] = gpd.read_file(path).to_crs(epsg=UTM_EPSG)
        else:
            citywide[name] = None

    channels = args.channels or store.channel_names
    unknown = [c for c in channels if c not in store.channel_names]
    if unknown:
        raise SystemExit(f"no such channels: {unknown}; have {store.channel_names}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, tile_row in selected.iterrows():
        inspect(tile_row, tiles, store, channels, citywide,
                out_dir / f"inspect_{tile_row.cell_id}.png")


if __name__ == "__main__":
    main()
