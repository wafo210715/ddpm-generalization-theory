#!/usr/bin/env python3
"""POI-per-ring coverage report (B6 of the v4 pipeline map).

    Checkpoint: POI-per-ring coverage report.  The fraction of units whose
    ring is non-degenerate must be high before any training.  You asked for
    this check explicitly.

"Non-degenerate" is not a matter of taste. Shannon entropy over categories in a
ring needs at least two POIs *and* at least two categories; with fewer it is
zero, and a channel of zeros is not an indicator of anything. That is precisely
what happened to the pilot: at a 50 m ring, 97.5 % of its units failed the test
and the diversity channel was 97.5 % zeros.

The ring radius is an open decision (indicator specification, question 3), and
so is whether categories come from the coarse key scheme or the raw tag values
(question 2). This report therefore does not evaluate one setting. It sweeps
radii and reports both category schemes side by side, so the decision can be
made against the number that follows from it.

Three coverage levels are reported per radius:

    any POI            >= 1 POI in the ring        --- poi_count has a value
    countable          >= 2 POIs                   --- necessary for entropy
    non-degenerate     >= 2 POIs, >= 2 categories  --- entropy is defined

Usage:
    python report_poi_coverage.py --pois data/hcmc/raw/hcmc_pois_all.gpkg
    python report_poi_coverage.py --from_tiles --radii 50 100 200 300 500
"""

import argparse
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "hcmc" / "raw"
TILES_DIR = RAW_DIR / "tiles"
REPORTS_DIR = ROOT / "data" / "hcmc" / "processed" / "reports"

UTM_EPSG = 32648

# The pilot's radius is first, so its 97.5 % is reproduced rather than asserted.
DEFAULT_RADII_M = (50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0)

# Coverage is evaluated on a grid of sample points per tile rather than on all
# 1024^2 units: the answer is a fraction, and a 64x64 sample estimates it to
# well within a percentage point while running in seconds.
DEFAULT_SAMPLE_N = 64

CATEGORY_COLUMNS = ("category_key", "category_value")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def sample_points(bounds, n: int) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    xs = np.linspace(xmin, xmax, n, endpoint=False) + (xmax - xmin) / (2 * n)
    ys = np.linspace(ymax, ymin, n, endpoint=False) - (ymax - ymin) / (2 * n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def coverage_for_tile(
    pois: gpd.GeoDataFrame,
    bounds,
    radii,
    sample_n: int,
    category_columns=CATEGORY_COLUMNS,
) -> list[dict]:
    """Coverage at each radius, for one tile, under each category scheme."""
    from scipy.spatial import cKDTree

    points = sample_points(bounds, sample_n)
    rows = []

    if pois is None or len(pois) == 0:
        for radius in radii:
            for column in category_columns:
                rows.append({
                    "radius_m": radius, "category_column": column,
                    "n_units": len(points), "any_poi": 0.0,
                    "countable": 0.0, "non_degenerate": 0.0,
                    "mean_pois_in_ring": 0.0,
                })
        return rows

    poi_xy = np.column_stack([pois.geometry.x.to_numpy(), pois.geometry.y.to_numpy()])
    tree = cKDTree(poi_xy)
    labels = {
        column: pd.factorize(pois[column].fillna("unknown"))[0]
        for column in category_columns
        if column in pois.columns
    }

    for radius in radii:
        neighbours = tree.query_ball_point(points, r=radius)
        counts = np.array([len(m) for m in neighbours])

        for column, codes in labels.items():
            distinct = np.array([
                len(np.unique(codes[m])) if len(m) else 0 for m in neighbours
            ])
            rows.append({
                "radius_m": radius,
                "category_column": column,
                "n_units": len(points),
                "any_poi": float(np.mean(counts >= 1)),
                "countable": float(np.mean(counts >= 2)),
                "non_degenerate": float(np.mean((counts >= 2) & (distinct >= 2))),
                "mean_pois_in_ring": float(counts.mean()),
            })
    return rows


def load_tile_pois(cell_id: str) -> gpd.GeoDataFrame | None:
    path = TILES_DIR / cell_id / "pois.gpkg"
    if path.exists():
        return gpd.read_file(path).to_crs(epsg=UTM_EPSG)
    if (TILES_DIR / cell_id / "pois.empty").exists():
        return None
    raise FileNotFoundError(f"no POI layer for {cell_id}")




def coverage_map(selected, tiles_dir, radii, out_path, n=256, store_path=None):
    """Draw where the entropy channel is defined, per tile, per radius.

    A coverage fraction says how much of a channel is zero.  It cannot say
    whether the zeros are a coverage hole or genuinely empty ground, and those
    two want opposite responses --- one is a reason to find better data, the
    other is a typology.  Only the map separates them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    store = None
    if store_path:
        import sys as _sys
        ddpm = str(ROOT.parent / "ddpm-win-master")
        if ddpm not in _sys.path:
            _sys.path.insert(0, ddpm)
        from datasets.cell_tensors import CellTensorStore
        store = CellTensorStore(store_path)

    rows = list(selected.itertuples())
    fig, axes = plt.subplots(len(rows), 1 + len(radii),
                             figsize=(4.1 * (1 + len(radii)), 4.35 * len(rows)),
                             squeeze=False)

    for r, tile_row in enumerate(rows):
        frame = gpd.GeoDataFrame(geometry=[tile_row.geometry], crs="EPSG:4326")
        xmin, ymin, xmax, ymax = frame.to_crs(epsg=UTM_EPSG).total_bounds
        pois = load_tile_pois(tile_row.cell_id)

        ax = axes[r][0]
        if store is not None and tile_row.cell_id in set(store.cell_ids()):
            ax.imshow(store.read_cell(tile_row.cell_id)[0], cmap="gray_r",
                      interpolation="nearest", extent=[xmin, xmax, ymin, ymax])
        n_inside = 0
        if pois is not None and len(pois):
            xy = np.column_stack([pois.geometry.x, pois.geometry.y])
            inside = ((xy[:, 0] >= xmin) & (xy[:, 0] <= xmax)
                      & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax))
            n_inside = int(inside.sum())
            ax.scatter(xy[inside, 0], xy[inside, 1], s=16, c="#D9534F",
                       edgecolors="white", linewidths=.5, zorder=3)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_title(f"{tile_row.cell_id}\nfootprints + {n_inside} POIs in tile", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        xs = np.linspace(xmin, xmax, n, endpoint=False) + (xmax - xmin) / (2 * n)
        ys = np.linspace(ymax, ymin, n, endpoint=False) - (ymax - ymin) / (2 * n)
        gx, gy = np.meshgrid(xs, ys)
        query = np.column_stack([gx.ravel(), gy.ravel()])

        for c, radius in enumerate(radii):
            ax = axes[r][c + 1]
            if pois is None or len(pois) == 0:
                defined = np.zeros((n, n), dtype=bool)
            else:
                tree = cKDTree(np.column_stack([pois.geometry.x, pois.geometry.y]))
                codes = pd.factorize(pois["category_key"].fillna("unknown"))[0]
                neighbours = tree.query_ball_point(query, r=radius)
                defined = np.array([
                    len(m) >= 2 and len(np.unique(codes[m])) >= 2 for m in neighbours
                ]).reshape(n, n)
            ax.imshow(defined, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{radius:.0f} m ring - entropy defined on "
                         f"{defined.mean():.1%}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Where the POI entropy channel is actually defined\n"
                 "green = the unit's ring holds >=2 POIs in >=2 categories;  "
                 "red = the channel is zero there", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(out_path, dpi=105)
    log.info(f"Wrote {out_path}")


def fetched_cell_ids() -> set:
    """Tiles whose fetch actually completed.

    fetch.json is written only after every requested layer has landed, so a
    tile directory without one is an interrupted fetch. Counting it as data
    would be the v2 pipeline's silent-partial failure in a new place.
    """
    if not TILES_DIR.exists():
        return set()
    return {
        d.name for d in TILES_DIR.iterdir()
        if d.is_dir() and (d / "fetch.json").exists()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=str,
                        default=str(RAW_DIR / "hcmc_tiles_z15.gpkg"))
    parser.add_argument("--pois", type=str, default=None,
                        help="A city-wide POI layer, clipped per tile.")
    parser.add_argument("--from_tiles", action="store_true",
                        help="Use the per-tile fetches under raw/tiles instead.")
    parser.add_argument("--cell_ids", type=str, nargs="+", default=None)
    parser.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII_M))
    parser.add_argument("--sample_n", type=int, default=DEFAULT_SAMPLE_N)
    parser.add_argument("--buffer_m", type=float, default=600.0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--map", action="store_true",
                        help="Also draw where the channel is defined. A "
                             "fraction cannot tell a coverage hole from empty "
                             "ground; the map can.")
    parser.add_argument("--map_radii", type=float, nargs="+",
                        default=[50.0, 100.0, 300.0])
    parser.add_argument("--map_tiles", type=int, default=3,
                        help="How many tiles to draw, taken from the ones with "
                             "the most POIs so the figure is not all red.")
    parser.add_argument("--store", type=str,
                        default=str(ROOT / "data" / "hcmc" / "processed"
                                    / "cells_z15_1024"),
                        help="Cell store, for the footprint backdrop.")
    args = parser.parse_args()

    tiles = gpd.read_file(args.index)
    if args.cell_ids:
        selected = tiles[tiles["cell_id"].isin(args.cell_ids)]
    elif args.from_tiles:
        fetched = fetched_cell_ids()
        selected = tiles[tiles["cell_id"].isin(fetched)]
    else:
        selected = tiles
    if selected.empty:
        raise SystemExit("no tiles selected")

    citywide = None
    if args.pois:
        log.info(f"Loading {args.pois}")
        citywide = gpd.read_file(args.pois).to_crs(epsg=UTM_EPSG)
        log.info(f"  {len(citywide):,} POIs")

    log.info(f"Coverage over {len(selected)} tiles, "
             f"{args.sample_n}x{args.sample_n} sample units each")

    all_rows = []
    for _, tile_row in selected.iterrows():
        frame = gpd.GeoDataFrame(geometry=[tile_row.geometry], crs="EPSG:4326")
        bounds = tuple(frame.to_crs(epsg=UTM_EPSG).total_bounds)

        if citywide is not None:
            xmin, ymin, xmax, ymax = bounds
            pois = citywide.cx[
                xmin - args.buffer_m : xmax + args.buffer_m,
                ymin - args.buffer_m : ymax + args.buffer_m,
            ]
        else:
            pois = load_tile_pois(tile_row.cell_id)

        for row in coverage_for_tile(pois, bounds, args.radii, args.sample_n):
            row["cell_id"] = tile_row.cell_id
            all_rows.append(row)

    per_tile = pd.DataFrame(all_rows)
    summary = (
        per_tile.groupby(["radius_m", "category_column"])[
            ["any_poi", "countable", "non_degenerate", "mean_pois_in_ring"]
        ]
        .mean()
        .reset_index()
    )

    print()
    print(f"POI-per-ring coverage, {len(selected)} tiles, "
          f"{args.sample_n * args.sample_n:,} units per tile")
    print()
    header = (f"{'radius':>8} | {'categories':>15} | {'any POI':>8} | "
              f"{'>=2 POIs':>9} | {'entropy defined':>16} | {'mean POIs':>10}")
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        print(f"{row.radius_m:>7.0f}m | {row.category_column:>15} | "
              f"{row.any_poi:>7.1%} | {row.countable:>8.1%} | "
              f"{row.non_degenerate:>15.1%} | {row.mean_pois_in_ring:>10.1f}")

    print("\n'entropy defined' is the fraction B6 asks about: units with at "
          "least\ntwo POIs in at least two categories. Everything else is a "
          "zero in the\npoi_entropy channel.")

    out_dir = Path(args.out) if args.out else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    per_tile.to_parquet(out_dir / "poi_coverage_per_tile.parquet")
    summary.to_csv(out_dir / "poi_coverage_summary.csv", index=False)
    log.info(f"Wrote {out_dir}/poi_coverage_summary.csv and "
             f"{out_dir}/poi_coverage_per_tile.parquet")

    if args.map:
        busiest = (
            per_tile[per_tile.category_column == "category_key"]
            .groupby("cell_id")["mean_pois_in_ring"].max()
            .sort_values(ascending=False).head(args.map_tiles).index
        )
        coverage_map(
            selected[selected["cell_id"].isin(busiest)],
            TILES_DIR, args.map_radii,
            str(out_dir / "poi_ring_coverage_map.png"),
            store_path=args.store if os.path.exists(args.store) else None,
        )


if __name__ == "__main__":
    main()
