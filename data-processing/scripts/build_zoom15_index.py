#!/usr/bin/env python3
"""Zoom-15 tile index for the HCMC study area (B1 of the v4 pipeline map).

    Cut the study area into zoom-15 map tiles (1,201 m).  In generalization
    theory (Bonnaire et al.) the memorization onset grows linearly with
    sample count.  Four times the tiles of zoom 14 widens the observable
    generalization window.  Typology is assigned per cell.  1.2 km cells
    are purer than 2.4 km cells, which keeps mutants clean.  Reversible:
    four zoom-15 tiles merge into one zoom-14 tile.  Tile addressing
    standardizes other cities later.  Verified: 1,200 to 1,203 m across all
    of HCMC.  No overlapping windows: near-duplicates corrupt the k-NN gap
    ratio.

This replaces the pilot's 1 km fishnet, which was an arbitrary grid snapped to
UTM coordinates.  Web-Mercator tiles instead, because a tile has an address
that means the same thing in every city, four zoom-15 tiles fold back into one
zoom-14 tile exactly, and the tiling is a partition --- no unit belongs to two
cells, which is what the last sentence above is protecting.

The three claims in B1 that are checkable are checked here rather than assumed:
tile width across the study area, exact non-overlap, and the zoom-14 folding.

Output
------
    data/hcmc/raw/hcmc_tiles_z15.gpkg      one row per tile
    data/hcmc/raw/hcmc_tiles_z15.parquet   the same table without geometry

Columns: ``cell_id`` (``z_x_y``), ``z``, ``x``, ``y``, tile bounds in WGS84,
tile width and height in metres, the centroid, the area of the tile inside the
municipality boundary, and ``parent_z14`` for the folding.

Usage:
    python build_zoom15_index.py
    python build_zoom15_index.py --zoom 14 --min_coverage 0.5
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "hcmc" / "raw"

UTM_EPSG = 32648  # UTM zone 48N, the metric CRS for HCMC
DEFAULT_ZOOM = 15

# B1's verified range, in metres.  A tile outside it means the tiling maths or
# the study area has changed, and the 1.17 m unit no longer follows from 1024 px.
EXPECTED_WIDTH_M = (1195.0, 1210.0)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===========================================================================
# Web Mercator tile arithmetic
# ===========================================================================


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """The (x, y) slippy-map tile containing a coordinate."""
    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` of a tile, in degrees."""
    n = 2.0**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def tile_width_m(y: int, zoom: int) -> float:
    """Ground width of a tile at its centre latitude.

    Web Mercator tiles shrink with latitude: 40 075 017 * cos(lat) / 2^zoom.
    """
    _, south, _, north = tile_bounds(0, y, zoom)
    centre_lat = (south + north) / 2.0
    return 40_075_016.686 * math.cos(math.radians(centre_lat)) / (2.0**zoom)


# ===========================================================================
# Index
# ===========================================================================


def load_boundary(name: str = "hcmc_boundary_2025.gpkg") -> gpd.GeoDataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Boundary not found: {path}. Run build_boundary_grid.py first."
        )
    boundary = gpd.read_file(path)
    log.info(f"Loaded boundary: {len(boundary)} polygon(s), CRS={boundary.crs}")
    return boundary


def build_tile_index(
    boundary: gpd.GeoDataFrame, zoom: int, min_coverage: float
) -> gpd.GeoDataFrame:
    """Every tile at *zoom* whose intersection with the boundary is large enough.

    Args:
        min_coverage: Keep a tile when at least this fraction of it lies inside
            the municipality.  0.0 keeps every tile the boundary touches.
            Coverage is reported per tile either way, so the threshold can be
            revisited without rebuilding the index.
    """
    boundary_wgs = boundary.to_crs(epsg=4326)
    geometry = boundary_wgs.union_all()
    west, south, east, north = geometry.bounds

    x_min, y_max = lonlat_to_tile(west, south, zoom)
    x_max, y_min = lonlat_to_tile(east, north, zoom)
    log.info(
        f"Study area spans tiles x {x_min}..{x_max}, y {y_min}..{y_max} "
        f"at zoom {zoom} ({(x_max - x_min + 1) * (y_max - y_min + 1):,} candidates)"
    )

    records = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            w, s, e, n = tile_bounds(x, y, zoom)
            tile = box(w, s, e, n)
            if not tile.intersects(geometry):
                continue
            inside = tile.intersection(geometry)
            coverage = inside.area / tile.area
            if coverage < min_coverage:
                continue
            records.append(
                {
                    "cell_id": f"{zoom}_{x}_{y}",
                    "z": zoom,
                    "x": x,
                    "y": y,
                    "west": w,
                    "south": s,
                    "east": e,
                    "north": n,
                    "centre_lon": (w + e) / 2.0,
                    "centre_lat": (s + n) / 2.0,
                    "width_m": tile_width_m(y, zoom),
                    "coverage": coverage,
                    "parent_z14": f"{zoom - 1}_{x // 2}_{y // 2}",
                    "geometry": tile,
                }
            )

    if not records:
        raise RuntimeError("No tiles intersect the boundary")

    tiles = gpd.GeoDataFrame(records, crs="EPSG:4326")
    tiles = tiles.sort_values(["y", "x"]).reset_index(drop=True)

    # Height in metres has to be measured, not derived: a Mercator tile is
    # square in projected coordinates but not on the ground.
    utm = tiles.to_crs(epsg=UTM_EPSG)
    bounds = utm.bounds
    tiles["height_m"] = (bounds["maxy"] - bounds["miny"]).to_numpy()
    tiles["utm_width_m"] = (bounds["maxx"] - bounds["minx"]).to_numpy()
    tiles["area_inside_km2"] = (
        utm.geometry.intersection(
            boundary.to_crs(epsg=UTM_EPSG).union_all()
        ).area
        / 1e6
    ).to_numpy()

    log.info(f"Kept {len(tiles):,} tiles at coverage >= {min_coverage}")
    return tiles


# ===========================================================================
# The checks B1 states
# ===========================================================================


def verify(tiles: gpd.GeoDataFrame, zoom: int) -> dict:
    """Check B1's three claims and return what was measured."""
    log.info("=" * 60)
    log.info("Verification")
    log.info("=" * 60)

    widths = tiles["width_m"].to_numpy()
    lo, hi = float(widths.min()), float(widths.max())
    log.info(f"Tile width: {lo:.1f} m to {hi:.1f} m (spread {hi - lo:.1f} m)")
    if not (EXPECTED_WIDTH_M[0] <= lo and hi <= EXPECTED_WIDTH_M[1]):
        raise AssertionError(
            f"tile width {lo:.1f}-{hi:.1f} m is outside the expected "
            f"{EXPECTED_WIDTH_M[0]}-{EXPECTED_WIDTH_M[1]} m; the 1.17 m unit at "
            f"1024 px no longer follows"
        )

    unit_m = widths.mean() / 1024.0
    log.info(f"Unit at 1024 px: {unit_m:.3f} m")

    # Non-overlap: tile ids are unique and integer coordinates partition the
    # plane by construction, so uniqueness is the whole check.  B1 cares about
    # this because a near-duplicate cell corrupts the k-NN gap ratio: a
    # generated sample would sit equally close to two real cells that are
    # actually the same place.
    n_unique = tiles["cell_id"].nunique()
    if n_unique != len(tiles):
        raise AssertionError(f"{len(tiles) - n_unique} duplicate tile ids")
    log.info(f"No overlapping windows: {n_unique:,} unique tile ids")

    # Folding: every group of four children shares one zoom-14 parent.
    per_parent = tiles.groupby("parent_z14").size()
    log.info(
        f"Folds into {len(per_parent):,} zoom-14 parents "
        f"({per_parent.min()}-{per_parent.max()} children each)"
    )
    if per_parent.max() > 4:
        raise AssertionError("a zoom-14 parent has more than four zoom-15 children")

    heights = tiles["height_m"].to_numpy()
    log.info(f"Tile height in UTM: {heights.min():.1f} m to {heights.max():.1f} m")

    total_area = float(tiles["area_inside_km2"].sum())
    log.info(f"Total municipality area covered: {total_area:,.1f} km^2")

    log.info("All verifications passed")
    return {
        "n_tiles": int(len(tiles)),
        "width_m_min": lo,
        "width_m_max": hi,
        "unit_m_at_1024px": float(unit_m),
        "n_parents_z14": int(len(per_parent)),
        "area_inside_km2": total_area,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--boundary", default="hcmc_boundary_2025.gpkg",
                        help="boundary file in data/hcmc/raw to index against")
    parser.add_argument(
        "--min_coverage", type=float, default=0.0,
        help="Minimum share of a tile inside the municipality. 0.0 keeps every "
             "tile the boundary touches; coverage is a column either way, so "
             "the dataset can be filtered later without rebuilding.",
    )
    parser.add_argument("--out_prefix", type=str, default=None)
    args = parser.parse_args()

    boundary = load_boundary(args.boundary)
    tiles = build_tile_index(boundary, args.zoom, args.min_coverage)
    summary = verify(tiles, args.zoom)

    prefix = args.out_prefix or str(RAW_DIR / f"hcmc_tiles_z{args.zoom}")
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    tiles.to_file(f"{prefix}.gpkg", driver="GPKG")
    pd.DataFrame(tiles.drop(columns="geometry")).to_parquet(f"{prefix}.parquet")
    log.info(f"Wrote {prefix}.gpkg and {prefix}.parquet")

    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
