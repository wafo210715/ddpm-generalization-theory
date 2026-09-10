#!/usr/bin/env python3
"""Fetch OSM layers for a set of zoom-15 tiles (A3 / B3-B5 input).

The v2 pipeline's fetchers split the whole municipality into four quadrants and
ask Overpass for each in one query.  On this network that times out repeatedly
in the dense quadrants, and --- the part that matters --- the v2 code logs
"Skipping tile" and carries on, so a partial download is indistinguishable from
a complete one in the files it leaves behind.

This fetcher works tile by tile instead.  Each zoom-15 tile is a 1.2 km square,
which Overpass answers in a second or two, and a tile that cannot be fetched
after its retries raises rather than silently leaving a hole in the study area.

Every layer is buffered outward before fetching.  An indicator measured over a
ring of radius r at a unit on the tile edge reads data from up to r beyond it,
so a tile fetched to its own boundary would have a rim of units computed from
truncated neighbourhoods --- an artefact that would look like a real edge to
the model.

Output, per layer, under data/hcmc/raw/tiles/<cell_id>/:
    buildings.gpkg   OSM way["building"]
    pois.gpkg        amenity, shop, office, tourism, leisure
    roads.gpkg       highway=* centrelines

Usage:
    python fetch_tile_osm.py --sample 8 --seed 42
    python fetch_tile_osm.py --cell_ids 15_26093_15409 15_26094_15409
    python fetch_tile_osm.py --sample 8 --buffer_m 500 --layers buildings pois
"""

import argparse
import logging
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point, Polygon, box

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "hcmc" / "raw"
TILES_DIR = RAW_DIR / "tiles"

UTM_EPSG = 32648
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass rejects the default python-requests User-Agent with 406 Not
# Acceptable, so every query has to identify itself.
OSM_HTTP_HEADERS = {
    "User-Agent": (
        "ddpm-generalization-theory/1.0 "
        "(urban vitality research; contact wafo210715@gmail.com)"
    )
}

# The ring and decay radii are open decisions (see
# discussion/indicator_specification.md), so the fetch buffer is set well above
# any of the candidates rather than tied to one of them.  300 m is the plan's
# "a few hundred metres" with room over it.
DEFAULT_BUFFER_M = 400.0

# POI categories, as in build_hcmc_pipeline_v2.py.
POI_KEYS = ("amenity", "shop", "office", "tourism", "leisure")

LAYERS = ("buildings", "pois", "roads")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===========================================================================
# Overpass
# ===========================================================================


def overpass(query: str, attempts: int = 5, timeout: int = 300) -> dict:
    """POST a query, retrying with backoff.  Raises rather than returning empty.

    The distinction matters: an empty result is a legitimate answer for a tile
    with no buildings on it, and a failed request is not.  Conflating the two
    is what makes a partial download invisible.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                OVERPASS_URL, data={"data": query},
                headers=OSM_HTTP_HEADERS, timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network, HTTP status, or JSON
            last_error = exc
            wait = 15 * (attempt + 1)
            log.warning(f"    attempt {attempt + 1}/{attempts} failed ({exc}); "
                        f"retrying in {wait}s")
            if attempt < attempts - 1:
                time.sleep(wait)
    raise RuntimeError(f"Overpass failed after {attempts} attempts: {last_error}")


def _bbox_clause(bounds: tuple[float, float, float, float]) -> str:
    """Overpass bbox from ``(west, south, east, north)`` degrees."""
    west, south, east, north = bounds
    return f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"


def fetch_buildings(bounds) -> gpd.GeoDataFrame:
    bbox = _bbox_clause(bounds)
    data = overpass(
        f'[out:json][timeout:300];\n'
        f'(way["building"]{bbox}; relation["building"]{bbox};);\n'
        f'out geom;'
    )
    records = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        for polygon in _element_polygons(element):
            records.append({
                "geometry": polygon,
                "building": tags.get("building", "yes"),
                "levels": _to_float(tags.get("building:levels")),
                "height": _to_float(tags.get("height")),
            })
    return _frame(records, "Polygon")


def fetch_pois(bounds) -> gpd.GeoDataFrame:
    bbox = _bbox_clause(bounds)
    parts = []
    for key in POI_KEYS:
        parts.append(f'node["{key}"]{bbox};')
        parts.append(f'way["{key}"]{bbox};')
    data = overpass(
        f'[out:json][timeout:300];\n({"".join(parts)});\nout center;'
    )
    records = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        if element["type"] == "node":
            lon, lat = element.get("lon"), element.get("lat")
        else:
            centre = element.get("center") or {}
            lon, lat = centre.get("lon"), centre.get("lat")
        if lon is None or lat is None:
            continue

        # A POI can carry several of the keys; the first present decides its
        # category, in the fixed POI_KEYS order, so the assignment is
        # deterministic rather than dictionary-order dependent.
        category_key = next((k for k in POI_KEYS if k in tags), None)
        if category_key is None:
            continue
        records.append({
            "geometry": Point(lon, lat),
            "category_key": category_key,
            "category_value": tags.get(category_key),
            "name": tags.get("name"),
        })
    return _frame(records, "Point")


def fetch_roads(bounds) -> gpd.GeoDataFrame:
    bbox = _bbox_clause(bounds)
    data = overpass(
        f'[out:json][timeout:300];\nway["highway"]{bbox};\nout geom;'
    )
    records = []
    for element in data.get("elements", []):
        nodes = element.get("geometry") or []
        if len(nodes) < 2:
            continue
        tags = element.get("tags", {})
        records.append({
            "geometry": LineString([(n["lon"], n["lat"]) for n in nodes]),
            "highway": tags.get("highway"),
            "name": tags.get("name"),
        })
    return _frame(records, "LineString")


def _element_polygons(element: dict):
    """Polygons of a way or a multipolygon relation, skipping broken geometry."""
    if element["type"] == "way":
        nodes = element.get("geometry") or []
        if len(nodes) < 3:
            return
        coords = [(n["lon"], n["lat"]) for n in nodes]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon = Polygon(coords)
        if polygon.is_valid and polygon.area > 1e-12:
            yield polygon
        return

    for member in element.get("members", []):
        if member.get("role") != "outer":
            continue
        nodes = member.get("geometry") or []
        if len(nodes) < 3:
            continue
        coords = [(n["lon"], n["lat"]) for n in nodes]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon = Polygon(coords)
        if polygon.is_valid and polygon.area > 1e-12:
            yield polygon


def _to_float(value):
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _frame(records, geom_type: str) -> gpd.GeoDataFrame:
    if not records:
        # An empty layer is a real answer.  Keep the schema so downstream code
        # does not branch on emptiness.
        return gpd.GeoDataFrame(
            {"geometry": gpd.GeoSeries([], dtype="geometry")}, crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ===========================================================================
# Per tile
# ===========================================================================


def buffered_bounds(tile_row, buffer_m: float) -> tuple[float, float, float, float]:
    """Tile bounds grown by *buffer_m* on the ground, back in degrees.

    Buffering in UTM rather than in degrees, so the margin is the same distance
    on every side.
    """
    tile = gpd.GeoDataFrame(
        geometry=[tile_row.geometry], crs="EPSG:4326"
    ).to_crs(epsg=UTM_EPSG)
    grown = tile.buffer(buffer_m).to_crs(epsg=4326)
    west, south, east, north = grown.total_bounds
    return west, south, east, north


def fetch_tile(tile_row, buffer_m: float, layers, overwrite: bool) -> dict:
    cell_id = tile_row.cell_id
    out_dir = TILES_DIR / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fetchers = {"buildings": fetch_buildings, "pois": fetch_pois, "roads": fetch_roads}
    counts = {}
    bounds = buffered_bounds(tile_row, buffer_m)

    for layer in layers:
        path = out_dir / f"{layer}.gpkg"
        if path.exists() and not overwrite:
            counts[layer] = len(gpd.read_file(path))
            log.info(f"  {layer}: {counts[layer]} cached")
            continue

        frame = fetchers[layer](bounds)
        counts[layer] = len(frame)
        if len(frame):
            frame.to_file(path, driver="GPKG")
        else:
            # GeoPackage cannot hold an empty layer with no schema; record the
            # emptiness so a later run does not mistake it for a missing fetch.
            (out_dir / f"{layer}.empty").write_text("0\n", encoding="utf-8")
        log.info(f"  {layer}: {counts[layer]} features")

    (out_dir / "fetch.json").write_text(
        pd.Series({
            "cell_id": cell_id,
            "buffer_m": buffer_m,
            "bounds": list(bounds),
            "fetched_utc": pd.Timestamp.utcnow().isoformat(),
            **{f"n_{k}": v for k, v in counts.items()},
        }).to_json(indent=2),
        encoding="utf-8",
    )
    return counts



# ===========================================================================
# Study-area fetch, in chunks
# ===========================================================================


def fetch_citywide(layer: str, boundary_path, chunks: int, out_path,
                   pause_s: float = 3.0, overwrite: bool = False):
    """Fetch one layer over the whole study area, in a grid of chunks.

    The v2 pipeline split the municipality into four and asked Overpass for
    each in one query, which times out in the dense quadrants.  A finer split
    answers reliably: each chunk is small enough to return in seconds, and a
    chunk that will not come back after its retries raises instead of leaving
    a hole.

    Chunks are fetched serially with a pause between them.  Overpass offers
    two slots and rate-limits hard past a few hundred queries; going slowly is
    the difference between finishing and being blocked for hours.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        existing = gpd.read_file(out_path)
        log.info(f"{out_path} exists with {len(existing):,} features; "
                 f"pass --overwrite to refetch")
        return existing

    boundary = gpd.read_file(boundary_path).to_crs(epsg=4326)
    west, south, east, north = boundary.total_bounds
    geometry = boundary.union_all()

    fetchers = {"buildings": fetch_buildings, "pois": fetch_pois, "roads": fetch_roads}
    lat_step = (north - south) / chunks
    lon_step = (east - west) / chunks

    frames, skipped = [], []
    total = chunks * chunks
    done = 0
    for i in range(chunks):
        for j in range(chunks):
            done += 1
            box_bounds = (west + j * lon_step, south + i * lat_step,
                          west + (j + 1) * lon_step, south + (i + 1) * lat_step)
            # Skip chunks that do not touch the municipality at all.
            if not box(*box_bounds).intersects(geometry):
                skipped.append(box_bounds)
                continue

            log.info(f"  chunk {done}/{total} "
                     f"({box_bounds[1]:.3f},{box_bounds[0]:.3f})")
            frame = fetchers[layer](box_bounds)
            log.info(f"    {len(frame):,} features")
            if len(frame):
                frames.append(frame)
            time.sleep(pause_s)

    if not frames:
        raise RuntimeError(f"no {layer} anywhere in the study area")

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    # Chunks share no edges for point layers, but ways can be returned by two
    # neighbouring boxes, so drop exact geometry duplicates.
    before = len(merged)
    merged = merged.drop_duplicates(subset="geometry")
    log.info(f"{layer}: {before:,} fetched, {len(merged):,} after de-duplication, "
             f"{len(skipped)} chunks outside the boundary")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(out_path, driver="GPKG")
    log.info(f"Wrote {out_path}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=str,
                        default=str(RAW_DIR / "hcmc_tiles_z15.gpkg"))
    parser.add_argument("--cell_ids", type=str, nargs="+", default=None)
    parser.add_argument("--sample", type=int, default=None,
                        help="Fetch this many tiles, sampled across the "
                             "study area by coverage rank rather than at "
                             "random, so the sample spans dense and sparse.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_m", type=float, default=DEFAULT_BUFFER_M)
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--citywide", choices=LAYERS, default=None,
                        help="Fetch this layer over the whole study area in a "
                             "grid of chunks, instead of per tile.")
    parser.add_argument("--chunks", type=int, default=8,
                        help="Grid resolution for --citywide; 8 means 64 boxes.")
    parser.add_argument("--pause_s", type=float, default=3.0,
                        help="Pause between chunk queries. Overpass rate-limits "
                             "hard; going slowly is what finishes.")
    args = parser.parse_args()

    if args.citywide:
        fetch_citywide(
            args.citywide,
            RAW_DIR / "hcmc_boundary_full.gpkg",
            args.chunks,
            RAW_DIR / f"hcmc_{args.citywide}_all.gpkg",
            pause_s=args.pause_s,
            overwrite=args.overwrite,
        )
        return

    tiles = gpd.read_file(args.index)
    log.info(f"Tile index: {len(tiles):,} tiles")

    if args.cell_ids:
        selected = tiles[tiles["cell_id"].isin(args.cell_ids)]
        missing = set(args.cell_ids) - set(selected["cell_id"])
        if missing:
            raise SystemExit(f"not in the index: {sorted(missing)}")
    elif args.sample:
        full = tiles[tiles["coverage"] > 0.99]
        if len(full) < args.sample:
            full = tiles
        selected = full.sample(args.sample, random_state=args.seed)
    else:
        raise SystemExit("pass --cell_ids or --sample")

    log.info(f"Fetching {len(selected)} tiles with a {args.buffer_m:.0f} m buffer")
    totals = {layer: 0 for layer in args.layers}
    for _, tile_row in selected.iterrows():
        log.info(f"{tile_row.cell_id} "
                 f"({tile_row.centre_lat:.4f}, {tile_row.centre_lon:.4f})")
        counts = fetch_tile(tile_row, args.buffer_m, args.layers, args.overwrite)
        for layer, count in counts.items():
            totals[layer] += count

    log.info("=" * 60)
    for layer, total in totals.items():
        log.info(f"{layer}: {total:,} features across {len(selected)} tiles")
    log.info(f"Written under {TILES_DIR}")


if __name__ == "__main__":
    main()
