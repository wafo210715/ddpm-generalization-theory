#!/usr/bin/env python3
"""Rasterize zoom-15 tiles into multi-channel cell tensors (B3-B5, B7, B8).

Each channel is computed **at its own natural scale** and only then resampled
onto the common 1024 px grid, which is B7:

    Each channel is computed at its natural scale first.  Alignment happens
    only at the end.  Liu's image-to-image work fused independently rendered
    layers the same way.

In practice that means three different things per channel:

  * ``building_binary`` is rasterized directly at the target grid.  Its natural
    scale *is* the polygon boundary, and at a 1.17 m unit a 3 m alley is two to
    three pixels wide --- the whole reason B2 chose that unit.
  * ``building_density`` and ``road_density`` are computed on a coarse grid
    with a disc of the indicator's own radius, then resampled up.  A field
    defined by a 100 m disc has nothing to say at 1.17 m that resampling does
    not already say.
  * ``accessibility_gravity`` is solved on the road graph, at graph nodes,
    with no grid at all until the last step.

Channel selection is provisional.  The rows here implement candidates from
``discussion/indicator_specification.md``; A1 has not been decided, and neither
have the radii, so every scale is a command-line argument and the defaults are
placeholders that exist to make the machinery runnable, not recommendations.

Output is a ``CellTensorStore`` in data space, ready for
``store.fit_pipeline()`` once B10's marginals have chosen the transforms.

Usage:
    python rasterize_tiles_v4.py --cell_ids 15_26097_15392 --img_size 1024
    python rasterize_tiles_v4.py --all_fetched --img_size 512
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "hcmc" / "raw"
TILES_DIR = RAW_DIR / "tiles"
PROC_DIR = ROOT / "data" / "hcmc" / "processed"

# Make the ddpm-win-master package importable for CellTensorStore.
_DDPM_ROOT = ROOT.parent / "ddpm-win-master"
if str(_DDPM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DDPM_ROOT))

UTM_EPSG = 32648

# --- provisional scales -----------------------------------------------------
# None of these is decided.  See discussion/indicator_specification.md, open
# question 3: the pilot's evidence is that these are the parameters that decide
# whether a channel carries signal at all (150 m flattened built form, 50 m
# emptied diversity), so they are arguments rather than constants.
DEFAULT_DENSITY_RADIUS_M = 100.0
DEFAULT_DECAY_RADIUS_M = 300.0
DEFAULT_RING_RADIUS_M = 300.0

CHANNEL_NAMES = [
    "building_binary",
    "building_density",
    "accessibility_gravity",
    "poi_count",
    "poi_entropy",
    "road_density",
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===========================================================================
# Grid helpers
# ===========================================================================


def tile_utm_bounds(tile_row) -> tuple[float, float, float, float]:
    """``(xmin, ymin, xmax, ymax)`` of a tile in UTM metres."""
    frame = gpd.GeoDataFrame(geometry=[tile_row.geometry], crs="EPSG:4326")
    return tuple(frame.to_crs(epsg=UTM_EPSG).total_bounds)


def unit_centres(bounds, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre coordinates of an ``n x n`` grid over *bounds*, in UTM metres.

    Returns ``(xs, ys)`` of length n each.  ``ys`` runs north to south so that
    row 0 is the top of the raster, which is what an image viewer and a
    north-up map both expect.
    """
    xmin, ymin, xmax, ymax = bounds
    xs = np.linspace(xmin, xmax, n, endpoint=False) + (xmax - xmin) / (2 * n)
    ys = np.linspace(ymax, ymin, n, endpoint=False) - (ymax - ymin) / (2 * n)
    return xs, ys


def resample_to(grid: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resample a coarse grid up to ``size x size``.

    B7's alignment step.  Cubic rather than bilinear or nearest: the fields
    resampled here are smooth by construction --- a disc average, a ring count,
    a decay field --- and a low-order interpolation from a coarse solve grid
    leaves visible facets that the descriptor would then read as spatial
    structure that is not in the data.  Blocking was plainly visible in the
    first 1024 px run at order=1.
    """
    if grid.shape == (size, size):
        return grid
    from scipy.ndimage import zoom

    factor = size / grid.shape[0]
    out = zoom(grid, factor, order=3, grid_mode=True, mode="nearest")[:size, :size]

    # Cubic interpolation overshoots at sharp edges, which put entropies below
    # zero and counts below zero in the first run.  The resampled field cannot
    # legitimately leave the range that was actually solved, so clamp to it:
    # this removes the ringing without touching anything interior.
    return np.clip(out, grid.min(), grid.max())


def coarse_n(bounds, radius_m: float, img_size: int, per_radius: int = 8) -> int:
    """Grid size at which a radius-``r`` field is worth computing.

    ``per_radius`` samples across one radius: fine enough that resampling adds
    nothing, coarse enough that a 300 m field is not solved a million times per
    tile.  Never finer than the target grid.

    Four samples per radius is enough in principle --- the field is bandlimited
    at its own radius --- but it left visible facets after interpolation, and a
    facet is indistinguishable from real structure to everything downstream.
    Eight costs four times the solve on a grid that is still tiny beside the
    target, and the facets go.
    """
    width = bounds[2] - bounds[0]
    n = int(np.ceil(per_radius * width / radius_m))
    return int(min(max(n, 8), img_size))


# ===========================================================================
# Channels
# ===========================================================================


def rasterize_building_binary(buildings, bounds, img_size: int) -> np.ndarray:
    """1 where a unit's centre falls inside a footprint.

    Computed straight at the target grid: the natural scale of a footprint is
    the footprint, and this is the channel that sets the 1.17 m floor.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    if buildings is None or len(buildings) == 0:
        return np.zeros((img_size, img_size), dtype=np.float64)

    xmin, ymin, xmax, ymax = bounds
    transform = from_bounds(xmin, ymin, xmax, ymax, img_size, img_size)
    burned = rasterize(
        ((geom, 1) for geom in buildings.geometry if geom is not None),
        out_shape=(img_size, img_size),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return burned.astype(np.float64)


def disc_density_from_mask(
    mask: np.ndarray, unit_m: float, radius_m: float
) -> np.ndarray:
    """Share of a disc of *radius_m* around each unit that the mask covers.

    Computed by convolution on the fine mask, which is exact up to the unit
    size and far cheaper than intersecting polygons with a disc per unit.
    """
    from scipy.signal import fftconvolve

    r_px = max(int(round(radius_m / unit_m)), 1)
    yy, xx = np.mgrid[-r_px : r_px + 1, -r_px : r_px + 1]
    kernel = ((yy**2 + xx**2) <= r_px**2).astype(np.float64)
    kernel /= kernel.sum()
    return np.clip(fftconvolve(mask, kernel, mode="same"), 0.0, 1.0)


def rasterize_road_density(roads, bounds, img_size: int, radius_m: float) -> np.ndarray:
    """Street length per unit area within a disc, in m per m^2."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    if roads is None or len(roads) == 0:
        return np.zeros((img_size, img_size), dtype=np.float64)

    xmin, ymin, xmax, ymax = bounds
    unit_m = (xmax - xmin) / img_size
    transform = from_bounds(xmin, ymin, xmax, ymax, img_size, img_size)
    burned = rasterize(
        ((geom, 1) for geom in roads.geometry if geom is not None),
        out_shape=(img_size, img_size),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(np.float64)

    # Each burned unit stands for roughly one unit_m of centreline over an area
    # of unit_m^2, so a covered fraction f of the disc means f * disc_area /
    # unit_m metres of street in disc_area of ground: a length density of
    # f / unit_m, in metres per square metre.
    covered = disc_density_from_mask(burned, unit_m, radius_m)
    return covered / unit_m


def gravity_on_network(
    roads, pois, bounds, img_size: int, decay_radius_m: float
) -> np.ndarray:
    """Gravity accessibility solved on the road graph (B4).

        Accessibility: gravity model on the road network, decay of a few
        hundred metres.  Exponential decay yields a smooth field.

    POIs are snapped to their nearest graph node; a multi-source Dijkstra from
    every POI-bearing node, cut off at three decay lengths, gives each node its
    score; each unit takes the score of its nearest node.

    Falls back to straight-line decay when there is no usable road graph, and
    says so, because a silently Euclidean "network gravity" would be a
    different indicator wearing the same name.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    xs, ys = unit_centres(bounds, img_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    if pois is None or len(pois) == 0:
        return np.zeros((img_size, img_size), dtype=np.float64)

    poi_xy = np.column_stack([pois.geometry.x.to_numpy(), pois.geometry.y.to_numpy()])

    if roads is None or len(roads) == 0:
        log.warning("    no roads: falling back to straight-line decay")
        return _euclidean_gravity(grid_x, grid_y, poi_xy, decay_radius_m)

    graph = nx.Graph()
    for line in roads.geometry:
        if line is None or line.is_empty:
            continue
        coords = list(line.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if length > 0:
                graph.add_edge(a, b, weight=length)

    if graph.number_of_nodes() < 2:
        log.warning("    road graph too small: falling back to straight-line decay")
        return _euclidean_gravity(grid_x, grid_y, poi_xy, decay_radius_m)

    nodes = np.array(graph.nodes())
    node_tree = cKDTree(nodes)

    _, poi_node_idx = node_tree.query(poi_xy)
    sources = [tuple(nodes[i]) for i in np.unique(poi_node_idx)]
    weight_per_node = pd.Series(poi_node_idx).value_counts()

    cutoff = 3.0 * decay_radius_m
    scores = np.zeros(len(nodes))
    node_index = {tuple(node): i for i, node in enumerate(nodes)}
    for source_idx in np.unique(poi_node_idx):
        source = tuple(nodes[source_idx])
        weight = float(weight_per_node[source_idx])
        lengths = nx.single_source_dijkstra_path_length(
            graph, source, cutoff=cutoff, weight="weight"
        )
        for node, distance in lengths.items():
            scores[node_index[node]] += weight * np.exp(-distance / decay_radius_m)

    _, nearest_node = node_tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    return scores[nearest_node].reshape(img_size, img_size)


def _euclidean_gravity(grid_x, grid_y, poi_xy, decay_radius_m: float) -> np.ndarray:
    field = np.zeros(grid_x.shape, dtype=np.float64)
    for x, y in poi_xy:
        distance = np.hypot(grid_x - x, grid_y - y)
        field += np.exp(-distance / decay_radius_m)
    return field


def poi_ring_stats(
    pois, bounds, n: int, ring_radius_m: float, category_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """POI count and Shannon entropy over a ring, on an ``n x n`` grid.

    Returns ``(count, entropy)``.  Entropy needs at least two POIs in at least
    two categories; below that it is zero, and B6's coverage report is about
    exactly how often that happens.
    """
    from scipy.spatial import cKDTree

    count = np.zeros((n, n), dtype=np.float64)
    entropy = np.zeros((n, n), dtype=np.float64)
    if pois is None or len(pois) == 0:
        return count, entropy

    poi_xy = np.column_stack([pois.geometry.x.to_numpy(), pois.geometry.y.to_numpy()])
    categories = pd.factorize(pois[category_column].fillna("unknown"))[0]

    xs, ys = unit_centres(bounds, n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    tree = cKDTree(poi_xy)
    neighbours = tree.query_ball_point(query, r=ring_radius_m)

    flat_count = np.zeros(len(query))
    flat_entropy = np.zeros(len(query))
    for i, members in enumerate(neighbours):
        flat_count[i] = len(members)
        if len(members) < 2:
            continue
        labels = categories[members]
        _, counts = np.unique(labels, return_counts=True)
        if len(counts) < 2:
            continue
        p = counts / counts.sum()
        flat_entropy[i] = float(-(p * np.log(p)).sum())

    return flat_count.reshape(n, n), flat_entropy.reshape(n, n)


# ===========================================================================
# Per tile
# ===========================================================================


# City-wide layers, when one has been downloaded, clipped per tile.  Loaded
# once and reused: reading a 26 MB GeoPackage per tile would dominate the run.
_CITYWIDE_CACHE: dict[str, gpd.GeoDataFrame] = {}


def load_citywide(path: str) -> gpd.GeoDataFrame:
    if path not in _CITYWIDE_CACHE:
        log.info(f"  loading city-wide layer {path}")
        _CITYWIDE_CACHE[path] = gpd.read_file(path).to_crs(epsg=UTM_EPSG)
    return _CITYWIDE_CACHE[path]


def load_layer(
    cell_id: str,
    layer: str,
    bounds: tuple[float, float, float, float] | None = None,
    citywide_path: str | None = None,
    buffer_m: float = 400.0,
    missing: str = "error",
) -> gpd.GeoDataFrame | None:
    """A layer for one tile in UTM, or None when there is genuinely nothing.

    Per-tile fetches take precedence; a city-wide layer is clipped to the tile
    plus *buffer_m* when no per-tile fetch exists.  The buffer is the same
    argument as in fetch_tile_osm: an indicator measured over a ring reads
    beyond the tile edge, and a layer clipped flush would leave a rim of units
    computed from truncated neighbourhoods.
    """
    directory = TILES_DIR / cell_id
    path = directory / f"{layer}.gpkg"
    if path.exists():
        return gpd.read_file(path).to_crs(epsg=UTM_EPSG)
    if (directory / f"{layer}.empty").exists():
        return None

    if citywide_path:
        frame = load_citywide(citywide_path)
        xmin, ymin, xmax, ymax = bounds
        window = frame.cx[
            xmin - buffer_m : xmax + buffer_m, ymin - buffer_m : ymax + buffer_m
        ]
        return window if len(window) else None

    if missing == "empty":
        # Recorded in the manifest by the caller.  A store built this way is
        # partial by construction, and the channels that depend on the missing
        # layer are all-zero -- which the marginals report will flag as an atom
        # rather than let pass as a sparse indicator.
        return None

    raise FileNotFoundError(
        f"{layer} for {cell_id} has not been fetched; run fetch_tile_osm.py, "
        f"pass --citywide_{layer}, or --missing_layers empty to accept "
        f"all-zero channels"
    )


def rasterize_tile(tile_row, img_size: int, args) -> np.ndarray:
    """One tile's ``(C, H, W)`` tensor, in data space."""
    cell_id = tile_row.cell_id
    bounds = tile_utm_bounds(tile_row)
    unit_m = (bounds[2] - bounds[0]) / img_size

    buildings = load_layer(cell_id, "buildings", bounds,
                           args.citywide_buildings, args.buffer_m,
                           args.missing_layers)
    pois = load_layer(cell_id, "pois", bounds, args.citywide_pois,
                      args.buffer_m, args.missing_layers)
    roads = load_layer(cell_id, "roads", bounds, args.citywide_roads,
                       args.buffer_m, args.missing_layers)

    # 1. building_binary --- natural scale is the polygon, so straight at target.
    binary = rasterize_building_binary(buildings, bounds, img_size)

    # 2. building_density --- a disc of its own radius over the fine mask.
    density = disc_density_from_mask(binary, unit_m, args.density_radius_m)

    # 3. accessibility_gravity --- solved on the graph, at a coarse grid, then up.
    n_access = coarse_n(bounds, args.decay_radius_m, img_size)
    access_coarse = gravity_on_network(
        roads, pois, bounds, n_access, args.decay_radius_m
    )
    access = resample_to(access_coarse, img_size)

    # 4, 5. poi_count and poi_entropy --- ring of its own radius, then up.
    n_ring = coarse_n(bounds, args.ring_radius_m, img_size)
    count_coarse, entropy_coarse = poi_ring_stats(
        pois, bounds, n_ring, args.ring_radius_m, args.poi_category_column
    )
    poi_count = resample_to(count_coarse, img_size)
    poi_entropy = resample_to(entropy_coarse, img_size)

    # 6. road_density --- a disc of the density radius over a burned centreline.
    road_density = rasterize_road_density(
        roads, bounds, img_size, args.density_radius_m
    )

    log.info(
        f"    grids: binary at {img_size}, density at {img_size}, "
        f"access at {n_access}, rings at {n_ring}"
    )
    return np.stack(
        [binary, density, access, poi_count, poi_entropy, road_density]
    )



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
    parser.add_argument("--cell_ids", type=str, nargs="+", default=None)
    parser.add_argument("--all_fetched", action="store_true",
                        help="Every tile with a fetched layer set on disk.")
    parser.add_argument("--img_size", type=int, default=1024,
                        help="B2 puts the experiments at 1024; 512 calibrates "
                             "and 128/256 debug.")
    parser.add_argument("--density_radius_m", type=float,
                        default=DEFAULT_DENSITY_RADIUS_M)
    parser.add_argument("--decay_radius_m", type=float,
                        default=DEFAULT_DECAY_RADIUS_M)
    parser.add_argument("--ring_radius_m", type=float, default=DEFAULT_RING_RADIUS_M)
    parser.add_argument("--poi_category_column", type=str, default="category_key",
                        help="category_key is the coarse scheme (amenity, shop, "
                             "office, tourism, leisure); category_value is the "
                             "raw tag value. Which one is open question 2.")
    parser.add_argument("--citywide_buildings", type=str, default=None,
                        help="Clip this layer per tile when no per-tile fetch "
                             "exists, e.g. data/hcmc/raw/hcmc_buildings.gpkg.")
    parser.add_argument("--citywide_pois", type=str, default=None)
    parser.add_argument("--citywide_roads", type=str, default=None)
    parser.add_argument("--buffer_m", type=float, default=400.0,
                        help="Margin kept around a tile when clipping a "
                             "city-wide layer, so ring-based indicators on the "
                             "tile edge still see their whole neighbourhood.")
    parser.add_argument("--missing_layers", choices=["error", "empty"],
                        default="error",
                        help="What to do when a layer was never fetched. "
                             "'empty' produces all-zero channels and records "
                             "the gap in the manifest; the store is then "
                             "partial by construction and says so.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32"])
    args = parser.parse_args()

    from datasets.cell_tensors import CellTensorStore

    tiles = gpd.read_file(args.index)
    if args.cell_ids:
        selected = tiles[tiles["cell_id"].isin(args.cell_ids)]
    elif args.all_fetched:
        fetched = fetched_cell_ids()
        selected = tiles[tiles["cell_id"].isin(fetched)]
    else:
        raise SystemExit("pass --cell_ids or --all_fetched")
    if selected.empty:
        raise SystemExit("no tiles selected")

    out_root = args.out or str(PROC_DIR / f"cells_z15_{args.img_size}")
    store = CellTensorStore.create(
        out_root,
        channel_names=CHANNEL_NAMES,
        img_size=args.img_size,
        dtype=args.dtype,
        metadata={
            "zoom": 15,
            "density_radius_m": args.density_radius_m,
            "decay_radius_m": args.decay_radius_m,
            "ring_radius_m": args.ring_radius_m,
            "poi_category_column": args.poi_category_column,
            "buffer_m": args.buffer_m,
            "missing_layers": args.missing_layers,
            "unit_m": None,  # filled below, once a tile has fixed it
            "provisional": (
                "Channel selection and every radius are candidates from "
                "discussion/indicator_specification.md; A1 is not decided."
            ),
        },
        exist_ok=True,
    )

    log.info(f"Rasterizing {len(selected)} tiles at {args.img_size} px into {out_root}")
    for _, tile_row in selected.iterrows():
        log.info(f"  {tile_row.cell_id}")
        tensor = rasterize_tile(tile_row, args.img_size, args)
        store.write_cell(tile_row.cell_id, tensor)

    unit_m = float(selected.iloc[0]["width_m"]) / args.img_size
    store.manifest["metadata"]["unit_m"] = unit_m
    store._save_manifest()

    log.info("=" * 60)
    log.info(f"Wrote {len(store)} cells, unit {unit_m:.3f} m")
    log.info(f"Store: {out_root}")


if __name__ == "__main__":
    main()
