#!/usr/bin/env python3
"""
HCMC Built Environment Data Pipeline v2
========================================

Redesigned pipeline addressing critical GAPs from v1:
  GAP-3: Cell naming is {row:03d}_{col:03d}.png (not sequential {label:04d})
  GAP-4: Building channel is CONTINUOUS density [0,1] (not binary 0/255)
  GAP-6: Per-channel GLOBAL 95th percentile normalization (not per-cell max)

Channels (Jacobs' conditions for urban vitality):
  B = Building density (continuous footprint coverage via buffer ring intersection)
  G = POI diversity (Shannon entropy of categories within buffer)
  R = Accessibility (gravity-model via pandana network analysis)

Architecture:
  - 1km grid cells divided into 20x20 sub-grid (50m calculation units)
  - 150m buffer rings around each sub-cell center for density/entropy
  - Per-channel global 95th percentile normalization
  - Upsample 20x20 -> 128x128 via bilinear interpolation (scipy.ndimage.zoom)
  - Save as RGB PNGs: R=accessibility, G=diversity, B=density
"""

import os
import sys
import time
import logging
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box, Point, Polygon
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import zoom as ndimage_zoom
from PIL import Image
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "hcmc"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CELLS_DIR = PROC_DIR / "cells"
CHANNELS_DIR = PROC_DIR / "channels"
REPORTS_DIR = PROC_DIR / "reports"

UTM_EPSG = 32648
CALC_UNIT_M = 50
BUFFER_M = 150
SUBDIV = 20
TARGET_PX = 128
NORM_PERCENTILE = 95

ACCESSIBILITY_BETA = 0.3
ACCESSIBILITY_CUTOFF_M = 3000
SNAP_MAX_DIST_M = 500

HCMC_BBOX = (106.357, 10.140, 107.028, 11.160)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===========================================================================
# Step 1: Load fishnet grid
# ===========================================================================
def load_grid():
    """Load fishnet grid from hcmc_grid_1km.gpkg. Verify 8436 cells."""
    grid_path = RAW_DIR / "hcmc_grid_1km.gpkg"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid not found: {grid_path}. Run build_boundary_grid.py first."
        )
    grid = gpd.read_file(grid_path)
    assert len(grid) == 8436, f"Expected 8436 cells, got {len(grid)}"
    assert "cell_id" in grid.columns
    sample = grid["cell_id"].iloc[0]
    assert "_" in sample, f"cell_id format unexpected: {sample}"
    log.info(f"Loaded grid: {len(grid)} cells, CRS={grid.crs}")
    return grid


# ===========================================================================
# Step 2: Fetch OSM data (reuse v1 Overpass patterns)
# ===========================================================================
def fetch_buildings(grid_gdf):
    """Fetch building polygons from Overpass API for the full grid extent.
    Uses v1 tile-splitting pattern. Cache to hcmc_buildings.gpkg."""
    cache_path = RAW_DIR / "hcmc_buildings.gpkg"
    if cache_path.exists():
        buildings = gpd.read_file(cache_path)
        log.info(f"Loaded {len(buildings)} cached buildings")
        return buildings

    log.info("Fetching buildings from Overpass API for full grid extent...")
    grid_wgs84 = grid_gdf.to_crs(epsg=4326)
    bounds = grid_wgs84.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

    overpass_url = "https://overpass-api.de/api/interpreter"
    lat_step = (bbox[2] - bbox[0]) / 2
    lon_step = (bbox[3] - bbox[1]) / 2
    tiles = []
    for i in range(2):
        for j in range(2):
            tile = (
                bbox[0] + i * lat_step, bbox[1] + j * lon_step,
                bbox[0] + (i + 1) * lat_step, bbox[1] + (j + 1) * lon_step,
            )
            tiles.append(tile)

    all_records = []
    for t_idx, tile in enumerate(tiles):
        log.info(f"  Querying tile {t_idx + 1}/{len(tiles)}")
        query = (
            f'[out:json][timeout:300][maxsize:536870912];\n'
            f'(way["building"]({tile[0]:.4f},{tile[1]:.4f},{tile[2]:.4f},{tile[3]:.4f}););\n'
            f'out geom;'
        )
        data = {"elements": []}
        for attempt in range(3):
            try:
                resp = requests.post(overpass_url, data={"data": query}, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                log.warning(f"    Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(30 * (attempt + 1))
                else:
                    log.warning(f"    Skipping tile {t_idx + 1}")

        for elem in data.get("elements", []):
            if elem["type"] != "way":
                continue
            geom_nodes = elem.get("geometry", [])
            if len(geom_nodes) < 3:
                continue
            coords = [(n["lon"], n["lat"]) for n in geom_nodes]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                poly = Polygon(coords)
                if not poly.is_valid or poly.area < 1e-10:
                    continue
                all_records.append({
                    "geometry": poly,
                    "building": elem.get("tags", {}).get("building", "yes"),
                })
            except Exception:
                continue
        log.info(f"    Buildings so far: {len(all_records)}")

    if not all_records:
        raise RuntimeError("No buildings found from Overpass API")

    buildings = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
    buildings.to_file(cache_path, driver="GPKG")
    log.info(f"Total buildings: {len(buildings)}")
    return buildings


def fetch_pois(grid_gdf):
    """Fetch POI nodes/ways from Overpass API.
    Categories: amenity, shop, office, tourism, leisure. Cache to hcmc_pois_all.gpkg."""
    cache_path = RAW_DIR / "hcmc_pois_all.gpkg"
    if cache_path.exists():
        pois = gpd.read_file(cache_path)
        log.info(f"Loaded {len(pois)} cached POIs")
        return pois

    log.info("Fetching POIs from Overpass API...")
    grid_wgs84 = grid_gdf.to_crs(epsg=4326)
    bounds = grid_wgs84.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = (
        f'[out:json][timeout:300][maxsize:536870912];\n'
        f'(\n'
        f'  node["amenity"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  node["shop"~"supermarket|convenience|mall|clothes|electronics|bakery"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  node["office"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  node["tourism"~"hotel|hostel|museum|attraction|viewpoint|artwork"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  node["leisure"~"park|garden|playground|sports_centre|swimming_pool|fitness_centre"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  way["amenity"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  way["shop"~"supermarket|convenience|mall|clothes|electronics|bakery"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  way["office"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  way["tourism"~"hotel|hostel|museum|attraction|viewpoint|artwork"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f'  way["leisure"~"park|garden|playground|sports_centre|swimming_pool|fitness_centre"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});\n'
        f');\n'
        f'out center;'
    )

    log.info("  Sending Overpass query (nodes + ways)...")
    data = {}
    for attempt in range(3):
        try:
            resp = requests.post(overpass_url, data={"data": query}, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            log.warning(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(30 * (attempt + 1))
            else:
                raise

    records = []
    for elem in data["elements"]:
        tags = elem.get("tags", {})
        if elem["type"] == "node":
            lat, lon = elem.get("lat"), elem.get("lon")
        else:
            center = elem.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        category, poi_type = "unknown", "unknown"
        if "amenity" in tags:
            category, poi_type = tags["amenity"], "amenity"
        elif "shop" in tags:
            category, poi_type = tags["shop"], "shop"
        elif "office" in tags:
            category, poi_type = tags["office"], "office"
        elif "tourism" in tags:
            category, poi_type = tags["tourism"], "tourism"
        elif "leisure" in tags:
            category, poi_type = tags["leisure"], "leisure"
        records.append({
            "geometry": Point(lon, lat),
            "poi_category": category,
            "poi_type": poi_type,
            "osm_type": elem["type"],
        })

    if not records:
        raise RuntimeError("No POIs found from Overpass API")

    pois = gpd.GeoDataFrame(records, crs="EPSG:4326")
    pois.to_file(cache_path, driver="GPKG")
    log.info(f"Total POIs: {len(pois)}")
    return pois


def fetch_network(grid_gdf):
    """Fetch pedestrian street network using osmnx. Cache to graphml."""
    import osmnx as ox

    cache_path = RAW_DIR / "hcmc_pedestrian_network.graphml"
    if cache_path.exists():
        log.info("Loading cached street network")
        return ox.load_graphml(cache_path)

    log.info("Downloading pedestrian street network from OSM...")
    grid_wgs84 = grid_gdf.to_crs(epsg=4326)
    bounds = grid_wgs84.total_bounds
    polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
    G = ox.graph_from_polygon(polygon, network_type="walk", simplify=True)
    G = ox.project_graph(G, to_crs=f"EPSG:{UTM_EPSG}")
    ox.save_graphml(G, cache_path)
    log.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ===========================================================================
# Step 3: Compute channels (50m calculation units with buffer rings)
# ===========================================================================
def compute_building_density(buildings_utm, grid_utm):
    """CONTINUOUS building density per sub-cell (GAP-4 fix).
    For each sub-cell center, create 150m buffer, compute building footprint
    coverage as fraction of buffer area. Returns dict {cell_id: 20x20 array}."""
    log.info("Computing B channel (building density) -- CONTINUOUS, not binary...")
    t0 = time.time()

    cache_path = PROC_DIR / "b_maps_raw.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            b_maps = pickle.load(f)
        log.info(f"  Loaded cached building density ({len(b_maps)} cells)")
        return b_maps

    buildings_sindex = buildings_utm.sindex
    buffer_area = np.pi * BUFFER_M ** 2
    building_geoms = buildings_utm.geometry.values

    b_maps = {}
    n_cells = len(grid_utm)

    for idx in range(n_cells):
        cell_geom = grid_utm.geometry.iloc[idx]
        cell_id = grid_utm["cell_id"].iloc[idx]
        minx, miny, maxx, maxy = cell_geom.bounds

        ext_bounds = (minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M)
        possible_idxs = list(buildings_sindex.intersection(ext_bounds))

        if not possible_idxs:
            b_maps[cell_id] = np.zeros((SUBDIV, SUBDIV), dtype=np.float64)
            if (idx + 1) % 500 == 0:
                log.info(f"  {idx + 1}/{n_cells} cells ({time.time()-t0:.0f}s)")
            continue

        candidates = building_geoms[possible_idxs]
        density_map = np.zeros((SUBDIV, SUBDIV), dtype=np.float64)

        for si in range(SUBDIV):
            cy = miny + (si + 0.5) * CALC_UNIT_M
            for sj in range(SUBDIV):
                cx = minx + (sj + 0.5) * CALC_UNIT_M
                buf = Point(cx, cy).buffer(BUFFER_M)
                total_area = 0.0
                for bgeom in candidates:
                    if buf.intersects(bgeom):
                        total_area += buf.intersection(bgeom).area
                density_map[si, sj] = total_area / buffer_area

        b_maps[cell_id] = density_map
        if (idx + 1) % 500 == 0:
            log.info(f"  {idx + 1}/{n_cells} cells ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    all_vals = np.concatenate([m.flatten() for m in b_maps.values()])
    log.info(f"  Building density done: {elapsed:.0f}s, range [{all_vals.min():.4f}, {all_vals.max():.4f}]")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(b_maps, f)
    log.info(f"  Cached to {cache_path}")
    return b_maps


def compute_poi_entropy(pois_utm, grid_utm):
    """POI Shannon entropy per sub-cell with buffer ring.
    For each sub-cell center with 150m buffer:
      - Count POIs per category within buffer
      - H = -sum(p_i * log2(p_i)), normalized by max_entropy.
    Returns dict {cell_id: 20x20 float64 array}."""
    log.info("Computing G channel (POI entropy)...")
    t0 = time.time()

    cache_path = PROC_DIR / "g_maps_raw.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            g_maps = pickle.load(f)
        log.info(f"  Loaded cached POI entropy ({len(g_maps)} cells)")
        return g_maps

    pois_sindex = pois_utm.sindex
    poi_geoms = pois_utm.geometry.values
    poi_cats = pois_utm["poi_category"].values
    n_categories = pois_utm["poi_category"].nunique()
    max_entropy = np.log2(n_categories) if n_categories > 1 else 1.0

    g_maps = {}
    n_cells = len(grid_utm)

    for idx in range(n_cells):
        cell_geom = grid_utm.geometry.iloc[idx]
        cell_id = grid_utm["cell_id"].iloc[idx]
        minx, miny, maxx, maxy = cell_geom.bounds

        ext_bounds = (minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M)
        possible_idxs = list(pois_sindex.intersection(ext_bounds))

        if not possible_idxs:
            g_maps[cell_id] = np.zeros((SUBDIV, SUBDIV), dtype=np.float64)
            if (idx + 1) % 500 == 0:
                log.info(f"  {idx + 1}/{n_cells} cells ({time.time()-t0:.0f}s)")
            continue

        cand_geoms = poi_geoms[possible_idxs]
        cand_cats = poi_cats[possible_idxs]
        entropy_map = np.zeros((SUBDIV, SUBDIV), dtype=np.float64)

        for si in range(SUBDIV):
            cy = miny + (si + 0.5) * CALC_UNIT_M
            for sj in range(SUBDIV):
                cx = minx + (sj + 0.5) * CALC_UNIT_M
                buf = Point(cx, cy).buffer(BUFFER_M)
                bb = buf.bounds
                nearby_cats = []
                for k in range(len(cand_geoms)):
                    pg = cand_geoms[k]
                    if bb[0] <= pg.x <= bb[2] and bb[1] <= pg.y <= bb[3]:
                        if buf.contains(pg):
                            nearby_cats.append(cand_cats[k])
                if len(nearby_cats) >= 2:
                    counts = Counter(nearby_cats)
                    freqs = np.array(list(counts.values()), dtype=np.float64)
                    probs = freqs / freqs.sum()
                    entropy_map[si, sj] = scipy_entropy(probs, base=2) / max_entropy

        g_maps[cell_id] = entropy_map
        if (idx + 1) % 500 == 0:
            log.info(f"  {idx + 1}/{n_cells} cells ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    all_vals = np.concatenate([m.flatten() for m in g_maps.values()])
    log.info(f"  POI entropy done: {elapsed:.0f}s, range [{all_vals.min():.4f}, {all_vals.max():.4f}]")

    with open(cache_path, "wb") as f:
        pickle.dump(g_maps, f)
    log.info(f"  Cached to {cache_path}")
    return g_maps


def compute_accessibility(pois_utm, G, grid_utm):
    """Gravity-model accessibility using pandana.
    Multi-bin exponential decay (same as v1): bins at [300,600,1000,2000,3000]m.
    Returns dict {cell_id: 20x20 float64 array} (raw, unnormalized)."""
    import pandana as pdna

    log.info("Computing R channel (accessibility)...")
    t0 = time.time()

    cache_path = PROC_DIR / "r_maps_raw.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            r_maps = pickle.load(f)
        log.info(f"  Loaded cached accessibility ({len(r_maps)} cells)")
        return r_maps

    log.info("  Building pandana network...")
    nodes_df = pd.DataFrame({
        "x": pd.Series({n: G.nodes[n]["x"] for n in G.nodes()}),
        "y": pd.Series({n: G.nodes[n]["y"] for n in G.nodes()}),
    })

    edges_list = []
    for u, v, data in G.edges(data=True):
        length = data.get("length", 1.0)
        edges_list.append((u, v, length))
        if not G.has_edge(v, u):
            edges_list.append((v, u, length))
    edges_df = pd.DataFrame(edges_list, columns=["from", "to", "length"])

    net = pdna.Network(
        nodes_df["x"], nodes_df["y"],
        edges_df["from"], edges_df["to"],
        edges_df[["length"]],
    )

    poi_node_ids = net.get_node_ids(pois_utm.geometry.x.values, pois_utm.geometry.y.values)
    net.set(poi_node_ids, name="poi")
    net.precompute(ACCESSIBILITY_CUTOFF_M)

    log.info("  Computing per-node accessibility...")
    bins = np.array([300, 600, 1000, 2000, ACCESSIBILITY_CUTOFF_M])
    bin_centers_km = (bins / 2) / 1000.0
    weights = np.exp(-ACCESSIBILITY_BETA * bin_centers_km)

    access_per_node = pd.Series(0.0, index=nodes_df.index)
    prev_count = None
    for i, (radius, weight) in enumerate(zip(bins, weights)):
        log.info(f"    Bin {i + 1}/{len(bins)}: radius={radius}m, weight={weight:.4f}")
        counts = net.aggregate(radius, type="count", name="poi", decay="flat")
        if prev_count is not None:
            marginal = counts - prev_count
        else:
            marginal = counts
        prev_count = counts.copy()
        access_per_node = access_per_node.add(marginal * weight, fill_value=0)

    log.info(f"  Node access: min={access_per_node.min():.3f}, max={access_per_node.max():.3f}")

    log.info("  Mapping sub-cell centers to network nodes...")
    r_maps = {}
    n_cells = len(grid_utm)

    for idx in range(n_cells):
        cell_geom = grid_utm.geometry.iloc[idx]
        cell_id = grid_utm["cell_id"].iloc[idx]
        minx, miny, maxx, maxy = cell_geom.bounds

        centers_x = []
        centers_y = []
        for si in range(SUBDIV):
            cy = miny + (si + 0.5) * CALC_UNIT_M
            for sj in range(SUBDIV):
                cx = minx + (sj + 0.5) * CALC_UNIT_M
                centers_x.append(cx)
                centers_y.append(cy)

        node_ids = net.get_node_ids(centers_x, centers_y)
        access_map = np.zeros((SUBDIV, SUBDIV), dtype=np.float64)

        for k in range(SUBDIV * SUBDIV):
            si = k // SUBDIV
            sj = k % SUBDIV
            nid = node_ids[k]
            if nid < 0:
                continue
            try:
                node_x = nodes_df.loc[nid, "x"]
                node_y = nodes_df.loc[nid, "y"]
                dist = np.sqrt((centers_x[k] - node_x) ** 2 + (centers_y[k] - node_y) ** 2)
                if dist <= SNAP_MAX_DIST_M:
                    access_map[si, sj] = access_per_node.get(nid, 0.0)
            except KeyError:
                pass

        r_maps[cell_id] = access_map
        if (idx + 1) % 500 == 0:
            log.info(f"  {idx + 1}/{n_cells} cells ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    all_vals = np.concatenate([m.flatten() for m in r_maps.values()])
    log.info(f"  Accessibility done: {elapsed:.0f}s, range [{all_vals.min():.4f}, {all_vals.max():.4f}]")

    with open(cache_path, "wb") as f:
        pickle.dump(r_maps, f)
    log.info(f"  Cached to {cache_path}")
    return r_maps


# ===========================================================================
# Step 4: Normalize (per-channel GLOBAL percentile) -- GAP-6 fix
# ===========================================================================
def normalize_channels(b_maps, g_maps, r_maps, percentile=NORM_PERCENTILE):
    """Per-channel global percentile normalization.
    1. Flatten ALL B values across ALL cells -> B_flat
    2. B_p95 = np.percentile(B_flat, 95)
    3. B_norm = np.clip(B / B_p95, 0, 1) * 255
    4. Repeat for G, R
    """
    log.info(f"Normalizing channels (per-channel global p{percentile})...")

    def _normalize(maps, name):
        all_vals = np.concatenate([m.flatten() for m in maps.values()])
        p = np.percentile(all_vals, percentile)
        if p <= 0:
            nonzero = all_vals[all_vals > 0]
            if len(nonzero) > 0:
                p = np.percentile(nonzero, percentile)
            else:
                p = 1.0
        log.info(f"  {name}: p{percentile}={p:.6f}, range=[{all_vals.min():.6f}, {all_vals.max():.6f}]")
        return {k: np.clip(v / p, 0, 1) * 255 for k, v in maps.items()}, p

    b_norm, b_p = _normalize(b_maps, "B (density)")
    g_norm, g_p = _normalize(g_maps, "G (diversity)")
    r_norm, r_p = _normalize(r_maps, "R (accessibility)")
    return b_norm, g_norm, r_norm, (b_p, g_p, r_p)


# ===========================================================================
# Step 5: Upsample and fuse
# ===========================================================================
def save_cells_and_channels(b_norm, g_norm, r_norm, grid_utm):
    """Upsample 20x20 -> 128x128, stack RGB, save per-cell and per-channel PNGs.
    RGB channel order: R=accessibility, G=diversity, B=density
    Filename: {row:03d}_{col:03d}.png (GAP-3 fix)"""
    log.info("Upsampling and saving cell images...")

    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    (CHANNELS_DIR / "density").mkdir(parents=True, exist_ok=True)
    (CHANNELS_DIR / "diversity").mkdir(parents=True, exist_ok=True)
    (CHANNELS_DIR / "accessibility").mkdir(parents=True, exist_ok=True)

    zoom_factor = TARGET_PX / SUBDIV  # 128 / 20 = 6.4
    cell_ids = sorted(b_norm.keys())
    saved = 0
    empty = 0

    for cell_id in cell_ids:
        b = b_norm.get(cell_id)
        g = g_norm.get(cell_id)
        r = r_norm.get(cell_id)
        if b is None or g is None or r is None:
            continue

        b_up = ndimage_zoom(b, zoom_factor, order=1)
        g_up = ndimage_zoom(g, zoom_factor, order=1)
        r_up = ndimage_zoom(r, zoom_factor, order=1)

        b_uint8 = np.clip(b_up, 0, 255).astype(np.uint8)
        g_uint8 = np.clip(g_up, 0, 255).astype(np.uint8)
        r_uint8 = np.clip(r_up, 0, 255).astype(np.uint8)

        rgb = np.stack([r_uint8, g_uint8, b_uint8], axis=-1)

        # Skip cells with no meaningful data (all channels essentially zero)
        if rgb.max() < 10:
            empty += 1
            continue

        filename = f"{cell_id}.png"
        Image.fromarray(rgb, mode="RGB").save(CELLS_DIR / filename)
        Image.fromarray(b_uint8, mode="L").save(CHANNELS_DIR / "density" / filename)
        Image.fromarray(g_uint8, mode="L").save(CHANNELS_DIR / "diversity" / filename)
        Image.fromarray(r_uint8, mode="L").save(CHANNELS_DIR / "accessibility" / filename)

        saved += 1
        if saved % 500 == 0:
            log.info(f"  Saved {saved}/{len(cell_ids)} cells")

    log.info(f"  Saved {saved} RGB cells to {CELLS_DIR}")
    log.info(f"  Empty (>90% black): {empty}/{saved} ({100*empty/max(saved,1):.1f}%)")
    return saved, empty


# ===========================================================================
# Step 6: Quality report
# ===========================================================================
def generate_quality_report(grid_utm, b_maps, g_maps, r_maps,
                            b_norm, g_norm, r_norm, percentiles,
                            n_saved, n_empty):
    """Generate comprehensive data quality report."""
    log.info("Generating quality report...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    b_p, g_p, r_p = percentiles
    lines = []
    def p(s=""): lines.append(s)

    p("=" * 70)
    p("HCMC Built Environment Pipeline v2 -- Quality Report")
    p("=" * 70)
    p("Key fixes: GAP-3 (naming), GAP-4 (continuous density), GAP-6 (global norm)")
    p(f"Grid cells: {len(grid_utm)}, Calc unit: {CALC_UNIT_M}m, Buffer: {BUFFER_M}m")
    p(f"Output: {TARGET_PX}x{TARGET_PX}x3 uint8 RGB PNGs")
    p(f"Normalization: per-channel global p{NORM_PERCENTILE}")
    p("")
    p(f"Saved cells: {n_saved}, Empty: {n_empty} ({100*n_empty/max(n_saved,1):.1f}%)")
    p("")

    b_all = np.concatenate([m.flatten() for m in b_maps.values()])
    g_all = np.concatenate([m.flatten() for m in g_maps.values()])
    r_all = np.concatenate([m.flatten() for m in r_maps.values()])

    p("--- Raw Channel Statistics ---")
    hdr = "{:20s} {:>12s} {:>12s} {:>12s}".format("", "B(Density)", "G(Entropy)", "R(Access)")
    p(hdr)
    p("{:20s} {:>12.6f} {:>12.6f} {:>12.6f}".format("Mean", b_all.mean(), g_all.mean(), r_all.mean()))
    p("{:20s} {:>12.6f} {:>12.6f} {:>12.6f}".format("Std", b_all.std(), g_all.std(), r_all.std()))
    p("{:20s} {:>12.6f} {:>12.6f} {:>12.6f}".format("Min", b_all.min(), g_all.min(), r_all.min()))
    p("{:20s} {:>12.6f} {:>12.6f} {:>12.6f}".format("Max", b_all.max(), g_all.max(), r_all.max()))
    p("{:20s} {:>12.6f} {:>12.6f} {:>12.6f}".format("P95", b_p, g_p, r_p))
    nz_b = (b_all > 0).sum()
    nz_g = (g_all > 0).sum()
    nz_r = (r_all > 0).sum()
    p("{:20s} {:>12d} {:>12d} {:>12d}".format("Non-zero", nz_b, nz_g, nz_r))
    nzp_b = 100*nz_b/len(b_all)
    nzp_g = 100*nz_g/len(g_all)
    nzp_r = 100*nz_r/len(r_all)
    p("{:20s} {:>11.2f}% {:>11.2f}% {:>11.2f}%".format("Non-zero %", nzp_b, nzp_g, nzp_r))
    p("")

    # Cross-channel correlations (on per-cell means)
    p("--- Cross-Channel Correlations (per-cell mean) ---")
    cell_ids = sorted(b_maps.keys())
    b_means = np.array([b_maps[c].mean() for c in cell_ids])
    g_means = np.array([g_maps[c].mean() for c in cell_ids])
    r_means = np.array([r_maps[c].mean() for c in cell_ids])
    p(f"  B vs G: {np.corrcoef(b_means, g_means)[0,1]:.4f}")
    p(f"  B vs R: {np.corrcoef(b_means, r_means)[0,1]:.4f}")
    p(f"  G vs R: {np.corrcoef(g_means, r_means)[0,1]:.4f}")
    p("")

    # Moran I attempt
    p("--- Spatial Autocorrelation (Moran I) ---")
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran
        for ch_name, ch_means in [("B(Density)", b_means), ("G(Entropy)", g_means), ("R(Access)", r_means)]:
            gdf_v = gpd.GeoDataFrame({"value": ch_means}, geometry=[grid_utm.geometry.iloc[i] for i, c in enumerate(cell_ids) if c in b_maps])
            if len(gdf_v) > 5:
                w = Queen.from_dataframe(gdf_v)
                w.transform = "r"
                mi = Moran(gdf_v["value"].values, w)
                p(f"  {ch_name}: I={mi.I:.4f}, p={mi.p_sim:.4f}")
    except Exception as e:
        p(f"  Moran I failed: {e}")
    p("")

    report = "\n".join(lines)
    report_path = REPORTS_DIR / "data_quality_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Report saved: {report_path}")
    for line in lines:
        log.info(line)
    return report


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Orchestrate all pipeline steps."""
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: Load grid
    log.info("=" * 50 + " Step 1: Load Grid")
    grid_gdf = load_grid()
    grid_utm = grid_gdf.to_crs(UTM_EPSG)

    # Step 2: Fetch OSM data
    log.info("=" * 50 + " Step 2: Fetch OSM Data")
    buildings = fetch_buildings(grid_gdf)
    pois = fetch_pois(grid_gdf)
    G = fetch_network(grid_gdf)
    buildings_utm = buildings.to_crs(UTM_EPSG)
    pois_utm = pois.to_crs(UTM_EPSG)

    # Step 3: Compute channels
    log.info("=" * 50 + " Step 3: Compute Channels")
    b_maps = compute_building_density(buildings_utm, grid_utm)
    g_maps = compute_poi_entropy(pois_utm, grid_utm)
    r_maps = compute_accessibility(pois_utm, G, grid_utm)

    # Step 4: Normalize
    log.info("=" * 50 + " Step 4: Normalize")
    b_norm, g_norm, r_norm, percentiles = normalize_channels(b_maps, g_maps, r_maps)

    # Step 5: Upsample and save
    log.info("=" * 50 + " Step 5: Save Cells")
    n_saved, n_empty = save_cells_and_channels(b_norm, g_norm, r_norm, grid_utm)

    # Step 6: Quality report
    log.info("=" * 50 + " Step 6: Quality Report")
    generate_quality_report(grid_utm, b_maps, g_maps, r_maps,
                            b_norm, g_norm, r_norm, percentiles,
                            n_saved, n_empty)

    elapsed = time.time() - t0
    log.info(f"\nPipeline v2 complete! {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"Output: {CELLS_DIR}/ ({n_saved} PNG files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCMC Data Pipeline v2")
    parser.add_argument("--force", action="store_true", help="Force recompute (ignore cache)")
    args = parser.parse_args()
    if args.force:
        for p in ["b_maps_raw.pkl", "g_maps_raw.pkl", "r_maps_raw.pkl"]:
            f = PROC_DIR / p
            if f.exists():
                f.unlink()
                log.info(f"Deleted cache: {f}")
    main()
