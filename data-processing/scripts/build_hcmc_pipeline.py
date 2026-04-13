#!/usr/bin/env python3
"""
HCMC Built Environment Data Pipeline (Day 1, revised)

Builds a 3-channel (BGR) built environment grid for Ho Chi Minh City urban core.
Channels map to Jacobs' (1961) conditions for urban vitality:

  B channel: Building footprint coverage (binary raster from OSM)
             → Jacobs' "Concentration" → Physical fabric density
  G channel: POI Shannon entropy per 1km cell
             → Jacobs' "Mixed primary uses" → Functional diversity
  R channel: Gravity-model accessibility (continuous exponential decay via pandana)
             → Jacobs' "Small blocks" → Spatial connectivity

The DDPM learns the generative manifold of built environment configurations,
not human behavior.

Output:
  data/hcmc/processed/hcmc_vitality_grid.npy   — shape (rows, cols, 3), float32, [0,1]
  data/hcmc/processed/cropped/                  — per-cell B/G/R subfolders
  data/hcmc/processed/fused/                    — per-cell RGB + grid composite
  data/hcmc/reports/                            — quality report + heatmaps
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.ops import polygonize
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import osmnx as ox
import pandana as pdna
from scipy.stats import entropy
from scipy.interpolate import griddata
from scipy.ndimage import zoom as ndimage_zoom
from libpysal.weights import Queen
from esda.moran import Moran
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "hcmc"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORT_DIR = DATA_DIR / "reports"

GRID_RES_KM = 1.0
UTM_EPSG = 32648  # UTM zone 48N

# HCMC bounding boxes (WGS84: minx, miny, maxx, maxy)
# Urban core (old bbox from plot_poi_distribution.py) — ~653 km^2
HCMC_BBOX_URBAN = (106.58, 10.70, 106.85, 10.90)
# Full municipality (from OSM Relation 1973756, pre-2025 merger) — ~2,407 km^2
HCMC_BBOX = (106.357, 10.140, 107.028, 11.160)

CELL_PX = 256
PIXEL_RES = GRID_RES_KM * 1000 / CELL_PX  # ~3.90625 m/pixel
CELLS_256_DIR = PROC_DIR / "cells_256x256"

# Accessibility: continuous exponential decay
# β=0.3/km: 500m→0.86, 1km→0.74, 2km→0.55, 3km→0.41
# Literature range for pedestrian: 0.1-0.5 (Osth et al., 2016; Stepniak & Rosik, 2018)
ACCESSIBILITY_BETA = 0.3  # per km
ACCESSIBILITY_CUTOFF_M = 3000  # soft cutoff at 3/β

# Square cell threshold: keep cells with >= 90% of full 1km² area
SQUARE_CELL_THRESHOLD = 0.9

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===========================================================================
# Step 1: Get HCMC urban core boundary
# ===========================================================================
def get_hcmc_boundary():
    """Get HCMC urban core boundary using bounding box."""
    boundary_path = RAW_DIR / "hcmc_boundary.gpkg"
    if boundary_path.exists():
        log.info("Loading cached HCMC boundary")
        return gpd.read_file(boundary_path)

    log.info("Creating HCMC urban core boundary from bbox...")
    bbox_geom = box(*HCMC_BBOX)
    gdf = gpd.GeoDataFrame(
        {"name": ["HCMC urban core"]}, geometry=[bbox_geom], crs="EPSG:4326"
    )

    gdf.to_file(boundary_path, driver="GPKG")
    area_km2 = gdf.to_crs(UTM_EPSG).area.iloc[0] / 1e6
    log.info(f"Boundary area: {area_km2:.1f} km^2")
    return gdf


# ===========================================================================
# Step 2: Build 1km grid
# ===========================================================================
def build_1km_grid(boundary_gdf, rework=False):
    """Create a 1km grid aligned to HCMC boundary in UTM.

    When rework=True, regenerate the grid without boundary clipping
    so all cells are rectangular and perfectly tileable for raster output.
    """
    grid_path = RAW_DIR / "hcmc_1km_grid.gpkg"
    if grid_path.exists() and not rework:
        log.info("Loading cached 1km grid")
        gdf = gpd.read_file(grid_path)
        return gdf

    log.info("Building 1km grid...")
    boundary_utm = boundary_gdf.to_crs(UTM_EPSG)
    bounds = boundary_utm.total_bounds

    x_min = np.floor(bounds[0] / 1000) * 1000
    y_min = np.floor(bounds[1] / 1000) * 1000
    x_max = np.ceil(bounds[2] / 1000) * 1000
    y_max = np.ceil(bounds[3] / 1000) * 1000

    grid_cells = []
    for x in np.arange(x_min, x_max, GRID_RES_KM * 1000):
        for y in np.arange(y_min, y_max, GRID_RES_KM * 1000):
            cell = box(x, y, x + GRID_RES_KM * 1000, y + GRID_RES_KM * 1000)
            grid_cells.append(cell)

    grid_gdf = gpd.GeoDataFrame({"geometry": grid_cells}, crs=f"EPSG:{UTM_EPSG}")

    if not rework:
        grid_gdf = gpd.overlay(grid_gdf, boundary_utm, how="intersection")

    grid_gdf["cell_id"] = range(len(grid_gdf))

    if rework:
        rework_path = RAW_DIR / "hcmc_1km_grid_full.gpkg"
        grid_gdf.to_file(rework_path, driver="GPKG")
        log.info(f"Saved full grid (no clip): {rework_path}")
    else:
        grid_gdf.to_file(grid_path, driver="GPKG")

    log.info(f"Grid: {len(grid_gdf)} cells")
    return grid_gdf


# ===========================================================================
# Step 3: Extract OSM building footprints (B channel)
# ===========================================================================
def extract_buildings(boundary_gdf):
    """Extract OSM building polygons via Overpass API.

    Downloads building footprint polygons for the HCMC urban core.
    Buildings are stored as polygon geometries for rasterization.
    """
    building_path = RAW_DIR / "hcmc_buildings.gpkg"
    if building_path.exists():
        log.info("Loading cached building footprints")
        buildings = gpd.read_file(building_path)
        log.info(f"Loaded {len(buildings)} buildings")
        return buildings

    log.info("Extracting building footprints from Overpass API...")
    boundary_wgs84 = boundary_gdf.to_crs(epsg=4326)
    bounds = boundary_wgs84.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

    overpass_url = "https://overpass-api.de/api/interpreter"

    # Split bbox into smaller tiles to avoid Overpass timeout
    # The full bbox is ~27km x 22km — split into ~4 tiles
    lon_step = (bbox[2] - bbox[0]) / 2
    lat_step = (bbox[3] - bbox[1]) / 2
    tiles = []
    for i in range(2):
        for j in range(2):
            tile = (
                bbox[0] + i * lat_step,
                bbox[1] + j * lon_step,
                bbox[0] + (i + 1) * lat_step,
                bbox[1] + (j + 1) * lon_step,
            )
            tiles.append(tile)

    all_records = []
    for t_idx, tile in enumerate(tiles):
        log.info(
            f"  Querying tile {t_idx + 1}/{len(tiles)}: ({tile[0]:.4f},{tile[1]:.4f},{tile[2]:.4f},{tile[3]:.4f})"
        )
        query = f"""
        [out:json][timeout:300][maxsize:536870912];
        (
          way["building"]({tile[0]:.4f},{tile[1]:.4f},{tile[2]:.4f},{tile[3]:.4f});
        );
        out geom;
        """

        for attempt in range(3):
            try:
                resp = requests.post(overpass_url, data={"data": query}, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                log.warning(f"    Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    import time

                    time.sleep(30 * (attempt + 1))
                else:
                    log.warning(f"    Skipping tile {t_idx + 1} after 3 attempts")
                    data = {"elements": []}

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
                all_records.append(
                    {
                        "geometry": poly,
                        "building": elem.get("tags", {}).get("building", "yes"),
                    }
                )
            except Exception:
                continue

        log.info(f"    Buildings so far: {len(all_records)}")

    if not all_records:
        raise RuntimeError("No buildings found from Overpass API")

    buildings = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
    buildings.to_file(building_path, driver="GPKG")
    log.info(f"Total buildings: {len(buildings)}")
    return buildings


def compute_building_channel(grid_gdf, buildings):
    """Rasterize building footprints as binary (1=building, 0=no building) per grid cell.

    Uses rasterio.features.rasterize to burn building polygons into each cell's
    bounding box. Each cell gets a binary value: 1 if any building footprint exists
    within the cell, 0 otherwise.

    At finer resolutions (future 500m), this would produce actual building shape
    patterns that the DDPM image encoder can learn from.
    """
    log.info("Computing B channel (building footprint coverage)...")

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    buildings_utm = buildings.to_crs(UTM_EPSG)

    # For each grid cell, check if any building intersects it
    building_values = np.zeros(len(grid_gdf), dtype=np.float32)

    for i, cell_geom in enumerate(grid_utm.geometry):
        # Check intersection with buildings
        hits = buildings_utm.geometry.intersects(cell_geom)
        if hits.any():
            building_values[i] = 1.0

    # Already binary [0, 1], no normalization needed
    log.info(
        f"  Building coverage: {(building_values > 0).sum()}/{len(building_values)} cells have buildings"
    )
    return building_values, building_values


# ===========================================================================
# Step 4: Extract OSM POI data
# ===========================================================================
def extract_pois(boundary_gdf):
    """Extract POIs using direct Overpass API queries.

    Includes both nodes (point POIs) and ways (area POIs, using centroids).
    Uses `out center` to get centroid coordinates for ways.
    """
    poi_path = RAW_DIR / "hcmc_pois_all.gpkg"
    if poi_path.exists():
        log.info("Loading cached POI data")
        pois = gpd.read_file(poi_path)
        log.info(f"Loaded {len(pois)} POIs")
        return pois

    log.info("Extracting POIs from Overpass API...")
    boundary_wgs84 = boundary_gdf.to_crs(epsg=4326)
    bounds = boundary_wgs84.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:300][maxsize:536870912];
    (
      node["amenity"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      node["shop"~"supermarket|convenience|mall|clothes|electronics|bakery"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      node["office"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      node["tourism"~"hotel|hostel|museum|attraction|viewpoint|artwork"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      node["leisure"~"park|garden|playground|sports_centre|swimming_pool|fitness_centre"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      way["amenity"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      way["shop"~"supermarket|convenience|mall|clothes|electronics|bakery"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      way["office"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      way["tourism"~"hotel|hostel|museum|attraction|viewpoint|artwork"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
      way["leisure"~"park|garden|playground|sports_centre|swimming_pool|fitness_centre"]({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f});
    );
    out center;
    """

    log.info("  Sending Overpass query (nodes + ways)...")
    resp = requests.post(overpass_url, data={"data": query}, timeout=600)
    resp.raise_for_status()
    data = resp.json()

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

        category = "unknown"
        poi_type = "unknown"
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

        records.append(
            {
                "geometry": Point(lon, lat),
                "poi_category": category,
                "poi_type": poi_type,
                "osm_type": elem["type"],
            }
        )

    if not records:
        raise RuntimeError("No POIs found from Overpass API")

    pois = gpd.GeoDataFrame(records, crs="EPSG:4326")
    pois.to_file(poi_path, driver="GPKG")

    log.info(f"Total POIs: {len(pois)}")
    for pt, count in pois["poi_type"].value_counts().items():
        log.info(f"  {pt}: {count}")
    return pois


# ===========================================================================
# Step 5: Compute G channel (POI Shannon entropy per grid cell)
# ===========================================================================
def compute_poi_entropy_channel(grid_gdf, pois):
    """Compute Shannon entropy of POI categories per grid cell.

    Shannon entropy H = -Σ p_i * log2(p_i) measures the diversity of POI categories
    in each 1km cell. Maximized when categories are evenly distributed.

    Directly quantifies Jacobs' (1961) "mixed primary uses" condition.

    Limitations:
    - Semantic content is lost: "restaurant" and "cafe" count as different as
      "restaurant" and "hospital". Category granularity affects absolute values.
    - Binary presence (a POI is counted or not), not weighted by area or capacity.
    - Only meaningful at aggregate spatial scale (1km). At pixel level (100m),
      most cells have 0-1 POIs, making entropy undefined.

    Future work: learned POI embeddings, multi-channel per super-category,
    GNN-based spatial encoding. Out of scope for current sprint.
    """
    log.info("Computing G channel (POI Shannon entropy)...")

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    pois_utm = pois.to_crs(UTM_EPSG)

    # Use intersects so POIs at cell boundaries are captured
    joined = gpd.sjoin(
        pois_utm, grid_utm[["cell_id", "geometry"]], how="left", predicate="intersects"
    )
    cell_col = "cell_id_right" if "cell_id_right" in joined.columns else "cell_id"

    joined = joined[~joined[cell_col].isna()].copy()
    joined["_cell_id"] = joined[cell_col].astype(int)

    entropy_values = np.zeros(len(grid_gdf), dtype=np.float64)

    for cid in joined["_cell_id"].unique():
        cell_pois = joined[joined["_cell_id"] == cid]
        counts = cell_pois["poi_category"].value_counts().values.astype(float)
        probs = counts / counts.sum()
        entropy_values[cid] = entropy(probs, base=2)

    n_categories = pois["poi_category"].nunique()
    max_entropy = np.log2(n_categories) if n_categories > 1 else 1.0
    entropy_norm = entropy_values / max_entropy if max_entropy > 0 else entropy_values

    log.info(
        f"  Entropy range: {entropy_values.min():.2f} - {entropy_values.max():.2f}, "
        f"non-zero: {(entropy_values > 0).sum()}/{len(entropy_values)}"
    )
    return entropy_norm.astype(np.float32), entropy_values


# ===========================================================================
# Step 6: Download OSM street network
# ===========================================================================
def download_street_network(boundary_gdf):
    """Download pedestrian-mode street network from OSM."""
    net_path = RAW_DIR / "hcmc_pedestrian_network.graphml"
    if net_path.exists():
        log.info("Loading cached street network")
        return ox.load_graphml(net_path)

    log.info("Downloading pedestrian street network from OSM...")
    boundary_wgs84 = boundary_gdf.to_crs(epsg=4326)
    geom = boundary_wgs84.geometry.iloc[0]

    G = ox.graph_from_polygon(geom, network_type="walk", simplify=True, retain_all=True)
    G = ox.project_graph(G, to_crs=f"EPSG:{UTM_EPSG}")

    ox.save_graphml(G, net_path)
    log.info(
        f"Street network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


# ===========================================================================
# Step 7: Compute R channel (Continuous exponential decay accessibility)
# ===========================================================================
def compute_accessibility_channel(grid_gdf, pois, G):
    """Compute gravity-model accessibility with continuous exponential decay.

    A_i = Σ_j exp(-β × d_ij)

    where d_ij is network walking distance in km, β = 0.3/km.

    Continuous decay avoids artificial discontinuities of stepped decay
    (e.g., 499m → weight 1.0, 501m → 0.5). Exponential is standard in
    transport geography (Geurs & Van Wee, 2004; Hansen, 1959).

    β = 0.3/km: 500m→0.86, 1km→0.74, 2km→0.55, 3km→0.41
    Soft cutoff at 3km (3/β): contribution <5% of peak.

    All POIs weighted equally (W_j = 1): no calibration data available.
    Weighting by type would conflate with Channel G (diversity).
    """
    log.info(
        f"Computing R channel (continuous exponential decay, beta={ACCESSIBILITY_BETA})..."
    )

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    pois_utm = pois.to_crs(UTM_EPSG)

    # Build pandana network
    nodes_df = pd.DataFrame(
        {
            "x": pd.Series({n: G.nodes[n]["x"] for n in G.nodes()}),
            "y": pd.Series({n: G.nodes[n]["y"] for n in G.nodes()}),
        }
    )

    edges_list = []
    for u, v, data in G.edges(data=True):
        length = data.get("length", 1.0)
        edges_list.append((u, v, length))
        if not G.has_edge(v, u):
            edges_list.append((v, u, length))

    edges_df = pd.DataFrame(edges_list, columns=["from", "to", "length"])

    net = pdna.Network(
        nodes_df["x"],
        nodes_df["y"],
        edges_df["from"],
        edges_df["to"],
        edges_df[["length"]],
    )

    # Set POIs on network
    poi_node_ids = net.get_node_ids(
        pois_utm.geometry.x.values, pois_utm.geometry.y.values
    )
    net.set(poi_node_ids, name="poi")

    # Precompute for max distance
    net.precompute(ACCESSIBILITY_CUTOFF_M)

    # Get grid centroid node IDs
    centroid_node_ids = net.get_node_ids(
        grid_utm.geometry.centroid.x.values,
        grid_utm.geometry.centroid.y.values,
    )

    # Compute accessibility using pandana aggregate with flat counts
    # at multiple distance bins, then apply continuous exponential weights.
    # pandana's decay only works with sum/mean, so we use flat counts
    # and weight them ourselves: A_i = Σ_j exp(-beta * d_ij)
    # Approximate with distance bins: at each radius r, all POIs within r
    # are counted. The marginal POIs between r_(k-1) and r_k get weight
    # exp(-beta * r_k). Sum over bins = A_i.

    net.precompute(ACCESSIBILITY_CUTOFF_M)

    # Use pandana's built-in exponential decay with sum type
    # pandana: exp(-distance/radius) * value, with type='sum'
    # We need to pass actual variable values (not just count)
    # pandana's exponential decay with aggregate() is unreliable (returns empty).
    # Use flat counts at multiple distance bins and apply continuous exponential
    # weights in post-processing.
    # A_i = Σ_r count_r * exp(-beta * r_km)
    # This approximates the continuous integral well with enough bins.

    net.precompute(ACCESSIBILITY_CUTOFF_M)

    # Distance bins: 500, 1500, 3000m (3 bins, ~1 hour for first run, cached after)
    bins = np.array([500, 1500, ACCESSIBILITY_CUTOFF_M])
    bin_centers_km = (bins / 2) / 1000.0  # midpoint of each bin in km
    weights = np.exp(-ACCESSIBILITY_BETA * bin_centers_km)

    accessibility = np.zeros(len(grid_utm), dtype=np.float64)
    prev_count = None

    for i, (radius, weight) in enumerate(zip(bins, weights)):
        log.info(f"    Bin {i + 1}/{len(bins)}: radius={radius}m, weight={weight:.4f}")
        counts = net.aggregate(radius, type="count", name="poi", decay="flat")
        if prev_count is not None:
            marginal = counts - prev_count
        else:
            marginal = counts
        prev_count = counts.copy()

        # Vectorized extraction: use .loc with array of node IDs
        values = marginal.loc[centroid_node_ids].values
        accessibility += values * weight

    access_norm = (
        accessibility / accessibility.max()
        if accessibility.max() > 0
        else accessibility
    )

    log.info(
        f"  Accessibility range: {accessibility.min():.2f} - {accessibility.max():.2f}, "
        f"non-zero: {(accessibility > 0).sum()}/{len(accessibility)}"
    )
    return access_norm.astype(np.float32), accessibility


# ===========================================================================
# Step 8: Assemble 3-channel grid (original scalar version)
# ===========================================================================


def rasterize_buildings(buildings, x_min, y_min, x_max, y_max, n_cols, n_rows):
    """Burn building polygons into a full-city binary canvas.

    Returns uint8 array (n_rows, n_cols) with 255=building, 0=empty.
    """
    log.info("Rasterizing buildings to full-city canvas...")

    transform = from_bounds(x_min, y_min, x_max, y_max, n_cols, n_rows)
    grid_extent = box(x_min, y_min, x_max, y_max)

    buildings_utm = buildings.to_crs(UTM_EPSG)
    clipped = buildings_utm[buildings_utm.geometry.intersects(grid_extent)].copy()

    log.info(
        f"  Buildings: {len(clipped)} of {len(buildings_utm)} clipped to grid extent"
    )
    log.info(f"  Grid extent: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    if len(clipped) > 0:
        log.info(f"  Clipped bounds: {clipped.total_bounds}")
    log.info(
        f"  Rasterize params: n_cols={n_cols}, n_rows={n_rows}, transform={transform}"
    )
    log.info(f"  Pixel resolution: {(x_max - x_min) / n_cols:.4f} m/px")

    shapes = ((geom, 255) for geom in clipped.geometry)

    canvas = rasterize(
        shapes=shapes,
        out_shape=(n_rows, n_cols),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    log.info(f"  Canvas: {canvas.shape}, building pixels: {(canvas > 0).sum()}")
    return canvas


def rasterize_poi_entropy(pois, grid_gdf, n_cols, n_rows):
    """Compute per-cell POI entropy density maps, upsample to 256x256.

    For each 1km cell:
      1. Divide into 20x20 sub-cells (50m each)
      2. Count POIs per category in each sub-cell
      3. Compute Shannon entropy per sub-cell
      4. Bilinear interpolate to 256x256

    Returns float32 array (n_rows * CELL_PX, n_cols * CELL_PX) normalized to [0,1].
    """
    log.info("Rasterizing POI entropy density...")

    SUBDIV = 20
    pois_utm = pois.to_crs(UTM_EPSG)

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values

    unique_x = np.unique(np.floor(cx / 1000) * 1000)
    unique_y = np.unique(np.floor(cy / 1000) * 1000)
    n_grid_rows = len(unique_y)
    n_grid_cols = len(unique_x)

    x_to_col = {v: i for i, v in enumerate(unique_x)}
    y_to_row = {v: i for i, v in enumerate(unique_y)}

    half_cell = GRID_RES_KM * 500
    sub_size = GRID_RES_KM * 1000 / SUBDIV

    poi_x = pois_utm.geometry.x.values
    poi_y = pois_utm.geometry.y.values
    poi_cats = pois_utm["poi_category"].values

    canvas_h = n_grid_rows * CELL_PX
    canvas_w = n_grid_cols * CELL_PX
    full_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    grid_cells_list = list(grid_gdf.iterrows())
    total = len(grid_cells_list)
    log.info(
        f"  Total POIs: {len(pois_utm)}, unique categories: {pois_utm['poi_category'].nunique()}"
    )
    log.info(f"  Grid cells to process: {total}")

    for idx, (i, row) in enumerate(grid_cells_list):
        if (idx + 1) % 100 == 0 or idx == 0:
            log.info(f"  Cell {idx + 1}/{total}")

        centroid = row.geometry.centroid
        cell_x0 = np.floor(cx[i] / 1000) * 1000
        cell_y0 = np.floor(cy[i] / 1000) * 1000
        grid_col = x_to_col.get(cell_x0)
        grid_row = y_to_row.get(cell_y0)
        if grid_col is None or grid_row is None:
            continue

        entropy_map = np.zeros((SUBDIV, SUBDIV), dtype=np.float32)

        left = cell_x0 - half_cell
        bottom = cell_y0 - half_cell

        sub_x0 = left + np.arange(SUBDIV) * sub_size
        sub_y0 = bottom + np.arange(SUBDIV) * sub_size

        for si in range(SUBDIV):
            for sj in range(SUBDIV):
                sx0 = sub_x0[sj]
                sy0 = sub_y0[si]
                mask = (
                    (poi_x >= sx0)
                    & (poi_x < sx0 + sub_size)
                    & (poi_y >= sy0)
                    & (poi_y < sy0 + sub_size)
                )
                if mask.sum() == 0:
                    continue
                cats = poi_cats[mask]
                counts = pd.Series(cats).value_counts().values.astype(float)
                probs = counts / counts.sum()
                entropy_map[si, sj] = entropy(probs, base=2)

        if entropy_map.max() > 0:
            entropy_map /= entropy_map.max()

        entropy_256 = ndimage_zoom(entropy_map, CELL_PX / SUBDIV, order=1)

        px_row = grid_row * CELL_PX
        px_col = grid_col * CELL_PX
        h = min(CELL_PX, full_canvas.shape[0] - px_row)
        w = min(CELL_PX, full_canvas.shape[1] - px_col)
        full_canvas[px_row : px_row + h, px_col : px_col + w] = entropy_256[:h, :w]

    log.info(
        f"  Entropy canvas: {full_canvas.shape}, range [{full_canvas.min():.3f}, {full_canvas.max():.3f}]"
    )
    return full_canvas


def rasterize_accessibility(pois, G, x_min, y_min, x_max, y_max, canvas_w, canvas_h):
    """Compute accessibility surface via pandana, interpolate to pixel grid.

    Returns float32 array (n_rows * CELL_PX, n_cols * CELL_PX) normalized to [0,1].
    """
    log.info("Rasterizing accessibility surface...")

    pois_utm = pois.to_crs(UTM_EPSG)

    nodes_df = pd.DataFrame(
        {
            "x": pd.Series({n: G.nodes[n]["x"] for n in G.nodes()}),
            "y": pd.Series({n: G.nodes[n]["y"] for n in G.nodes()}),
        }
    )

    edges_list = []
    for u, v, data in G.edges(data=True):
        length = data.get("length", 1.0)
        edges_list.append((u, v, length))
        if not G.has_edge(v, u):
            edges_list.append((v, u, length))

    edges_df = pd.DataFrame(edges_list, columns=["from", "to", "length"])

    net = pdna.Network(
        nodes_df["x"],
        nodes_df["y"],
        edges_df["from"],
        edges_df["to"],
        edges_df[["length"]],
    )

    poi_node_ids = net.get_node_ids(
        pois_utm.geometry.x.values, pois_utm.geometry.y.values
    )
    net.set(poi_node_ids, name="poi")

    net.precompute(ACCESSIBILITY_CUTOFF_M)

    try:
        log.info("    Trying pandana exponential decay (type=sum)...")
        access_per_node = net.aggregate(
            ACCESSIBILITY_CUTOFF_M, type="sum", decay="exponential", name="poi"
        )
        if access_per_node.dropna().max() == 0:
            raise ValueError("Exponential decay returned all zeros")
        log.info("    Exponential decay succeeded")
    except Exception:
        log.info("    Falling back to multi-bin approximation...")
        bins = np.array([300, 600, 1000, 2000, ACCESSIBILITY_CUTOFF_M])
        bin_centers_km = (bins / 2) / 1000.0
        weights = np.exp(-ACCESSIBILITY_BETA * bin_centers_km)

        access_per_node = pd.Series(0.0, index=nodes_df.index)
        prev_count = None
        for radius, weight in zip(bins, weights):
            counts = net.aggregate(radius, type="count", name="poi", decay="flat")
            if prev_count is not None:
                marginal = counts - prev_count
            else:
                marginal = counts
            prev_count = counts.copy()
            access_per_node = access_per_node.add(marginal * weight, fill_value=0)

    node_x = nodes_df["x"].values
    node_y = nodes_df["y"].values
    node_vals = np.asarray(
        access_per_node.reindex(nodes_df.index).fillna(0), dtype=np.float64
    )
    log.info(
        f"  Node values: min={node_vals.min():.3f}, max={node_vals.max():.3f}, mean={node_vals.mean():.3f}, nonzero={np.count_nonzero(node_vals)}"
    )

    pixel_xs = np.linspace(x_min + PIXEL_RES / 2, x_max - PIXEL_RES / 2, canvas_w)
    pixel_ys = np.linspace(y_min + PIXEL_RES / 2, y_max - PIXEL_RES / 2, canvas_h)

    grid_x, grid_y = np.meshgrid(pixel_xs, pixel_ys)

    log.info("    Interpolating from network nodes to pixel grid...")
    interp_vals = griddata(
        points=(node_x, node_y),
        values=node_vals,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=0,
    )

    interp_vals = np.nan_to_num(interp_vals, nan=0.0)

    if interp_vals.max() > 0:
        interp_vals = interp_vals / interp_vals.max()

    full_canvas = interp_vals.astype(np.float32)
    log.info(
        f"  Accessibility canvas: {full_canvas.shape}, range [{full_canvas.min():.3f}, {full_canvas.max():.3f}]"
    )
    return full_canvas


def create_cell_images(b_canvas, g_canvas, r_canvas, grid_gdf, n_cols, n_rows):
    """Slice full-city canvases into per-cell 256x256 RGB PNGs.

    Output: CELLS_256_DIR/{label:04d}.png
    """
    log.info("Creating per-cell 256x256 RGB images...")

    CELLS_256_DIR.mkdir(parents=True, exist_ok=True)

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values

    unique_x = np.unique(np.floor(cx / 1000) * 1000)
    unique_y = np.unique(np.floor(cy / 1000) * 1000)
    x_to_col = {v: i for i, v in enumerate(unique_x)}
    y_to_row = {v: i for i, v in enumerate(unique_y)}

    log.info(
        f"  Canvas shapes: B={b_canvas.shape}, G={g_canvas.shape}, R={r_canvas.shape}"
    )
    log.info(f"  Grid: {len(unique_y)} rows x {len(unique_x)} cols")
    log.info(
        f"  Expected canvas size: {len(unique_y) * CELL_PX} x {len(unique_x) * CELL_PX}"
    )
    log.info(f"  unique_y range: [{unique_y.min()}, {unique_y.max()}]")
    log.info(f"  unique_x range: [{unique_x.min()}, {unique_x.max()}]")

    half_cell = GRID_RES_KM * 500
    saved = 0
    empty = 0

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

        g_max = g_cell.max()
        g_uint8 = (
            np.clip(g_cell / g_max * 255, 0, 255).astype(np.uint8)
            if g_max > 0
            else g_cell.astype(np.uint8)
        )

        r_max = r_cell.max()
        r_uint8 = (
            np.clip(r_cell / r_max * 255, 0, 255).astype(np.uint8)
            if r_max > 0
            else r_cell.astype(np.uint8)
        )

        rgb = np.stack([r_uint8, g_uint8, b_cell], axis=-1)

        label = f"{saved:04d}"
        out_path = CELLS_256_DIR / f"{label}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path)
        saved += 1

        if rgb.max() == 0:
            empty += 1

    log.info(f"  Saved {saved} images to {CELLS_256_DIR}")
    log.info(f"  Empty (all-black): {empty}/{saved}")
    return saved


def assemble_grid(grid_gdf, b_channel, g_channel, r_channel):
    """Assemble channels into a 2D grid array."""
    log.info("Assembling 3-channel grid...")

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values

    unique_x = np.unique(np.round(cx / 1000) * 1000)
    unique_y = np.unique(np.round(cy / 1000) * 1000)
    rows, cols = len(unique_y), len(unique_x)

    log.info(f"Grid dimensions: {rows} x {cols} = {rows * cols} cells")

    x_to_col = {v: i for i, v in enumerate(unique_x)}
    y_to_row = {v: i for i, v in enumerate(unique_y)}

    b_grid = np.zeros((rows, cols), dtype=np.float32)
    g_grid = np.zeros((rows, cols), dtype=np.float32)
    r_grid = np.zeros((rows, cols), dtype=np.float32)
    mask_grid = np.zeros((rows, cols), dtype=bool)

    for i in range(len(grid_gdf)):
        cx_km = np.round(cx[i] / 1000) * 1000
        cy_km = np.round(cy[i] / 1000) * 1000
        if cy_km in y_to_row and cx_km in x_to_col:
            r, c = y_to_row[cy_km], x_to_col[cx_km]
            b_grid[r, c] = b_channel[i]
            g_grid[r, c] = g_channel[i]
            r_grid[r, c] = r_channel[i]
            mask_grid[r, c] = True

    vitality_grid = np.stack([b_grid, g_grid, r_grid], axis=-1)
    return vitality_grid, mask_grid, unique_y, unique_x


# ===========================================================================
# Step 9: Filter square cells and crop
# ===========================================================================
def crop_square_cells(grid_gdf, b_channel, g_channel, r_channel):
    """Filter to square cells only (>90% area), save cropped per-cell data.

    Output:
      processed/cropped/B_buildings/   — building presence per cell
      processed/cropped/G_entropy/     — entropy value per cell
      processed/cropped/R_accessibility/ — accessibility value per cell
      processed/cropped/labels.txt     — index → cell_id, row, col
    """
    log.info("Cropping square cells...")

    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    areas = grid_utm.geometry.area
    full_area = (GRID_RES_KM * 1000) ** 2  # 1,000,000 m^2

    # Filter square cells
    square_mask = areas >= SQUARE_CELL_THRESHOLD * full_area
    square_cells = grid_gdf[square_mask].copy()
    n_square = len(square_cells)
    n_irregular = len(grid_gdf) - n_square

    log.info(f"  Square cells: {n_square}, Irregular (skipped): {n_irregular}")

    # Get grid positions
    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values

    unique_x = np.unique(np.round(cx / 1000) * 1000)
    unique_y = np.unique(np.round(cy / 1000) * 1000)
    x_to_col = {v: i for i, v in enumerate(unique_x)}
    y_to_row = {v: i for i, v in enumerate(unique_y)}

    # Create output directories
    cropped_dir = PROC_DIR / "cropped"
    for ch_name in ["B_buildings", "G_entropy", "R_accessibility"]:
        (cropped_dir / ch_name).mkdir(parents=True, exist_ok=True)

    # Build label mapping and save per-cell data
    records = []
    for idx, (i, row) in enumerate(square_cells.iterrows()):
        label = f"{idx:04d}"
        cx_km = np.round(cx[i] / 1000) * 1000
        cy_km = np.round(cy[i] / 1000) * 1000
        grid_row = y_to_row.get(cy_km, -1)
        grid_col = x_to_col.get(cx_km, -1)

        # Save channel values (at 1km, each cell is one value)
        b_val = b_channel[i]
        g_val = g_channel[i]
        r_val = r_channel[i]

        records.append(
            {
                "label": label,
                "cell_id": i,
                "grid_row": grid_row,
                "grid_col": grid_col,
                "B_buildings": b_val,
                "G_entropy": g_val,
                "R_accessibility": r_val,
            }
        )

        # Save as small numpy arrays (single value at 1km resolution)
        np.save(
            cropped_dir / "B_buildings" / f"{label}.npy",
            np.array([b_val], dtype=np.float32),
        )
        np.save(
            cropped_dir / "G_entropy" / f"{label}.npy",
            np.array([g_val], dtype=np.float32),
        )
        np.save(
            cropped_dir / "R_accessibility" / f"{label}.npy",
            np.array([r_val], dtype=np.float32),
        )

    # Save label mapping as CSV
    labels_df = pd.DataFrame(records)
    labels_df.to_csv(cropped_dir / "labels.txt", index=False)

    log.info(f"  Saved {n_square} cells to {cropped_dir}")
    return square_cells, labels_df


# ===========================================================================
# Step 10: Fuse RGB per cell
# ===========================================================================
def fuse_rgb(grid_gdf, labels_df, b_channel, g_channel, r_channel):
    """Fuse B+G+R channels into per-cell RGB and composite grid visualization.

    Output:
      processed/fused/hcmc_cell_data.csv  — per-cell B, G, R values
      processed/fused/hcmc_grid_rgb.png    — composite grid visualization
    """
    log.info("Fusing RGB images...")

    fused_dir = PROC_DIR / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    # Save per-cell CSV
    labels_df.to_csv(fused_dir / "hcmc_cell_data.csv", index=False)

    # Create composite grid visualization
    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values

    unique_x = np.unique(np.round(cx / 1000) * 1000)
    unique_y = np.unique(np.round(cy / 1000) * 1000)
    rows, cols = len(unique_y), len(unique_x)

    x_to_col = {v: i for i, v in enumerate(unique_x)}
    y_to_row = {v: i for i, v in enumerate(unique_y)}

    rgb_grid = np.zeros((rows, cols, 3), dtype=np.float32)
    mask_grid = np.zeros((rows, cols), dtype=bool)

    for _, rec in labels_df.iterrows():
        i = rec["cell_id"]
        cx_km = np.round(cx[i] / 1000) * 1000
        cy_km = np.round(cy[i] / 1000) * 1000
        if cy_km in y_to_row and cx_km in x_to_col:
            r, c = y_to_row[cy_km], x_to_col[cx_km]
            rgb_grid[r, c, 0] = rec["R_accessibility"]
            rgb_grid[r, c, 1] = rec["G_entropy"]
            rgb_grid[r, c, 2] = rec["B_buildings"]
            mask_grid[r, c] = True

    # Save composite PNG
    display = rgb_grid.copy()
    for ch in range(3):
        display[:, :, ch][~mask_grid] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(display, aspect="equal", interpolation="nearest")
    ax.set_title(
        "HCMC Built Environment Composite (R=Access, G=Entropy, B=Buildings)",
        fontsize=12,
    )
    fig.savefig(fused_dir / "hcmc_grid_rgb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"  Fused data saved to {fused_dir}")
    log.info(f"  Cell data: {fused_dir / 'hcmc_cell_data.csv'}")


# ===========================================================================
# Step 11: Data quality report
# ===========================================================================
def generate_quality_report(
    grid_gdf,
    buildings,
    pois,
    b_raw,
    g_raw,
    r_raw,
    b_norm,
    g_norm,
    r_norm,
    vitality_grid,
    mask_grid,
):
    """Generate comprehensive data quality report."""
    log.info("Generating data quality report...")

    report_path = REPORT_DIR / "data_quality_report.txt"
    lines = []

    def p(s=""):
        lines.append(s)

    p("=" * 70)
    p("HCMC Built Environment Data - Quality Report")
    p("=" * 70)
    p()
    p("Channels (Jacobs' conditions for urban vitality):")
    p("  B: Building footprint coverage (binary) -> Concentration")
    p("  G: POI Shannon entropy -> Mixed primary uses")
    p("  R: Gravity accessibility (exp decay, beta=0.3/km) -> Small blocks")
    p()

    # Grid stats
    p("--- Grid Statistics ---")
    p(f"Total grid cells:        {len(grid_gdf)}")
    p(f"Grid dimensions:         {vitality_grid.shape[0]} x {vitality_grid.shape[1]}")
    p(f"Grid resolution:         {GRID_RES_KM} km")
    area_km2 = grid_gdf.to_crs(UTM_EPSG).area.sum() / 1e6
    p(f"Total area:              {area_km2:.1f} km^2")
    p(
        f"Coverage:                {mask_grid.sum()} / {mask_grid.size} cells ({100 * mask_grid.sum() / mask_grid.size:.1f}%)"
    )

    # Square vs irregular
    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    areas = grid_utm.geometry.area
    full_area = (GRID_RES_KM * 1000) ** 2
    n_square = (areas >= SQUARE_CELL_THRESHOLD * full_area).sum()
    p(f"Square cells (>90%):     {n_square} ({100 * n_square / len(grid_gdf):.1f}%)")
    p(f"Irregular cells:         {len(grid_gdf) - n_square}")
    p()

    # Building stats
    p("--- Building Statistics ---")
    p(f"Total building footprints: {len(buildings)}")
    buildings_utm = buildings.to_crs(UTM_EPSG)
    total_building_area = buildings_utm.geometry.area.sum() / 1e6
    p(f"Total building area:      {total_building_area:.2f} km^2")
    p(f"Building types:           {buildings['building'].nunique()}")
    for bt, cnt in buildings["building"].value_counts().head(10).items():
        p(f"  {bt:30s}: {cnt:6d}")
    p()

    # POI stats
    p("--- POI Statistics ---")
    p(f"Total POIs:              {len(pois)}")
    p(f"POI categories:          {pois['poi_category'].nunique()}")
    p(f"POI types:               {pois['poi_type'].nunique()}")
    pois_utm = pois.to_crs(UTM_EPSG)
    joined = gpd.sjoin(
        pois_utm, grid_utm[["cell_id", "geometry"]], how="left", predicate="intersects"
    )
    cell_col = "cell_id_right" if "cell_id_right" in joined.columns else "cell_id"
    joined = joined[~joined[cell_col].isna()].copy()
    poi_per_cell = (
        joined.groupby(cell_col).size().reindex(range(len(grid_gdf)), fill_value=0)
    )
    p(f"Mean POIs/cell:          {poi_per_cell.mean():.1f}")
    p(f"Median POIs/cell:        {poi_per_cell.median():.1f}")
    p(f"Max POIs/cell:           {poi_per_cell.max():.0f}")
    p(
        f"Zero-POI cells:          {(poi_per_cell == 0).sum()} ({100 * (poi_per_cell == 0).mean():.1f}%)"
    )
    p()

    # Channel stats
    p("--- Channel Statistics (raw) ---")
    p(f"{'':25s} {'B (Buildings)':>15s} {'G (Entropy)':>15s} {'R (Access)':>15s}")
    p(f"{'Mean':25s} {b_raw.mean():>15.4f} {g_raw.mean():>15.3f} {r_raw.mean():>15.3f}")
    p(f"{'Std':25s} {b_raw.std():>15.4f} {g_raw.std():>15.3f} {r_raw.std():>15.3f}")
    p(f"{'Min':25s} {b_raw.min():>15.4f} {g_raw.min():>15.3f} {r_raw.min():>15.3f}")
    p(f"{'Max':25s} {b_raw.max():>15.4f} {g_raw.max():>15.3f} {r_raw.max():>15.3f}")
    p(
        f"{'Non-zero %':25s} {100 * (b_raw > 0).mean():>14.1f}% {100 * (g_raw > 0).mean():>14.1f}% {100 * (r_raw > 0).mean():>14.1f}%"
    )
    p()

    # Moran's I
    p("--- Spatial Autocorrelation (Moran's I) ---")
    try:
        for ch_name, ch_data in [
            ("B (Buildings)", b_norm),
            ("G (Entropy)", g_norm),
            ("R (Accessibility)", r_norm),
        ]:
            gdf_vals = gpd.GeoDataFrame(
                {"value": ch_data}, geometry=grid_utm.geometry.values
            )
            w = Queen.from_dataframe(gdf_vals)
            w.transform = "r"
            mi = Moran(gdf_vals["value"].values, w)
            p(f"  {ch_name:25s}: I={mi.I:.4f}, p={mi.p_sim:.4f}, z={mi.z_sim:.4f}")
    except Exception as e:
        p(f"  Moran's I failed: {e}")
    p()

    # Cross-channel correlations
    p("--- Cross-Channel Correlations ---")
    p(f"  B (Bldg) vs G (Ent):    {np.corrcoef(b_norm, g_norm)[0, 1]:.4f}")
    p(f"  B (Bldg) vs R (Access): {np.corrcoef(b_norm, r_norm)[0, 1]:.4f}")
    p(f"  G (Ent)  vs R (Access): {np.corrcoef(g_norm, r_norm)[0, 1]:.4f}")
    p()

    # Output
    p("--- Output Grid ---")
    p(f"Shape: {vitality_grid.shape}")
    p(f"Dtype: {vitality_grid.dtype}")
    if mask_grid.any():
        p(
            f"Value range: [{vitality_grid[mask_grid].min():.4f}, {vitality_grid[mask_grid].max():.4f}]"
        )
        p(f"Mean (masked): {vitality_grid[mask_grid].mean():.4f}")
    p()

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Report saved: {report_path}")
    for line in lines:
        log.info(line)
    return report


# ===========================================================================
# Step 12: Visualize channels
# ===========================================================================
def visualize_channels(vitality_grid, mask_grid):
    """Create heatmap visualizations for each channel and composite."""
    log.info("Creating channel heatmaps...")

    display = vitality_grid.copy()
    for c in range(3):
        display[:, :, c][~mask_grid] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    titles = [
        ("B: Building Footprint Coverage", "Blues"),
        ("G: POI Shannon Entropy", "Greens"),
        ("R: Gravity Accessibility (exp decay)", "Reds"),
    ]

    for idx, (title, cmap) in enumerate(titles):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(
            display[:, :, idx], cmap=cmap, aspect="equal", interpolation="nearest"
        )
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized [0,1]")

    ax = axes[1, 1]
    rgb = np.zeros((*vitality_grid.shape[:2], 3), dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = display[:, :, c]
    ax.imshow(rgb, aspect="equal", interpolation="nearest")
    ax.set_title("Composite: Built Environment (BGR)", fontsize=12)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "hcmc_channel_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Heatmaps saved: {REPORT_DIR / 'hcmc_channel_heatmaps.png'}")

    # Individual channel images
    names = ["buildings", "poi_entropy", "accessibility"]
    cmaps = ["Blues", "Greens", "Reds"]
    for c, (name, cmap) in enumerate(zip(names, cmaps)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(
            display[:, :, c], cmap=cmap, aspect="equal", interpolation="nearest"
        )
        ax.set_title(f"HCMC - {name.replace('_', ' ').title()}", fontsize=14)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Normalized [0,1]")
        plt.tight_layout()
        fig.savefig(REPORT_DIR / f"hcmc_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main_rework():
    """Reworked pipeline: produces 256x256 spatial raster images per 1km cell."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CELLS_256_DIR, exist_ok=True)

    t0 = time.time()

    # Step 1: Boundary
    log.info("=" * 50 + "\nStep 1: HCMC Boundary")
    boundary_gdf = get_hcmc_boundary()

    # Step 2: Grid (no boundary clipping for perfect tiling)
    log.info("=" * 50 + "\nStep 2: 1km Grid (full rectangular)")
    grid_gdf = build_1km_grid(boundary_gdf, rework=True)

    # Step 3: Buildings
    log.info("=" * 50 + "\nStep 3: Loading Building Footprints")
    buildings = extract_buildings(boundary_gdf)

    # Step 4: POIs
    log.info("=" * 50 + "\nStep 4: Loading POIs")
    pois = extract_pois(boundary_gdf)

    # Step 5: Street network
    log.info("=" * 50 + "\nStep 5: Loading Street Network")
    G = download_street_network(boundary_gdf)

    # Compute grid dimensions
    grid_utm = grid_gdf.to_crs(UTM_EPSG)
    bounds = grid_utm.total_bounds
    x_min = np.floor(bounds[0] / 1000) * 1000
    y_min = np.floor(bounds[1] / 1000) * 1000
    x_max = np.ceil(bounds[2] / 1000) * 1000
    y_max = np.ceil(bounds[3] / 1000) * 1000

    cx = grid_utm.geometry.centroid.x.values
    cy = grid_utm.geometry.centroid.y.values
    unique_x = np.unique(np.floor(cx / 1000) * 1000)
    unique_y = np.unique(np.floor(cy / 1000) * 1000)
    n_cols = len(unique_x)
    n_rows = len(unique_y)

    log.info(f"Grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} cells")
    log.info(f"Pixel resolution: {PIXEL_RES:.4f} m/px")
    log.info(f"Canvas size: {n_cols * CELL_PX} x {n_rows * CELL_PX} px")
    log.info(f"Extent: ({x_min}, {y_min}) to ({x_max}, {y_max}) UTM48N")

    # Step 6: B channel — Building footprint binary raster
    log.info("=" * 50 + "\nStep 6: B Channel (Building Raster)")
    b_canvas = rasterize_buildings(
        buildings, x_min, y_min, x_max, y_max, n_cols * CELL_PX, n_rows * CELL_PX
    )
    b_canvas = np.flipud(b_canvas).copy()
    log.info(
        f"  B canvas: shape={b_canvas.shape}, dtype={b_canvas.dtype}, nonzero={(b_canvas > 0).sum()}"
    )

    # Step 7: G channel — POI entropy density
    log.info("=" * 50 + "\nStep 7: G Channel (POI Entropy Density)")
    g_canvas = rasterize_poi_entropy(pois, grid_gdf, n_cols, n_rows)
    log.info(
        f"  G canvas: shape={g_canvas.shape}, dtype={g_canvas.dtype}, nonzero={(g_canvas > 0).sum()}"
    )

    # Step 8: R channel — Accessibility surface
    log.info("=" * 50 + "\nStep 8: R Channel (Accessibility Surface)")
    canvas_w = n_cols * CELL_PX
    canvas_h = n_rows * CELL_PX
    r_canvas = rasterize_accessibility(
        pois, G, x_min, y_min, x_max, y_max, canvas_w, canvas_h
    )
    log.info(
        f"  R canvas: shape={r_canvas.shape}, dtype={r_canvas.dtype}, nonzero={(r_canvas > 0).sum()}"
    )

    np.save(PROC_DIR / "b_canvas.npy", b_canvas)
    np.save(PROC_DIR / "g_canvas.npy", g_canvas)
    np.save(PROC_DIR / "r_canvas.npy", r_canvas)
    log.info("  Saved intermediate canvases as .npy files")

    # Step 9: Fuse and crop per-cell images
    log.info("=" * 50 + "\nStep 9: Creating Per-Cell 256x256 Images")
    n_saved = create_cell_images(b_canvas, g_canvas, r_canvas, grid_gdf, n_cols, n_rows)

    elapsed = time.time() - t0
    log.info(f"\nReworked pipeline complete! {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log.info(f"Output: {CELLS_256_DIR}/ ({n_saved} PNG files)")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    t0 = time.time()

    # Step 1: Boundary
    log.info("=" * 50 + "\nStep 1: HCMC Boundary")
    boundary_gdf = get_hcmc_boundary()

    # Step 2: Grid
    log.info("=" * 50 + "\nStep 2: 1km Grid")
    grid_gdf = build_1km_grid(boundary_gdf)

    # Step 3: B channel (buildings)
    log.info("=" * 50 + "\nStep 3: B Channel (Building Footprints)")
    buildings = extract_buildings(boundary_gdf)
    b_norm, b_raw = compute_building_channel(grid_gdf, buildings)

    # Step 4: POIs
    log.info("=" * 50 + "\nStep 4: OSM POI Extraction")
    pois = extract_pois(boundary_gdf)

    # Step 5: G channel (entropy)
    log.info("=" * 50 + "\nStep 5: G Channel (POI Entropy)")
    g_norm, g_raw = compute_poi_entropy_channel(grid_gdf, pois)

    # Step 6: Street network
    log.info("=" * 50 + "\nStep 6: OSM Street Network")
    G = download_street_network(boundary_gdf)

    # Step 7: R channel (accessibility)
    log.info("=" * 50 + "\nStep 7: R Channel (Accessibility)")
    r_norm, r_raw = compute_accessibility_channel(grid_gdf, pois, G)

    # Step 8: Assemble grid
    log.info("=" * 50 + "\nStep 8: Assembling Grid")
    vitality_grid, mask_grid, unique_y, unique_x = assemble_grid(
        grid_gdf, b_norm, g_norm, r_norm
    )

    grid_path = PROC_DIR / "hcmc_vitality_grid.npy"
    np.save(grid_path, vitality_grid)
    log.info(f"Saved: {grid_path} (shape: {vitality_grid.shape})")

    # Step 9: Crop square cells
    log.info("=" * 50 + "\nStep 9: Cropping Square Cells")
    square_cells, labels_df = crop_square_cells(grid_gdf, b_norm, g_norm, r_norm)

    # Step 10: Fuse RGB
    log.info("=" * 50 + "\nStep 10: Fusing RGB")
    fuse_rgb(grid_gdf, labels_df, b_norm, g_norm, r_norm)

    # Step 11: Report
    log.info("=" * 50 + "\nStep 11: Quality Report")
    generate_quality_report(
        grid_gdf,
        buildings,
        pois,
        b_raw,
        g_raw,
        r_raw,
        b_norm,
        g_norm,
        r_norm,
        vitality_grid,
        mask_grid,
    )

    # Step 12: Visualize
    log.info("=" * 50 + "\nStep 12: Channel Heatmaps")
    visualize_channels(vitality_grid, mask_grid)

    elapsed = time.time() - t0
    log.info(f"\nPipeline complete! {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log.info(f"Output: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCMC Built Environment Data Pipeline")
    parser.add_argument(
        "--rework",
        action="store_true",
        help="Produce 256x256 spatial raster images per 1km cell",
    )
    args = parser.parse_args()

    if args.rework:
        main_rework()
    else:
        main()
