#!/usr/bin/env python3
"""
HCMC Full Municipality Boundary & 1km Fishnet Grid Generator

Fetches the pre-merger HCMC municipality boundary from Overpass API
(OSM Relation 1973756 as of 2024, before the 2025 Binh Duong/Ba Ria-Vung Tau
merger), generates a 1km fishnet grid covering the full municipality extent,
and saves both as GeoPackage files.

Context: Vietnam merged HCMC with Binh Duong and Ba Ria-Vung Tau provinces
in April 2025 (Resolution 60-NQ/TW). The current OSM boundary (36,423 km^2)
now covers the merged megacity. We use the historical pre-merger boundary
(~2,400 km^2) which corresponds to the original HCMC municipality.

Output:
  data/hcmc/raw/hcmc_boundary_full.gpkg  -- single polygon, EPSG:4326
  data/hcmc/raw/hcmc_grid_1km.gpkg       -- fishnet cells, EPSG:4326
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "hcmc" / "raw"

UTM_EPSG = 32648  # UTM zone 48N for HCMC
GRID_RES_M = 1000  # 1 km grid cells

# OSM Relation 1973756 = Ho Chi Minh City (admin_level=4, ISO3166-2=VN-SG)
# We use [date:"2024-06-01"] to get the pre-merger boundary.
HCMC_OSM_RELATION_ID = 1973756
HCMC_HISTORICAL_DATE = "2024-06-01T00:00:00Z"

# Overpass rejects the default python-requests User-Agent with 406 Not
# Acceptable, so every query has to identify itself.  Contact details are part
# of the Overpass usage policy, not decoration.
OSM_HTTP_HEADERS = {
    "User-Agent": (
        "ddpm-generalization-theory/1.0 "
        "(urban vitality research; contact wafo210715@gmail.com)"
    )
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===========================================================================
# Ring builder: connect way segments into closed polygon rings
# ===========================================================================
def _build_rings(segments):
    """Connect line segments end-to-end into closed rings.

    Each segment is a list of (lon, lat) tuples. Returns a list of
    closed rings (first point == last point, >= 4 points).
    """
    rings = []
    remaining = list(segments)

    while remaining:
        ring = list(remaining.pop(0))
        changed = True
        while changed:
            changed = False
            for i, seg in enumerate(remaining):
                if not seg:
                    continue
                if ring[-1] == seg[0]:
                    ring.extend(seg[1:])
                    remaining[i] = []
                    changed = True
                elif ring[-1] == seg[-1]:
                    ring.extend(reversed(seg[:-1]))
                    remaining[i] = []
                    changed = True
                elif ring[0] == seg[-1]:
                    ring = seg[:-1] + ring
                    remaining[i] = []
                    changed = True
                elif ring[0] == seg[0]:
                    ring = list(reversed(seg[1:])) + ring
                    remaining[i] = []
                    changed = True
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
        remaining = [s for s in remaining if s]
    return rings


# ===========================================================================
# Step 1: Get HCMC full municipality boundary
# ===========================================================================
def get_hcmc_boundary():
    """Get pre-merger HCMC municipality boundary from Overpass API.

    Uses the historical version of OSM Relation 1973756 (date=2024-06-01)
    to get the boundary BEFORE Vietnam's 2025 administrative merger that
    absorbed Binh Duong and Ba Ria-Vung Tau into HCMC.

    The pre-merger HCMC covers ~2,400 km^2 (official: 2,095 km^2, OSM
    boundary is slightly larger due to maritime/coastal areas in Can Gio).
    """
    boundary_path = RAW_DIR / "hcmc_boundary_full.gpkg"
    if boundary_path.exists():
        log.info("Loading cached HCMC full boundary")
        return gpd.read_file(boundary_path)

    log.info(
        f"Fetching pre-merger HCMC boundary from Overpass "
        f"(relation {HCMC_OSM_RELATION_ID}, date={HCMC_HISTORICAL_DATE})..."
    )

    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:120][date:"{HCMC_HISTORICAL_DATE}"];
    relation({HCMC_OSM_RELATION_ID});
    out geom;
    """

    for attempt in range(3):
        try:
            r = requests.post(url, data={"data": query},
                              headers=OSM_HTTP_HEADERS, timeout=180)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                import time

                time.sleep(30 * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to fetch boundary after 3 attempts")

    # Parse relation and build polygon from outer way members
    rel = None
    for elem in data.get("elements", []):
        if elem.get("type") == "relation":
            rel = elem
            break

    if not rel:
        raise RuntimeError("No relation found in Overpass response")

    log.info(
        f"  Relation: {rel['id']}, name={rel.get('tags', {}).get('name:en', 'N/A')}"
    )

    # Extract outer way segments
    outer_segments = []
    for member in rel.get("members", []):
        if member.get("type") == "way" and member.get("role") == "outer":
            geom = member.get("geometry", [])
            if len(geom) >= 2:
                coords = [(n["lon"], n["lat"]) for n in geom]
                outer_segments.append(coords)

    log.info(f"  Outer segments: {len(outer_segments)}")

    # Build closed rings
    rings = _build_rings(outer_segments)
    log.info(f"  Closed rings: {len(rings)}")

    # Build polygons
    polygons = []
    for ring in rings:
        try:
            poly = Polygon(ring)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
            elif not poly.is_valid:
                poly = poly.buffer(0)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
        except Exception:
            pass

    if not polygons:
        raise RuntimeError("Failed to build valid polygons from OSM data")

    # Use MultiPolygon if there are multiple parts (islands etc.)
    if len(polygons) == 1:
        geom = polygons[0]
    else:
        geom = MultiPolygon(polygons)

    gdf = gpd.GeoDataFrame(
        {"name": ["Ho Chi Minh City (pre-merger 2024)"]},
        geometry=[geom],
        crs="EPSG:4326",
    )

    os.makedirs(RAW_DIR, exist_ok=True)
    gdf.to_file(boundary_path, driver="GPKG")

    area_km2 = gdf.to_crs(UTM_EPSG).area.iloc[0] / 1e6
    log.info(f"Boundary area: {area_km2:.1f} km^2")
    log.info(f"Saved to: {boundary_path}")
    return gdf


# ===========================================================================
# Step 2: Generate 1km fishnet grid
# ===========================================================================
def build_fishnet_grid(boundary_gdf):
    """Generate 1km fishnet grid covering the full HCMC municipality extent.

    Grid is aligned to integer UTM coordinates divisible by 1000m.
    The fishnet extends beyond the boundary (NOT clipped) to ensure
    complete spatial coverage for data collection.
    """
    grid_path = RAW_DIR / "hcmc_grid_1km.gpkg"
    if grid_path.exists():
        log.info("Loading cached 1km fishnet grid")
        return gpd.read_file(grid_path)

    log.info("Building 1km fishnet grid...")

    # Work in UTM for metric grid alignment
    boundary_utm = boundary_gdf.to_crs(UTM_EPSG)
    bounds = boundary_utm.total_bounds  # [xmin, ymin, xmax, ymax]

    # Snap to 1000m multiples
    x_min = np.floor(bounds[0] / GRID_RES_M) * GRID_RES_M
    y_min = np.floor(bounds[1] / GRID_RES_M) * GRID_RES_M
    x_max = np.ceil(bounds[2] / GRID_RES_M) * GRID_RES_M
    y_max = np.ceil(bounds[3] / GRID_RES_M) * GRID_RES_M

    log.info(f"Boundary UTM bounds: {bounds}")
    log.info(f"Snapped grid extent: ({x_min}, {y_min}) to ({x_max}, {y_max})")

    # Generate fishnet cells
    xs = np.arange(x_min, x_max, GRID_RES_M)
    ys = np.arange(y_min, y_max, GRID_RES_M)
    n_cols = len(xs)
    n_rows = len(ys)

    cells = []
    cell_ids = []
    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            cell = box(x, y, x + GRID_RES_M, y + GRID_RES_M)
            cells.append(cell)
            cell_ids.append(f"{row_idx:03d}_{col_idx:03d}")

    grid_gdf = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "geometry": cells},
        crs=f"EPSG:{UTM_EPSG}",
    )

    # Convert to EPSG:4326 for output (coordinates derived from UTM)
    grid_gdf = grid_gdf.to_crs("EPSG:4326")

    os.makedirs(RAW_DIR, exist_ok=True)
    grid_gdf.to_file(grid_path, driver="GPKG")

    log.info(f"Grid: {len(grid_gdf)} cells ({n_rows} rows x {n_cols} cols)")
    log.info(f"Saved to: {grid_path}")
    return grid_gdf


# ===========================================================================
# Step 3: Verification
# ===========================================================================
def verify_outputs():
    """Run QA verification on outputs."""
    log.info("Running verification...")

    # Verify boundary
    boundary_path = RAW_DIR / "hcmc_boundary_full.gpkg"
    assert boundary_path.exists(), f"Boundary file not found: {boundary_path}"

    gdf = gpd.read_file(boundary_path)
    assert len(gdf) == 1, f"Expected 1 polygon, got {len(gdf)}"

    area = gdf.to_crs(UTM_EPSG).area.iloc[0] / 1e6
    log.info(f"Boundary area: {area:.1f} km^2")
    assert 1500 < area < 5000, (
        f"Area {area:.1f} km^2 outside expected range (1500-5000)"
    )

    # Verify grid
    grid_path = RAW_DIR / "hcmc_grid_1km.gpkg"
    assert grid_path.exists(), f"Grid file not found: {grid_path}"

    grid = gpd.read_file(grid_path).to_crs(UTM_EPSG)
    bounds = grid.total_bounds
    xmin, ymin = bounds[0], bounds[1]

    # Check alignment (round to handle floating-point from CRS conversion)
    xmin_r = round(xmin)
    ymin_r = round(ymin)
    assert xmin_r % 1000 == 0, f"xmin {xmin_r} not divisible by 1000"
    assert ymin_r % 1000 == 0, f"ymin {ymin_r} not divisible by 1000"
    log.info(f"Grid aligned: xmin={xmin_r}, ymin={ymin_r}, {len(grid)} cells")

    # Check cell_id format
    sample_id = grid["cell_id"].iloc[0]
    assert "_" in sample_id, f"cell_id format unexpected: {sample_id}"
    parts = sample_id.split("_")
    assert len(parts) == 2, f"cell_id should be row_col: {sample_id}"

    log.info("All verifications passed!")
    return True


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Step 1: Get boundary
    log.info("=" * 60)
    log.info("Step 1: HCMC Full Municipality Boundary")
    log.info("=" * 60)
    boundary = get_hcmc_boundary()

    # Step 2: Build grid
    log.info("=" * 60)
    log.info("Step 2: 1km Fishnet Grid")
    log.info("=" * 60)
    grid = build_fishnet_grid(boundary)

    # Step 3: Verify
    log.info("=" * 60)
    log.info("Step 3: Verification")
    log.info("=" * 60)
    verify_outputs()

    log.info("=" * 60)
    log.info("DONE -- boundary and grid ready")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
