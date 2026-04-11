"""Diagnostic: test each rasterize function independently to find where data is lost."""

import sys, os

sys.path.insert(0, r"C:\sd\ddpm-generalization-theory\data-processing\scripts")

import numpy as np
import geopandas as gpd
from pathlib import Path

# Import pipeline config
ROOT = Path(r"C:\sd\ddpm-generalization-theory\data-processing")
DATA_DIR = ROOT / "data" / "hcmc"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
UTM_EPSG = 32648
GRID_RES_KM = 1.0
CELL_PX = 256
PIXEL_RES = GRID_RES_KM * 1000 / CELL_PX

# Load raw data
print("=== Loading raw data ===")
buildings = gpd.read_file(RAW_DIR / "hcmc_buildings.gpkg")
print(f"Buildings: {len(buildings)} rows, CRS={buildings.crs}")

pois = gpd.read_file(RAW_DIR / "hcmc_pois_all.gpkg")
print(f"POIs: {len(pois)} rows, CRS={pois.crs}, cols={pois.columns.tolist()[:5]}")

grid = gpd.read_file(RAW_DIR / "hcmc_1km_grid.gpkg")
print(f"Grid: {len(grid)} cells, CRS={grid.crs}")

# Check building extents
print("\n=== Building extent check ===")
b_utm = buildings.to_crs(UTM_EPSG)
print(f"Building bounds (UTM): {b_utm.total_bounds}")
print(f"Building bounds (orig): {buildings.total_bounds}")

g_utm = grid.to_crs(UTM_EPSG)
print(f"Grid bounds (UTM): {g_utm.total_bounds}")

# Check POI extents
print("\n=== POI extent check ===")
p_utm = pois.to_crs(UTM_EPSG)
print(f"POI bounds (UTM): {p_utm.total_bounds}")

# Check if buildings overlap with grid
from shapely.geometry import box

gb = g_utm.total_bounds
grid_box = box(gb[0], gb[1], gb[2], gb[3])
overlap = b_utm[b_utm.geometry.intersects(grid_box)]
print(f"Buildings overlapping grid bbox: {len(overlap)} / {len(b_utm)}")

# Check if POIs overlap with grid
poverlap = p_utm[p_utm.geometry.intersects(grid_box)]
print(f"POIs overlapping grid bbox: {len(poverlap)} / {len(p_utm)}")

# Test rasterize_buildings logic
print("\n=== Test rasterize_buildings logic ===")
from rasterio.transform import from_bounds
from rasterio.features import rasterize

x_min = np.floor(gb[0] / 1000) * 1000
y_min = np.floor(gb[1] / 1000) * 1000
x_max = np.ceil(gb[2] / 1000) * 1000
y_max = np.ceil(gb[3] / 1000) * 1000

cx = g_utm.geometry.centroid.x.values
cy = g_utm.geometry.centroid.y.values
unique_x = np.unique(np.floor(cx / 1000) * 1000)
unique_y = np.unique(np.floor(cy / 1000) * 1000)
n_cols = len(unique_x)
n_rows = len(unique_y)

canvas_w = n_cols * CELL_PX
canvas_h = n_rows * CELL_PX

print(f"Grid extent: ({x_min}, {y_min}) to ({x_max}, {y_max})")
print(f"Grid dims: {n_rows} rows x {n_cols} cols")
print(f"Canvas dims: {canvas_h} x {canvas_w} pixels")
print(f"Pixel res: {PIXEL_RES:.4f} m/px")

transform = from_bounds(x_min, y_min, x_max, y_max, canvas_w, canvas_h)
print(f"Transform: {transform}")

# Test with a few buildings
test_buildings = overlap.head(100)
shapes = ((geom, 255) for geom in test_buildings.geometry)
canvas = rasterize(
    shapes=shapes,
    out_shape=(canvas_h, canvas_w),
    transform=transform,
    fill=0,
    dtype=np.uint8,
    all_touched=True,
)
print(
    f"Test canvas: shape={canvas.shape}, nonzero={np.count_nonzero(canvas)}, max={canvas.max()}"
)

# Test POI entropy for one cell
print("\n=== Test POI entropy for one cell ===")
from scipy.stats import entropy as scipy_entropy

test_cell = g_utm.iloc[0]
centroid = test_cell.geometry.centroid
cell_x0 = np.floor(centroid.x / 1000) * 1000
cell_y0 = np.floor(centroid.y / 1000) * 1000
half_cell = 500
sub_size = 50  # 1000/20

poi_x = p_utm.geometry.x.values
poi_y = p_utm.geometry.y.values
poi_cats = p_utm["poi_category"].values

print(f"Test cell centroid: ({centroid.x:.1f}, {centroid.y:.1f})")
print(
    f"Cell bounds: ({cell_x0 - half_cell}, {cell_y0 - half_cell}) to ({cell_x0 + half_cell}, {cell_y0 + half_cell})"
)

pois_in_cell = (
    (poi_x >= cell_x0 - half_cell)
    & (poi_x < cell_x0 + half_cell)
    & (poi_y >= cell_y0 - half_cell)
    & (poi_y < cell_y0 + half_cell)
)
print(f"POIs in this cell: {pois_in_cell.sum()} / {len(pois_utm)}")

if pois_in_cell.sum() > 0:
    print(f"Categories in cell: {np.unique(poi_cats[pois_in_cell])[:10]}")
else:
    # Find a cell that HAS POIs
    print("This cell has no POIs. Finding one that does...")
    for idx in range(len(g_utm)):
        c = g_utm.geometry.centroid.iloc[idx]
        cx0 = np.floor(c.x / 1000) * 1000
        cy0 = np.floor(c.y / 1000) * 1000
        mask = (
            (poi_x >= cx0 - half_cell)
            & (poi_x < cx0 + half_cell)
            & (poi_y >= cy0 - half_cell)
            & (poi_y < cy0 + half_cell)
        )
        if mask.sum() > 5:
            print(f"Cell {idx}: ({c.x:.1f}, {c.y:.1f}) has {mask.sum()} POIs")
            cats = np.unique(poi_cats[mask])
            print(f"  Categories: {cats[:10]}")
            break
    else:
        print("WARNING: No cell has more than 5 POIs!")

print("\n=== Diagnostic complete ===")
