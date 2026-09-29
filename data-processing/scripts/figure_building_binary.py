"""Issue #2, round 1: one dense tile, footprints as vectors and as the pixel-centre raster.

Crops a 150 m window so individual houses are visible, and draws the same window
at 256, 512 and 1024 px per tile.  Writes discussion/layer_cards_fig/building_binary.png.
"""
import pathlib
import numpy as np, geopandas as gpd, shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_origin

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / "data-processing/data/hcmc/raw"
OUT = REPO / "discussion/layer_cards_fig/building_binary.png"
OUT.parent.mkdir(exist_ok=True)
UTM, CELL, WIN = 3405, "15_26084_15389", 150.0     # densest of the 40 probe tiles, 4 321 buildings

tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM).set_index("cell_id")
t = tiles.loc[CELL].geometry
minx, miny, maxx, maxy = t.bounds
b = gpd.read_parquet(RAW / "overture_tu/buildings.parquet", bbox=None).to_crs(UTM)
b = b.iloc[b.sindex.query(t, predicate="intersects")]
geoms = shapely.intersection(b.geometry.values, t)

cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
wx0, wy0, wx1, wy1 = cx - WIN / 2, cy - WIN / 2, cx + WIN / 2, cy + WIN / 2

fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
gpd.GeoSeries(geoms).plot(ax=ax[0], color="#333", edgecolor="#e85d04", linewidth=0.4)
ax[0].set_xlim(wx0, wx1); ax[0].set_ylim(wy0, wy1)
ax[0].set_title(f"Overture outlines ({len(geoms):,} in the tile)")
for a, r in zip(ax[1:], [256, 512, 1024]):
    tr = from_origin(minx, maxy, (maxx - minx) / r, (maxy - miny) / r)
    m = rasterize(((g, 1) for g in geoms), out_shape=(r, r), transform=tr, fill=0, dtype="uint8")
    a.imshow(m, cmap="gray_r", extent=(minx, maxx, miny, maxy), interpolation="nearest", vmin=0, vmax=1)
    a.set_xlim(wx0, wx1); a.set_ylim(wy0, wy1)
    a.set_title(f"{r} px per tile · {1201 / r:.2f} m unit")
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
fig.suptitle(f"building_binary · tile {CELL} · central {WIN:.0f} m window", y=0.99)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
