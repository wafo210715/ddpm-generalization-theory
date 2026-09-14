"""Compute the candidate channels city-wide, per tile, so the data can be looked at.

This is the B10 checkpoint's input: what each indicator actually looks like once
the city is cut into zoom-15 tiles.  Two passes.

1. A city-wide 20 m raster of the polygon and line layers, reduced to one mean
   per tile per channel.  At 20 m a tile is 60x60 cells, which is enough for a
   tile-level summary and cheap enough to run over 1.49 M buildings in one go.
   Pixel-centre sampling on a regular grid is an unbiased estimator of area
   fraction once averaged over a tile.
2. WorldCover read directly (10 m, already a raster) and places counted by join.

Writes data/hcmc/processed/preview/tile_channels.parquet
"""
import pathlib, json
import numpy as np, geopandas as gpd, pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview"
OUT.mkdir(parents=True, exist_ok=True)
UTM, RES = 3405, 20.0

print("loading …", flush=True)
tiles = gpd.read_file(RAW / "hcmc_tiles_z15.gpkg").to_crs(UTM)
blds = gpd.read_parquet(RAW / "overture/buildings.parquet").to_crs(UTM)
hts = gpd.read_parquet(RAW / "heights_3dglobfp.parquet").to_crs(UTM)
roads = gpd.read_file(RAW / "osm/roads.gpkg").to_crs(UTM)
places = gpd.read_parquet(RAW / "overture/places.parquet").to_crs(UTM)
print(f"  buildings {len(blds):,} | heights {len(hts):,} | roads {len(roads):,} | places {len(places):,}", flush=True)

minx, miny, maxx, maxy = tiles.total_bounds
minx, miny = np.floor(minx / RES) * RES, np.floor(miny / RES) * RES
W = int(np.ceil((maxx - minx) / RES)); H = int(np.ceil((maxy - miny) / RES))
tr = from_origin(minx, miny + H * RES, RES, RES)
print(f"  city grid {W} x {H} at {RES:.0f} m = {W*H/1e6:.1f} M cells", flush=True)

def burn(gdf, values=None, dtype="float32"):
    shapes = zip(gdf.geometry, (values if values is not None else np.ones(len(gdf))))
    return rasterize(shapes, out_shape=(H, W), transform=tr, fill=0,
                     dtype=dtype, merge_alg=rasterio.enums.MergeAlg.replace)

print("burning buildings …", flush=True)
r_build = burn(blds)
print("burning heights …", flush=True)
r_height = burn(hts, hts["Height"].astype("float32").values)
print("burning roads …", flush=True)
r_road = burn(roads)

# tile index -> slice of the city grid
tiles["col0"] = ((tiles.bounds.minx - minx) / RES).round().astype(int)
tiles["row0"] = ((miny + H * RES - tiles.bounds.maxy) / RES).round().astype(int)
NC = int(round(1201.0 / RES))            # cells across one tile

# places per tile
pj = gpd.sjoin(places[["geometry"]], tiles[["cell_id", "geometry"]], predicate="within")
pcount = pj.groupby("cell_id").size()
# a POI kernel field needs the point tree
ptree = cKDTree(np.c_[places.geometry.x.values, places.geometry.y.values])

print("reducing per tile …", flush=True)
wc_path = REPO / "data/external/worldcover/ESA_WorldCover_10m_2021_v200_N09E105_Map.tif"
wc = rasterio.open(wc_path)
GREEN = {10, 20, 30, 40, 95}     # tree, shrub, grass, crop, mangrove
rows = []
for t in tiles.itertuples():
    r0, c0 = t.row0, t.col0
    sl = (slice(max(r0,0), min(r0+NC,H)), slice(max(c0,0), min(c0+NC,W)))
    b = r_build[sl]; hgt = r_height[sl]; rd = r_road[sl]
    if b.size == 0:
        continue
    cov = float(b.mean())
    occupied = hgt > 0
    rec = {
        "cell_id": t.cell_id, "x": t.x, "y": t.y,
        "build": cov,
        "height": float(hgt[occupied].mean()) if occupied.any() else 0.0,
        "far": float(cov * (hgt[occupied].mean() / 3.0)) if occupied.any() else 0.0,
        "road": float(rd.mean()),
        "poi": float(pcount.get(t.cell_id, 0)) / 1.443,       # per km2
    }
    # WorldCover for this tile, read in its own CRS
    wb = transform_bounds(UTM, wc.crs, *t.geometry.bounds)
    win = rasterio.windows.from_bounds(*wb, transform=wc.transform)
    try:
        a = wc.read(1, window=win, boundless=True, fill_value=0)
        rec["green"] = float(np.isin(a, list(GREEN)).mean())
        rec["water"] = float((a == 80).mean())
    except Exception:
        rec["green"] = rec["water"] = float("nan")
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_parquet(OUT / "tile_channels.parquet")
print(f"\nwrote {len(df):,} tiles -> tile_channels.parquet\n", flush=True)
print(df[["build","height","far","road","poi","green","water"]].describe().round(3).to_string())
np.save(OUT / "_city_build.npy", r_build.astype("uint8"))
json.dump({"minx":minx,"miny":miny,"W":W,"H":H,"res":RES}, open(OUT/"_city_grid.json","w"))
