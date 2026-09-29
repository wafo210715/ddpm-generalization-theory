"""Issue #2, round 1 (23 Sep 2026): what the footprint layer can and cannot say.

Three questions, all on the 3 010 cells (3 020 urban tiles minus Con Dao) unless
stated:

  shape     footprint size and width by density band.  Overture is almost all
            machine-learned roofprints here, and a tube house is 4-6 m wide, so
            the width of the narrow side says whether houses come out one by one
            or fused into a row.
  check     Overture footprint cover against WorldCover's built-up class, an
            independent Sentinel product, per tile.  Tiles WorldCover calls
            built but Overture leaves bare are candidate omissions.
  pixel     on the 40 probe tiles (seed 0, as report_descriptor_probe.py):
            what the pixel-centre rule keeps at 256, 512 and 1024 px, against
            the vector truth -- buildings lost, separate buildings fused into
            one blob, and the width of the gaps between buildings.
"""
import sys, pathlib, json
import numpy as np, pandas as pd, geopandas as gpd, shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "data/hcmc/raw"
REP = REPO / "data/hcmc/processed/reports"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else REP / "building_footprints.json")
UTM = 3405
CON_DAO = set("15_26086_15589 15_26086_15590 15_26086_15591 15_26087_15589 15_26087_15590 "
              "15_26088_15588 15_26088_15589 15_26088_15590 15_26089_15586 15_26089_15589".split())
BANDS = [(100, 250), (250, 1000), (1000, 2000), (2000, 10**9)]
RES = [256, 512, 1024]

src = pd.read_parquet(REP / "tile_sources_urban2025.parquet")
cells = src[~src.cell_id.isin(CON_DAO)].reset_index(drop=True)
assert len(cells) == 3010
tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
tiles = tiles[tiles.cell_id.isin(cells.cell_id)].merge(cells[["cell_id", "buildings", "builtup_frac"]])
tile_area = tiles.geometry.area.median()

print("loading buildings …", flush=True)
b = gpd.read_parquet(RAW / "overture_tu/buildings.parquet").to_crs(UTM)
b["area"] = b.geometry.area
attrs = {c: float(b[c].notna().mean()) for c in ["subtype", "class", "height", "num_floors"]}
rect = shapely.minimum_rotated_rectangle(b.geometry.values)
xy, idx = shapely.get_coordinates(rect, return_index=True)
first = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
p0, p1, p2 = xy[first], xy[first + 1], xy[first + 2]
s1, s2 = np.hypot(*(p1 - p0).T), np.hypot(*(p2 - p1).T)
b["short"], b["long"] = np.minimum(s1, s2), np.maximum(s1, s2)

cent = gpd.GeoDataFrame(b[["area", "short", "long"]], geometry=b.geometry.centroid, crs=UTM)
j = gpd.sjoin(cent, tiles[["cell_id", "buildings", "geometry"]], predicate="within")
print(f"  {len(b):,} buildings, {len(j):,} with centroid in the 3 010 cells", flush=True)

def shape_stats(d):
    a = d["area"].values
    return {"n": int(len(d)), "median_area_m2": round(float(np.median(a)), 1),
            "share_under_22m2": round(float((a < 22).mean()), 3),     # one 4.69 m pixel
            "share_over_500m2": round(float((a > 500).mean()), 3),
            "area_share_over_500m2": round(float(a[a > 500].sum() / a.sum()), 3),
            "median_short_side_m": round(float(np.median(d.short)), 1),
            "share_short_side_le_6m": round(float((d.short <= 6).mean()), 3),
            "median_long_short_ratio": round(float(np.median(d.long / np.maximum(d.short, .01))), 2)}

shape = {"all": shape_stats(j)}
for lo, hi in BANDS:
    shape[f"{lo}-{hi if hi < 10**9 else ''}"] = shape_stats(j[(j.buildings >= lo) & (j.buildings < hi)])

cov = j.groupby("cell_id")["area"].sum() / tile_area
tiles["cover"] = tiles.cell_id.map(cov).fillna(0)
rho = float(tiles[["cover", "builtup_frac"]].corr(method="spearman").iloc[0, 1])
tiles["ratio"] = tiles.cover / tiles.builtup_frac.clip(lower=1e-3)
built = tiles[tiles.builtup_frac >= 0.3]
low = built[built.ratio < 0.15].sort_values("ratio")
check = {"spearman_cover_vs_worldcover_builtup": round(rho, 3),
         "median_cover": round(float(tiles.cover.median()), 3),
         "median_builtup": round(float(tiles.builtup_frac.median()), 3),
         "median_ratio_where_builtup_ge_0.3": round(float(built.ratio.median()), 3),
         "tiles_builtup_ge_0.3": int(len(built)),
         "tiles_builtup_ge_0.3_with_cover_under_0.15_of_it": int(len(low)),
         "lowest": low[["cell_id", "buildings", "cover", "builtup_frac"]].head(12).round(3).to_dict("records")}

# --- pixel: the 40 probe tiles -------------------------------------------------
rng = np.random.default_rng(0)
samp = cells.iloc[rng.choice(len(cells), 40, replace=False)].cell_id
sidx = b.sindex
pix = {r: {"lost": 0, "fused": 0, "blobs_raster": 0, "cover": []} for r in RES}
tot_bld, tot_blobs, gaps, true_cover = 0, 0, [], []
for cid in samp:
    t = tiles.loc[tiles.cell_id == cid].geometry.iloc[0]
    minx, miny, maxx, maxy = t.bounds
    sub = b.iloc[sidx.query(t, predicate="intersects")]
    sub = sub[sub.geometry.centroid.within(t)]           # buildings that belong to this tile
    geoms = shapely.intersection(sub.geometry.values, t)
    blobs = shapely.get_parts(shapely.union_all(geoms))  # touching houses are one blob in the vector too
    blobs = blobs[shapely.area(blobs) > 0]
    tot_bld += len(geoms); tot_blobs += len(blobs)
    true_cover.append(shapely.area(shapely.union_all(geoms)) / t.area)
    tree = shapely.STRtree(blobs)
    _, d = tree.query_nearest(blobs, exclusive=True, return_distance=True, all_matches=False)
    gaps.append(d)
    for r in RES:
        tr = from_origin(minx, maxy, (maxx - minx) / r, (maxy - miny) / r)
        ids = rasterize(zip(geoms, range(1, len(geoms) + 1)), out_shape=(r, r), transform=tr, fill=0, dtype="int32")
        bid = rasterize(zip(blobs, range(1, len(blobs) + 1)), out_shape=(r, r), transform=tr, fill=0, dtype="int32")
        pix[r]["lost"] += len(geoms) - (len(np.unique(ids)) - 1)
        lab, n = ndimage.label(ids > 0, structure=np.ones((3, 3)))
        pix[r]["blobs_raster"] += n
        pairs = pd.DataFrame({"lab": lab[bid > 0], "blob": bid[bid > 0]}).drop_duplicates()
        per_lab = pairs.groupby("lab").blob.nunique()
        pix[r]["fused"] += int(pairs[pairs.lab.isin(per_lab[per_lab > 1].index)].blob.nunique())
        pix[r]["cover"].append(float((ids > 0).mean()))
    print(f"  {cid}: {len(geoms)} buildings, {len(blobs)} blobs", flush=True)

gaps = np.concatenate(gaps)
pixel = {"tiles": 40, "buildings": tot_bld, "vector_blobs": tot_blobs,
         "true_cover_mean": round(float(np.mean(true_cover)), 4),
         "gap_to_nearest_blob_m": {"median": round(float(np.median(gaps)), 2),
                                   "share_under_1.17": round(float((gaps < 1.17).mean()), 3),
                                   "share_under_2.35": round(float((gaps < 2.35).mean()), 3),
                                   "share_under_4.69": round(float((gaps < 4.69).mean()), 3)},
         "by_res": {r: {"unit_m": round(1201 / r, 2),
                        "buildings_lost": round(v["lost"] / tot_bld, 3),
                        "blobs_fused_into_another": round(v["fused"] / tot_blobs, 3),
                        "raster_blobs_per_vector_blob": round(v["blobs_raster"] / tot_blobs, 3),
                        "cover_mean": round(float(np.mean(v["cover"])), 4)} for r, v in pix.items()}}

res = {"overture_release": "2026-08-19.0", "attribute_fill": attrs,
       "shape": shape, "check": check, "pixel": pixel}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
