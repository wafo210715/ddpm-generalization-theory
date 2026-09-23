"""A1 per-tile source presence survey, 15 Sep 2026.

Note: junctions here still use the endpoint-only rule that A1 found to undercount
twelvefold; build_layer_preview.prepare() carries the corrected degree rule.
Per-tile source presence on the full 2025 urban set (>=100 Overture buildings, intersects).
Answers, per layer, 'is the source there at all in this cell' -- the absence question A1 has to decide."""
import sys, pathlib, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask as rmask
from scipy.spatial import cKDTree
from shapely.geometry import box
sys.path.insert(0, "/home/wafo/Documents/repo/ddpm-generalization-theory/data-processing/scripts")
from build_layer_preview import POI_CLASSES, vertices
R = pathlib.Path("/home/wafo/Documents/repo/ddpm-generalization-theory/data-processing/data")
RAW = R/"hcmc/raw"; OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/wafo/Documents/repo/ddpm-generalization-theory/data-processing/data/hcmc/processed/reports/tile_sources_urban2025.parquet"); UTM = 3405
tiles = gpd.read_file(RAW/"hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
blds = gpd.read_parquet(RAW/"overture_tu/buildings.parquet").to_crs(UTM)
cnt = gpd.sjoin(blds[["geometry"]], tiles[["cell_id","geometry"]], predicate="intersects").groupby("cell_id").size()
tiles["buildings"] = tiles.cell_id.map(cnt).fillna(0).astype(int)
hts = gpd.read_parquet(RAW/"heights_3dglobfp_tu.parquet").to_crs(UTM)
hcnt = gpd.sjoin(hts[["geometry"]], tiles[["cell_id","geometry"]], predicate="intersects").groupby("cell_id").size()
tiles["heights"] = tiles.cell_id.map(hcnt).fillna(0).astype(int)
places = gpd.read_parquet(RAW/"overture_tu/places.parquet").to_crs(UTM)
cat = places.category.astype(str); cls = pd.Series("other", index=places.index)
for n, pat in reversed(POI_CLASSES): cls[cat.str.contains(pat, regex=True, na=False)] = n
places["cls"] = cls
pj = gpd.sjoin(places[["geometry","cls"]], tiles[["cell_id","geometry"]], predicate="within")
tiles["places"] = tiles.cell_id.map(pj.groupby("cell_id").size()).fillna(0).astype(int)
tiles["places_cls"] = tiles.cell_id.map(pj[pj.cls!="other"].groupby("cell_id").size()).fillna(0).astype(int)
tiles["n_classes"] = tiles.cell_id.map(pj[pj.cls!="other"].groupby("cell_id").cls.nunique()).fillna(0).astype(int)
roads = gpd.read_file(RAW/"osm_tu/roads.gpkg").to_crs(UTM)
rj = gpd.sjoin(roads[["geometry"]], tiles[["cell_id","geometry"]], predicate="intersects")
tiles["road_ways"] = tiles.cell_id.map(rj.groupby("cell_id").size()).fillna(0).astype(int)
from collections import Counter
ends = Counter()
for g in roads.geometry:
    for p in (g.geoms if g.geom_type.startswith("Multi") else [g]):
        c = list(p.coords)
        for e in (c[0], c[-1]): ends[(round(e[0],1), round(e[1],1))] += 1
j = np.array([k for k, v in ends.items() if v >= 3]); jt = cKDTree(j)
stops = gpd.read_file(RAW/"osm_tu/transit_stops.gpkg").to_crs(UTM)
st = cKDTree(np.c_[stops.geometry.x, stops.geometry.y])
water = gpd.read_file(RAW/"osm_tu/water_areas.gpkg").to_crs(UTM)
green = gpd.read_file(RAW/"osm_tu/green_areas.gpkg").to_crs(UTM)
parks = green[green["leisure"].isin(["park","garden"])] if "leisure" in green else green
print("parks", len(parks), "of green", len(green), "water", len(water), "stops", len(stops), "junctions", len(j), flush=True)
wt = vertices(water, 3); pt = vertices(parks, 3)
sev = roads[roads["highway"].isin(["motorway","trunk","motorway_link"])]
sv = vertices(sev, 2)
pop = rasterio.open(R/"external/worldpop/vnm_ppp_2020_constrained.tif")
wc = rasterio.open(R/"external/worldcover/ESA_WorldCover_10m_2021_v200_N09E105_Map.tif")
map_ = pd.read_parquet(R/"hcmc/processed/preview2025/mapillary_per_cell.parquet").set_index("cell_id")
urban = tiles[tiles.buildings >= 100].reset_index(drop=True)
rows = []
for i, t in enumerate(urban.itertuples()):
    g = t.geometry; minx, miny, maxx, maxy = g.bounds; cx, cy = (minx+maxx)/2, (miny+maxy)/2
    rec = dict(cell_id=t.cell_id, buildings=t.buildings, heights=t.heights, places=t.places,
               places_cls=t.places_cls, n_classes=t.n_classes, road_ways=t.road_ways)
    rec["junctions_in"] = int(((j[:,0]>=minx)&(j[:,0]<=maxx)&(j[:,1]>=miny)&(j[:,1]<=maxy)).sum())
    rec["stops_in"] = int(((stops.geometry.x>=minx)&(stops.geometry.x<=maxx)&(stops.geometry.y>=miny)&(stops.geometry.y<=maxy)).sum())
    rec["stop_nearest_m"] = float(st.query([cx, cy])[0])
    rec["water_nearest_m"] = float(wt.query([cx, cy])[0]) if wt else np.nan
    rec["park_nearest_m"] = float(pt.query([cx, cy])[0]) if pt else np.nan
    rec["sever_nearest_m"] = float(sv.query([cx, cy])[0]) if sv else np.nan
    gg = gpd.GeoSeries([g], crs=UTM)
    try:
        a, _ = rmask(pop, gg.to_crs(pop.crs).geometry, crop=True, nodata=-1)
        a = a[0]; rec["pop_sum"] = float(a[a>0].sum()); rec["pop_px_zero"] = float((a==0).mean())
    except ValueError:
        rec["pop_sum"] = np.nan; rec["pop_px_zero"] = np.nan
    try:
        a, _ = rmask(wc, gg.to_crs(wc.crs).geometry, crop=True, nodata=0)
        a = a[0]; v = a[a>0]; rec["green_frac"] = float(np.isin(v,[10,20,30,40,95]).mean()) if v.size else np.nan
        rec["water_frac"] = float((v==80).mean()) if v.size else np.nan
        rec["builtup_frac"] = float((v==50).mean()) if v.size else np.nan
    except ValueError:
        rec["green_frac"] = rec["water_frac"] = rec["builtup_frac"] = np.nan
    rec["mly_images"] = int(map_.loc[t.cell_id, "images"]) if t.cell_id in map_.index else 0
    rows.append(rec)
    if i % 500 == 0: print(f"  {i}/{len(urban)}", flush=True)
df = pd.DataFrame(rows); df.to_parquet(OUT); print(df.describe().T.to_string())
