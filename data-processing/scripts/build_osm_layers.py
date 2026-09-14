"""Consolidate the raw PBF extracts into the layers the channels are built from.

The raw extract splits by OSM geometry type, which is not how the indicators are
organised.  A POI is a POI whether it was mapped as a node or as a shop polygon,
and a transit stop is one stop even though OSM records it as both a highway node
and a public_transport platform.  This produces one clean layer per subject.
"""
import pathlib
import geopandas as gpd
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "data/hcmc/raw/osm_2025"
OUT = REPO / "data/hcmc/raw/osm_2025"
POI_KEYS = ["amenity", "shop", "office", "tourism", "leisure"]
# nodes tagged highway=* that are not transit and not POIs
FURNITURE = {"crossing", "traffic_signals", "street_lamp", "turning_circle",
             "motorway_junction", "turning_loop", "stop", "give_way", "speed_camera"}


def tag_of(df, keys):
    """First non-null value across keys, plus which key it came from."""
    key = pd.Series(pd.NA, index=df.index, dtype="object")
    val = pd.Series(pd.NA, index=df.index, dtype="object")
    for k in keys:
        if k not in df.columns:
            continue
        m = val.isna() & df[k].notna()
        key[m], val[m] = k, df.loc[m, k]
    return key, val


pts = gpd.read_file(SRC / "points.gpkg")
blds = gpd.read_file(SRC / "buildings.gpkg")
lu = gpd.read_file(SRC / "landuse.gpkg")

# ---- POIs: nodes + tagged building polygons + tagged leisure polygons ---------
frames = []
for src, label in [(pts, "node"), (blds, "building"), (lu, "area")]:
    k, v = tag_of(src, POI_KEYS)
    m = v.notna()
    if label != "node":                      # polygons become their centroid
        g = src.loc[m].to_crs(3405).centroid.to_crs(4326)
    else:
        g = src.loc[m].geometry
    frames.append(gpd.GeoDataFrame(
        {"category_key": k[m], "category_value": v[m],
         "name": src.loc[m]["name"] if "name" in src.columns else pd.NA,
         "src": label},
        geometry=g.values, crs=4326))
pois = pd.concat(frames, ignore_index=True)
pois = gpd.GeoDataFrame(pois, geometry="geometry", crs=4326)
pois.to_file(OUT / "pois_all.gpkg", driver="GPKG")
print(f"pois_all: {len(pois):,}  ({pois.src.value_counts().to_dict()})")

# ---- transit stops: one row per stop, deduplicated by position ----------------
is_transit = pd.Series(False, index=pts.index)
if "highway" in pts.columns:
    is_transit |= pts.highway.eq("bus_stop")
if "public_transport" in pts.columns:
    is_transit |= pts.public_transport.isin(["platform", "station", "stop_position"])
if "railway" in pts.columns:
    is_transit |= pts.railway.isin(["station", "subway_entrance", "halt", "tram_stop"])
tr = pts[is_transit].copy()
tr["mode"] = "bus"
if "railway" in tr.columns:
    tr.loc[tr.railway.notna(), "mode"] = "rail"
m = tr.to_crs(3405)
tr["cluster"] = list(zip((m.geometry.x // 15).astype(int), (m.geometry.y // 15).astype(int)))
tr = tr.drop_duplicates("cluster").drop(columns=["cluster"])
tr[["osm_id", "name", "mode", "geometry"]].to_file(OUT / "transit_stops.gpkg", driver="GPKG")
print(f"transit_stops: {len(tr):,}  ({tr['mode'].value_counts().to_dict()})")

# ---- street furniture: crossings, signals, lamps ------------------------------
if "highway" in pts.columns:
    fur = pts[pts.highway.isin(FURNITURE)][["osm_id", "highway", "geometry"]]
    fur.to_file(OUT / "street_furniture.gpkg", driver="GPKG")
    print(f"street_furniture: {len(fur):,}  ({fur.highway.value_counts().head(4).to_dict()})")

# ---- water: natural=water polygons + waterway lines ---------------------------
wat = lu[lu.natural.isin(["water", "wetland", "bay"])] if "natural" in lu.columns else lu.iloc[:0]
wat.to_file(OUT / "water_areas.gpkg", driver="GPKG")
print(f"water_areas: {len(wat):,}")

# ---- green: parks, gardens, forest, grass ------------------------------------
green_leisure = lu.leisure.isin(["park", "garden", "pitch", "playground", "golf_course", "nature_reserve"]) if "leisure" in lu.columns else False
green_landuse = lu.landuse.isin(["grass", "forest", "meadow", "village_green", "recreation_ground", "greenfield"]) if "landuse" in lu.columns else False
green_natural = lu.natural.isin(["wood", "grassland", "scrub", "heath"]) if "natural" in lu.columns else False
grn = lu[green_leisure | green_landuse | green_natural]
grn.to_file(OUT / "green_areas.gpkg", driver="GPKG")
print(f"green_areas: {len(grn):,}")
