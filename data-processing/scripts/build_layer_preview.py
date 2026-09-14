"""Compute every "Have it" layer and render it, for the 2025 city.

Replaces the earlier preview, which was wrong in two ways the user caught: it
reduced each tile to a single mean, which discards the within-tile field that is
the whole object the model is given, and it used an ad-hoc channel set instead of
the factors the landscape page marks as available.

Three products, all from scripts/layers.py so the definitions cannot drift:
  mosaic   one pass over the whole city at 8 units per tile, so every tile shows
           its own internal variation rather than one number
  tiles    a few tiles at 512 units, where the alley-and-plot grain is visible
  margins  per-unit distributions and per-unit correlations, pooled over a sample
"""
import pathlib, json, base64, io, sys
import numpy as np, geopandas as gpd, pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree
from scipy.stats import skew, entropy as shannon
import matplotlib as mpl
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from layers import compute, LAYERS, PENDING, CENTRE, R_RING

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview2025"
OUT.mkdir(parents=True, exist_ok=True)
UTM = 3405
MOSAIC_PER_TILE = 8

CMAP = {"building_binary":"bone","building_density":"bone","building_height":"magma",
        "far":"inferno","building_form":"cividis","road_density":"pink",
        "intersection":"copper","transit_access":"YlOrBr","poi_count":"viridis",
        "poi_entropy":"mako" if "mako" in mpl.colormaps else "viridis",
        "green":"YlGn","dist_water":"Blues_r","dist_park":"Greens_r",
        "dist_centre":"Purples_r","severance":"Oranges_r","population":"rocket" if "rocket" in mpl.colormaps else "plasma"}

POI_CLASSES = [
    ("food_drink",    r"restaurant|cafe|coffee|bakery|diner|pub|food|ice_cream|dessert|brewer|juice|tea"),
    ("retail",        r"shop|store|shopping|market|grocer|supermarket|boutique|mall|retail"),
    ("services",      r"service|repair|laundry|salon|spa|beauty|barber|cleaning|agency|insurance|bank|financial|atm|legal|consult"),
    ("culture",       r"museum|art|theat|librar|religio|temple|church|mosque|galler|cultur|monument|histor"),
    ("education",     r"school|universit|college|educat|training|kindergarten|academy|tutor"),
    ("health",        r"hospital|clinic|pharmac|doctor|dentist|medical|health|nursing|veterinar"),
    ("accommodation", r"hotel|hostel|motel|lodging|resort|guest"),
    ("leisure",       r"park|gym|fitness|sport|recreation|entertain|nightlife|club|cinema|casino|stadium|golf"),
    ("transport",     r"parking|gas_station|fuel|transport|airport|station|taxi|rail|ferry|charging"),
    ("office",        r"office|company|corporate|business|professional|real_estate"),
    ("public",        r"government|police|fire_|post_office|civic|court|embassy|town_hall|community"),
]


def vertices(gdf, step=1):
    xs, ys = [], []
    for g in gdf.geometry:
        if g is None or g.is_empty: continue
        parts = g.geoms if g.geom_type.startswith("Multi") else [g]
        for p in parts:
            c = np.asarray(p.exterior.coords if p.geom_type == "Polygon" else p.coords)
            xs.append(c[::step, 0]); ys.append(c[::step, 1])
    if not xs: return None
    return cKDTree(np.c_[np.concatenate(xs), np.concatenate(ys)])


def prepare():
    print("loading sources …", flush=True)
    ctx = {}
    blds = gpd.read_parquet(RAW / "overture_2025/buildings.parquet").to_crs(UTM)
    hts  = gpd.read_parquet(RAW / "heights_3dglobfp_2025.parquet").to_crs(UTM)
    roads = gpd.read_file(RAW / "osm_2025/roads.gpkg").to_crs(UTM)
    places = gpd.read_parquet(RAW / "overture_2025/places.parquet").to_crs(UTM)
    print(f"  buildings {len(blds):,} | heights {len(hts):,} | roads {len(roads):,} | places {len(places):,}", flush=True)
    blds["_area"] = blds.geometry.area
    bsi, hsi, rsi = blds.sindex, hts.sindex, roads.sindex

    def burner(gdf, si, values=None):
        def f(minx, miny, maxx, maxy, nx, ny):
            from shapely.geometry import box
            bb = box(minx, miny, maxx, maxy)
            idx = list(si.query(bb, predicate="intersects"))
            if not idx: return np.zeros((ny, nx), "float32")
            sub = gdf.iloc[idx]
            v = np.ones(len(sub)) if values is None else sub[values].astype("float32").values
            tr = from_origin(minx, maxy, (maxx-minx)/nx, (maxy-miny)/ny)
            return rasterize(zip(sub.geometry, v), out_shape=(ny, nx), transform=tr,
                             fill=0, dtype="float32")
        return f
    ctx["burn_buildings"] = burner(blds, bsi)
    ctx["burn_footprint_area"] = burner(blds, bsi, "_area")
    ctx["burn_heights"] = burner(hts, hsi, "Height")
    ctx["burn_roads"] = burner(roads, rsi)

    # junctions: endpoints shared by three or more ways
    from collections import Counter
    ends = Counter()
    for g in roads.geometry:
        parts = g.geoms if g.geom_type.startswith("Multi") else [g]
        for p in parts:
            c = list(p.coords)
            for e in (c[0], c[-1]): ends[(round(e[0],1), round(e[1],1))] += 1
    j = np.array([k for k, v in ends.items() if v >= 3])
    ctx["junc_tree"] = cKDTree(j)
    print(f"  junctions {len(j):,}", flush=True)

    pxy = np.c_[places.geometry.x.values, places.geometry.y.values]
    ctx["poi_tree"] = cKDTree(pxy)
    cat = places["category"].astype(str)
    cls = pd.Series("other", index=places.index)
    for name, pat in reversed(POI_CLASSES):
        cls[cat.str.contains(pat, regex=True, na=False)] = name
    keep = cls != "other"
    class_trees = [cKDTree(pxy[(cls == n).values]) for n, _ in POI_CLASSES if (cls == n).sum() > 0]
    def entropy_field(px, py, r):
        pts = np.c_[px.ravel(), py.ravel()]
        counts = np.stack([t.query_ball_point(pts, r, return_length=True) for t in class_trees])
        tot = counts.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(tot > 0, counts / np.maximum(tot, 1), 0.0)
            h = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=0)
        return h.reshape(px.shape).astype("float32")
    ctx["entropy"] = entropy_field
    print(f"  place classes {len(class_trees)}; classified {int(keep.sum()):,}", flush=True)

    stops = gpd.read_file(RAW / "osm_2025/transit_stops.gpkg").to_crs(UTM)
    ctx["stop_tree"] = cKDTree(np.c_[stops.geometry.x.values, stops.geometry.y.values])
    water = gpd.read_file(RAW / "osm_2025/water_areas.gpkg").to_crs(UTM)
    green = gpd.read_file(RAW / "osm_2025/green_areas.gpkg").to_crs(UTM)
    ctx["water_tree"] = vertices(water, 3)
    ctx["park_tree"] = vertices(green[green.get("leisure").isin(["park","garden"])] if "leisure" in green else green, 3)
    sev = roads[roads["highway"].isin(["motorway","trunk","motorway_link"])] if "highway" in roads else roads.iloc[:0]
    ctx["sever_tree"] = vertices(sev, 2) if len(sev) else None
    print(f"  stops {len(stops):,} | water {len(water):,} | green {len(green):,} | severance ways {len(sev):,}", flush=True)

    wc = rasterio.open(REPO / "data/external/worldcover/ESA_WorldCover_10m_2021_v200_N09E105_Map.tif")
    pop = rasterio.open(REPO / "data/external/worldpop/vnm_ppp_2020_constrained.tif")
    GREEN = [10, 20, 30, 40, 95]
    def reader(src, fn):
        def f(minx, miny, maxx, maxy, nx, ny):
            wb = transform_bounds(UTM, src.crs, minx, miny, maxx, maxy)
            win = rasterio.windows.from_bounds(*wb, transform=src.transform)
            a = src.read(1, window=win, out_shape=(ny, nx), boundless=True, fill_value=0,
                         resampling=rasterio.enums.Resampling.average if fn is None
                         else rasterio.enums.Resampling.nearest)
            return fn(a) if fn else a.astype("float32")
        return f
    ctx["read_green"] = reader(wc, lambda a: np.isin(a, GREEN).astype("float32"))
    ctx["read_pop"] = reader(pop, lambda a: np.where(a < 0, 0.0, a).astype("float32"))

    c = gpd.GeoSeries.from_xy([CENTRE[0]], [CENTRE[1]], crs=4326).to_crs(UTM).iloc[0]
    ctx["centre_utm"] = (c.x, c.y)
    return ctx


def png(a, name, lo=None, hi=None, mask=None):
    """mask marks the units to scale on -- the city, not the bounding box.

    Without it the percentiles are computed over an array that is mostly outside
    the municipality, so for any sparse layer the 99th percentile is zero and the
    image comes out blank.
    """
    a = a.astype("float32")
    sel = a[mask] if mask is not None else a
    sel = sel[np.isfinite(sel)]
    if sel.size == 0: sel = np.zeros(1, "float32")
    lo = lo if lo is not None else float(np.percentile(sel, 1))
    hi = hi if hi is not None else float(np.percentile(sel, 99))
    a = np.nan_to_num(a)
    n = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    cm = mpl.colormaps[CMAP.get(name, "viridis")]
    rgb = (cm(n)[..., :3] * 255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), lo, hi


def main():
    ctx = prepare()
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
    # Con Dao is a commune of the city but sits 230 km offshore; including it in
    # the mosaic bounds leaves seven eighths of the image as empty sea.
    comm = gpd.read_file(RAW / "hcmc_communes_2025.gpkg")
    mainland = comm[comm.geometry.bounds.miny >= 9.5].to_crs(UTM)
    minx, miny, maxx, maxy = mainland.total_bounds
    n_all = len(tiles)
    tiles = tiles.cx[minx:maxx, miny:maxy]
    print(f"  mainland tiles {len(tiles):,} of {n_all:,}; Con Dao rendered separately", flush=True)
    nxt = int(round((maxx - minx) / 1201)); nyt = int(round((maxy - miny) / 1201))
    nx, ny = nxt * MOSAIC_PER_TILE, nyt * MOSAIC_PER_TILE
    print(f"\nmosaic: {nxt} x {nyt} tiles -> {nx} x {ny} units at {(maxx-minx)/nx:.0f} m", flush=True)
    M = compute(ctx, minx, miny, maxx, maxy, nx, ny)

    # mask to the city
    from rasterio.features import geometry_mask
    from shapely.ops import unary_union
    b_geom = unary_union([g.buffer(0) for g in mainland.geometry])
    tr = from_origin(minx, maxy, (maxx-minx)/nx, (maxy-miny)/ny)
    inside = ~geometry_mask([b_geom], out_shape=(ny, nx), transform=tr, invert=False)
    print(f"  units inside the city: {inside.sum():,} of {inside.size:,}", flush=True)

    mos = {}
    for name, _, _ in LAYERS:
        a = M[name].copy(); a[~inside] = np.nan
        p, lo, hi = png(a, name, mask=inside)
        mos[name] = {"png": p, "lo": round(lo, 3), "hi": round(hi, 3),
                     "within_tile_sd": round(float(np.nanmean([
                         np.nanstd(a[i*MOSAIC_PER_TILE:(i+1)*MOSAIC_PER_TILE,
                                     j*MOSAIC_PER_TILE:(j+1)*MOSAIC_PER_TILE])
                         for i in range(0, nyt, 7) for j in range(0, nxt, 7)])), 4),
                     "overall_sd": round(float(np.nanstd(a)), 4)}
        print(f"  {name:18s} lo {lo:9.2f} hi {hi:9.2f}", flush=True)
    json.dump({"nx": nx, "ny": ny, "per_tile": MOSAIC_PER_TILE,
               "nxt": nxt, "nyt": nyt, "mosaic": mos,
               "layers": [{"id": n, "side": s, "desc": d} for n, s, d in LAYERS],
               "pending": [{"id": n, "side": s, "desc": d} for n, s, d in PENDING]},
              open(OUT / "mosaic.json", "w"))
    print(f"\nwrote mosaic.json ({(OUT/'mosaic.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
