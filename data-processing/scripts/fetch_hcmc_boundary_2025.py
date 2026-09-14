"""Build the current Ho Chi Minh City land boundary from its commune-level units.

The city absorbed Binh Duong and Ba Ria-Vung Tau on 1 July 2025.  The OSM
relation for the merged city cannot be used directly as a study area: it is one
polygon of 36 423 km2 whose centroid sits offshore, because it includes
territorial waters out to Con Dao.

The land area is the union of the 168 commune-level units (admin_level 6) the
city is now divided into.  Those polygons are also the districts to show on a
map, so they are kept individually as well as unioned.
"""
import pathlib, json
import requests, geopandas as gpd, pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data/hcmc/raw"
UA = {"User-Agent": "hcmc-vitality-research/1.0 (academic)"}
URL = "https://overpass-api.de/api/interpreter"

Q_IDS = """[out:json][timeout:300];
rel(1973756);map_to_area->.h;
rel(area.h)["boundary"="administrative"]["admin_level"="6"];
out ids tags;"""

def Q_GEOM(ids):
    return "[out:json][timeout:300];\nrelation(id:%s);\nout geom;" % ",".join(str(i) for i in ids)


def post(query, tries=4):
    import time
    for a in range(tries):
        try:
            r = requests.post(URL, data={"data": query}, headers=UA, timeout=400)
            r.raise_for_status()
            return r.json()["elements"]
        except Exception as e:
            wait = 20 * (a + 1)
            print(f"    retry {a+1}: {type(e).__name__} {str(e)[:70]}; sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("overpass failed after retries")

def rings(el):
    outers, inners = [], []
    for m in el.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        pts = [(p["lon"], p["lat"]) for p in m["geometry"]]
        (outers if m.get("role") != "inner" else inners).append(pts)
    # stitch open ways into closed rings
    def stitch(segs):
        segs = [list(s) for s in segs if len(s) > 1]
        out, cur = [], None
        while segs:
            if cur is None:
                cur = segs.pop(0)
            joined = False
            for i, s in enumerate(segs):
                if cur[-1] == s[0]:   cur += s[1:];               segs.pop(i); joined = True; break
                if cur[-1] == s[-1]:  cur += s[::-1][1:];         segs.pop(i); joined = True; break
                if cur[0]  == s[-1]:  cur = s[:-1] + cur;         segs.pop(i); joined = True; break
                if cur[0]  == s[0]:   cur = s[::-1][:-1] + cur;   segs.pop(i); joined = True; break
            if not joined or cur[0] == cur[-1]:
                if len(cur) > 3: out.append(cur)
                cur = None
        if cur and len(cur) > 3: out.append(cur)
        return out
    polys = []
    for o in stitch(outers):
        try: polys.append(Polygon(o))
        except Exception: pass
    holes = []
    for i in stitch(inners):
        try: holes.append(Polygon(i))
        except Exception: pass
    if not polys: return None
    g = unary_union([p.buffer(0) for p in polys])
    if holes:
        g = g.difference(unary_union([h.buffer(0) for h in holes]))
    return g

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("listing commune-level units …", flush=True)
    ids = [e["id"] for e in post(Q_IDS)]
    print(f"  {len(ids)} relations; fetching geometry in batches", flush=True)
    els = []
    B = 20
    for i in range(0, len(ids), B):
        chunk = ids[i:i+B]
        els += post(Q_GEOM(chunk))
        print(f"    {min(i+B,len(ids))}/{len(ids)}", flush=True)

    rows = []
    for e in els:
        g = rings(e)
        if g is None or g.is_empty:
            print("   skipped (no geometry):", e.get("tags", {}).get("name")); continue
        t = e.get("tags", {})
        rows.append({"osm_id": e["id"], "name": t.get("name"),
                     "name_en": t.get("name:en"), "type": t.get("place") or t.get("admin_level"),
                     "geometry": g})
    gdf = gpd.GeoDataFrame(rows, crs=4326)
    gdf["area_km2"] = gdf.to_crs(3405).area / 1e6
    gdf.to_file(OUT / "hcmc_communes_2025.gpkg", driver="GPKG")

    land = unary_union(gdf.geometry.values.tolist())
    b = gpd.GeoDataFrame({"name": ["Ho Chi Minh City (2025)"]}, geometry=[land], crs=4326)
    b.to_file(OUT / "hcmc_boundary_2025.gpkg", driver="GPKG")

    old = gpd.read_file(OUT / "hcmc_boundary_full.gpkg").to_crs(3405)
    print(f"\n  communes kept        : {len(gdf)}")
    print(f"  land area (2025)     : {b.to_crs(3405).area.sum()/1e6:,.1f} km2")
    print(f"  previous study area  : {old.area.sum()/1e6:,.1f} km2")
    print(f"  growth               : {b.to_crs(3405).area.sum()/old.area.sum():.2f}x")
    print(f"  bounds               : {[round(v,3) for v in b.total_bounds]}")
    print(f"\n  largest communes by area:")
    print(gdf.nlargest(5, 'area_km2')[['name','area_km2']].to_string(index=False))

if __name__ == "__main__":
    main()
