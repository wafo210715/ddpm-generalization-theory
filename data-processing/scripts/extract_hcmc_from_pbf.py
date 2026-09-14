"""Clip every OSM layer we need out of the Vietnam extract, in one pass.

Overpass rate-limits this host after a few hundred queries, which is why the
road and POI layers only ever covered ten tiles.  A Geofabrik country extract
has no such limit and carries every tag, so this replaces per-tile fetching for
all citywide OSM layers.

Writes to data/hcmc/raw/osm/<layer>.gpkg, clipped to the study-area polygon.
"""
import os, sys, pathlib
os.environ.setdefault("OGR_INTERLEAVED_READING", "YES")
os.environ.setdefault("OSM_MAX_TMPFILE_SIZE", "4000")   # MB

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PBF = REPO / "data/osm/vietnam-latest.osm.pbf"
OUT = REPO / "data/hcmc/raw/osm"
BOUNDARY = REPO / "data/hcmc/raw/hcmc_boundary_full.gpkg"


def parse_other_tags(s):
    """GDAL packs untyped OSM tags into an hstore string: "k"=>"v","k2"=>"v2"."""
    out = {}
    if not s:
        return out
    for part in str(s).split('","'):
        if "=>" not in part:
            continue
        k, _, v = part.partition("=>")
        out[k.strip('"')] = v.strip('"')
    return out


WANT_TAGS = ["building:levels", "height", "building:height", "start_date",
             "building:use", "amenity", "shop", "office", "tourism", "leisure",
             "landuse", "natural", "waterway", "railway", "public_transport",
             "lit", "sidewalk", "lanes", "surface", "maxspeed", "levels"]

LAYERS = {
    # out name      (pbf layer,        keep rows where any of these cols is set)
    "buildings":    ("multipolygons",  ["building"]),
    "landuse":      ("multipolygons",  ["landuse", "leisure", "natural"]),
    "roads":        ("lines",          ["highway"]),
    "waterways":    ("lines",          ["waterway"]),
    "points":       ("points",         None),   # POIs + transit, filtered later
}


def main():
    if not PBF.exists():
        sys.exit(f"missing {PBF} — download it first")
    OUT.mkdir(parents=True, exist_ok=True)
    boundary = gpd.read_file(BOUNDARY)
    poly = boundary.geometry.union_all()
    minx, miny, maxx, maxy = poly.bounds
    print(f"study area bbox: {minx:.3f} {miny:.3f} {maxx:.3f} {maxy:.3f}", flush=True)

    for name, (layer, required) in LAYERS.items():
        dst = OUT / f"{name}.gpkg"
        if dst.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        print(f"{name}: reading {layer} ...", flush=True)
        gdf = read_dataframe(PBF, layer=layer, bbox=(minx, miny, maxx, maxy))
        print(f"  {len(gdf):,} features in bbox", flush=True)

        # explode hstore tags into real columns
        if "other_tags" in gdf.columns:
            extra = gdf["other_tags"].map(parse_other_tags)
            for tag in WANT_TAGS:
                col = tag.replace(":", "_")
                if col not in gdf.columns:
                    vals = extra.map(lambda d, t=tag: d.get(t))
                    if vals.notna().any():
                        gdf[col] = vals
            gdf = gdf.drop(columns=["other_tags"])

        if required:
            mask = False
            for c in required:
                if c in gdf.columns:
                    mask = mask | gdf[c].notna()
            gdf = gdf[mask]
            print(f"  {len(gdf):,} after tag filter", flush=True)

        gdf = gdf[gdf.intersects(poly)]
        print(f"  {len(gdf):,} inside study area", flush=True)
        gdf = gdf.loc[:, [c for c in gdf.columns if gdf[c].notna().any() or c == "geometry"]]
        gdf.to_file(dst, driver="GPKG")
        print(f"  -> {dst.name}  ({dst.stat().st_size/1e6:.0f} MB)", flush=True)

    # split the points layer into POIs and transit
    pts = gpd.read_file(OUT / "points.gpkg")
    poi_keys = [c for c in ["amenity", "shop", "office", "tourism", "leisure"] if c in pts.columns]
    if poi_keys:
        m = False
        for c in poi_keys:
            m = m | pts[c].notna()
        pts[m].to_file(OUT / "pois.gpkg", driver="GPKG")
        print(f"pois: {m.sum():,}")
    tk = [c for c in ["public_transport", "railway", "highway"] if c in pts.columns]
    if tk:
        m = False
        for c in tk:
            m = m | pts[c].notna()
        pts[m].to_file(OUT / "transit_points.gpkg", driver="GPKG")
        print(f"transit points: {m.sum():,}")


if __name__ == "__main__":
    main()
