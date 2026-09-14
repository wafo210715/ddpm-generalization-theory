"""Build the current HCMC land boundary from the Vietnam OSM extract.

The city absorbed Binh Duong and Ba Ria-Vung Tau on 1 July 2025.  The OSM
relation for the merged city is unusable as a study area: one polygon of
36 423 km2 whose centroid sits offshore, because it carries territorial waters.
The land area is the union of the 168 commune-level units (admin_level 6) the
city is now divided into, and those polygons are also what a district map needs.

Read from the local country extract rather than Overpass, which times out on
these relations even in batches of twenty.
"""
import os, pathlib
os.environ.setdefault("OGR_INTERLEAVED_READING", "YES")
os.environ.setdefault("OSM_MAX_TMPFILE_SIZE", "4000")
import geopandas as gpd, pandas as pd
from pyogrio import read_dataframe
from shapely.ops import unary_union

REPO = pathlib.Path(__file__).resolve().parent.parent
PBF = REPO / "data/osm/vietnam-latest.osm.pbf"
OUT = REPO / "data/hcmc/raw"
BBOX = (105.9, 8.4, 107.8, 11.7)      # mainland plus Con Dao

# the 168 commune ids, from the relation's own membership
IDS = pathlib.Path(__file__).parent / "_hcmc_commune_ids.txt"


def main():
    print("reading admin boundaries from the country extract …", flush=True)
    g = read_dataframe(PBF, layer="multipolygons", bbox=BBOX,
                       columns=["osm_id", "osm_way_id", "name", "admin_level", "boundary", "other_tags"])
    print(f"  {len(g):,} multipolygons in bbox", flush=True)
    # NB: g.boundary is GeoSeries.boundary, the geometry -- the column needs g["boundary"]
    g = g[(g["boundary"] == "administrative") & (g["admin_level"] == "6")]
    print(f"  {len(g):,} at admin_level 6", flush=True)

    ids = {int(x) for x in IDS.read_text().split()} if IDS.exists() else None
    if ids:
        g = g[g["osm_id"].astype("float").astype("Int64").isin(ids)]
        print(f"  {len(g):,} matching the HCMC membership list", flush=True)

    g = g.reset_index(drop=True)
    g["area_km2"] = g.to_crs(3405).area / 1e6
    g.to_file(OUT / "hcmc_communes_2025.gpkg", driver="GPKG")

    land = unary_union([x.buffer(0) for x in g.geometry])
    b = gpd.GeoDataFrame({"name": ["Ho Chi Minh City (2025)"]}, geometry=[land], crs=4326)
    b.to_file(OUT / "hcmc_boundary_2025.gpkg", driver="GPKG")

    old = gpd.read_file(OUT / "hcmc_boundary_full.gpkg").to_crs(3405)
    new_km2 = b.to_crs(3405).area.sum() / 1e6
    print(f"\n  communes            : {len(g)}")
    print(f"  land area 2025      : {new_km2:,.1f} km2")
    print(f"  previous study area : {old.area.sum()/1e6:,.1f} km2")
    print(f"  growth              : {new_km2/(old.area.sum()/1e6):.2f}x")
    print(f"  bounds              : {[round(v,3) for v in b.total_bounds]}")
    print("\n  largest communes:")
    print(g.nlargest(6, "area_km2")[["name", "area_km2"]].to_string(index=False))


if __name__ == "__main__":
    main()
