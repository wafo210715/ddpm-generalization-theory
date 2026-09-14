"""Does Overture POI density track built density, where OSM's did not?

B6 found OSM POI coverage uncorrelated with built form: the second-densest tile
in the city, 6 866 buildings, held four POIs.  That made a POI channel encode
mapping effort rather than urban function.  Overture returns 17x more places, so
the question is whether the relationship to built density is now sane, and
whether a small ring is finally defined often enough to be a channel.
"""
import geopandas as gpd, numpy as np, pandas as pd

tiles = gpd.read_file("data/hcmc/raw/hcmc_tiles_z15.gpkg").set_index("cell_id")
blds = gpd.read_file("data/hcmc/raw/osm/buildings.gpkg")
ov = gpd.read_parquet("data/hcmc/raw/overture/places.parquet")
osm = gpd.read_file("data/hcmc/raw/osm/pois_all.gpkg")
print(f"overture places {len(ov):,} | osm pois {len(osm):,} | osm buildings {len(blds):,}\n")

t = tiles.reset_index()[["cell_id", "geometry"]]
def per_tile(pts, label):
    j = gpd.sjoin(pts[["geometry"]].to_crs(4326), t, how="inner", predicate="within")
    return j.groupby("cell_id").size().rename(label)

cnt = pd.concat([per_tile(blds, "buildings"),
                 per_tile(ov, "overture"),
                 per_tile(osm, "osm")], axis=1).fillna(0)
cnt = cnt.reindex(t.cell_id, fill_value=0)

built = cnt[cnt.buildings >= 100]
print(f"tiles with >=100 OSM buildings: {len(built)}")
print(f"  median places per tile   overture {built.overture.median():8.0f}   osm {built.osm.median():6.0f}")
print(f"  mean                     overture {built.overture.mean():8.0f}   osm {built.osm.mean():6.0f}")
print(f"  tiles with zero places   overture {(built.overture==0).sum():8d}   osm {(built.osm==0).sum():6d}")

print("\nSpearman correlation with building count, over tiles that have any building:")
any_b = cnt[cnt.buildings > 0]
for c in ["overture", "osm"]:
    print(f"  {c:9s} rho = {any_b.buildings.corr(any_b[c], method='spearman'):.3f}")

print("\nThe tile B6 singled out (2nd densest, 6 866 buildings, 4 OSM POIs):")
for tid in ["15_26098_15391", "15_26097_15392", "15_26098_15390"]:
    if tid in cnt.index:
        r = cnt.loc[tid]
        print(f"  {tid}: {r.buildings:6.0f} buildings | overture {r.overture:6.0f} | osm {r.osm:4.0f}")

# density per km2 on built-up tiles
print(f"\nplaces per km2 on built-up tiles: overture {built.overture.mean()/1.443:.0f}, osm {built.osm.mean()/1.443:.0f}")
