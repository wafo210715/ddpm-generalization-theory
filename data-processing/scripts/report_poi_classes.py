"""Can the POI layer carry the functional classes the literature's factors need?

Several factors in the BE table are not "POI density" but a *subclass* density:
cultural POIs (Qin 2025, Q. Li 2021), food and life-service (Tan 2026),
night-time economy (Shi 2023), third places (Collins 2024).  Splitting a sparse
layer into classes multiplies the sparsity, so this counts what each class has
citywide and on the tiles that are actually built up.
"""
import geopandas as gpd, pandas as pd, numpy as np

CLASSES = {
    "food_drink":  {"restaurant","cafe","fast_food","bar","pub","food_court","ice_cream","biergarten"},
    "retail":      None,   # every shop=* value
    "culture":     {"place_of_worship","museum","theatre","arts_centre","library","cinema","community_centre","gallery","artwork","attraction","temple"},
    "education":   {"school","kindergarten","university","college","language_school","driving_school"},
    "health":      {"pharmacy","clinic","hospital","doctors","dentist","veterinary"},
    "public_admin":{"government","townhall","police","post_office","courthouse","fire_station","embassy"},
    "finance":     {"bank","atm","bureau_de_change"},
    "leisure_rec": {"park","pitch","swimming_pool","garden","sports_centre","fitness_centre","playground","stadium","golf_course"},
    "accommodation":{"hotel","hostel","guest_house","motel","apartment"},
    "office":      None,   # every office=* value
    "transport":   {"parking","fuel","bus_station","taxi","car_rental","charging_station","ferry_terminal"},
}

pois = gpd.read_file("data/hcmc/raw/hcmc_pois_all.gpkg")
tiles = gpd.read_file("data/hcmc/raw/hcmc_tiles_z15.gpkg")
blds  = gpd.read_file("data/hcmc/raw/hcmc_buildings.gpkg")

def classify(row):
    if row.category_key == "shop":   return "retail"
    if row.category_key == "office": return "office"
    v = row.category_value
    for name, vals in CLASSES.items():
        if vals and v in vals: return name
    return "other"

pois["fclass"] = pois.apply(classify, axis=1)

print("=== citywide POIs per functional class (2 406 km²)")
vc = pois.fclass.value_counts()
for k, v in vc.items():
    print(f"  {k:15s} {v:6d}   {v/2406.5:6.2f} per km²")
print(f"  {'TOTAL':15s} {len(pois):6d}   {len(pois)/2406.5:6.2f} per km²")

# join to tiles
tj = tiles.copy()
pj = gpd.sjoin(pois, tj[["tile_id","geometry"]] if "tile_id" in tj else tj, how="inner", predicate="within")
bj = gpd.sjoin(blds.copy(), tj[["tile_id","geometry"]] if "tile_id" in tj else tj, how="inner", predicate="intersects")

key = "tile_id" if "tile_id" in tj.columns else [c for c in tj.columns if c != "geometry"][0]
print(f"\n(tile key column: {key}; tile columns: {list(tj.columns)})")

bcount = bj.groupby(key).size().rename("buildings")
built = bcount[bcount >= 100].index          # the 139 tiles
print(f"\ntiles with >=100 buildings: {len(built)}")

sub = pj[pj[key].isin(built)]
tab = sub.pivot_table(index=key, columns="fclass", aggfunc="size", fill_value=0)
tab = tab.reindex(built, fill_value=0)

print("\n=== per built-up tile (1.44 km² each), POIs of each class")
rows = []
for c in sorted(tab.columns):
    s = tab[c]
    rows.append((c, s.median(), s.mean(), (s == 0).mean()*100, (s >= 5).mean()*100))
rows.sort(key=lambda r: -r[2])
print(f"{'class':15s} {'median':>7s} {'mean':>7s} {'tiles w/ 0':>11s} {'tiles w/ >=5':>13s}")
for c, med, mean, z, f in rows:
    print(f"{c:15s} {med:7.0f} {mean:7.1f} {z:10.0f}% {f:12.0f}%")

tot = tab.sum(axis=1)
print(f"\nall classes combined: median {tot.median():.0f}, mean {tot.mean():.1f} POIs per built-up tile")
print(f"tiles with 0 POIs among the {len(built)} built-up: {(tot==0).sum()}")
