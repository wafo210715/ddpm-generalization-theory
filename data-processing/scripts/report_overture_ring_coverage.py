"""Redo B6's ring-coverage checkpoint on Overture places instead of OSM.

B6 asked on what fraction of units a Shannon entropy over the ring is even
defined -- at least two places in at least two categories.  On OSM the answer was
1.1% at 50 m and 5.9% at 100 m, which killed the diversity channel and pushed the
ring out to 300 m, where it mostly reports distance to the nearest mapped
cluster.  Overture holds 17x more places, so the checkpoint is re-run here.

Two efficiency notes.  Coverage is a fraction of units, so it is estimated on a
128x128 sub-grid per tile rather than the full 1024x1024 -- unbiased and 64x
cheaper.  And the two-category test is evaluated per class with return_length,
which avoids materialising a neighbour list per sample point.
"""
import geopandas as gpd, numpy as np, pandas as pd
from scipy.spatial import cKDTree

N = 128
RINGS = [50, 100, 200, 300, 500]
UTM = 3405

tiles = gpd.read_file("data/hcmc/raw/hcmc_tiles_z15.gpkg").to_crs(UTM)
blds = gpd.read_parquet("data/hcmc/raw/overture/buildings.parquet").to_crs(UTM)
places = gpd.read_parquet("data/hcmc/raw/overture/places.parquet").to_crs(UTM)
places = places[places.category.notna()].copy()
# Overture ships 1 065 leaf categories.  Entropy over 1 065 classes is not what
# the literature means by functional mix, and one KD-tree per class is far too
# slow, so they fold into the functional taxonomy those papers actually use.
CLASSES = [
    ("food_drink",    r"restaurant|cafe|coffee|bakery|diner|_bar$|^bar_|pub|food|ice_cream|dessert|brewer|juice|tea"),
    ("retail",        r"shop|store|shopping|market|grocer|supermarket|boutique|mall|retail"),
    ("services",      r"service|repair|laundry|salon|spa|beauty|barber|cleaning|agency|insurance|bank|financial|atm|legal|consult"),
    ("culture",       r"museum|art|theat|librar|religio|temple|church|mosque|shrine|galler|cultur|monument|histor"),
    ("education",     r"school|universit|college|educat|training|kindergarten|academy|tutor"),
    ("health",        r"hospital|clinic|pharmac|doctor|dentist|medical|health|nursing|veterinar"),
    ("accommodation", r"hotel|hostel|motel|lodging|resort|guest"),
    ("leisure",       r"park|gym|fitness|sport|recreation|entertain|nightlife|club|cinema|casino|amusement|stadium|golf"),
    ("transport",     r"parking|gas_station|fuel|transport|airport|station|taxi|rail|ferry|charging"),
    ("office",        r"office|company|corporate|business|professional|real_estate"),
    ("public",        r"government|police|fire_|post_office|civic|court|embassy|town_hall|community"),
]
cat = places.category.astype(str)
cls = pd.Series("other", index=places.index)
for name, pat in reversed(CLASSES):          # earlier entries win ties
    cls[cat.str.contains(pat, regex=True, na=False)] = name
places["cls"] = cls
places = places[places.cls != "other"]
classes = sorted(places.cls.unique())
print(f"places {len(places):,} in {len(classes)} functional classes", flush=True)
print(places.cls.value_counts().to_string(), flush=True)

trees = {c: cKDTree(np.c_[g.geometry.x.values, g.geometry.y.values])
         for c, g in places.groupby("cls")}

bpt = gpd.sjoin(blds[["geometry"]], tiles[["cell_id", "geometry"]], predicate="intersects")
bcount = bpt.groupby("cell_id").size()

rows = []
for i, t in enumerate(tiles.itertuples()):
    minx, miny, maxx, maxy = t.geometry.bounds
    gx, gy = np.meshgrid(np.linspace(minx, maxx, N), np.linspace(miny, maxy, N))
    pts = np.c_[gx.ravel(), gy.ravel()]
    rec = {"cell_id": t.cell_id, "buildings": int(bcount.get(t.cell_id, 0))}
    for r in RINGS:
        total = np.zeros(len(pts), dtype=np.int32)
        nclass = np.zeros(len(pts), dtype=np.int16)
        for c, tree in trees.items():
            k = tree.query_ball_point(pts, r, return_length=True)
            total += k
            nclass += (k > 0)
        rec[f"r{r}"] = float(((total >= 2) & (nclass >= 2)).mean())
    rows.append(rec)
    if i % 200 == 0:
        print(f"  {i}/{len(tiles)} tiles", flush=True)

df = pd.DataFrame(rows)
out = "data/hcmc/processed/reports/overture_ring_coverage.parquet"
df.to_parquet(out)

def block(sub, label):
    print(f"\n{label} (n={len(sub)})", flush=True)
    print("  ring   entropy defined   tiles >=80% defined")
    for r in RINGS:
        c = sub[f"r{r}"]
        print(f"  {r:4d} m       {100*c.mean():5.1f} %           {100*(c>=0.8).mean():5.1f} %")

block(df, "all 1 881 tiles")
block(df[df.buildings > 0], "tiles with any Overture building")
block(df[df.buildings >= 100], "tiles with >=100 Overture buildings")
print(f"\nwritten to {out}")
