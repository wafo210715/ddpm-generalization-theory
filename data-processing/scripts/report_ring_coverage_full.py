"""B6 on the full 2025 urban tile set: ring coverage over every tile with >=100 Overture buildings.

Written for A1, 15 Sep 2026.  The 57 % / 79.7 % figures on the corrections page were
measured on the 138 tiles that OSM had mapped densely; this runs the same test on all 3 020.
Ring coverage on the FULL 2025 urban tile set (>=100 Overture buildings), Overture places over the tile union.
Same test as report_overture_ring_coverage.py: share of units whose ring holds >=2 places in >=2 of the 11 classes.
Also records: share of units with >=1 place in ring (count non-zero), per tile."""
import sys, pathlib, numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/wafo/Documents/repo/ddpm-generalization-theory/data-processing/scripts")
from build_layer_preview import POI_CLASSES
RAW = pathlib.Path("/home/wafo/Documents/repo/ddpm-generalization-theory/data-processing/data/hcmc/raw")
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/wafo/Documents/repo/ddpm-generalization-theory/data-processing/data/hcmc/processed/reports/ring_coverage_urban2025.parquet")
N = 64; RINGS = [50, 100, 150, 200, 300, 500]; UTM = 3405
tiles = gpd.read_file(RAW/"hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
blds = gpd.read_parquet(RAW/"overture_tu/buildings.parquet").to_crs(UTM)
places = gpd.read_parquet(RAW/"overture_tu/places.parquet").to_crs(UTM)
cat = places.category.astype(str)
cls = pd.Series("other", index=places.index)
for name, pat in reversed(POI_CLASSES):
    cls[cat.str.contains(pat, regex=True, na=False)] = name
places["cls"] = cls
print("places", len(places), "classified", int((cls!="other").sum()), flush=True)
allxy = np.c_[places.geometry.x, places.geometry.y]
all_tree = cKDTree(allxy)
trees = {c: cKDTree(np.c_[g.geometry.x, g.geometry.y]) for c, g in places[places.cls!="other"].groupby("cls")}
cnt = gpd.sjoin(blds[["geometry"]], tiles[["cell_id","geometry"]], predicate="intersects").groupby("cell_id").size()
tiles["buildings"] = tiles.cell_id.map(cnt).fillna(0).astype(int)
urban = tiles[tiles.buildings >= 100].reset_index(drop=True)
print("urban tiles", len(urban), flush=True)
rows = []
for i, t in enumerate(urban.itertuples()):
    minx, miny, maxx, maxy = t.geometry.bounds
    res = (maxx-minx)/N
    ys, xs = np.mgrid[0:N, 0:N]
    pts = np.c_[(minx+(xs+.5)*res).ravel(), (maxy-(ys+.5)*res).ravel()]
    rec = {"cell_id": t.cell_id, "buildings": t.buildings}
    for r in RINGS:
        total = np.zeros(len(pts), "int32"); nclass = np.zeros(len(pts), "int16")
        for c, tree in trees.items():
            k = tree.query_ball_point(pts, r, return_length=True); total += k; nclass += (k > 0)
        rec[f"def{r}"] = float(((total>=2)&(nclass>=2)).mean())
        rec[f"any{r}"] = float((all_tree.query_ball_point(pts, r, return_length=True) > 0).mean())
    rows.append(rec)
    if i % 300 == 0: print(f"  {i}/{len(urban)}", flush=True)
df = pd.DataFrame(rows); df.to_parquet(OUT)
print("\nring  entropy-defined%  tiles>=80%def  tiles<10%def   any-place%  tiles 0 any")
for r in RINGS:
    d = df[f"def{r}"]; a = df[f"any{r}"]
    print(f"{r:4d}  {100*d.mean():6.1f}  {100*(d>=.8).mean():6.1f}  {100*(d<.1).mean():6.1f}   {100*a.mean():6.1f}  {100*(a==0).mean():6.1f}")
