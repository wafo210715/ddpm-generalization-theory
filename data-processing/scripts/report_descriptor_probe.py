"""Issue #2 part 3, 23 Sep 2026: lag-1 Moran's I and the by-construction rules at 256 px.

Written for discussion/analysis_map.md section 4.

Preview code (layers.compute via build_layer_tiles.tile_layers, junction degree
rule), so these are preview numbers, not B8 numbers.  Sample: 40 tiles drawn at
random (seed 0) from the 3 010 cells = 3 020 urban tiles minus Con Dao.
"""
import sys, pathlib, json
import numpy as np, pandas as pd, geopandas as gpd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data-processing/scripts"))
sys.path.insert(0, str(REPO / "ddpm-win-master"))
from analysis.spatial import morans_i
from layers import LAYERS, native_grid, up, R_RING
from build_layer_preview import prepare, RAW, UTM
from build_layer_tiles import tile_layers, BUFFER_M

N = 256
CON_DAO = set("15_26086_15589 15_26086_15590 15_26086_15591 15_26087_15589 15_26087_15590 "
              "15_26088_15588 15_26088_15589 15_26088_15590 15_26089_15586 15_26089_15589".split())
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "data-processing/data/hcmc/processed/reports/descriptor_probe_256.json")

src = pd.read_parquet(REPO / "data-processing/data/hcmc/processed/reports/tile_sources_urban2025.parquet")
cells = src[~src.cell_id.isin(CON_DAO)].reset_index(drop=True)
assert len(cells) == 3010, len(cells)
rng = np.random.default_rng(0)
samp = cells.iloc[rng.choice(len(cells), 40, replace=False)]
tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM).set_index("cell_id")

ctx = prepare()
names = [n for n, _, _ in LAYERS]
mi = {n: [] for n in names}
viol = {k: [0, 0] for k in ["height_without_building", "building_without_height",
                            "far_without_density", "density_without_far",
                            "entropy_without_two_places", "entropy_above_ln11"]}
count_floor = 2 / (np.pi * (R_RING / 1000) ** 2)       # 2 places in a 300 m ring, per km2
per_tile = []
for i, (cid, nb) in enumerate(zip(samp.cell_id, samp.buildings)):
    L = tile_layers(ctx, tiles.loc[cid].geometry, N)
    L = {k: np.nan_to_num(v) for k, v in L.items()}
    for n in names:
        mi[n].append(morans_i(L[n], 8))
    b, h, d, f = L["building_binary"], L["building_height"], L["building_density"], L["far"]
    c, e = L["poi_count"], L["poi_entropy"]
    pairs = {"height_without_building": ((h > 0) & (b == 0), h > 0),
             "building_without_height": ((b == 1) & (h == 0), b == 1),
             "far_without_density": ((f > 0) & (d == 0), f > 0),
             "density_without_far": ((d > 0) & (f == 0), d > 0),
             "entropy_without_two_places": ((e > 0) & (c < count_floor), e > 0),
             "entropy_above_ln11": (e > np.log(11) + 1e-4, np.ones_like(e, bool))}
    row = {"cell_id": cid, "buildings": int(nb)}
    for k, (bad, base) in pairs.items():
        viol[k][0] += int(bad.sum()); viol[k][1] += int(base.sum())
        row[k] = int(bad.sum())
    per_tile.append(row)
    print(f"{i+1}/40 {cid} {nb}", flush=True)

# Null: white noise on each layer's native grid, read onto 256 the same way.
ext_tile = 1201.0
pad = int(round(N * BUFFER_M / ext_tile)); ext = ext_tile + 2 * BUFFER_M; NT = N + 2 * pad
scales = {"building_binary": None, "building_height": None, "green": 10.0,
          "building_density": 100.0, "far": 100.0, "road_density": 100.0, "intersection": 100.0,
          "dist_water": 100.0, "population": 100.0, "building_form": 150.0,
          "poi_count": 300.0, "poi_entropy": 300.0, "dist_park": 300.0, "severance": 300.0,
          "transit_access": 500.0, "dist_centre": 1000.0}
null = {}
for n, s in scales.items():
    g = native_grid(ext, s, NT)
    vals = []
    for r in range(20):
        z = np.random.default_rng(r).standard_normal((g, g)).astype("float32")
        vals.append(morans_i(up(z, NT, NT)[pad:pad + N, pad:pad + N], 8))
    null[n] = {"native_per_tile": round(g * ext_tile / ext, 1), "noise_I": float(np.median(vals))}

res = {"sample": "40 tiles, random seed 0, from the 3 010 cells", "n": N,
       "buildings_in_sample": {"median": int(samp.buildings.median()), "min": int(samp.buildings.min()),
                               "max": int(samp.buildings.max())},
       "morans_i": {n: {"median": float(np.nanmedian(v)), "p10": float(np.nanpercentile(v, 10)),
                        "p90": float(np.nanpercentile(v, 90)), **null[n]} for n, v in mi.items()},
       "violations": {k: {"bad_units": a, "of_units": b, "share": (a / b if b else None)}
                      for k, (a, b) in viol.items()},
       "tiles_with_violation": {k: int(sum(r[k] > 0 for r in per_tile)) for k in viol},
       "count_floor_per_km2": count_floor}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
