"""Tile renders and per-unit distributions, for the 2025 city.

The companion to build_layer_preview.py, which does the city mosaic.  Same
prepare() and the same layers.compute(), so a value shown here and a value shown
on the mosaic are the same quantity.

Tiles are rendered with a buffer and then cropped, so a 100 m disc or a 300 m
ring near a tile edge still reads real neighbours rather than the edge of an
array.  That buffer is why the dataset build fetches beyond the tile too.
"""
import pathlib, json, sys
import numpy as np, geopandas as gpd, pandas as pd
from scipy.stats import skew

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from layers import compute, LAYERS
from build_layer_preview import prepare, png, OUT, RAW, UTM

BUFFER_M = 400.0          # beyond the tile, so neighbourhood scales are honest
RENDER_N = 512
MARGIN_N = 128
N_SAMPLE = 80


def tile_layers(ctx, geom, N):
    minx, miny, maxx, maxy = geom.bounds
    f = BUFFER_M / (maxx - minx)
    pad = int(round(N * f))
    M = compute(ctx, minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M,
                N + 2 * pad)
    return {k: v[pad:pad + N, pad:pad + N] for k, v in M.items()}


def main():
    ctx = prepare()
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM).set_index("cell_id")

    # rank tiles by built density using the mosaic we already computed
    mj = json.load(open(OUT / "mosaic.json"))
    print("picking tiles across the density range …", flush=True)
    blds = gpd.read_parquet(RAW / "overture_tu/buildings.parquet").to_crs(UTM)
    j = gpd.sjoin(blds[["geometry"]], tiles.reset_index()[["cell_id", "geometry"]],
                  predicate="intersects")
    cnt = j.groupby("cell_id").size().sort_values()
    built = cnt[cnt >= 50]
    picks = [built.index[int(q * (len(built) - 1))] for q in (0.999, 0.93, 0.72, 0.45, 0.18)]
    picks = list(dict.fromkeys(picks))
    print("  ", picks, flush=True)

    gallery = []
    for cid in picks:
        L = tile_layers(ctx, tiles.loc[cid].geometry, RENDER_N)
        imgs = {}
        for name, _, _ in LAYERS:
            p, lo, hi = png(L[name], name)
            imgs[name] = p
        gallery.append({"cell_id": cid, "buildings": int(cnt.get(cid, 0)), "png": imgs})
        print("   rendered", cid, flush=True)

    rng = np.random.default_rng(0)
    samp = list(rng.choice(built.index.values, size=min(N_SAMPLE, len(built)), replace=False))
    pool = {n: [] for n, _, _ in LAYERS}
    for i, cid in enumerate(samp):
        L = tile_layers(ctx, tiles.loc[cid].geometry, MARGIN_N)
        for n, _, _ in LAYERS:
            pool[n].append(L[n].ravel())
        if i % 20 == 0: print(f"   marginal {i}/{len(samp)}", flush=True)
    pool = {k: np.nan_to_num(np.concatenate(v)) for k, v in pool.items()}

    # Per-CELL as well as per-unit.  A unit is one pixel; a cell is one whole
    # tile, which is one training example.  B10 needs the per-unit marginal,
    # because the transform and standardization act on unit values.  B10b needs
    # the per-cell distribution, because that is the object E3 compares against
    # generated cells.  They answer different questions and both belong here.
    per_cell = {n: [] for n, _, _ in LAYERS}
    for cid in samp:
        L = tile_layers(ctx, tiles.loc[cid].geometry, MARGIN_N)
        for n, _, _ in LAYERS:
            per_cell[n].append(float(np.nan_to_num(L[n]).mean()))
    cellm = {}
    for n, v in per_cell.items():
        v = np.array(v)
        hist, edges = np.histogram(np.log1p(np.abs(v)), bins=24)
        cellm[n] = {"mean": round(float(v.mean()), 3),
                    "sd": round(float(v.std()), 3),
                    "min": round(float(v.min()), 3),
                    "max": round(float(v.max()), 3),
                    "hist": hist.tolist(),
                    "edges": [round(float(e), 3) for e in edges]}
    cellM = pd.DataFrame(per_cell)
    cellcorr = cellM.corr(method="spearman").fillna(0).round(3)

    marg = {}
    for k, v in pool.items():
        nz = v[v > 0]
        hist, edges = np.histogram(np.log1p(np.abs(v)), bins=40)
        marg[k] = {"zero_pct": round(float((v == 0).mean() * 100), 2),
                   "skew": round(float(skew(v)), 2),
                   "mean": round(float(v.mean()), 3),
                   "p50": round(float(np.median(nz)) if nz.size else 0.0, 3),
                   "p99": round(float(np.percentile(v, 99)), 3),
                   "max": round(float(v.max()), 3),
                   "hist": hist.tolist(),
                   "edges": [round(float(e), 3) for e in edges]}
    M = pd.DataFrame(pool)
    corr = M.corr(method="spearman").fillna(0).round(3)

    json.dump({"gallery": gallery, "marginals": marg,
               "corr": {"cols": [n for n, _, _ in LAYERS], "m": corr.values.tolist()},
               "cell_marginals": cellm,
               "cell_corr": {"cols": [n for n, _, _ in LAYERS], "m": cellcorr.values.tolist()},
               "n_sample_tiles": len(samp), "render_n": RENDER_N,
               "buffer_m": BUFFER_M},
              open(OUT / "tiles.json", "w"))
    print("\nper-unit distributions:")
    print(pd.DataFrame(marg).T[["zero_pct", "skew", "mean", "p99", "max"]].to_string())
    print(f"\nwrote tiles.json ({(OUT/'tiles.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
