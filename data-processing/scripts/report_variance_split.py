"""How much of each layer's variance lives inside a tile, at the real unit?

The earlier version of this number was computed on the city mosaic at 150 m,
which is a rendering size, not a sampling unit.  That understates the answer for
exactly the layers whose information is in fine detail, so it is redone here at
the sampling units the model would actually be trained on.

Law of total variance over a tile sample:
    total  =  E[ Var(units within a tile) ]  +  Var( tile means )
The first term is what a tile mean throws away; the second is what it keeps.
Reported at four resolutions, which also answers whether 128 px is a cheaper
version of the same experiment or a different one.
"""
import pathlib, json, sys
import numpy as np, geopandas as gpd, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from layers import LAYERS
from build_layer_preview import prepare, RAW, OUT, UTM
from build_layer_tiles import tile_layers

# from one value per tile up to the sampling unit, so the transition is visible
RESOLUTIONS = [2, 4, 8, 16, 32, 64, 128, 512, 1024]
N_TILES = 40


def main():
    ctx = prepare()
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM).set_index("cell_id")
    blds = gpd.read_parquet(RAW / "overture_2025/buildings.parquet").to_crs(UTM)
    j = gpd.sjoin(blds[["geometry"]], tiles.reset_index()[["cell_id", "geometry"]],
                  predicate="intersects")
    cnt = j.groupby("cell_id").size()
    built = cnt[cnt >= 100]
    rng = np.random.default_rng(1)
    samp = list(rng.choice(built.index.values, size=min(N_TILES, len(built)), replace=False))
    print(f"sampling {len(samp)} built-up tiles of {len(built):,}\n", flush=True)

    result = {}
    for N in RESOLUTIONS:
        means = {n: [] for n, _, _ in LAYERS}
        wvars = {n: [] for n, _, _ in LAYERS}
        for i, cid in enumerate(samp):
            L = tile_layers(ctx, tiles.loc[cid].geometry, N)
            for n, _, _ in LAYERS:
                a = np.nan_to_num(L[n]).astype("float64")
                means[n].append(a.mean()); wvars[n].append(a.var())
            if i % 10 == 0:
                print(f"  {N} px: {i}/{len(samp)}", flush=True)
        res = {}
        for n, _, _ in LAYERS:
            within = float(np.mean(wvars[n]))
            between = float(np.var(means[n]))
            tot = within + between
            res[n] = {"within": within, "between": between,
                      "share": round(within / tot, 4) if tot > 0 else 0.0}
        result[N] = res
        print(f"  {N} px done\n", flush=True)

    json.dump({"resolutions": RESOLUTIONS, "n_tiles": len(samp), "split": result},
              open(OUT / "variance_split.json", "w"))

    print("share of variance INSIDE a tile, by sampling unit\n")
    hdr = " ".join(f"{N:>5d}" for N in RESOLUTIONS)
    print(f"{'layer':18s} {hdr}   metres per px")
    order = sorted([n for n, _, _ in LAYERS], key=lambda n: -result[1024][n]["share"])
    for n in order:
        row = " ".join(f"{100*result[N][n]['share']:5.1f}" for N in RESOLUTIONS)
        print(f"{n:18s} {row}")
    print(f"\n{'m/px':18s} " + " ".join(f"{1201/N:5.0f}" for N in RESOLUTIONS))


if __name__ == "__main__":
    main()
