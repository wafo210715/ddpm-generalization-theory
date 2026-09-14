"""Rasterize sample tiles, render them, and pool the per-unit marginals.

Three outputs, all feeding the B10 checkpoint:

  1. PNGs of each channel for a handful of tiles spanning the density range, so
     the data can be looked at rather than described.
  2. Pooled per-unit marginals across a larger tile sample: zero fraction, skew,
     percentiles, histogram.  B10 chooses each channel's transform from these.
     Gaussianity is not the target and is not expected.
  3. The correlation matrix between channels -- the C1 correlationist baseline,
     run on our own data, which is also what flags near-duplicate channels.

Plus a resolution ladder on one tile (128/256/512/1024) for the open question of
whether 128 is too lossy to carry the fabric.
"""
import pathlib, json, base64, io
import numpy as np, geopandas as gpd, pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree
from scipy.stats import skew
import matplotlib as mpl
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview"
IMG = OUT / "tiles"; IMG.mkdir(parents=True, exist_ok=True)
UTM = 3405
CH = ["build", "height", "road", "poi", "green", "water"]
CMAP = {"build":"bone", "height":"magma", "road":"pink",
        "poi":"viridis", "green":"YlGn", "water":"Blues"}

tiles = gpd.read_file(RAW / "hcmc_tiles_z15.gpkg").to_crs(UTM).set_index("cell_id")
blds  = gpd.read_parquet(RAW / "overture/buildings.parquet").to_crs(UTM)
hts   = gpd.read_parquet(RAW / "heights_3dglobfp.parquet").to_crs(UTM)
roads = gpd.read_file(RAW / "osm/roads.gpkg").to_crs(UTM)
places= gpd.read_parquet(RAW / "overture/places.parquet").to_crs(UTM)
wc    = rasterio.open(REPO/"data/external/worldcover/ESA_WorldCover_10m_2021_v200_N09E105_Map.tif")
GREEN = [10,20,30,40,95]
ptree = cKDTree(np.c_[places.geometry.x.values, places.geometry.y.values])
bsi, hsi, rsi = blds.sindex, hts.sindex, roads.sindex

def channels(cid, N):
    g = tiles.loc[cid].geometry
    minx, miny, maxx, maxy = g.bounds
    res = (maxx-minx)/N
    tr = from_origin(minx, maxy, res, res)
    out = {}
    sub = blds.iloc[list(bsi.query(g, predicate="intersects"))]
    out["build"] = rasterize(zip(sub.geometry, np.ones(len(sub))), out_shape=(N,N),
                             transform=tr, fill=0, dtype="float32") if len(sub) else np.zeros((N,N),"float32")
    sh = hts.iloc[list(hsi.query(g, predicate="intersects"))]
    out["height"] = rasterize(zip(sh.geometry, sh["Height"].astype("float32")), out_shape=(N,N),
                              transform=tr, fill=0, dtype="float32") if len(sh) else np.zeros((N,N),"float32")
    sr = roads.iloc[list(rsi.query(g, predicate="intersects"))]
    out["road"] = rasterize(zip(sr.geometry, np.ones(len(sr))), out_shape=(N,N),
                            transform=tr, fill=0, dtype="float32") if len(sr) else np.zeros((N,N),"float32")
    ys, xs = np.mgrid[0:N, 0:N]
    px = minx + (xs+0.5)*res; py = maxy - (ys+0.5)*res
    k = ptree.query_ball_point(np.c_[px.ravel(), py.ravel()], 100.0, return_length=True)
    out["poi"] = (k.reshape(N,N) / (np.pi*100**2/1e6)).astype("float32")   # per km2
    wb = transform_bounds(UTM, wc.crs, minx, miny, maxx, maxy)
    win = rasterio.windows.from_bounds(*wb, transform=wc.transform)
    a = wc.read(1, window=win, out_shape=(N,N), boundless=True, fill_value=0,
                resampling=rasterio.enums.Resampling.nearest)
    out["green"] = np.isin(a, GREEN).astype("float32")
    out["water"] = (a == 80).astype("float32")
    return out

def png(arr, name, hi=None):
    a = arr.astype("float32")
    hi = hi if hi is not None else (np.percentile(a[a>0], 99) if (a>0).any() else 1.0)
    n = np.clip(a/max(hi,1e-6), 0, 1)
    rgb = (mpl.colormaps[CMAP[name]](n)[..., :3]*255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

summ = pd.read_parquet(OUT/"tile_channels.parquet").set_index("cell_id")
built = summ[summ.build > 0.02].sort_values("build")
picks = [built.index[int(q*(len(built)-1))] for q in (0.99, 0.85, 0.6, 0.35, 0.12)]
picks = list(dict.fromkeys(picks))
print("rendering tiles:", picks, flush=True)

gallery = []
for cid in picks:
    c = channels(cid, 512)
    gallery.append({"cell_id": cid,
                    "stats": {k: round(float(summ.loc[cid, k]), 4) for k in ["build","height","far","road","poi","green","water"]},
                    "png": {k: png(c[k], k) for k in CH}})
    print("  ", cid, flush=True)

# resolution ladder on the densest pick
ladder = []
for N in (128, 256, 512, 1024):
    c = channels(picks[0], N)
    b = c["build"]
    ladder.append({"N": N, "m_per_px": round(1201.0/N, 2),
                   "coverage": round(float(b.mean()), 4),
                   "edges": round(float(np.abs(np.diff(b,axis=1)).mean()+np.abs(np.diff(b,axis=0)).mean()), 4),
                   "png": png(b, "build")})
    print("  ladder", N, flush=True)

# pooled marginals over a wider sample
rng = np.random.default_rng(0)
samp = list(rng.choice(built.index.values, size=min(60, len(built)), replace=False))
pool = {k: [] for k in CH}
for i, cid in enumerate(samp):
    c = channels(cid, 128)
    for k in CH: pool[k].append(c[k].ravel())
    if i % 15 == 0: print(f"  marginal {i}/{len(samp)}", flush=True)
pool = {k: np.concatenate(v) for k, v in pool.items()}

marg = {}
for k, v in pool.items():
    nz = v[v > 0]
    hist, edges = np.histogram(np.log1p(v), bins=40)
    marg[k] = {"zero_pct": round(float((v==0).mean()*100), 2),
               "skew": round(float(skew(v)), 2),
               "mean": round(float(v.mean()), 3),
               "p50": round(float(np.median(nz)) if nz.size else 0.0, 3),
               "p99": round(float(np.percentile(v, 99)), 3),
               "max": round(float(v.max()), 3),
               "hist": hist.tolist(), "edges": [round(float(e),3) for e in edges]}
M = pd.DataFrame({k: v for k, v in pool.items()})
corr = M.corr(method="spearman").round(3)

json.dump({"gallery": gallery, "ladder": ladder, "marginals": marg,
           "corr": {"cols": CH, "m": corr.values.tolist()},
           "n_sample_tiles": len(samp)}, open(OUT/"preview.json","w"))
print("\nmarginals:"); print(pd.DataFrame(marg).T[["zero_pct","skew","mean","p99","max"]].to_string())
print("\nspearman correlation between channels:"); print(corr.to_string())
print(f"\nwrote preview.json ({(OUT/'preview.json').stat().st_size/1e6:.1f} MB)")
