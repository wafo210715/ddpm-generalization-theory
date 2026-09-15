"""Count street view imagery in every tile of the study area.

Mapillary serves image points as vector tiles at zoom 14, and each zoom-14 tile
is exactly the four zoom-15 tiles of our grid, so a point's quadrant within the
tile gives its cell without any coordinate conversion.

Per cell this records how many images there are, what share are panoramic, and
when they were taken -- the last because imagery from 2019 and imagery from 2025
are different measurements of a city that rebuilds this fast.

Needs MAPILLARY_TOKEN in the environment.  The token is a credential: it is read
from .env, which is gitignored, and is never written into any output.
"""
import os, sys, json, time, pathlib
import numpy as np, pandas as pd, geopandas as gpd, requests
from mapbox_vector_tile import decode

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview2025"
CACHE = OUT / "mapillary_tiles.parquet"
URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/14/{x}/{y}"

tok = os.environ.get("MAPILLARY_TOKEN")
if not tok:
    sys.exit("MAPILLARY_TOKEN not set (source .env first)")


def fetch(px, py, tries=3):
    for a in range(tries):
        try:
            r = requests.get(URL.format(x=px, y=py), params={"access_token": tok}, timeout=120)
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return r.content
        except Exception as e:
            if a == tries - 1:
                print(f"    give up {px}/{py}: {type(e).__name__}", flush=True)
                return None
            time.sleep(4 * (a + 1))


def main():
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg")[["cell_id", "x", "y"]]
    tiles["px"], tiles["py"] = tiles.x // 2, tiles.y // 2
    parents = tiles.groupby(["px", "py"]).size().reset_index()[["px", "py"]]
    print(f"{len(tiles):,} cells under {len(parents):,} zoom-14 parents", flush=True)

    rows, t0, failed = [], time.time(), 0
    for i, r in enumerate(parents.itertuples()):
        px, py = int(r.px), int(r.py)
        raw = fetch(px, py)
        if raw is None:
            failed += 1
            continue
        counts = {}
        if len(raw) > 50:
            d = decode(raw)
            lay = d.get("image")
            if lay:
                ext = lay.get("extent", 4096)
                half = ext / 2
                for f in lay["features"]:
                    cx, cy = f["geometry"]["coordinates"][:2]
                    # MVT y grows downward, matching web-mercator tile y
                    qx, qy = (0 if cx < half else 1), (0 if cy < half else 1)
                    key = (2 * px + qx, 2 * py + qy)
                    p = f["properties"]
                    a = counts.setdefault(key, [0, 0, []])
                    a[0] += 1
                    a[1] += 1 if p.get("is_pano") else 0
                    ts = p.get("captured_at")
                    if ts:
                        a[2].append(ts)
        for (gx, gy), (n, npano, ts) in counts.items():
            rows.append({"x": gx, "y": gy, "images": n, "pano": npano,
                         "median_ts": float(np.median(ts)) if ts else np.nan})
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(parents)}  {el:5.0f}s  eta {el/max(i,1)*(len(parents)-i)/60:4.1f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE)
    print(f"\nfailed parents: {failed}")

    j = tiles.merge(df, on=["x", "y"], how="left")
    j["images"] = j["images"].fillna(0).astype(int)
    j["pano"] = j["pano"].fillna(0).astype(int)
    j["year"] = pd.to_datetime(j["median_ts"], unit="ms", errors="coerce").dt.year
    j.to_parquet(OUT / "mapillary_per_cell.parquet")

    tot = int(j.images.sum())
    print(f"\nimages in the study area : {tot:,}")
    print(f"cells with any imagery   : {int((j.images>0).sum()):,} of {len(j):,}"
          f"  ({100*(j.images>0).mean():.1f} %)")
    print(f"median images per covered cell: {int(j[j.images>0].images.median()):,}")
    print(f"panoramic share          : {100*j.pano.sum()/max(tot,1):.1f} %")
    print("\ncapture year of the median image, by cell:")
    print(j[j.images>0].year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
