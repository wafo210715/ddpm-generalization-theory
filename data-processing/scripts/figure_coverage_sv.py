"""Add the street view layer to the coverage figure, on the same grid and scale."""
import pathlib, json, base64, io
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview2025"


def tile_png(df, col, x0, y0, nx, ny, cmap, vmax, log=True):
    a = np.full((ny, nx), np.nan, "float32")
    ok = (df.y - y0 >= 0) & (df.y - y0 < ny) & (df.x - x0 >= 0) & (df.x - x0 < nx)
    d = df[ok]
    a[(d.y - y0).values, (d.x - x0).values] = d[col].values
    v = np.nan_to_num(a)
    n = np.clip((np.log1p(v) / np.log1p(vmax)) if log else ((v - vmax[0]) / (vmax[1] - vmax[0])), 0, 1)
    rgb = (mpl.colormaps[cmap](n)[..., :3] * 255).astype("uint8")
    rgb[np.isnan(a)] = 246
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((nx * 4, ny * 4), Image.NEAREST).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    fig = json.load(open(OUT / "coverage_figure.json"))
    m = pd.read_parquet(OUT / "mapillary_per_cell.parquet")
    t = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg")[["cell_id", "x", "y"]]
    bl = pd.read_parquet(OUT / "tile_channels.parquet") if (OUT / "tile_channels.parquet").exists() else None

    x0, y0, nx, ny = int(t.x.min()), int(t.y.min()), fig["nx"], fig["ny"]
    vmax = float(m.images.quantile(0.995))
    fig["img"]["mapillary"] = tile_png(m, "images", x0, y0, nx, ny, "magma", vmax)

    yr = m[m.images > 0].copy()
    fig["img"]["mly_year"] = tile_png(yr, "year", x0, y0, nx, ny, "cividis",
                                      (2015.0, 2025.0), log=False)

    covered = m[m.images > 0]
    fig["sv"] = {
        "total": int(m.images.sum()),
        "cells_any": int((m.images > 0).sum()),
        "cells": int(len(m)),
        "pct_cells": round(100 * (m.images > 0).mean(), 1),
        "median_per_covered": int(covered.images.median()) if len(covered) else 0,
        "pano_pct": round(100 * m.pano.sum() / max(int(m.images.sum()), 1), 1),
        "vmax": round(vmax, 0),
        "years": {str(int(k)): int(v) for k, v in
                  covered.year.value_counts().sort_index().items() if not np.isnan(k)},
        "brackets": [{"label": lab, "n": int(((m.images >= lo) & (m.images < hi)).sum())}
                     for lab, lo, hi in [("none", 0, 1), ("1–99", 1, 100),
                                         ("100–999", 100, 1000), ("1 000–9 999", 1000, 10000),
                                         ("10 000+", 10000, 1e12)]],
    }

    # how does imagery line up with the fabric, and with places?
    join = t.merge(m[["x", "y", "images"]], on=["x", "y"], how="left").fillna({"images": 0})
    ovt = gpd.read_parquet(RAW / "overture_tu/places.parquet")[["geometry"]]
    bld = gpd.read_parquet(RAW / "overture_tu/buildings.parquet")[["geometry"]]
    tg = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg")[["cell_id", "geometry"]]
    for g, name in [(ovt, "places"), (bld, "buildings")]:
        c = gpd.sjoin(g.to_crs(4326), tg.to_crs(4326), predicate="within").groupby("cell_id").size()
        join[name] = join.cell_id.map(c).fillna(0)
    built = join[join.buildings >= 100]
    fig["sv"]["spearman_buildings"] = round(float(join.images.corr(join.buildings, method="spearman")), 3)
    fig["sv"]["spearman_places"] = round(float(join.images.corr(join.places, method="spearman")), 3)
    fig["sv"]["built_covered_pct"] = round(100 * (built.images > 0).mean(), 1)
    fig["sv"]["built_median"] = int(built.images.median())
    fig["sv"]["built_zero"] = int((built.images == 0).sum())
    fig["sv"]["built_tiles"] = int(len(built))

    json.dump(fig, open(OUT / "coverage_figure.json", "w"))
    s = fig["sv"]
    print(f"images            {s['total']:,}")
    print(f"cells with any    {s['cells_any']:,} of {s['cells']:,}  ({s['pct_cells']} %)")
    print(f"median per covered{s['median_per_covered']:,}")
    print(f"panoramic         {s['pano_pct']} %")
    print(f"built-up covered  {s['built_covered_pct']} %  (zero in {s['built_zero']:,} of {s['built_tiles']:,})")
    print(f"spearman vs buildings {s['spearman_buildings']}, vs places {s['spearman_places']}")
    print("years:", s["years"])


if __name__ == "__main__":
    main()
