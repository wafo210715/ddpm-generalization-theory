"""Whole-city coverage: how many places sit in each tile, and from which source.

Renders the tile grid itself -- one pixel per zoom-15 tile -- so the shape is the
municipality and the value is what a cell would actually receive.  Counts are
drawn on a log scale, because place density spans four orders of magnitude and a
linear ramp shows only District 1.
"""
import pathlib, json, base64, io
import numpy as np, geopandas as gpd, pandas as pd
import matplotlib as mpl
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview2025"
UTM = 3405


def tile_png(df, col, x0, y0, nx, ny, cmap, vmax):
    a = np.full((ny, nx), np.nan, "float32")
    a[(df.y - y0).values, (df.x - x0).values] = df[col].values
    n = np.clip(np.log1p(np.nan_to_num(a)) / np.log1p(vmax), 0, 1)
    rgb = (mpl.colormaps[cmap](n)[..., :3] * 255).astype("uint8")
    rgb[np.isnan(a)] = 246                                  # outside the tile set
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((nx * 4, ny * 4), Image.NEAREST).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg")
    t = tiles[["cell_id", "x", "y", "geometry"]].copy()
    ov = gpd.read_parquet(RAW / "overture_tu/places.parquet")[["geometry"]]
    osm = gpd.read_file(RAW / "osm_tu/pois_all.gpkg")[["geometry"]]
    bld = gpd.read_parquet(RAW / "overture_tu/buildings.parquet")[["geometry"]]
    print(f"overture {len(ov):,} | osm {len(osm):,} | buildings {len(bld):,}", flush=True)

    def per_tile(g, name):
        j = gpd.sjoin(g.to_crs(4326), t.to_crs(4326), predicate="within")
        return j.groupby("cell_id").size().rename(name)

    df = t.set_index("cell_id").join(
        pd.concat([per_tile(ov, "overture"), per_tile(osm, "osm"),
                   per_tile(bld, "buildings")], axis=1)).fillna(0).reset_index()

    x0, y0 = int(df.x.min()), int(df.y.min())
    nx, ny = int(df.x.max()) - x0 + 1, int(df.y.max()) - y0 + 1
    mainland = df[df.y - y0 < ny * 0.75]                    # drop the Con Dao strip
    y1 = int(mainland.y.max()) - y0 + 1
    print(f"grid {nx} x {ny}; mainland rows {y1}", flush=True)
    m = mainland.copy()

    vmax = float(df.overture.quantile(0.995))
    imgs = {
        "overture": tile_png(m, "overture", x0, y0, nx, y1, "viridis", vmax),
        "osm":      tile_png(m, "osm",      x0, y0, nx, y1, "viridis", vmax),
        "buildings":tile_png(m, "buildings",x0, y0, nx, y1, "bone_r",
                             float(df.buildings.quantile(0.995))),
    }

    def brackets(col):
        s = df[col]
        return [{"label": lab, "n": int(((s >= lo) & (s < hi)).sum())}
                for lab, lo, hi in [("none", 0, 1), ("1–9", 1, 10), ("10–99", 10, 100),
                                    ("100–999", 100, 1000), ("1 000+", 1000, 1e9)]]

    built = df[df.buildings >= 100]
    out = {
        "n_tiles": len(df), "nx": nx, "ny": y1,
        "totals": {"overture": int(df.overture.sum()), "osm": int(df.osm.sum()),
                   "buildings": int(df.buildings.sum())},
        "vmax": round(vmax, 1),
        "img": imgs,
        "brackets": {"overture": brackets("overture"), "osm": brackets("osm")},
        "built_tiles": len(built),
        "median_on_built": {"overture": float(built.overture.median()),
                            "osm": float(built.osm.median())},
        "zero_on_built": {"overture": int((built.overture == 0).sum()),
                          "osm": int((built.osm == 0).sum())},
        "spearman_with_buildings": {
            "overture": round(float(df[df.buildings > 0].buildings.corr(
                df[df.buildings > 0].overture, method="spearman")), 3),
            "osm": round(float(df[df.buildings > 0].buildings.corr(
                df[df.buildings > 0].osm, method="spearman")), 3)},
    }
    json.dump(out, open(OUT / "coverage_figure.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "img"}, indent=1)[:1200])
    print(f"\nwrote coverage_figure.json ({(OUT/'coverage_figure.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
