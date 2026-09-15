"""Extract 3D-GloBFP's Vietnam shapefile and clip it to the study area.

This is the source that makes floor-area ratio possible.  OSM tags storeys on
3.9% of HCMC buildings and height on 0.37%, and Overture is worse at 0.02%, so a
per-building height dataset is the only route to FAR.  3D-GloBFP ships one
shapefile per country, ~4.5 GB uncompressed for Vietnam, of which HCMC is a
small fraction.

The archive is RAR5; libarchive reads it with no external binary.
"""
import pathlib, shutil
import libarchive
import geopandas as gpd

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "data/external/3dglobfp/Vietnam.rar"
WORK = REPO / "data/external/3dglobfp/vietnam"
OUT = REPO / "data/hcmc/raw/heights_3dglobfp_tu.parquet"
BOUNDARY = REPO / "data/hcmc/raw/hcmc_fetch_area.gpkg"   # the tile union, not the municipality


def extract() -> pathlib.Path:
    shp = WORK / "Vietnam.shp"
    if shp.exists():
        print(f"already extracted ({shp.stat().st_size/1e9:.1f} GB)", flush=True)
        return shp
    WORK.mkdir(parents=True, exist_ok=True)
    print("extracting ...", flush=True)
    cwd = pathlib.Path.cwd()
    try:
        import os
        os.chdir(WORK)
        libarchive.extract_file(str(SRC))
    finally:
        import os
        os.chdir(cwd)
    print(f"  extracted ({shp.stat().st_size/1e9:.1f} GB)", flush=True)
    return shp


def main():
    shp = extract()
    b = gpd.read_file(BOUNDARY)
    poly = b.geometry.union_all()
    minx, miny, maxx, maxy = poly.bounds
    print(f"reading within bbox {minx:.3f} {miny:.3f} {maxx:.3f} {maxy:.3f} ...", flush=True)
    g = gpd.read_file(shp, bbox=(minx, miny, maxx, maxy))
    print(f"  {len(g):,} buildings in bbox; columns {list(g.columns)}", flush=True)
    g = g[g.intersects(poly)]
    print(f"  {len(g):,} inside the study area", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT)
    print(f"  -> {OUT.name} ({OUT.stat().st_size/1e6:.0f} MB)", flush=True)

    hcol = next((c for c in g.columns if c.lower() in ("height", "h", "bh", "hgt")), None)
    if hcol:
        h = g[hcol].astype(float)
        print(f"\nheight column '{hcol}': min {h.min():.1f}, median {h.median():.1f}, "
              f"mean {h.mean():.1f}, p99 {h.quantile(.99):.1f}, max {h.max():.1f} m")
        print(f"  non-null: {100*h.notna().mean():.1f}%")


if __name__ == "__main__":
    main()
