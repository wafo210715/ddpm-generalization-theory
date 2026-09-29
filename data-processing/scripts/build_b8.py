"""B8 at 256 px, built one layer per round (issue #2, parts 1-2).

Each round decides one layer and adds its function here.  A layer is written as
one array over every cell, so a round can finish without the others:

    <OUT>/cells.parquet          cell order and geometry: cell_id, z, x, y, bounds
                                 in EPSG:3857, building count, density band
    <OUT>/layers/<name>.npy      (N, H, W) in data space, in cells.parquet order
    <OUT>/layers/<name>.json     provenance: source, rule, script, date, check

When the last grid is decided, `assemble` stacks the grids into the
CellTensorStore that ddpm-win-master trains from (datasets/cell_tensors.py).

Pixel grid.  A cell is its own zoom-15 web-Mercator tile, rasterized on the
tile's exact square in EPSG:3857, so the cells tile the city with no overlap and
no gap.  The preview (layers.py) rasterized each tile's bounding box in UTM
instead; there a tile is rotated 0.2-0.3 degrees and its box covers 0.6-1.2 %
more ground, 1-2 px of overlap per edge at 256.  Web Mercator is spherical, so
a tile is 0.65 % shorter north-south than east-west on the ground: at 256 px a
pixel is 4.69 m by 4.66 m at the centre.  Metric radii in later layers are set
in ground metres per cell for that reason.

Usage:
    python build_b8.py cells
    python build_b8.py building_binary
    python build_b8.py check building_binary
"""
import sys, json, time, pathlib, datetime
import numpy as np, pandas as pd, geopandas as gpd, shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "data/hcmc/raw"
REP = REPO / "data/hcmc/processed/reports"
N = 256
OUT = REPO / f"data/hcmc/processed/b8_{N}"
MERC = 3857
R_EARTH = 6378137.0
CON_DAO = set("15_26086_15589 15_26086_15590 15_26086_15591 15_26087_15589 15_26087_15590 "
              "15_26088_15588 15_26088_15589 15_26088_15590 15_26089_15586 15_26089_15589".split())
BANDS = [(100, 250, "100-249"), (250, 1000, "250-999"), (1000, 2000, "1000-1999"), (2000, 10**9, "2000+")]


# --- cells ---------------------------------------------------------------------

def tile_bounds(z, x, y):
    """Exact EPSG:3857 bounds of a slippy-map tile."""
    size = 2 * np.pi * R_EARTH / 2 ** z
    minx = -np.pi * R_EARTH + x * size
    maxy = np.pi * R_EARTH - y * size
    return minx, maxy - size, minx + size, maxy


def build_cells():
    """The 3 010 training cells (A1 section 3.1), sorted, with exact tile bounds."""
    src = pd.read_parquet(REP / "tile_sources_urban2025.parquet")[["cell_id", "buildings"]]
    src = src[~src.cell_id.isin(CON_DAO)]
    t = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg")
    c = t[["cell_id", "z", "x", "y"]].merge(src, on="cell_id").sort_values("cell_id").reset_index(drop=True)
    assert len(c) == 3010, len(c)
    b = np.array([tile_bounds(z, x, y) for z, x, y in zip(c.z, c.x, c.y)])
    c["minx"], c["miny"], c["maxx"], c["maxy"] = b.T
    # the index geometry and the formula must agree, or the tiles are not what we think
    g = t.set_index("cell_id").loc[c.cell_id].to_crs(MERC).geometry.bounds.values
    assert np.abs(g - b).max() < 0.01, np.abs(g - b).max()
    c["band"] = pd.cut(c.buildings, [lo for lo, _, _ in BANDS] + [10**9], right=False,
                       labels=[lab for _, _, lab in BANDS]).astype(str)
    # Round 1: who drew the cell's footprints.  Google draws buildings at 0.45x
    # Microsoft's median size inside the same cells, so the share is a nuisance
    # variable every footprint-derived layer should be read against.
    prov = pd.read_parquet(REP / "building_sources_per_cell.parquet")
    c = c.merge(prov, on="cell_id", how="left", validate="1:1")
    assert c.filter(like="share_").notna().all().all()
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_parquet(OUT / "cells.parquet")
    print(f"wrote {OUT / 'cells.parquet'}: {len(c)} cells", c.band.value_counts().to_dict())
    return c


def load_cells():
    return pd.read_parquet(OUT / "cells.parquet")


def transform(cell, n=N):
    return from_origin(cell.minx, cell.maxy, (cell.maxx - cell.minx) / n, (cell.maxy - cell.miny) / n)


def write_layer(name, arr, meta):
    (OUT / "layers").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "layers" / f"{name}.npy", arr)
    meta = {"layer": name, "shape": list(arr.shape), "dtype": str(arr.dtype), "unit_px": N,
            "script": "data-processing/scripts/build_b8.py",
            "written": datetime.datetime.now().isoformat(timespec="seconds"), **meta}
    json.dump(meta, open(OUT / "layers" / f"{name}.json", "w"), indent=1)
    print(f"wrote layers/{name}.npy {arr.shape} {arr.dtype}")


# --- layers --------------------------------------------------------------------

def load_buildings():
    return gpd.read_parquet(RAW / "overture_tu/buildings.parquet", columns=["id", "geometry"]).to_crs(MERC)


def building_binary(cells):
    """Round 1, decided 23 Sep: 1 if the pixel centre falls inside an Overture
    footprint, else 0.  Every building that intersects the tile is burned, not
    only those whose centroid lies inside, so the layer is continuous across
    cell edges."""
    b = load_buildings()
    si = b.sindex
    out = np.zeros((len(cells), N, N), np.uint8)
    t0 = time.time()
    for i, cell in enumerate(cells.itertuples()):
        box = shapely.box(cell.minx, cell.miny, cell.maxx, cell.maxy)
        geoms = b.geometry.values[si.query(box, predicate="intersects")]
        if len(geoms):
            out[i] = rasterize(((g, 1) for g in geoms), out_shape=(N, N), transform=transform(cell),
                               fill=0, dtype="uint8", all_touched=False)
        if i % 500 == 0:
            print(f"  {i}/{len(cells)}  {time.time() - t0:.0f}s", flush=True)
    write_layer("building_binary", out, {
        "source": "Overture Maps buildings, release 2026-08-19.0, overture_tu/buildings.parquet",
        "rule": "pixel centre inside a footprint (rasterio all_touched=False)",
        "values": "0/1", "decided": "issue #2 round 1, 23 Sep 2026; discussion/layer_cards.md"})
    return out


LAYERS = {"building_binary": building_binary}


# --- checks --------------------------------------------------------------------

def check_building_binary(cells):
    """Pixel-centre sampling is an unbiased estimate of cover, so each cell's
    raster cover should sit close to the true share of the tile under a
    footprint.  Compared on every cell."""
    a = np.load(OUT / "layers/building_binary.npy", mmap_mode="r")
    assert a.shape == (len(cells), N, N) and set(np.unique(a[:50])) <= {0, 1}
    b = load_buildings(); si = b.sindex
    rows = []
    for i, cell in enumerate(cells.itertuples()):
        box = shapely.box(cell.minx, cell.miny, cell.maxx, cell.maxy)
        geoms = b.geometry.values[si.query(box, predicate="intersects")]
        true = shapely.area(shapely.intersection(shapely.union_all(geoms), box)) / shapely.area(box)
        rows.append((cell.cell_id, cell.band, float(a[i].mean()), float(true)))
    d = pd.DataFrame(rows, columns=["cell_id", "band", "raster", "vector"])
    d["diff"] = d.raster - d.vector
    res = {"cells": len(d), "mean_cover_raster": round(d.raster.mean(), 4), "mean_cover_vector": round(d.vector.mean(), 4),
           "abs_diff_median": round(d["diff"].abs().median(), 4), "abs_diff_p99": round(d["diff"].abs().quantile(.99), 4),
           "abs_diff_max": round(d["diff"].abs().max(), 4),
           "cells_with_no_building_pixel": int((d.raster == 0).sum()),
           "by_band_mean_cover": d.groupby("band").raster.mean().round(4).to_dict()}
    meta = json.load(open(OUT / "layers/building_binary.json"))
    meta["check"] = res
    json.dump(meta, open(OUT / "layers/building_binary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


CHECKS = {"building_binary": check_building_binary}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cells":
        build_cells()
    elif cmd == "check":
        CHECKS[sys.argv[2]](load_cells())
    else:
        LAYERS[cmd](load_cells())
