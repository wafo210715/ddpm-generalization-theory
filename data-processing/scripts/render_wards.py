"""Render each ward on the tiles it actually sits on, at one unit for the whole city.

Three corrections to the earlier district renders.

  ONE UNIT EVERYWHERE.  The previous version fitted each area's bounding box to a
  fixed image width, so the metres-per-pixel fell out differently for every
  district.  The unit is a property of the grid, not of the thing being drawn, so
  it is fixed here at PX_PER_TILE for every ward.

  WHOLE TILES, SQUARE.  A ward is not a rectangle and the model does not see one.
  Each render is the square block of zoom-15 tiles covering the ward, padded out
  with real neighbouring tiles rather than blank space, with the tile grid drawn
  so the cell structure is visible and the ward outlined in red.

  WARD NAMES.  Districts were abolished on 1 July 2025.  The city is 113 wards
  (phuong), 54 communes (xa) and one special zone, so the old numbers are gone.
"""
import pathlib, json, base64, io, sys
import numpy as np, geopandas as gpd
import matplotlib as mpl
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from layers import compute, LAYERS
from build_layer_preview import prepare, RAW, OUT, UTM, CMAP

PX_PER_TILE = 320                      # 1201 m / 320 = 3.75 m per unit, everywhere
BUFFER_M = 400.0
TILE_M = 1201.0

WARDS = [
    "Phường Sài Gòn", "Phường Bến Thành", "Phường Cầu Ông Lãnh", "Phường Tân Định",
    "Phường Bàn Cờ", "Phường Xuân Hòa", "Phường Nhiêu Lộc",
    "Phường Vĩnh Hội", "Phường Khánh Hội", "Phường Xóm Chiếu",
    "Phường Chợ Quán", "Phường An Đông", "Phường Chợ Lớn",
    "Phường Bình Tây", "Phường Bình Tiên", "Phường Bình Phú",
    "Phường Phú Lâm", "Phường Minh Phụng",
    "Phường Gò Vấp", "Phường An Nhơn", "Phường Thông Tây Hội",
    "Phường Hạnh Thông", "Phường An Hội Tây", "Phường An Hội Đông",
]


def encode(a, name, mask_in, tiles_px, outline_px):
    """Colour-map, then draw the tile grid and the ward outline on top."""
    a = a.astype("float32")
    sel = a[mask_in]; sel = sel[np.isfinite(sel)]
    if sel.size == 0: sel = np.zeros(1, "float32")
    lo, hi = float(np.percentile(sel, 1)), float(np.percentile(sel, 99))
    n = np.clip((np.nan_to_num(a) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (mpl.colormaps[CMAP.get(name, "viridis")](n)[..., :3] * 255).astype("uint8")
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for gx in tiles_px["xs"]:
        d.line([(gx, 0), (gx, im.height)], fill=(140, 150, 158), width=1)
    for gy in tiles_px["ys"]:
        d.line([(0, gy), (im.width, gy)], fill=(140, 150, 158), width=1)
    for ring in outline_px:
        if len(ring) > 2:
            d.line(ring + [ring[0]], fill=(214, 44, 32), width=3, joint="curve")
    buf = io.BytesIO()
    if name == "building_binary":
        im.save(buf, format="PNG", optimize=True); fmt = "png"
    else:
        im.save(buf, format="JPEG", quality=78, optimize=True, progressive=True, subsampling=1)
        fmt = "jpeg"
    return base64.b64encode(buf.getvalue()).decode(), lo, hi, buf.getbuffer().nbytes, fmt


def main():
    only = sys.argv[1:] or WARDS
    ctx = prepare()
    comm = gpd.read_file(RAW / "hcmc_communes_2025.gpkg").to_crs(UTM)
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM)

    out, total_mb = {}, 0.0
    for wname in only:
        sub = comm[comm["name"] == wname]
        if sub.empty:
            print(f"  !! {wname} not found"); continue
        geom = sub.geometry.iloc[0].buffer(0)
        hit = tiles[tiles.intersects(geom)]
        x0, x1 = int(hit.x.min()), int(hit.x.max())
        y0, y1 = int(hit.y.min()), int(hit.y.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        s = max(w, h)                                   # square, in whole tiles
        x0 -= (s - w) // 2; y0 -= (s - h) // 2          # centred, real neighbours
        blk = tiles[(tiles.x >= x0) & (tiles.x < x0 + s) & (tiles.y >= y0) & (tiles.y < y0 + s)]
        minx, miny, maxx, maxy = blk.total_bounds
        # square in metres too, from the tile grid rather than the ward's shape
        maxx = minx + s * TILE_M; miny = maxy - s * TILE_M
        N = s * PX_PER_TILE
        mpp = (maxx - minx) / N

        pad = int(round(N * BUFFER_M / (maxx - minx)))
        M = compute(ctx, minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M,
                    N + 2 * pad)
        crop = {k: v[pad:pad + N, pad:pad + N] for k, v in M.items()}

        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin
        tr = from_origin(minx, maxy, mpp, mpp)
        mask_in = ~geometry_mask([geom], out_shape=(N, N), transform=tr, invert=False)

        grid_px = {"xs": [round(i * PX_PER_TILE) for i in range(1, s)],
                   "ys": [round(i * PX_PER_TILE) for i in range(1, s)]}
        polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        outline = []
        for p in polys:
            outline.append([(int((px - minx) / mpp), int((maxy - py) / mpp))
                            for px, py in p.exterior.coords])

        imgs, nb = {}, 0
        for nm, _, _ in LAYERS:
            p, lo, hi, b, fmt = encode(crop[nm], nm, mask_in, grid_px, outline)
            imgs[nm] = {"img": p, "fmt": fmt, "lo": round(lo, 3), "hi": round(hi, 3)}
            nb += b
        total_mb += nb / 1e6
        out[wname] = {"km2": round(geom.area / 1e6, 2), "tiles_covering": int(len(hit)),
                      "block": f"{s} × {s}", "px": N, "m_per_px": round(mpp, 2),
                      "cells": [f"{int(r.x)}/{int(r.y)}" for r in blk.itertuples()],
                      "layers": imgs}
        print(f"{wname:26s} {geom.area/1e6:5.2f} km2  {len(hit):2d} tiles  "
              f"block {s}x{s}  {N}px  {mpp:.2f} m/px  {nb/1e6:4.1f} MB", flush=True)

    dst = OUT / "wards.json"
    prev = json.load(open(dst)) if dst.exists() and len(only) < len(WARDS) else {}
    prev.update(out)
    json.dump(prev, open(dst, "w"))
    print(f"\nwrote wards.json ({dst.stat().st_size/1e6:.1f} MB), "
          f"{len(prev)} wards, {total_mb:.1f} MB of imagery")


if __name__ == "__main__":
    main()
