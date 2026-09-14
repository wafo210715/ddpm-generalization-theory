"""Render the familiar districts at the real sampling unit.

The whole-city mosaic has to be drawn at 150 m per unit, because at 1.17 m the
city would be about 580 000 pixels across.  A district is small enough to draw at
the unit the model actually sees, which is the only way to check by eye that the
layers describe the place rather than merely covering it.

The 2025 restructure abolished districts, so each familiar name below is the set
of wards that now covers it.
"""
import pathlib, json, base64, io, sys
import numpy as np, geopandas as gpd
import matplotlib as mpl
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from layers import compute, LAYERS
from build_layer_preview import prepare, RAW, OUT, UTM, CMAP

UNIT_M = 1.1730           # the sampling unit at 1024 px per zoom-15 tile
BUFFER_M = 400.0
MAX_PX = 2200             # lands near 2.35 m/px, the tier that still resolves an alley

AREAS = {
    "District 1":  ["Phường Sài Gòn", "Phường Bến Thành", "Phường Cầu Ông Lãnh", "Phường Tân Định"],
    "District 3":  ["Phường Bàn Cờ", "Phường Xuân Hòa", "Phường Nhiêu Lộc"],
    "District 4":  ["Phường Vĩnh Hội", "Phường Khánh Hội", "Phường Xóm Chiếu"],
    "District 5":  ["Phường Chợ Quán", "Phường An Đông", "Phường Chợ Lớn"],
    "District 6":  ["Phường Bình Tây", "Phường Bình Tiên", "Phường Bình Phú",
                    "Phường Phú Lâm", "Phường Minh Phụng"],
    "Gò Vấp":      ["Phường Gò Vấp", "Phường An Nhơn", "Phường Thông Tây Hội",
                    "Phường Hạnh Thông", "Phường An Hội Tây", "Phường An Hội Đông"],
}


def render(a, name, mask):
    a = a.astype("float32")
    sel = a[mask]; sel = sel[np.isfinite(sel)]
    if sel.size == 0: sel = np.zeros(1, "float32")
    lo, hi = float(np.percentile(sel, 1)), float(np.percentile(sel, 99))
    n = np.clip((np.nan_to_num(a) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (mpl.colormaps[CMAP.get(name, "viridis")](n)[..., :3] * 255).astype("uint8")
    rgb[~mask] = 245                                            # outside the area
    buf = io.BytesIO()
    # Footprints are a hard-edged mask, where PNG is both smaller and sharper.
    # Everything else is a continuous field, where JPEG is several times smaller
    # at a quality no one can see the difference from at this scale.
    if name == "building_binary":
        Image.fromarray(rgb).convert("L").save(buf, format="PNG", optimize=True)
        fmt = "png"
    else:
        Image.fromarray(rgb).save(buf, format="JPEG", quality=82, optimize=True,
                                  progressive=True, subsampling=1)
        fmt = "jpeg"
    return base64.b64encode(buf.getvalue()).decode(), lo, hi, buf.getbuffer().nbytes, fmt


def main():
    only = sys.argv[1:] or list(AREAS)
    ctx = prepare()
    comm = gpd.read_file(RAW / "hcmc_communes_2025.gpkg").to_crs(UTM)
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from shapely.ops import unary_union

    out = {}
    for label in only:
        wards = AREAS[label]
        sub = comm[comm["name"].isin(wards)]
        missing = set(wards) - set(sub["name"])
        if missing: print(f"  !! {label}: not found {missing}", flush=True)
        geom = unary_union([g.buffer(0) for g in sub.geometry])
        minx, miny, maxx, maxy = geom.bounds
        w, h = maxx - minx, maxy - miny
        nx = int(round(w / UNIT_M)); ny = int(round(h / UNIT_M))
        scale = min(1.0, MAX_PX / max(nx, ny))
        nx, ny = int(nx * scale), int(ny * scale)
        mpp = w / nx
        print(f"\n{label}: {len(sub)} wards, {geom.area/1e6:.1f} km2, "
              f"{w/1000:.1f} x {h/1000:.1f} km -> {nx} x {ny} px at {mpp:.2f} m/px", flush=True)

        pad = int(round(nx * BUFFER_M / w))
        M = compute(ctx, minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M,
                    nx + 2 * pad, ny + 2 * pad)
        tr = from_origin(minx, maxy, w / nx, h / ny)
        mask = ~geometry_mask([geom], out_shape=(ny, nx), transform=tr, invert=False)

        imgs, total = {}, 0
        for n_, _, _ in LAYERS:
            crop = M[n_][pad:pad + ny, pad:pad + nx]
            p, lo, hi, nb, fmt = render(crop, n_, mask)
            imgs[n_] = {"img": p, "fmt": fmt, "lo": round(lo, 3), "hi": round(hi, 3)}
            total += nb
        out[label] = {"wards": list(sub["name"]), "km2": round(geom.area / 1e6, 2),
                      "nx": nx, "ny": ny, "m_per_px": round(mpp, 2),
                      "km_w": round(w / 1000, 2), "km_h": round(h / 1000, 2),
                      "layers": imgs}
        print(f"   images {total/1e6:.1f} MB", flush=True)

    dst = OUT / "districts.json"
    prev = json.load(open(dst)) if dst.exists() and len(only) < len(AREAS) else {}
    prev.update(out)
    json.dump(prev, open(dst, "w"))
    print(f"\nwrote districts.json ({dst.stat().st_size/1e6:.1f} MB) with {len(prev)} areas")


if __name__ == "__main__":
    main()
