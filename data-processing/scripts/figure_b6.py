"""Draw B6: where a diversity ring is non-degenerate, on OSM and on Overture.

B6 asks one question of every unit: put a ring of radius r around it, and does
that ring hold at least two places in at least two categories?  Below that,
Shannon entropy is undefined and the pilot wrote zero.

On OSM the answer was 1.1 % of units at 50 m, which is what pushed the ring out
to 300 m -- and at 300 m the defined region becomes unions of discs around
clusters, so the channel mostly reports distance to the nearest mapped cluster,
which the accessibility channel already says.  This renders that argument
instead of asserting it: the same tiles, the same radii, both sources.
"""
import pathlib, json, base64, io, sys
import numpy as np, geopandas as gpd, pandas as pd
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import matplotlib as mpl

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
OUT = REPO / "data/hcmc/processed/preview2025"
UTM, N, RADII = 3405, 320, [50, 100, 200, 300]

CLASSES = [
    ("food_drink",    r"restaurant|cafe|coffee|bakery|diner|pub|food|ice_cream|dessert|brewer|juice|tea"),
    ("retail",        r"shop|store|shopping|market|grocer|supermarket|boutique|mall|retail"),
    ("services",      r"service|repair|laundry|salon|spa|beauty|barber|cleaning|agency|insurance|bank|financial|atm|legal|consult"),
    ("culture",       r"museum|art|theat|librar|religio|temple|church|mosque|galler|cultur|monument|histor"),
    ("education",     r"school|universit|college|educat|training|kindergarten|academy|tutor"),
    ("health",        r"hospital|clinic|pharmac|doctor|dentist|medical|health|nursing|veterinar"),
    ("accommodation", r"hotel|hostel|motel|lodging|resort|guest"),
    ("leisure",       r"park|gym|fitness|sport|recreation|entertain|nightlife|club|cinema|casino|stadium|golf"),
    ("transport",     r"parking|gas_station|fuel|transport|airport|station|taxi|rail|ferry|charging"),
    ("office",        r"office|company|corporate|business|professional|real_estate"),
    ("public",        r"government|police|fire_|post_office|civic|court|embassy|town_hall|community"),
]


def classify(series):
    cls = pd.Series("other", index=series.index)
    for name, pat in reversed(CLASSES):
        cls[series.astype(str).str.contains(pat, regex=True, na=False)] = name
    return cls


def trees(pts, cls):
    out = []
    for name, _ in CLASSES:
        m = (cls == name).values
        if m.sum():
            out.append(cKDTree(pts[m]))
    return out


def defined_mask(tr, minx, miny, maxx, maxy, r, n):
    res = (maxx - minx) / n
    ys, xs = np.mgrid[0:n, 0:n]
    px = (minx + (xs + .5) * res).ravel(); py = (maxy - (ys + .5) * res).ravel()
    q = np.c_[px, py]
    total = np.zeros(len(q), "int32"); nclass = np.zeros(len(q), "int16")
    for t in tr:
        k = t.query_ball_point(q, r, return_length=True)
        total += k; nclass += (k > 0)
    return ((total >= 2) & (nclass >= 2)).reshape(n, n)


def png(arr, kind, dots=None, minx=None, maxy=None, res=None):
    if kind == "mask":
        rgb = np.where(arr[..., None], np.array([[[34, 102, 68]]]), np.array([[[240, 242, 243]]]))
        im = Image.fromarray(rgb.astype("uint8"))
    else:
        im = Image.new("RGB", (N, N), (240, 242, 243))
    if dots is not None:
        d = ImageDraw.Draw(im)
        for x, y in dots:
            cx, cy = (x - minx) / res, (maxy - y) / res
            d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(28, 94, 134))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
    blds = gpd.read_parquet(RAW / "overture_tu/buildings.parquet").to_crs(UTM)
    ov = gpd.read_parquet(RAW / "overture_tu/places.parquet").to_crs(UTM)
    osm = gpd.read_file(RAW / "osm_tu/pois_all.gpkg").to_crs(UTM)
    print(f"overture {len(ov):,} | osm {len(osm):,}", flush=True)

    cnt = gpd.sjoin(blds[["geometry"]], tiles[["cell_id", "geometry"]],
                    predicate="intersects").groupby("cell_id").size().sort_values()
    # pick across the URBAN range, not across every tile that holds a shed:
    # the 70th percentile of all built tiles is still peri-urban here.
    urban = cnt[cnt >= 100]
    picks = [urban.index[-1],
             urban.index[int(len(urban) * .80)],
             urban.index[int(len(urban) * .45)]]
    labels = ["dense core", "inner urban", "outer urban"]
    print(f"urban tiles (>=100 buildings): {len(urban):,}", flush=True)

    ov_cls = classify(ov["category"]); osm_cls = classify(osm["category_value"])
    ov_pts = np.c_[ov.geometry.x, ov.geometry.y]
    osm_pts = np.c_[osm.geometry.x, osm.geometry.y]
    T = {"Overture": trees(ov_pts, ov_cls), "OpenStreetMap": trees(osm_pts, osm_cls)}
    P = {"Overture": (ov_pts, ov_cls), "OpenStreetMap": (osm_pts, osm_cls)}

    out = {"radii": RADII, "tiles": []}
    for cid, lab in zip(picks, labels):
        g = tiles.set_index("cell_id").loc[cid].geometry
        minx, miny, maxx, maxy = g.bounds
        res = (maxx - minx) / N
        rec = {"cell_id": cid, "label": lab, "buildings": int(cnt[cid]), "sources": {}}
        for src in ("OpenStreetMap", "Overture"):
            pts, cls = P[src]
            inside = ((pts[:, 0] >= minx) & (pts[:, 0] <= maxx) &
                      (pts[:, 1] >= miny) & (pts[:, 1] <= maxy) & (cls != "other").values)
            dots = pts[inside]
            masks, shares = {}, {}
            for r in RADII:
                m = defined_mask(T[src], minx, miny, maxx, maxy, r, N)
                masks[str(r)] = png(m, "mask")
                shares[str(r)] = round(float(m.mean()) * 100, 1)
            rec["sources"][src] = {
                "places": int(inside.sum()),
                "classes": int(len(set(cls[inside]))),
                "dots": png(None, "dots", dots, minx, maxy, res),
                "masks": masks, "share": shares}
            print(f"  {lab:12s} {src:14s} {int(inside.sum()):5d} places  "
                  + "  ".join(f"{r}m {shares[str(r)]:5.1f}%" for r in RADII), flush=True)
        out["tiles"].append(rec)

    json.dump(out, open(OUT / "b6_figure.json", "w"))
    print(f"\nwrote b6_figure.json ({(OUT/'b6_figure.json').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
