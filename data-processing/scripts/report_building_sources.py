"""Issue #2, round 1 (23 Sep 2026): which provider drew each footprint, and does it show.

Overture fills its building layer in a fixed order -- OSM, Esri, high-precision
Google, Microsoft, low-precision Google -- so the provider can change from one
patch of city to the next.  If the providers draw buildings differently, that
change is in the footprint layer and a model trained on it will learn it as if
it were the city.  Two questions:

  style     size, width and elongation of footprints by provider
  patches   each cell's provider mix, drawn on the tile grid

Sources from fetch_overture_hcmc.py building_sources, joined on id.  Cells are
the 3 010 (3 020 urban tiles minus Con Dao); buildings by centroid.
Writes reports/building_sources.json and discussion/layer_cards_fig/building_sources.png.
"""
import pathlib, json
import numpy as np, pandas as pd, geopandas as gpd, shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / "data-processing/data/hcmc/raw"
REP = REPO / "data-processing/data/hcmc/processed/reports"
FIG = REPO / "discussion/layer_cards_fig/building_sources.png"
UTM = 3405
CON_DAO = set("15_26086_15589 15_26086_15590 15_26086_15591 15_26087_15589 15_26087_15590 "
              "15_26088_15588 15_26088_15589 15_26088_15590 15_26089_15586 15_26089_15589".split())
SHORT = {"Microsoft ML Buildings": "Microsoft", "Google Open Buildings": "Google", "OpenStreetMap": "OSM"}

src = pd.read_parquet(REP / "tile_sources_urban2025.parquet")[["cell_id", "buildings"]]
src = src[~src.cell_id.isin(CON_DAO)]
tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
tiles = tiles.merge(src, on="cell_id", how="left")
cells = tiles[tiles.buildings.notna()]
assert len(cells) == 3010

b = gpd.read_parquet(RAW / "overture_tu/buildings.parquet", columns=["id", "geometry"]).to_crs(UTM)
s = pd.read_parquet(RAW / "overture_tu/building_sources.parquet")
b = b.merge(s, on="id", how="left", validate="1:1")
assert b.source.notna().all()
b["prov"] = b.source.map(SHORT)
b["area"] = b.geometry.area
rect = shapely.minimum_rotated_rectangle(b.geometry.values)
xy, idx = shapely.get_coordinates(rect, return_index=True)
first = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
s1 = np.hypot(*(xy[first + 1] - xy[first]).T); s2 = np.hypot(*(xy[first + 2] - xy[first + 1]).T)
b["short"], b["long"] = np.minimum(s1, s2), np.maximum(s1, s2)

cent = gpd.GeoDataFrame(b[["prov", "area", "short", "long", "source_confidence"]],
                        geometry=b.geometry.centroid, crs=UTM)
j = gpd.sjoin(cent, cells[["cell_id", "geometry"]], predicate="within")

style = {}
for p, g in j.groupby("prov"):
    a = g["area"].values
    style[p] = {"n": int(len(g)), "share_of_buildings": round(len(g) / len(j), 3),
                "share_of_footprint_area": round(float(a.sum() / j["area"].sum()), 3),
                "median_area_m2": round(float(np.median(a)), 1),
                "share_under_22m2": round(float((a < 22).mean()), 3),
                "median_short_side_m": round(float(g.short.median()), 1),
                "median_long_short_ratio": round(float((g.long / g.short.clip(lower=.01)).median()), 2)}
conf = j.loc[j.prov == "Google", "source_confidence"]
style["Google"]["confidence_quantiles_10_50_90"] = [round(float(q), 3) for q in conf.quantile([.1, .5, .9])]

mix = j.groupby(["cell_id", "prov"]).size().unstack(fill_value=0)
share = mix.div(mix.sum(axis=1), axis=0)
cells = cells.merge(share, left_on="cell_id", right_index=True, how="left").fillna({"Google": 0, "Microsoft": 0, "OSM": 0})
share.add_prefix("share_").reset_index().to_parquet(REP / "building_sources_per_cell.parquet")
dom = share.max(axis=1)
patches = {"cells": int(len(share)),
           "cells_one_provider_ge_90pct": int((dom >= .9).sum()),
           "cells_mixed_no_provider_over_60pct": int((dom <= .6).sum()),
           "median_google_share": round(float(share.Google.median()), 3),
           "cells_google_ge_90pct": int((share.Google >= .9).sum()),
           "cells_microsoft_ge_90pct": int((share.Microsoft >= .9).sum()),
           "cells_osm_ge_50pct": int((share.OSM >= .5).sum())}

# Same cell, same fabric, different provider: in cells where both Google and
# Microsoft drew at least 50 buildings, compare their medians inside the cell.
# If the gap survives here it is the provider's drawing, not the neighbourhood.
med = j[j.prov.isin(["Google", "Microsoft"])].groupby(["cell_id", "prov"])
both = med.size().unstack().dropna()
both = both[(both.Google >= 50) & (both.Microsoft >= 50)].index
m_area = med["area"].median().unstack().loc[both]
m_short = med["short"].median().unstack().loc[both]
small = j[j.prov.isin(["Google", "Microsoft"])].assign(u22=lambda d: d["area"] < 22)
m_small = small.groupby(["cell_id", "prov"]).u22.mean().unstack().loc[both]
within = {"cells_with_both_ge_50": int(len(both)),
          "median_of_cell_ratio_google_over_microsoft_area": round(float((m_area.Google / m_area.Microsoft).median()), 3),
          "cells_where_google_median_smaller": round(float((m_area.Google < m_area.Microsoft).mean()), 3),
          "median_cell_short_side_m": {"Google": round(float(m_short.Google.median()), 2),
                                       "Microsoft": round(float(m_short.Microsoft.median()), 2)},
          "median_cell_share_under_22m2": {"Google": round(float(m_small.Google.median()), 3),
                                           "Microsoft": round(float(m_small.Microsoft.median()), 3)}}

out = {"style": style, "patches": patches, "within_cell": within}
json.dump(out, open(REP / "building_sources.json", "w"), indent=1)
print(json.dumps(out, indent=1))

main = tiles[~tiles.cell_id.isin(CON_DAO) & (tiles.geometry.centroid.y > tiles.geometry.centroid.y.median() - 120_000)]
fig, ax = plt.subplots(1, 2, figsize=(18, 10))
for a, col, title in [(ax[0], "Google", "share of the cell's buildings drawn by Google"),
                      (ax[1], "OSM", "share drawn by OpenStreetMap (the rest is Microsoft)")]:
    main.plot(ax=a, facecolor="#f2f2f0", edgecolor="#e2e2de", linewidth=0.1)
    cells.plot(ax=a, column=col, cmap="PuOr_r" if col == "Google" else "Greens", vmin=0, vmax=1,
               linewidth=0, legend=True, legend_kwds={"shrink": 0.6})
    a.set_title(title, fontsize=13, loc="left"); a.set_axis_off(); a.set_aspect("equal")
    x0, y0, x1, y1 = main.total_bounds; a.set_xlim(x0, x1); a.set_ylim(y0, y1)
fig.suptitle("Round 1 · who drew the footprints, per training cell (Overture 2026-08-19.0)", x=0.01, ha="left", fontsize=15)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
print("wrote", FIG)
