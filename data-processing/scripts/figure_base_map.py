"""Issue #2, round 0: the base map -- which tiles are training cells.

Every zoom-15 tile of the 2025 city, the 3 010 cells coloured by building-count
band (A1's bands; the >= 1 000 band is C7's "core" mutant), the tiles left out
greyed, Con Dao's ten excluded cells in an inset, commune boundaries and water
for orientation.  Writes discussion/layer_cards_fig/base_map.png and prints the
band counts the card quotes.
"""
import pathlib
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / "data-processing/data/hcmc/raw"
REP = REPO / "data-processing/data/hcmc/processed/reports"
OUT = REPO / "discussion/layer_cards_fig/base_map.png"
OUT.parent.mkdir(exist_ok=True)
UTM = 3405
CENTRE = (106.6983, 10.7769)
CON_DAO = set("15_26086_15589 15_26086_15590 15_26086_15591 15_26087_15589 15_26087_15590 "
              "15_26088_15588 15_26088_15589 15_26088_15590 15_26089_15586 15_26089_15589".split())
BANDS = [(100, 250, "100–249", "#fde0c5"), (250, 1000, "250–999", "#f59e72"),
         (1000, 2000, "1 000–1 999", "#d9453a"), (2000, 10**9, "2 000 +", "#7a0d2e")]

tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg").to_crs(UTM)
src = pd.read_parquet(REP / "tile_sources_urban2025.parquet")[["cell_id", "buildings"]]
tiles = tiles.merge(src, on="cell_id", how="left")
tiles["urban"] = tiles.buildings.notna()
tiles["condao"] = tiles.cell_id.isin(CON_DAO)
cells = tiles[tiles.urban & ~tiles.condao]
assert len(cells) == 3010, len(cells)

communes = gpd.read_file(RAW / "hcmc_communes_2025.gpkg").to_crs(UTM)
boundary = gpd.read_file(RAW / "hcmc_boundary_2025.gpkg").to_crs(UTM)
water = gpd.read_file(RAW / "osm_tu/water_areas.gpkg").to_crs(UTM)
water = water[water.geometry.area > 2e4]               # rivers and lakes, not ponds
wetland = water[water.natural == "wetland"]             # 396 km2 that dist_water treats as water (round 12)
water = water[water.natural != "wetland"]
centre = gpd.GeoSeries(gpd.points_from_xy([CENTRE[0]], [CENTRE[1]]), crs=4326).to_crs(UTM)

rows = []
for lo, hi, lab, _ in BANDS:
    n = int(((cells.buildings >= lo) & (cells.buildings < hi)).sum())
    rows.append((lab, n))
print("tiles in the city", len(tiles), "| >=100 buildings", int(tiles.urban.sum()),
      "| Con Dao excluded", int((tiles.urban & tiles.condao).sum()), "| cells", len(cells))
print("bands", rows, "| core >= 1000:", int((cells.buildings >= 1000).sum()))
print(f"cell area {cells.geometry.area.sum() / 1e6:,.0f} km2; median tile {cells.geometry.area.median() / 1e6:.3f} km2")

main = tiles[~tiles.condao & (tiles.geometry.centroid.y > centre.y.iloc[0] - 120_000)]
fig, ax = plt.subplots(figsize=(13, 14))
main.plot(ax=ax, facecolor="#f2f2f0", edgecolor="#d8d8d4", linewidth=0.15)
water.clip(main.total_bounds).plot(ax=ax, color="#bcd7ea", linewidth=0)
wetland.clip(main.total_bounds).plot(ax=ax, facecolor="#dfe8dc", edgecolor="#9fb59a", hatch="....", linewidth=0)
for lo, hi, lab, col in BANDS:
    cells[(cells.buildings >= lo) & (cells.buildings < hi)].plot(ax=ax, facecolor=col, edgecolor="white",
                                                               linewidth=0.1, alpha=0.95)
communes.boundary.clip(main.total_bounds).plot(ax=ax, color="#555", linewidth=0.25)
boundary.boundary.clip(main.total_bounds).plot(ax=ax, color="#111", linewidth=0.9)
centre.plot(ax=ax, marker="*", color="#1f6feb", markersize=160, zorder=5)
ax.annotate("Bến Thành", (centre.x.iloc[0], centre.y.iloc[0]), xytext=(8, 8),
            textcoords="offset points", fontsize=10, color="#1f6feb")
x0, y0, x1, y1 = main.total_bounds
ax.plot([x0 + 4000, x0 + 14000], [y0 + 4000, y0 + 4000], color="#111", lw=2.5)
ax.text(x0 + 9000, y0 + 5200, "10 km", ha="center", fontsize=10)
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_axis_off()

handles = [Patch(facecolor=c, label=f"{l} buildings · {n:,} cells") for (_, _, l, c), (_, n) in zip(BANDS, rows)]
handles += [Patch(facecolor="#f2f2f0", edgecolor="#bbb", label=f"< 100 buildings · {int((~tiles.urban).sum()):,} tiles, not trained on"),
            Patch(facecolor="#bcd7ea", label="water (OSM, > 2 ha)"),
            Patch(facecolor="#dfe8dc", edgecolor="#9fb59a", hatch="....", label="wetland (OSM), which dist_water counts as water"),
            Line2D([], [], color="#555", lw=0.8, label="commune boundary, 2025"),
            Line2D([], [], color="#111", lw=1.5, label="city boundary, 2025")]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=10, frameon=True, title="zoom-15 tiles, 1 201 m",
          title_fontsize=10)
ax.set_title(f"Round 0 · the training cells: {len(cells):,} of {len(tiles):,} tiles", fontsize=15, loc="left")

cd = tiles[tiles.condao | (tiles.geometry.centroid.y < centre.y.iloc[0] - 120_000)]
ins = fig.add_axes([0.66, 0.66, 0.28, 0.20])
cd.plot(ax=ins, facecolor="#f2f2f0", edgecolor="#ccc", linewidth=0.2)
cd[cd.condao].plot(ax=ins, facecolor="none", edgecolor="#b3261e", hatch="////", linewidth=0.8)
boundary.boundary.clip(cd.total_bounds).plot(ax=ins, color="#111", linewidth=0.6)
ins.set_title(f"Côn Đảo, 230 km offshore:\n{int(tiles.condao.sum())} cells excluded (no WorldCover tile)",
              fontsize=9, color="#b3261e")
ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
