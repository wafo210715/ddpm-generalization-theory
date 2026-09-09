# HCMC data status — zoom-15 retiling, first rasterization

Run on 9 Sep 2026 on the Linux box. Covers A3's data move, B1's retiling, B3–B7
for the built-form channels, and what B6 and B10 can and cannot say yet.

## What was rebuilt

The pilot's raw vectors were not on this machine, so everything below was
re-downloaded rather than copied.

| | |
|---|---|
| Municipality boundary | OSM relation 1973756 at `[date:"2024-06-01"]`, the pre-merger HCMC. **2 406.5 km²**, matching the previous run exactly. |
| 1 km fishnet | 8 436 cells, reproduced for reference; superseded by the tile index below. |
| Buildings | 114 443 footprints, city-wide, complete (all four quadrants returned). |
| POIs | **Incomplete — see below.** |
| Roads | Per-tile only, and only for two rural tiles. |

### The v2 fetcher fails silently, and did

`build_hcmc_pipeline_v2.py` splits the municipality into four quadrants, and on
a failed quadrant it logs `Skipping POI tile 3` and carries on. POI tile 3 is
the dense north-eastern quadrant — the city core. Had the run completed, it
would have written `hcmc_pois_all.gpkg` with the core missing and nothing in
the file to say so.

Two changes address this. Both scripts now send a User-Agent, without which
Overpass answers 406 Not Acceptable and the pipeline cannot fetch at all. And
`fetch_tile_osm.py` fetches tile by tile — a 1.2 km square answers in seconds —
and **raises** rather than skipping, so a partial download cannot be mistaken
for a complete one. Each tile is fetched with a 400 m buffer, because an
indicator measured over a ring reads beyond the tile edge and a flush clip
would leave a rim of units computed from truncated neighbourhoods.

### Overpass then rate-limited this host

After the city-wide building fetch and about a dozen tile queries, both of
`overpass-api.de`'s addresses became unreachable from this machine while every
other site stayed up. That is a temporary block, not an outage. The POI and
road fetches, and therefore B6's coverage report, are the work this stops.

## The zoom-15 tile index

`build_zoom15_index.py` builds it and checks B1's three claims rather than
assuming them.

| | |
|---|---|
| Tiles intersecting the municipality | **1 881** |
| Tile width | **1 199.9 m to 1 203.9 m** — B1's stated 1 200–1 203 m, confirmed |
| Unit at 1024 px | **1.174 m** |
| Overlap | none; 1 881 unique ids, a partition by construction |
| Zoom-14 parents | 516, with 1 to 4 children each — the folding B1 relies on |
| Area covered | 2 406.5 km², the whole municipality |

Written to `data/hcmc/raw/hcmc_tiles_z15.{gpkg,parquet}`.

### How many of those tiles are usable is a smaller number

Joining the 114 443 buildings to the index:

| | tiles | share |
|---|---:|---:|
| With at least one building | **617** | 32.8 % |
| With at least 100 buildings | 139 | 7.4 % |
| Median buildings per tile | 0 | |
| Densest tile (`15_26097_15392`) | 7 177 | |

This matters for B1's argument. Zoom 15 does give roughly four times the tiles
of zoom 14, but the dataset that can actually be trained on is bounded by the
tiles that hold data, and in OSM that is around 600, not 1 881. Generalization
theory has the memorization onset growing linearly with sample count, so the
observable window is set by that 600, and the plan's expectation of a wide
window at zoom 15 should be read against it.

Whether the sparse tiles are genuinely empty ground — Cần Giờ is mangrove — or
gaps in OSM coverage is not answered here, and the two have different
consequences: empty ground is a typology, missing data is a hole.

## First rasterization at 1024 px

Eight tiles spanning the density range, rasterized by `rasterize_tiles_v4.py`
into a `CellTensorStore` at 1024 px, 1.173 m per unit, six candidate channels.
Only the two built-form channels have data behind them; the four that need POIs
or roads are all-zero and the store's manifest records that it is partial.

Each channel is computed at its own scale first and resampled last, per B7:
footprints straight at 1024, the density disc by convolution on the fine mask,
the ring and gravity fields on a 17 × 17 grid — four samples across their own
radius — then resampled up.

### What the marginals say

| channel | zero | skew | Moran's I | between-cell variance |
|---|---:|---:|---:|---:|
| `building_binary` | 86.6 % | 2.14 | **0.871** | 12.3 % |
| `building_density` (100 m) | 26.0 % | 1.04 | **1.000** | 46.4 % |

Two findings, and they are the two the plan predicted.

**B3 works.** At a 1.17 m unit the footprint channel has Moran's I of 0.80 to
0.93 across the four tiles pictured, against the pilot's near-zero. The alley
and rowhouse grain is present and separable; this is what B2 chose the unit
for, and it is now measured rather than argued.

**A smooth field gains nothing from the fine unit.** The 100 m-radius density
channel comes out at Moran's I = 1.000 — every unit is indistinguishable from
its neighbours. That is exactly the companion page's point stated as a number:
"reading a smooth field at more points does not change what the field means."
It also raises a question the specification table already lists as open, since
a channel that is perfectly redundant at 1.17 m is paying full resolution cost
for information a 25 m grid would carry.

The two channels are not redundant with each other: bivariate Moran's I between
them is 0.31 on the densest tile, so the density field is a genuinely different
statement about the same source, not a blurred copy that the descriptor would
see as the same thing.

## What is blocked, and on what

- **B6, the POI-per-ring coverage report.** `report_poi_coverage.py` is written
  and sweeps ring radii from 50 m (the pilot's, so its 97.5 % is reproduced
  rather than asserted) to 500 m, under both the coarse category-key scheme and
  the raw tag values, reporting three coverage levels per radius. It needs POI
  data, which needs Overpass. Run it with
  `--pois data/hcmc/raw/hcmc_pois_all.gpkg` or `--from_tiles` once the block
  lifts.

  It has been exercised on synthetic POIs scattered uniformly over three tiles
  at roughly HCMC-core density, purely to confirm the script runs: 40.9 % of
  units had entropy defined at 50 m, 96.3 % at 100 m. **Do not read those as
  an answer.** Uniform scatter is the best case for ring coverage; real POIs
  cluster along streets and around markets, which leaves large parts of a tile
  empty at every radius, and it is that clustering that produced the pilot's
  97.5 %. The real numbers will be worse, and how much worse is the thing B6
  is for.
- **The four POI and road channels**, for the same reason.
- **Accessibility on the road network.** Implemented — multi-source Dijkstra
  from POI-snapped graph nodes, exponential decay, cut off at three decay
  lengths — and it falls back to straight-line decay only with a warning,
  because a silently Euclidean "network gravity" would be a different indicator
  under the same name. Not yet run on real roads.

## Reproducing this

```bash
cd data-processing/scripts
V=../../ddpm-win-master/.venv/bin/python

$V build_boundary_grid.py
$V build_zoom15_index.py
$V fetch_tile_osm.py --sample 8 --seed 42
$V rasterize_tiles_v4.py --all_fetched --img_size 1024
$V report_poi_coverage.py --from_tiles
$V report_channel_marginals.py --store ../data/hcmc/processed/cells_z15_1024 --plot
```

Every radius in the rasterizer is a command-line argument with a placeholder
default. None of them is decided; see
[`indicator_specification.md`](indicator_specification.md), open question 3.
