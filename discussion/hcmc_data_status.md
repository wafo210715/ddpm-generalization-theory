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
| POIs | Per-tile, for the eight rasterized tiles (10 Sep). City-wide still incomplete. |
| Roads | Per-tile, for the eight rasterized tiles (10 Sep). |

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

## B6: the POI-per-ring coverage report, on real data

Overpass lifted the block on 10 Sep and the POI and road layers landed for the
eight rasterized tiles. This is the checkpoint B6 asks for, and it does not
pass.

Averaged over ten tiles, the fraction of units whose ring holds at least two
POIs in at least two categories — the condition for Shannon entropy to be
defined at all:

| ring | any POI | ≥ 2 POIs | **entropy defined** | mean POIs in ring |
|---:|---:|---:|---:|---:|
| 50 m | 8.5 % | 2.3 % | **1.1 %** | 0.1 |
| 100 m | 22.6 % | 10.7 % | **5.9 %** | 0.5 |
| 200 m | 44.9 % | 30.2 % | **20.0 %** | 2.0 |
| 300 m | 60.5 % | 44.9 % | **37.0 %** | 4.6 |
| 500 m | 77.6 % | 65.4 % | **58.3 %** | 12.5 |

The pilot's 50 m ring gave 2.5 % coverage; this sample gives 1.1 %, so the
pilot's finding reproduces and is if anything optimistic. Using the raw tag
values instead of the five coarse category keys lifts the entropy column by a
few points (1.7 %, 9.5 %, 28.8 %, 42.7 %, 63.3 %) because more distinct
categories make the two-category test easier to pass — worth knowing for open
question 2, though it does not change the shape of the problem.

### Per tile, which is where it gets interesting

Fraction of units with entropy defined:

| tile | 50 m | 100 m | 200 m | 300 m | 500 m | POIs per 300 m ring | buildings |
|---|---:|---:|---:|---:|---:|---:|---:|
| `15_26093_15400` | 2.0 % | 16.7 % | 58.9 % | 93.4 % | 100 % | 12.8 | 126 |
| `15_26097_15392` | 6.9 % | 31.4 % | 68.8 % | 88.5 % | 100 % | 20.2 | 7 177 |
| `15_26102_15388` | 0.7 % | 3.3 % | 22.9 % | 60.6 % | 100 % | 3.8 | 442 |
| `15_26097_15391` | 0.4 % | 3.2 % | 26.8 % | 60.3 % | 94.6 % | 4.4 | 1 096 |
| `15_26104_15399` | 0.2 % | 1.0 % | 8.6 % | 25.3 % | 65.6 % | 1.3 | 15 |
| `15_26098_15390` | 0.5 % | 3.3 % | 10.2 % | 23.0 % | 57.2 % | 2.1 | 2 988 |
| `15_26098_15391` | 0.0 % | 0.0 % | 4.0 % | 19.2 % | 63.6 % | 1.2 | 6 866 |
| three rural tiles | 0 % | 0 % | 0 % | 0 % | 0–1.6 % | 0–0.5 | 0–2 |

**POI coverage does not track built density.** `15_26098_15391` is the
second-densest tile in the whole study area — 6 866 buildings, unmistakably a
real neighbourhood — and OSM holds **four** POIs for it. `15_26093_15400` has
126 buildings and 12.8 POIs per ring. Whatever the POI layer is measuring in
HCMC, it is not evenly related to whether anyone lives or trades there. That is
an OSM completeness artefact, and it means a POI-derived channel would encode
mapping effort alongside urban function.

### The map is the part that matters

`report_poi_coverage.py --map` draws where the channel is defined, because a
coverage fraction cannot tell a mapping hole from genuinely empty ground and
the two call for opposite responses. Two things are visible in it that are not
visible in the table:

- The densely built tile with four POIs is red everywhere at 50 and 100 m. Its
  zeros are a coverage hole, not a quiet neighbourhood.
- At 300 m, where coverage finally looks acceptable, the green regions are
  **unions of discs around POI clusters**. At that radius the entropy channel is
  largely reporting distance to the nearest cluster of mapped POIs — which is
  close to what the accessibility channel already measures, and the pilot's 0.88
  correlation between diversity and accessibility now has an obvious mechanism.

Written to `data/hcmc/processed/reports/poi_ring_coverage_map.png`.

### What this does not settle

Choosing between the responses is A1's decision, not this report's. The options
the numbers leave open are a larger ring, plain counts instead of entropy, a
different POI source, treating activity as a cell-level condition, or dropping
the subject from this paper as A2 already contemplates.

One earlier number here should be discounted. Before the real POIs arrived this
report was exercised on uniformly scattered synthetic POIs and gave 96.3 %
coverage at a 100 m ring. The real figure is 5.9 %. Uniform scatter is the best
case for ring coverage and real POIs cluster hard; the synthetic run confirmed
the script runs and nothing else.

## What is still blocked

- **City-wide POIs and roads.** Only the eight rasterized tiles have them.
  Overpass rate-limits this host after a few hundred queries, so the full
  study area needs the tile-by-tile fetcher run in batches over time.
- **The four POI and road channels in the 1024 px store.** They can be
  rasterized now for the eight fetched tiles; the store on disk still has them
  as all-zero from before the block lifted.
- **Accessibility on the road network.** The roads are now fetched, so this can
  run; it has not yet been run on real roads.

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
