# What changed on 14–15 September, and what it invalidates

One page, so the corrected numbers live in a single place rather than being
re-derived from whichever document was read last. Everything here was measured,
and the scripts that produce each figure are named beside it.

**If you read one thing:** the pilot's picture of a small, sparse dataset in a
2 406 km² city is gone. The study area is 2.8× larger, the buildings 35× more
numerous, and the trainable tile count went up by a factor of twenty-two.

---

## 1. The study area is the current city, and it is measured by tiles

Ho Chi Minh City absorbed Bình Dương and Bà Rịa–Vũng Tàu on 1 July 2025. The
merged city's own OSM relation is unusable as a study area: one polygon of
36 423 km² whose centroid sits offshore, because it carries territorial waters
out to Côn Đảo. The land area is built instead as the union of the 168
commune-level units. Districts no longer exist; the city is 113 wards (phường),
54 communes (xã) and one special zone.

| | old | now |
|---|---:|---:|
| study area | 2 406.5 km² | **6 718.7 km²** |
| zoom-15 tiles | 1 881 | **5 170** |

`build_hcmc_boundary_2025.py`, `build_zoom15_index.py`

### The boundary only chooses the tiles; the tiles carry the data

Clipping source data to the municipal polygon left every straddling tile half
empty — 885 of 5 170 tiles, 699 km² of blank *inside* the tile set. A cell is a
training example, and a cell with a straight empty edge teaches the model that
cities have straight empty edges.

So the order is now: boundary → covering tiles → data over the whole tiles. The
study area for fetching is the **tile union**, 7 414.2 km², plus a 1 km buffer,
because a 300 m ring or a 100 m disc at the outer edge must read real
neighbours. `build_fetch_area.py`

---

## 2. OpenStreetMap was never measuring this city

| | OSM | Overture / 3D-GloBFP |
|---|---:|---:|
| buildings | 86 418 | **3 037 029** |
| footprint area | 23.6 km² | **425.0 km²** |
| places | 19 670 | **411 279** |
| buildings with a height | 3.9 % | **70 %** of Overture buildings, 82 % by area (see below) |

`fetch_overture_hcmc.py`, `prepare_building_heights.py`, `extract_hcmc_from_pbf.py`

**Corrected 15 Sep, A1.** The row above read "buildings with a height …
**100 %** (2 233 344)". That was 100 % of the 3D-GloBFP polygons, not of the
Overture buildings the footprint grid is drawn from; the two sets are different
outlines. In a 150-tile sample of the urban set, 70.3 % of Overture buildings
have their centroid inside a 3D-GloBFP polygon — 46 % of those under 30 m²,
85 % and more over 100 m² — and 82 % by footprint area. B8 joins height onto the
Overture footprint by centroid so the height and footprint grids share one mask.
`report_tile_sources.py`, and the A1 table.

---

## 3. The dataset is not small, and that changes the window argument

This is the correction with the most consequence, and it contradicts the
pipeline map's D3 reasoning directly.

| tiles holding | plan said | measured now |
|---|---:|---:|
| any building | 617 | **4 845** |
| ≥ 100 buildings | 139 | **3 020** |
| ≥ 500 buildings | — | 1 586 |
| ≥ 1 000 buildings | — | 938 |

D3 argued "the dataset is small … so the window may open and close early here."
On these numbers the trainable set is roughly twenty times what that sentence
assumed. Memorization onset grows with sample count, so the expected window is
wider, not narrower, and the step budget sized from the old figure will be
wrong in the direction of being too short.

---

## 4. B6 reverses: the sparsity was OSM's, not the city's

Fraction of units where a ring holds ≥ 2 places in ≥ 2 functional classes, on
the same tiles B6 reported:

| ring | OSM | Overture |
|---:|---:|---:|
| 50 m | 8.0 % | **57.0 %** |
| 100 m | 27.1 % | **79.7 %** |
| 300 m | 78.2 % | 97.6 % |

The 300 m ring B6 forced is no longer necessary on the core. `report_overture_ring_coverage.py`

**Corrected 15 Sep, A1.** The Overture column above was measured on the
**138 tiles holding ≥ 100 OpenStreetMap buildings** — the part of the city OSM
had mapped densely, hence "the same tiles B6 reported". It is not the urban
set. On the same 1 881-tile index restricted to tiles with ≥ 100 *Overture*
buildings (1 023 tiles) the on-disk parquet gives 14.3 % at 50 m and 26.6 % at
100 m. On the full 2025 urban set, all 3 020 tiles with ≥ 100 Overture
buildings (`report_ring_coverage_full.py`):

| ring | all 3 020 | ≥ 1 000 buildings (938) | ≥ 2 000 (426) |
|---:|---:|---:|---:|
| 50 m | 6.3 % | 18.8 % | 34 % |
| 100 m | 13.4 % | 37.9 % | 61 % |
| 300 m | 35.3 % | 78.5 % | 94 % |
| 500 m | 51.9 % | — | 99 % |

So the sentence "the 300 m ring is no longer necessary" held for the core and
not for the dataset. A1 keeps the ring at 300 m for both place grids, writes
absence as a value, and puts the ≥ 1 000-building core on the C7 mutant list;
the reasoning is in `indicator_specification.md` §3.2.

---

## 5. The channel set comes from the landscape page, not from a list of six

The old six — footprint binary, building density, road accessibility, place
diversity, place count, night lights — are superseded. Layers are now the "Have
it" rows of `hcmc_data_landscape.html`: **16 computable**, with perception
scores and age-sex structure still pending.

Each layer carries its **measurement scale** separately from its sampling unit,
which the earlier preview collapsed. Building density is footprint area within a
100 m disc; transit is a gravity field with a 500 m decay, not a presence mask;
places are counted in a 300 m ring and entropy runs over 11 functional classes
in that ring. Per B7 each is computed at four samples across its own scale and
read onto the common grid — for a 300 m ring at 1024 px that is a 17 × 17 grid,
the size the pilot already used. `layers.py`

### Where "six channels" came from, since it keeps confusing people

It was never a decision. The pilot wrote one RGB image per cell: **R** road
accessibility, **G** POI diversity, **B** building density. The v4 map then gave
each of those its own step — B3 buildings, B4 accessibility, B5 diversity — and
the "six channels" is that same triple with B and G split in two plus one
activity layer. Hence the names in `cells_channels_units.html`:

> B1 footprint binary · B2 building density · R road accessibility ·
> G1 POI diversity · G2 POI count · H night lights

Three colours, doubled, plus one. No indicator was chosen on its merits and no
count was argued for. An agent reading the map on another machine reported that
six seemed to come from nowhere, and it did.

B3–B5 should now be read as history: one step per colour of a file format the
project stopped using at B8. The layer set is the "Have it" rows of the data
landscape — 16 computable, 13 grids and 3 cell conditions, each with an explicit
measurement scale.

---

---

## 6. Grids and conditions are now sorted by measurement

Share of each layer's variance that lives **inside** a tile, at the sampling
unit, over 40 built-up tiles:

| | |
|---|---:|
| footprints | 89 % |
| footprint grain | 83 % |
| junctions | 69 % |
| height | 68 % |
| distance to centre | **0 %** |
| distance to park, severance | **1 %** |

A tile mean keeps only the remainder. The bottom group barely varies inside a
tile, so those are cell conditions rather than grids — the plan's three-way sort,
settled by measurement instead of assumption. `report_variance_split.py`

**One caution.** This share saturates early: footprints reach ~85 % by eight
units per tile and do not move from 128 px to 1024 px. Variance is fixed by how
much of a thing there is, and downsampling preserves that. So **variance cannot
decide 128 vs 512** — what changes across that range is arrangement, where edge
density falls about fivefold. That comparison needs a spatial statistic.

---

## 7. Two measurements that look alike and are not

At the sampling unit, "is there a building under this point" is the right
question, because a building spans many pixels. At 150 m, on a whole-city view,
it is not: a 60 m² building almost never contains the centre of a 150 m cell, so
point-sampling renders the city as a 2 % speckle. The city-wide view therefore
measures built form as **coverage by area** per cell, with height as an
area-weighted mean. Same indicator, two scales, two correct methods.

Likewise **per unit ≠ per cell**. A unit is one pixel; a cell is one tile and one
training example. Per-unit marginals choose each layer's transform at B10.
Per-cell distributions are B10b, and are what E3 compares against generated
cells. Both are now on the layer page.

---

## 8. What A1 measured on 15 September, and what it changed

All on the 3 020 urban tiles unless stated. `report_tile_sources.py`,
`report_ring_coverage_full.py`, `tile_sources_urban2025.parquet`,
`ring_coverage_urban2025.parquet`.

| finding | figure | consequence |
|---|---:|---|
| junctions were counted as way endpoints shared by ≥ 3 ways | 21 242 on the union; **254 512** by node degree ≥ 3 | `prepare()` patched; the intersection grid's per-unit zero fraction falls from 93.7 % to 62.0 % on the same 80 tiles |
| Overture buildings with a 3D-GloBFP height | **70 %** by centroid, 82 % by area | height joins onto Overture footprints at B8 |
| urban tiles with no Overture place | 549, **18.2 %** | place grids keep absence as a value |
| urban tiles with fewer than two functional classes | 1 072, 35.5 % | entropy written 0 where undefined; count disambiguates |
| urban tiles with no Mapillary image | 1 293, **42.8 %** | perception is a condition on covered cells only |
| urban tiles with no transit stop inside | 72.3 %; nearest > 2.5 km on 42.8 % | transit grid carried, flagged flattest |
| urban tiles entirely nodata in WorldPop constrained | 406, **13.4 %** | written as 0; flagged for C1 |
| urban tiles outside the WorldCover tile on disk | 10, all Côn Đảo | excluded → **3 010 cells** |
| store size, 13 grids × 3 010 cells | **10.3 GB at 256**, 41 GB at 512 | not 17.6 / 70, which assumed all 5 170 tiles |
| `building_form` as coded | coverage-weighted; one warehouse fills a 300 m square | B8 recomputes count-weighted over centroids |
| disc filters as coded | square boxes (`uniform_filter`) | B8 uses disc kernels |
| distance fields as coded | to every third polygon vertex; beads along canals | B8 uses distance transforms |
| place class map | 79 999 of 387 513 categorised places fall to "other" | B8 extends the map (`accommodation`, `bar`, `fashion` …) |

## Where this lands in the plan

- **A1** written and decided, 15 Sep: `indicator_specification.md`. 13 grids,
  3 conditions, 3 010 cells, 256 px first. Open question 4 is answered twice
  over: `building:levels` is 3.9 %, and irrelevant because 3D-GloBFP supplies
  height on 70 % of Overture buildings (82 % by area).
- **A2** answered: the human-side layers are population, night lights, gridded
  economy, Mapillary-derived perception and platform venue counts. All are
  conditions or derived fields, so the supply-side scope holds on measured grounds.
- **B1** restate the tile-count argument: 5 170 tiles, 3 020 trainable.
- **B6** re-run and reverse on the record.
- **D3** the step budget must be re-sized; "the dataset is small" no longer holds.
- **Rasterized channels** rebuild on Overture footprints over the tile union.

## Pages

- `hcmc_data_landscape.html` — every reviewed factor against what HCMC supplies
- `hcmc_wards.html` — 24 wards on their tiles, one unit, tile grid and boundary
- `hcmc_channels.html` — **deprecated**, see the note at its head
