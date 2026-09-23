# A1 · Indicator specification table — decided 15 September 2026

**Status: decided.** This is the table B8 builds from. Every number on this page
was measured on the data on disk on 15 September, and the script that produces
each one is named beside it. Where a figure contradicts an older document, this
page wins; the older text is quoted where it was corrected
([`v4_corrections_260915.md`](v4_corrections_260915.md), sections 2, 4 and 8).

The earlier draft of this page (OpenStreetMap era, six candidate channels,
1024 px assumed) is superseded in full. What it settled — the column
definitions and the rule that a unit is *read* at the pixel while an indicator
is *measured* at its own scale — is kept below.

---

## 1. What the table sits on

| | |
|---|---|
| Cell | one zoom-15 web-Mercator tile, 1 201 m at HCMC's latitude, one training example |
| Tile set | the **3 020** tiles that intersect ≥ 100 Overture buildings, **minus the 10 Côn Đảo tiles** (see §3) → **3 010 cells** |
| Fetch area | the tile union, 7 414 km², plus a 1 km buffer, so no ring or disc reads an array edge |
| Sampling unit, first store | **256 px per tile, 4.69 m**. 512 px (2.35 m) follows once a short run trains cleanly; the two stores are the resolution comparison |
| Computation | each layer at its own measurement scale, four samples across that scale (B7, `SAMPLES_PER_SCALE = 4`), read onto the common grid last |
| Storage | 13 grids as one float32 tensor per cell, in physical units, plus 3 conditions per cell in the manifest (B8) |
| Transform, standardization | per grid, fitted after the transform, parameters stored (B11) |

Two counts of "built-up tiles" are in circulation and both are right:
**3 020** counts buildings that *intersect* a tile (`report_tile_sources.py`,
the corrections page); **2 968** counts buildings *within* a tile
(`figure_coverage.py`, the coverage page). The tile set uses the first.

### Store size at the decided settings

| | 256 px | 512 px |
|---|---:|---:|
| one cell, 13 grids, float32 | 3.4 MB | 13.6 MB |
| 3 010 cells | **10.3 GB** | 41 GB |

The earlier figures of 17.6 GB and 70 GB assumed all 5 170 tiles, most of
which hold no buildings and are not in the training set.

---

## 2. Column definitions

**Layer** — the name carried in the store manifest and every descriptor key.

**Source** — the dataset on disk, specific enough to rebuild from.

**Formula** — what is computed at each measurement location. Where the code in
`layers.py` computes something other than the intended quantity, both are
written: *as computed* and *B8*. The B8 form is what the store must implement.

**Scale** — the distance the value is computed over. This is where the
indicator's meaning lives; the pixel size is not part of it.

**Native grid at 256** — how many samples the layer is computed on per tile
before being read onto the 256 grid (`native_grid()`), so the cost and the
smoothness are visible.

**Per-unit marginal** — zero fraction and skew of unit values pooled over 80
tiles holding ≥ 50 buildings, at 128 px (`build_layer_tiles.py`,
`tiles.json`). This is what B10 chose transforms from.

**Within-tile share** — share of the layer's variance that lives inside a tile
at 1024 px, 40 tiles ≥ 100 buildings (`report_variance_split.py`,
`variance_split.json`). The settled rule: a layer whose share is at the floor
is a **condition**; the rest are **grids**.

**Absent per cell** — the share of the 3 020 urban tiles where the *source*
has nothing at all in or near the tile (`report_tile_sources.py`). This is the
column the two open questions turned on.

**Kind** — grid or condition. Only grids count towards the channel count.

**Transform** — B10's reading of the marginal, handed to B11: `none`, `log1p`,
or `quantile`. Suggestions, fitted at B11, not fixed here.

---

## 3. The decisions, in order

### 3.1 Tile set: drop Côn Đảo, keep everything else

Ten tiles with ≥ 100 buildings lie on Côn Đảo, 230 km offshore and south of
9° N, outside the one WorldCover tile on disk (`N09E105`), so the vegetation
layer is undefined there and the island is a different fabric in any case.
They are excluded. Reversible by fetching `N06E105`. Cell ids:
`15_26086_15589 15_26086_15590 15_26086_15591 15_26087_15589 15_26087_15590
15_26088_15588 15_26088_15589 15_26088_15590 15_26089_15586 15_26089_15589`.

No other restriction. The place and perception questions below are decided
per layer, not by cutting tiles: the trainable set stays at 3 010 and the
density gradient inside it is the object of study, not a nuisance.

### 3.2 Places: keep both grids, absence is a value, ring 300 m

Measured on all 3 020 urban tiles (`report_ring_coverage_full.py`,
`ring_coverage_urban2025.parquet`), share of units whose ring holds ≥ 2 places
in ≥ 2 of the 11 functional classes — the condition for entropy to be defined:

| ring | all 3 020 | ≥ 500 bld (1 586) | ≥ 1 000 bld (938) | ≥ 2 000 bld (426) | tiles with < 10 % defined |
|---:|---:|---:|---:|---:|---:|
| 50 m | 6.3 % | 11.8 % | 18.8 % | 34 % | 86 % |
| 100 m | 13.4 % | 24.7 % | 37.9 % | 61 % | 72 % |
| 150 m | 19.6 % | 35.6 % | 52.2 % | 76 % | 62 % |
| 200 m | 25.3 % | 44.9 % | 63.2 % | 86 % | 53 % |
| **300 m** | **35.3 %** | **60.2 %** | **78.5 %** | **94 %** | **40 %** |
| 500 m | 51.9 % | — | — | 99 % | 26 % |

And per tile, from `report_tile_sources.py`: 18.2 % of urban tiles hold no
Overture place at all, 35.5 % hold fewer than two functional classes, 57.1 %
hold fewer than ten places. By density band the absence is almost entirely
below 500 buildings: 49 % of tiles in the 100–250 band have no place, 0 % of
tiles above 1 000.

The decision:

- **`poi_count` is a grid, ring 300 m.** A count of zero within 300 m is a
  measurement, not a gap: place count tracks building count at Spearman 0.83
  across tiles, so where Overture is empty the fabric is mostly empty too. The
  layer is written as is, zeros included.
- **`poi_entropy` is a grid, ring 300 m, written as 0 where undefined.** The
  undefined case (fewer than two places) and the true zero (one kind) are not
  separable in this grid alone. They are separable in the pair: a unit with
  `poi_count` = 0 and `poi_entropy` = 0 is undefined, a unit with count > 0
  and entropy 0 is single-use. That is the joint structure B10b exists to
  measure, and it is why the count grid ships. **E3 evaluates entropy only on
  units where the count is ≥ 2**, so the undefined mass does not score as
  agreement.
- **Ring radius 300 m for both.** 100 m defines entropy on 13 % of units over
  the set and 61 % even on the densest band; 500 m makes the grid a map of
  distance to the nearest cluster over a third of the city. At 300 m the
  channel is a field on the core (94 % on ≥ 2 000-building tiles), a partial
  field on the middle band, and a near-blank on the outer band — and the
  outer band is where the fabric is thin too. Same ring for count and
  entropy, so the two read the same neighbourhood.
- **The core is a C7 mutant, not the baseline.** Tiles ≥ 1 000 buildings (938)
  are where the place grids are dense on nearly every unit. That subset goes
  on the C7 list as the "core" mutant, alongside the full 3 010, so the
  place layers' contribution can be read from the difference rather than
  assumed.

One caution carried to C1: `poi_count` and `poi_entropy` correlate at 0.90
(Spearman, per unit) and 0.96 (per cell). Much of that is the shared zero mass
on the same units; C1 must recompute it on units where both are defined
before calling them duplicates.

### 3.3 Perception: a condition on covered cells only, not in the first store

Mapillary survey over all 5 170 tiles (`survey_mapillary.py`,
`mapillary_per_cell.parquet`): on the 3 020 urban tiles, **1 293 (42.8 %) have
no image** and 1 484 (49.1 %) have fewer than 100. Coverage correlates 0.64
with building count. Where imagery exists it is abundant (median 1 535 per
covered tile, 60 % panoramic, median capture 2023 or later).

Decision: perception scores, when built, attach as **a cell condition on the
1 727 covered urban tiles and are absent elsewhere**; never a grid; not part of
B8 at 256. A grid that is blank across two fifths of the training set would
teach the model that the absence is a feature of the city. The score is a
whole-cell summary in any case. It stays in `PENDING`.

### 3.4 Grid or condition: carried from the variance split

Within-tile share of variance at 1024 px, from `variance_split.json`:

| layer | share | kind |
|---|---:|---|
| building_binary | 89 % | grid |
| building_form | 83 % | grid |
| intersection | 69 % | grid (re-measure at B8, see row 7) |
| building_height | 68 % | grid |
| green | 55 % | grid |
| building_density | 45 % | grid |
| road_density | 41 % | grid |
| poi_entropy | 29 % | grid |
| far | 17 % | grid |
| poi_count | 11 % | grid |
| transit_access | 11 % | grid |
| population | 9 % | grid |
| dist_water | 3 % | grid — closest to the line |
| dist_park | 1 % | **condition** |
| severance | 1 % | **condition** |
| dist_centre | 0 % | **condition** |

The bottom three are near-constant across a tile and attach to the cell as
three numbers. Everything else is a grid. `dist_water` at 3 % is carried as a
grid on the settled rule; E3 will show whether the model learns anything
inside a tile from it, and demoting it then costs one column.

Resolution is not decided by this table. The share is flat from 128 to 1024
px; the 256 and 512 stores are compared on a spatial statistic (edge density,
Moran's I on the descriptor), not on variance.

---

## 4. The table

Thirteen grids, three conditions. Rows in store order.

### Built form — Overture buildings and 3D-GloBFP heights

| # | Layer | Source | Formula | Scale | Native grid at 256 | Per-unit zero · skew | Within-tile share | Absent per cell | Kind | Transform |
|--:|---|---|---|---|---:|---|---:|---|---|---|
| 1 | `building_binary` | Overture buildings, 3 037 029 over the tile union (`overture_tu/buildings.parquet`) | 1 if the unit centre falls inside a footprint, else 0 | the polygon | 256 | 92.5 % · 3.2 | 89 % | 0 % — every cell has ≥ 100 | grid | none |
| 2 | `building_density` | same | footprint area within a **100 m disc** ÷ disc area. *As computed:* `uniform_filter` on a ~25 m burn, i.e. a **200 m square box**. *B8:* a true disc kernel | 100 m | 49 | 38.6 % · 2.0 | 45 % | 0 % | grid | none (bounded 0–1) |
| 3 | `building_height` | 3D-GloBFP, 2 233 344 polygons with `Height` (`heights_3dglobfp_tu.parquet`) | height in metres of the building on this unit. *As computed:* burned from the 3D-GloBFP polygons, whose outlines differ from Overture's, so this grid's mask is not row 1's mask (91.5 % zero against 92.5 %). *B8:* join height to the Overture footprint by centroid and burn on row 1's polygons, so one mask serves both | the polygon | 256 | 91.5 % · 5.8 | 68 % | 0.5 % of cells have no 3D-GloBFP polygon; 15.6 % have fewer than half as many as Overture. In a 150-tile sample **70.3 % of Overture buildings** have a height by centroid match (46 % of those under 30 m², 85 % + over 100 m²; **82 % by area**) | grid | log1p |
| 4 | `far` | rows 1 + 3 | Σ over buildings within a 100 m disc of footprint × (height ÷ 3 m) ÷ disc area; 3 m per storey is the assumption. Same box-vs-disc note as row 2 | 100 m | 49 | 43.1 % · 4.0 | 17 % | as row 3; unmatched buildings contribute zero floor area, so FAR runs low where small buildings dominate | grid | log1p |
| 5 | `building_form` | Overture buildings | mean footprint area of buildings whose **centroid** lies within 150 m, m². *As computed:* disc mean of each building's area burned over its own pixels — a coverage-weighted mean in which one warehouse dominates a 300 m square (max 45 000 m², skew 14, the yellow blocks in the ward atlas). *B8:* count-weighted mean over centroids in a true 150 m disc | 150 m | 33 | 32.3 % · 14.4 | 83 % | 0 % | grid | log1p |

### Streets — OpenStreetMap over the tile union

| # | Layer | Source | Formula | Scale | Native grid at 256 | Per-unit zero · skew | Within-tile share | Absent per cell | Kind | Transform |
|--:|---|---|---|---|---:|---|---:|---|---|---|
| 6 | `road_density` | OSM `highway=*` centrelines, **236 119 ways on the union** (`osm_tu/roads.gpkg`; residential 113 349, service 81 906, footway 9 445 …) | street length within a 100 m disc ÷ disc area, km per km². *B8:* drop `highway=construction|proposed` (1 251 ways); keep service and footway, which are the alleys | 100 m | 49 | 27.5 % · 1.1 | 41 % | 0.1 % (2 cells) | grid | none |
| 7 | `intersection` | OSM road graph, **nodes of degree ≥ 3** (endpoint counts 1, interior vertex 2) | junctions within a 100 m disc ÷ disc area, per km². **Corrected 15 Sep:** the code counted only endpoints shared by ≥ 3 ways, which finds 21 242 junctions on the union; the degree rule finds **254 512**, twelve times as many. `build_layer_preview.prepare()` now carries the degree rule; the ward atlas and `tiles.json` predate it | 100 m | 49 | **62.0 % · 4.0** under the degree rule (93.7 % · 8.4 under the old one), same 80 tiles | 69 % (old rule; re-measure at B8) | 1.2 % of cells have no junction (35; was 948 under the old rule) | grid | log1p |
| 8 | `transit_access` | OSM transit stops, 6 288 on the union (`osm_tu/transit_stops.gpkg`) | Σ over the 8 nearest stops of exp(−d / 500 m) | 500 m decay | 10 | 0.0 % · 2.0 | 11 % | 72.3 % of cells have no stop inside; on **42.8 %** the nearest stop is > 2.5 km, so the field is < 0.01 across the whole cell | grid, carried — the flattest grid in the set. If E3 shows nothing learned inside a tile, this is the first to demote | log1p |

### Places — Overture over the tile union

| # | Layer | Source | Formula | Scale | Native grid at 256 | Per-unit zero · skew | Within-tile share | Absent per cell | Kind | Transform |
|--:|---|---|---|---|---:|---|---:|---|---|---|
| 9 | `poi_count` | Overture places, 411 279 (`overture_tu/places.parquet`), all of them | places within a 300 m ring ÷ ring area, per km² | 300 m | 17 | 48.0 % · 7.9 | 11 % | 18.2 % no place; 57.1 % fewer than 10 (§3.2) | grid; zero is a value | log1p |
| 10 | `poi_entropy` | Overture places with a category (387 513), folded into **11 functional classes** by the regex map in `build_layer_preview.POI_CLASSES` → 331 280 classified, **79 999 left as "other" and excluded** | −Σ p log p over the classes present in the 300 m ring; 0 where fewer than two places. *B8:* extend the class map — the largest unmapped raw categories are `car_dealer` 3 691, `fashion` 2 751, `electronics` 2 277, `interior_design` 1 887, `eat_and_drink` 1 681, `tattoo_and_piercing` 1 679, **`accommodation` 1 483** (the word itself is not in the accommodation pattern), `bar` 1 246, `karaoke` 1 092 | 300 m | 17 | 63.9 % · 1.1 | 29 % | defined on 35.3 % of units over the set, 94 % on ≥ 2 000-building tiles; on 40 % of cells defined on < 10 % of units (§3.2) | grid; 0 where undefined, disambiguated by row 9; E3 scores it only where count ≥ 2 | none (bounded 0 – ln 11 = 2.40) |

### Surfaces and distances

| # | Layer | Source | Formula | Scale | Native grid at 256 | Per-unit zero · skew | Within-tile share | Absent per cell | Kind | Transform |
|--:|---|---|---|---|---:|---|---:|---|---|---|
| 11 | `green` | ESA WorldCover 2021 v200, tile N09E105, 10 m | 1 if the class is tree 10, shrub 20, grass 30, **crop 40** or mangrove 95, else 0; nearest-neighbour read. Cropland counts on purpose: the layer is vegetation, not park | 10 m | 256 | 25.6 % · −1.0 (mean 0.72 on the sample; 0.69 per urban tile) | 55 % | 0.3 % — the 10 Côn Đảo tiles, now excluded | grid | none |
| 12 | `dist_water` | OSM water areas, 7 189 polygons (`osm_tu/water_areas.gpkg`) | distance in metres to the nearest water polygon. *As computed:* to the nearest of every third polygon **vertex**, which draws beads along a straight canal bank (visible in the An Đông panel). *B8:* rasterize the polygons and take a Euclidean distance transform | 100 m | 49 | 0 % · 1.7 | 3 % | 41 % of cells have no water vertex within a tile length of the centre; the field is still defined | grid, closest to the condition line | log1p, or clip at 2 km |
| 13 | `population` | WorldPop 2020 **constrained**, 100 m (`vnm_ppp_2020_constrained.tif`) | residents per hectare, nearest-neighbour read | 100 m | 49 | 68.8 % · 7.1 | 9 % | **13.4 % of cells (406) are entirely nodata** — the constrained product's settlement mask misses them although they hold 100–216 buildings; concentrated in the northern fringe. Elsewhere it is a 12 × 12 block field that restates the building mask (cell correlation 0.87 with footprints) | grid, carried. Two flags for C1: redundancy with rows 1–2, and the nodata cells, which B8 writes as 0 | log1p |

### Conditions — three numbers per cell

| # | Layer | Source | Formula | Scale | Within-tile share | Absent per cell | Kind |
|--:|---|---|---|---|---:|---|---|
| c1 | `dist_park` | OSM `leisure=park|garden`, 1 282 polygons | distance from the tile centre to the nearest park polygon, metres (B8: polygon distance, not vertex distance) | 300 m | 1 % | 76 % of cells have no park within a tile length; 49 % none within 3 km | condition |
| c2 | `dist_centre` | Bến Thành, 106.6983 E 10.7769 N | distance from the tile centre, metres | 1 km | 0 % | never | condition |
| c3 | `severance` | OSM `highway=motorway|trunk|motorway_link` as coded, 2 875 ways; B8 adds `trunk_link` → 3 718 | distance from the tile centre to the nearest such way, metres | 300 m | 1 % | 54 % of cells have none within 3 km | condition |

### Pending, and where they will attach

| Layer | State | Decision |
|---|---|---|
| `perception` | needs the Mapillary pull and a vision model | condition on the 1 727 covered urban cells only, absent on 1 293; never a grid (§3.3) |
| `age_sex` | needs the WorldPop age-sex rasters | condition, per-cell shares |

---

## 5. What B8 must do differently from `layers.py` today

Collected from the *B8* notes above, so the store is not built on the
preview's shortcuts:

1. **Disc kernels**, not `uniform_filter` boxes, for rows 2, 4, 5 and 6.
2. **Height on Overture footprints** by centroid join, so rows 1 and 3 share a
   mask (row 3).
3. **Count-weighted footprint grain** over centroids within 150 m (row 5).
4. **Junctions by node degree** — already patched in `prepare()` (row 7).
5. **Distance transforms** from rasterized polygons for `dist_water`, and
   polygon distance for the two distance conditions (rows 12, c1).
6. **Extend the place class map** for the nine raw categories listed in row 10,
   then re-run `report_ring_coverage_full.py` to confirm the 300 m figures
   move only upward.
7. **Exclude `highway=construction|proposed`** from row 6.
8. **Drop the 10 Côn Đảo tiles** and write the nodata WorldPop cells as 0.
9. Write the three conditions and the cell's building count into the manifest.

After the store exists, `tiles.json` and the ward atlas are re-rendered from
it, so the pictures and the numbers come from the same arrays.

---

## 6. Manifest fields, for the loader

`channel_names` = rows 1–13 in order; `img_channels` = 13; `img_size` = 256
first, 512 second; `conditions` = `dist_park, dist_centre, severance`;
`transforms` = the last column, fitted at B11; `unit_m` = 4.69 at 256;
`tile_set` = 3 010 cell ids with their building counts, so C6/C7 can cut
mutants without touching the vectors.
