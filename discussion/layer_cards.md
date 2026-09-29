# Factor cards · issue #2, parts 1 and 2

A card starts from a **factor**, not from a raster. The order within a card:
1. the factor, and the condition it stands for;
2. how the vitality literature measures it, read critically and cited to the papers;
3. the data source we have for HCMC;
4. what we compute from it, and what that cannot say (measured on disk, with the
   subset named);
5. where a better or new measure might sit;
6. what was decided.

The A1 table ([`indicator_specification.md`](indicator_specification.md)) stays
the store's specification; a card changes it only where a decision says so. Every
decided layer is built into B8 by
[`build_b8.py`](../data-processing/scripts/build_b8.py) and checked, so the store
is ready once the last grid is decided.

*Restructured 28 Sep. This page was first ordered as one raster layer per round,
in A1's store order. The user pointed out that `building_binary` is data, not an
indicator. The factor it serves is Jacobs' concentration, which the literature
measures in several competing ways. So the page now runs by factor, and the
layer order becomes the order of loops below.*

## The loop, per factor

| step | who | what |
|---:|---|---|
| 1 | you | bring the loop's papers into `references/papers/`, PDF plus text layer |
| 2 | me | per paper, extract how it measures the factor: formula, denominator or areal unit, scale, data source, effect found, method (linear or not), with line references. Check against the rows already in [`vitality_provenance.md`](vitality_provenance.md), which were extracted on the Mac |
| 3 | together | read each measure critically: what it captures, what it hides, what its data source biases. One entry per paper in `references/annotated_bibliography.md`, tagged *vitality* |
| 4 | me | compute the candidate measures on the HCMC data on disk, with figures, and state what our sources can and cannot support |
| 5 | you | decide which measures become grids or conditions, and whether a new one is worth trying |
| 6 | me | build it in B8, check it against the source, close the card, update A1 where it changed |

A paper that serves several loops gets one bibliography entry, with a section per
factor, filled in as each loop reaches it.

## The loops

Jacobs' four conditions for diversity give the spine. Loops D and E are factors
the later literature added.

| loop | factor | our layers (A1) | measures to examine | papers |
|---|---|---|---|---|
| **A** | **concentration**: built intensity | `building_binary` (data, done), `building_density`, `building_height`, `far`, `building_form` | building coverage per street block (BCR); GSI and FSI (Spacematrix); building count; mean height; entropy of building coverage size; morphological types; FAR thresholds | Xia 2020, Kim 2020, Wu 2023, Jiang 2024, Dogan & Lee 2024, Tan 2026, Qin 2025, Liu 2025, Chen 2021, Li 2022; minor: Zeng 2018, Li 2021, Collins 2024, Scepanovic 2021 |
| **B** | **short blocks**: streets and permeability | `road_density`, `intersection`; candidates: space syntax, block size | intersection density (definitions and denominators split, provenance §1.2); road density as length or count; integration and choice; block size; gamma index; street-orientation entropy | Scepanovic 2021, Kim 2020, Wu 2023, Tan 2026, Li 2022, Gómez-Varo 2022, Dogan & Lee 2024, Collins 2024, Li 2021, Cai 2025 |
| **C** | **mixed primary uses** | `poi_count`, `poi_entropy`; candidates: land-use mix, building-function mix | at least six entropy variants (provenance §1.1): count-based Shannon, exp(Shannon), floor-area-weighted, normalised, land-use-map entropy, balance indices | Liu 2019, Li 2021, Jiang 2024, Yue 2021, Cai 2025, Gómez-Varo 2022, Liu 2022, Wu 2023, Tan 2026, Dogan & Lee 2024, Chen 2024 |
| **D** | accessibility and distance | `transit_access`, `dist_water`; conditions `dist_centre`, `dist_park`, `severance` | gravity, Hansen and 2SFCA forms; decay choices; which parks count | Williams 2019, Wang 2025, Lee & Kang 2022, Mouratidis 2020, Qin 2025, Shi 2023, Li 2022, Scepanovic 2021 |
| **E** | green | `green`; candidate: Sentinel-2 NDVI | land-cover class, NDVI, green view from street imagery | Li 2022, Wu 2023, Chen 2024, Mouratidis 2020 |
| **F** | **concentration**: people | `population`; candidates: age-sex, night lights, perception | resident vs ambient population; whether WorldPop only restates the footprints | Chen 2021, Lee & Kang 2022, Pan 2021, Collins 2024, Liu 2025, Tan 2026 |
| **G** | **aged buildings** | none yet: a substitute | mean and spread of construction year; urbanisation epoch; community age | Gómez-Varo 2022, Yue 2021, Cai 2025, Xia 2020, Li 2022 |

**Read across every loop:** Jacobs 1961, the chapters on the four conditions;
Gómez-Varo 2022, whose JANE index is Jacobs made measurable; Doan & Zhang 2025,
the review that finds no agreed definition; and Yue 2021, the one study of
HCMC in the set.

**Generalization track**, for the argument, in parallel and at your pace:
Bonnaire et al. 2025; Li, Dai & Qu 2024; Wang & Vastola 2024; the Two Clocks
paper, 2026; Wang et al. 2024 (subspace clustering); Zhang et al. 2024
(reproducibility); Kadkhodaie et al. 2024. All are open on arXiv or OpenReview.
Tagged *generalization* in the bibliography.

## Now: loop A, concentration as built intensity

The papers are on disk in `references/papers/urban-vitality/` (29 Sep), each with
a text layer. The reading order, with the reason for each, is in
[`HANDOFF_factor_loop.md`](HANDOFF_factor_loop.md) under "The papers on disk":

1. Gómez-Varo 2022
2. Ye 2018
3. Xia 2020
4. Jiang 2024
5. Dogan & Lee 2024
6. Tan 2026
7. Doan 2025 (non-linear)
8. Wu 2023

Two changes from the 28 Sep list:
- **Jacobs 1961 isn't in the folder**, so Gómez-Varo, which operationalizes
  Jacobs, comes first.
- **Two new papers come in.** Ye 2018 is Spacematrix per street block, with
  block types. Doan 2025 is a second non-linear threshold study, to read beside
  Tan.

---

## Base map — which tiles are training cells

*Done 23 Sep 2026. `figure_base_map.py`, `build_b8.py cells` →
`b8_256/cells.parquet`.*

![the 3 010 training cells](layer_cards_fig/base_map.png)

**The cells.** The 2025 city has **5 170** zoom-15 tiles. **3 020** intersect
100 or more Overture buildings; the **10** on Côn Đảo are excluded (A1 §3.1),
leaving **3 010 cells**, 4 316 km². By building count:

| band | cells | |
|---|---:|---|
| 100–249 | 776 | the fringe: places are absent on half of these (A1 §3.2) |
| 250–999 | 1 296 | |
| 1 000–1 999 | 511 | with the next row, the 938-cell "core" mutant (C7) |
| 2 000 + | 427 | |

A1 quotes 788 tiles at "100–250" and 426 "above 2 000". Both reconcile exactly.
788 counts the 3 020 tiles with 250 included; take away the six Côn Đảo tiles in
that range and the six cells holding exactly 250, and 776 remain. 426 counts
strictly above 2 000, and one cell holds exactly 2 000. All ten Côn Đảo tiles
hold fewer than 1 000 buildings.

**The pixel grid.** Each cell is rasterized on its own web-Mercator tile, so
the cells fit edge to edge. On the ground a tile at the centre is 1 201 m east–west
and 1 194 m north–south: Web Mercator treats the Earth as a sphere, which costs
0.65 %. So a pixel at 256 is 4.69 × 4.66 m. The preview drew each tile's
bounding box in UTM instead, which is rotated 0.2–0.3° and 1–2 pixels wider
per edge; B8 does not. Distances in EPSG:3405 match ground distances to 0.3 m.

**Found while drawing it.** The OSM water layer that `dist_water` is computed
from includes **396 km² of `natural=wetland`**: 1 361 polygons, 31 % of its
area, most of it Cần Giờ's mangroves (hatched on the map). As coded,
`dist_water` treats mangrove as open water. This is carried to round 12.

---

## Loop A · data source: `building_binary` — is there a building on this pixel?

*Done 23 Sep 2026, as "round 1" before the page was restructured. Measured by `report_building_footprints.py` →
`reports/building_footprints.json`; figure by `figure_building_binary.py`.*

**What it is for.** The city's figure-ground plan: where the built mass sits
and where the open ground runs between it. It is the only layer that draws
individual buildings. Density, floor-area ratio and footprint grain are
computed from the same outlines, and at B8 the height layer is burned onto
them too, so four other grids inherit this card's limits.

**Where it comes from.** Overture Maps, release `2026-08-19.0`, 3 037 029
buildings over the tile union; 2 828 377 have their centroid in the 3 010 cells.
Overture fills its building layer in a fixed order: OpenStreetMap first, then
Esri Community Maps, then high-precision Google Open Buildings, then Microsoft,
then lower-precision Google, each adding only buildings that don't intersect one
already taken ([Overture January 2024 release notes](https://overturemaps.org/overture-january-2024-release-notes/)).
OSM holds 86 418 buildings in the city, so almost all of this layer is
**machine-learned roof outlines traced from satellite imagery**.

**Who drew it.** Fetched on 23 Sep (`fetch_overture_hcmc.py building_sources`,
same release, joined on id; `report_building_sources.py`). Over the 3 010 cells,
buildings by centroid:

| provider | buildings | footprint area | median outline | under one 256-px pixel | narrow side | dated |
|---|---:|---:|---:|---:|---:|---|
| Microsoft ML Buildings | 57.0 % | 54.4 % | 84 m² | 4.6 % | 6.6 m | 2021 |
| Google Open Buildings | 39.3 % | 35.1 % | **38 m²** | **27.1 %** | 4.7 m | 2023; model confidence 0.67 / 0.76 / 0.90 at the 10th / 50th / 90th percentile |
| OpenStreetMap | 3.7 % | 10.6 % | 89 m² | 7.3 % | 6.7 m | mostly 2018–2023 |

No Esri. **The providers draw differently, and it is not the neighbourhood.**
In the 2 692 cells where Google and Microsoft each drew 50 or more buildings,
Google's median outline is **0.45×** Microsoft's in the same cell, smaller in
98.8 % of them. Its narrow side is 4.8 m against 6.7 m, and 27 % of its outlines
are smaller than a pixel against 4 %. Google cuts the same fabric into roughly
twice as many, half-size pieces.

The mix also varies across the city in patches. The median cell is 43 % Google,
and 1 430 cells have no provider above 60 %. But whole districts lean one way:
south-west of the centre is almost all Google, the core mostly Microsoft. That is
what Overture's rule predicts, since it chooses Google's high-precision tier one
square S2 cell at a time.

![provider share per cell](layer_cards_fig/building_sources.png)

**Consequence.** Every layer computed from footprints carries the provider
patchwork as if it were a difference in fabric. It is strongest in footprint
grain (round 5), which averages outline size. For this layer it shows as
fragmentation, which at 256 px is partly hidden by fusion. Each cell's provider
shares are now in `b8_256/cells.parquet` (`share_Google`, `share_Microsoft`,
`share_OSM`), so any result can be checked against them. Whether to rebuild the
footprints from a single provider is a round-5 decision.

**How it is computed.** A pixel is 1 if its centre falls inside a footprint.
At 256 px per tile a pixel is 4.69 m.

**What it cannot say.**

| check | finding | subset |
|---|---|---|
| Completeness against an independent product | Overture footprint cover tracks WorldCover's built-up class (Sentinel, 10 m, 2021) at Spearman **0.95** per tile. Of the 879 tiles WorldCover calls ≥ 30 % built, **4** have Overture cover below 15 % of that: `15_26124_15422`, `15_26105_15397`, `15_26093_15371`, `15_26092_15393`. They should be looked at in imagery; the rest show no large holes | 3 010 cells |
| Size of a building | median footprint **64 m²**; **13.6 %** are smaller than one 256-px pixel (22 m²); **3.3 %** are over 500 m² and hold **36 %** of all footprint area. In the 100–1 000-building bands those large footprints hold 47–49 % of the area; on tiles with ≥ 2 000 buildings, 24 % | 3 010 cells, buildings by centroid |
| Width | narrow side median **5.8 m**, ≤ 6 m for **53 %**, the width of a tube house. The median length-to-width ratio is only **1.65**, though, where a full-depth tube house (4 × 15–20 m) is 4 or more. So the typical outline is a short roof section, not a whole house plot. Worth checking against imagery before anything is read into building shape | 3 010 cells |
| Gaps between buildings | between neighbouring separate buildings the median gap is **1.0 m**: **53 %** are under 1.17 m, 68 % under 2.35 m, 81 % under 4.69 m. Most sub-metre gaps are probably the seams a model leaves between adjacent roofs, not passages; the 2–4 m alleys B2 chose the 1.17 m unit for are the 13 % between 2.35 and 4.69 m | 40 probe tiles, 31 669 buildings, 28 204 separate blobs once touching outlines merge |
| What each pixel size keeps | cover is exact at every size (6.66 % against the true 6.74 %). What changes is which buildings stay separate: | 40 probe tiles |

| pixel | buildings with no pixel at all | separate buildings fused with a neighbour | blobs, per true blob |
|---:|---:|---:|---:|
| 256 px · 4.69 m | **5.2 %** | **67 %** | 0.44 |
| 512 px · 2.35 m | 0 % | 59 % | 0.58 |
| 1024 px · 1.17 m | 0 % | 48 % | 0.69 |

![building_binary at three pixel sizes](layer_cards_fig/building_binary.png)

*The densest of the 40 probe tiles, 4 321 buildings, central 150 m. At 256
px the plan reads as blocks, and streets narrower than about 5 m close. At
1 024 px the white seams inside blocks are the sub-metre gaps between
neighbouring roof outlines, drawn as if they were alleys.*

What the layer can carry, then: at 256 it is a **block plan**. Cover is right,
streets wider than a pixel survive, individual buildings mostly don't. Finer
pixels separate more buildings, but they also draw the outline seams, and the
binary layer can't tell a seam from an alley. The resolution comparison has to
reckon with that.

- **Function and height.** Overture carries a building class on **0.42 %** of
  buildings, a subtype on 0.49 %, a height on 0.014 % and a floor count on
  0.11 %. Function is absent (the landscape page's "building function mix: no
  source" holds), and height has to come from 3D-GloBFP.
- **Date.** The provider sets are dated 2021 (Microsoft) and 2023 (Google), but
  those are dataset dates. The capture date of the imagery under each building
  is still unknown.

**What else could stand in for it, or improve it.**

1. **Coverage fraction instead of 0/1.** Write each pixel as the share of its
   area under a footprint, by burning at 2 048 px and averaging 8 × 8 blocks.
   A 3 m alley then shows as a continuous line of partly covered pixels, about
   0.4–0.7 depending on how it falls across them. The centre rule draws the same
   alley as a broken line, or loses it. No building is lost, and cover stays
   exact. This is the same move the city
   mosaic already made at 150 m (commit `2e6180f`: "built form has to be
   measured by area, not sampled"), applied at the pixel. Cost: the layer stops
   being 0/1, and the "height only on a building" rule becomes "height only
   where coverage > 0".
2. **Re-fetch Overture's `sources` column.** Done, see "Who drew it" above.
3. **A single provider.** Google and Microsoft both publish their full sets,
   not only the pieces Overture kept. Taking one of them everywhere would remove
   the patchwork, at the cost of that provider's own errors. Whether either
   covers the whole city on its own is still to be checked. Decided at round 5.
4. **Google Open Buildings 2.5D Temporal.** Annual building presence,
   fractional counts and heights from 2016 to 2023, about 4 m effective
   resolution, from Sentinel-2. It covers South-East Asia
   ([Google Research](https://sites.research.google/gr/open-buildings/temporal/)).
   Three uses here: an independent completeness check for 2023; a
   **building-change layer**, which is the "building age" row the landscape
   page marks as a weak substitute (100 m GHSL epochs); and a second height
   source against 3D-GloBFP. It's a new download, so it belongs to the part-2
   pass over the substitute rows, not to this store.

**Decision, 23 Sep.**

- **0/1 by the pixel-centre rule**, as A1 had it. The coverage fraction was
  offered and declined.
- **Provider sources fetched.** They are in `cells.parquet` as per-cell shares.
- **Single provider or Overture's mix:** deferred to round 5.

**Built.** `build_b8.py building_binary` → `b8_256/layers/building_binary.npy`,
3 010 × 256 × 256, uint8, 197 MB, 40 s. Every building that intersects a tile
is burned, not only those whose centroid is inside, so buildings continue
across cell edges. `build_b8.py check building_binary`, on every cell:

| | |
|---|---|
| mean cover, raster against vector | **9.09 %** against **9.09 %** |
| per-cell difference | median 0.02, 99th percentile 0.15, max 0.34 percentage points |
| cells with no building pixel | 0 |
| mean cover by band | 1.8 % (100–249) · 6.4 % (250–999) · 13.9 % (1 000–1 999) · 24.9 % (2 000 +) |

So the per-unit zero fraction over the 3 010 cells is **90.9 %**. A1 row 1
quotes 92.5 %, which was measured on the 80 preview tiles with ≥ 50 buildings.

![4 × 4 cells stitched from the B8 array](layer_cards_fig/b8_building_binary_mosaic.png)

*Sixteen cells around Bến Thành, stitched straight from the array. The canal,
the river and the streets run across the cell edges with no step or seam.
That confirms the cells tile exactly.*
