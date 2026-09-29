# Acquisition plan for the vitality factors — what we can actually get for HCMC

> [!warning] Superseded in part, 15 September 2026
> This page predates the move to the current city and to Overture. The study
> area, tile count, source data and channel set have all changed; the figures
> below are the OpenStreetMap-era values and several are wrong by an order of
> magnitude. Read [`v4_corrections_260915.md`](v4_corrections_260915.md) first.
> What survives is the method and the reasoning, not the numbers.

Assembled 12 Sep 2026. One row per factor in the vitality indicator review's two
de-duplicated tables (§4 built environment, §5 human), naming the source we will
use, the substitute where the exact thing does not exist, and the limitation to
disclose in the paper.

This supersedes the earlier OSM-only coverage note, whose central conclusions
about POI and building coverage were wrong — not about OSM, but about HCMC.

## The finding that reframes everything

**OpenStreetMap is not the city.** Every sparsity result this project has
recorded — B6's 1.1 % entropy coverage, the "second-densest tile has four POIs"
anomaly, the 617 usable tiles that bound the training set — was a measurement of
OSM's completeness in Vietnam, not of Ho Chi Minh City's urban form.

| | OpenStreetMap | Overture Maps | ratio |
|---|---:|---:|---:|
| buildings in the study area | 86 418 | **1 488 287** | 17.2× |
| building footprint area | 23.6 km² | **179.0 km²** | 7.6× |
| places / POIs | 19 670 | **341 692** | 17.4× |
| tiles with ≥ 100 buildings | 138 | **1 023** | 7.4× |
| tiles with any building | 593 | **1 416** | 2.4× |
| median buildings per tile | 0 | **164** | — |

OSM holds 6 % of the city's buildings and 13 % of its built area. The median
Overture building is 60.5 m², which is the tube-house and alley fabric that OSM
has not mapped. Overture is free, ODbL, and queryable anonymously over public
parquet on S3; both layers are already on disk.

### The extra buildings are real, and they are not duplicates

Three checks, because a 17× claim carries the rest of this page:

- **Overture effectively contains OSM.** 99.1 % of sampled OSM buildings have an
  Overture centroid within 5 m, 99.8 % within 20 m. So the additional 1.4 M are
  genuinely additional fabric, not a different dataset of the same city.
- **No duplicate stacking.** Only 0.5 % of Overture buildings have another
  centroid within 3 m; the mean neighbour count at that radius is 1.01. The
  count is not inflated by repeated footprints.
- **The size distribution is plausible for HCMC.** Median 60 m², tenth
  percentile 19 m², ninetieth 231 m² — tube houses and alley plots. One caution:
  11.4 % of footprints are under 20 m², which is small enough to include sheds,
  canopies and machine-learning artefacts, so a minimum-area filter should be a
  parameter of the rasterizer rather than an assumption.

### It is not simply denser everywhere

If Overture merely added uniform density, it would not fix a coverage bias. Of
the 138 tiles with ≥ 100 OSM buildings, thirteen had ≤ 10 OSM POIs — the
mapping holes. In those the median goes from 6 places to 220, a 37× gain, against
22× in the well-mapped tiles, and none of the thirteen remains sparse. The gain
is largest exactly where OSM was blindest.

### B6's checkpoint, re-run

Fraction of units where a ring holds ≥ 2 places in ≥ 2 functional classes, so
that Shannon entropy is defined at all. Like for like, on the same 138 tiles B6
reported:

| ring | OSM (B6) | Overture | tiles ≥ 80 % defined |
|---:|---:|---:|---:|
| 50 m | 8.0 % | **57.0 %** | 27.5 % |
| 100 m | 27.1 % | **79.7 %** | 61.6 % |
| 200 m | 60.3 % | 93.6 % | 89.9 % |
| 300 m | 78.2 % | 97.6 % | 94.9 % |
| 500 m | 94.1 % | 99.7 % | 100 % |

The ring can come in from 300 m to 100 m on the core, which removes B6's
objection that at 300 m the channel mostly reports distance to the nearest
mapped cluster.

**But the dataset is no longer those 138 tiles.** Across the 1 023 tiles that
Overture shows as built up, coverage is 14.3 % at 50 m and 54.8 % at 300 m,
because that set includes peri-urban fabric OSM never mapped at all. The radius
decision therefore still depends on which tiles the dataset is made of, exactly
as B6 said — only now the trade is between a tighter ring and a wider city,
rather than between a tighter ring and nothing.

### This reaches the physics argument, not just the plumbing

B1 reasoned that the trainable set is bounded by the tiles that hold data, about
600 in OSM rather than 1 881, and that since memorization onset grows linearly
with sample count the observable generalization window is bounded with it. On
Overture the set is 1 023 tiles at the 100-building threshold and 1 416 with any
building. That roughly doubles n, and on the theory the window should widen with
it. The already-rasterized building density channel, the pilot's most solid
result, was computed from 13 % of the city's built area.

---

## §4 Built environment — acquisition table

Source chosen per factor, with the fallback and what to disclose.

| Factor | Source we will use | Substitute / fallback | Limitation to state |
|---|---|---|---|
| POI count & density | **Overture places**, 341 692, 1 065 categories | ShopeeFood for food specifically | Category schema is Overture's, not the papers' |
| POI entropy / mix | Overture places folded to 11 functional classes | — | 100 m ring on the core, wider elsewhere |
| POI functional subclasses | Overture categories; **ShopeeFood** for food and drink | Cốc Cốc's 14 categories as cross-check | Culture and public admin remain thin |
| Building density / coverage | **Overture buildings**, 1.49 M | OSM as a quality cross-check | ML-derived footprints, not surveyed |
| Building height | **3D-GloBFP**, per-building height, Vietnam file 1.68 GiB, CC BY 4.0, Zenodo 10.5281/zenodo.11397015 | GlobalBuildingAtlas 3 m (103 GiB tile, **CC BY-NC**); GHS-BUILT-H AGBH 100 m raster; Google Open Buildings 2.5D Temporal (4 m, 2016–2023, only post-2020 source) | 2020 vintage; SE Asian high-rise underestimated |
| FAR / FSI | Overture footprints × 3D-GloBFP height | GHS-BUILT-H AGBH as coarse fallback | Derived, not surveyed; carries the height error |
| Building age | **GHS-BUILT-S epochs**, 100 m, 1975–2030, `JRC/GHSL/P2023A/GHS_BUILT_S` | WSF Evolution 30 m, but **stops 2015** and so misses HCMC's biggest decade | Epoch of built-up appearance, not construction date; 100 m is ~12×12 per tile |
| Building form typology | Derived from Overture footprints | — | Derived measure |
| Building function mix | Overture `subtype`, 0.66 % populated | Infer from co-located places | Effectively unavailable; disclose as a gap |
| Road density | **OSM roads**, 141 858 ways | Overture transportation | Strong layer; within-tile CV 0.90 |
| Intersection density | OSM road topology | — | Sparse point field, needs a kernel |
| Space syntax | OSM road graph | — | Global integration needs a citywide graph, not per-tile fetches |
| Land-use mix (Shannon) | **ESA WorldCover 10 m** `ESA/WorldCover/v200`; cross-check Dynamic World and Esri annual | None. Overture land_use is **not** one — see below | Land cover is not land *use*; a built-up class cannot separate residential from commercial |
| Transit accessibility | **OSM transit**, 5 217 stops deduplicated | Overture infrastructure | Metro is effectively one line |
| Green / NDVI | **Sentinel-2 dry-season composite**, `COPERNICUS/S2_SR_HARMONIZED` | WorldCover tree/grass classes | Only 86 scenes/yr under 20 % cloud, 71 of them Jan–Apr; composite, never a date |
| Distance to park | Overture + OSM green, 3 729 areas | WorldCover | Thin source |
| Water distance | **OSM water** 2 882 areas + Overture water 4 335 | JRC Global Surface Water 30 m | 30 m too coarse for canals |
| Terrain | **Copernicus DEM GLO-30** `COPERNICUS/DEM/GLO30` | FABDEM (CC BY-**NC**), NASADEM | It is a surface model including buildings; city relief is under 10 m, so a condition, not a field |
| Distance to centre | Computed | — | Constant within a tile, so a condition |
| Severance (rail / motorway) | OSM | — | Weak signal |
| Street lighting | **VIIRS** `NOAA/VIIRS/DNB/ANNUAL_V22` | SDGSAT-1 at ~40 m if obtained; OSM `lit` is 0.6 % | VIIRS is ~2.4 pixels per tile, a condition |
| Sidewalks | **Derive from Mapillary** via Mask2Former sidewalk and curb classes | OSM sidewalk tags, 0.4 % | Derived from imagery, not surveyed |
| House / land price | **Scrape homedy.com or guland.vn** (robots.txt open) | — | See the terms section below; alley housing likely under-represented |

## §5 Human — acquisition table

| Factor | Source we will use | Limitation to state |
|---|---|---|
| Population density | **LandScan Global 1 km** | Coarse, ~1.4 cells per tile, but see circularity below |
| | WorldPop constrained 100 m as the fine alternative | Building-weighted, partly circular |
| Age & sex structure | **WorldPop age-sex 100 m**, 2020 | Unconstrained only; no building-masked variant exists |
| Night-time lights | **VIIRS annual V2.2** | ~2.4 pixels per tile, a cell condition |
| | SDGSAT-1 ~40 m if we pursue it | Manual portal, ~20-scene initial quota, proposal for more, HCMC archive depth unverified |
| Economic aggregate | **Chen et al. 2022 gridded GDP, 1 km**, 1992–2019 | Night-light calibrated, so partly the same signal as the NTL channel |
| | Meta Relative Wealth Index 2.4 km, Apr 2021 | Single snapshot; custom licence, not plain CC BY |
| Human capital / education | Vietnam GSO census, ward level **unconfirmed** | Dashboard blocks automated fetch; needs a manual or Vietnamese-language check |
| Check-ins / reviews | **ShopeeFood venue and review counts**, 164 561 food venues in HCMC | Food only; a demand weight on a supply stock, which is the review's own T7 pattern — do not present it as a demand measure |
| Perception scores | **Mapillary** imagery, measured dense in alleys, + an Apache-2.0 ViT perception model via ZenSVI | Imagery is solved; the model is the uncertainty. No Vietnamese city in Place Pulse, so validate locally |
| Crowd / footfall | No usable source | Google busyness is undocumented scraping; Meta mobility discontinued |
| Mobile signalling | None | Disclose as unavailable |
| Floating population / OD | None | Disclose as unavailable |

### Mapillary answers the alley question, and it was measured

The one claim the earlier note could not verify is now settled by a live probe of
the Mapillary graph API, sampling 150 m cells across six districts:

| district | cells with imagery | images per 150 m cell |
|---|---:|---:|
| D1 core, arterial | 14/14 | 34 – 1 768 |
| **D4, dense alleys** | **15/16** | 0 – 1 792 |
| **Bình Thạnh, alley fabric** | **16/16** | 28 – 1 709 |
| **Gò Vấp, alley fabric** | **15/16** | 0 – 1 211 |
| **D8, canal alleys** | **16/16** | 13 – 1 482 |
| Củ Chi, rural | 14/16 | 0 – 252 |

A 150 m cell in District 4 holding 1 772 images is not arterial coverage. The
cause is identified: Be Group put roughly 9 M images over 27 000 km onto
Mapillary from April 2023, shot on helmet-mounted 360° cameras from scooters,
which reach the hẻm that cars cannot. Capture is concentrated in 2024, and two
thirds of HCMC imagery is panoramic.

**Perception therefore moves from a hoped-for channel to a buildable one, and the
uncertainty moves from the imagery to the model.** A published multi-city study
using Mapillary for green-view in HCMC gave it the lowest image-availability
score of eleven cities; that extraction predates the Be Group wave and its
coverage verdict is now stale, though its method — 50 m sampling along the OSM
network, usability filtering — is still the one to follow.

Two practical notes. Dense-core queries at our tile size return HTTP 500 because
the result set is too large, so the fetch must be tiled finer. The binding rate
limit is 50 000 tile requests per day, not per minute. And the coverage numbers
above were gathered using Mapillary's own published demo token from their public
example repository; we must register our own free token before any real pull.

### The perception model is reusable, and HCMC is not in the training set

There is no need to train anything. A ViT-B/16 model fine-tuned on Place Pulse
2.0 is published under Apache-2.0 with per-dimension accuracies from 61.6 % on
"boring" to 77.1 % on "lively". ZenSVI, MIT-licensed, wraps download, panorama
conversion, Mask2Former segmentation for green-view and element counts, and the
perception scoring in one pipeline.

The caveat is real and should be disclosed. Place Pulse 2.0 covers 56 cities and
includes no Vietnamese one, and HCMC is also absent from the 688-city Global
Streetscapes dataset, which holds only Hanoi, Nha Trang, Huế and Cần Thơ for
Vietnam. Published work already shows Place Pulse models transferring poorly to
Asian streetscapes. A local validation on a few hundred HCMC images should be
budgeted before any perception number is reported.

### Overture does not rescue land use, because it is OSM again

Worth stating so nobody retries it. Overture's land_use theme looks promising —
6 726 polygons in the study area, 21 subtypes, 56 classes — but it covers only
306 km² of a 2 406 km² city, 12.7 %, and its residential class totals **45.0 km²,
the same 45 km² OSM has.** The theme is derived from OSM, so it inherits exactly
the gap it appeared to fill.

This matters for what we can claim. Land cover gives a built-up class at 10 m,
which is a real field, but it cannot separate residential from commercial fabric.
The Shannon land-use mix that ten of the reviewed papers use therefore has no
true HCMC equivalent. The honest substitute is a mix computed over Overture place
categories rather than over land parcels, and that difference should be named in
the paper rather than glossed as "land-use mix".

### The circularity trap in the population channel

Most gridded population products are dasymetric: they redistribute census counts
using **buildings** as the weighting layer. We already have a building channel,
so such a population channel would be partly a re-expression of it. Ranked from
worst to best:

1. Meta HRSL 30 m — weighted by Meta's own building detector, and frozen at 2020
2. WorldPop constrained 100 m — weighted by Maxar, Ecopia, Microsoft and Google footprints
3. GHS-POP 100 m — weighted by GHS-BUILT-S, whose classifier was trained partly on **OSM building delineation**, our own source
4. WorldPop age-sex, unconstrained — covariate-based, weak circularity
5. **LandScan Global 1 km — no building weighting at all**
6. GSO census — enumerated, not modelled, zero circularity

If we want one population channel that is not a restatement of the building
channel, LandScan is the only clean raster, and the price is 1 km. This needs to
be a stated choice in the paper, not a default.

### Administrative boundaries: check the vintage

Three products disagree because Vietnam changed shape twice. GADM v4.1 gives 322
wards and matches our pre-merger 2 406.5 km² study area. geoBoundaries stops at
district. The OCHA product reflects the July 2025 merger and has no wards yet.
Use GADM ADM3.

---

## Terms of service — three decisions that are yours

I have not done any of these; they are judgement calls about acceptable use.

1. **ShopeeFood and Cốc Cốc expose open, unauthenticated JSON APIs.** Both
   robots.txt files are fully open. Both terms prohibit commercial
   redistribution. Non-commercial research use is the defensible reading, and
   publishing the derived rasters rather than the raw listings stays well inside
   it. Cốc Cốc's restaurant vertical mirrors TripAdvisor, which adds a second
   licence layer.
2. **batdongsan.com.vn and nhatot.com sit behind Cloudflare bot challenges**,
   on robots.txt itself. Getting the largest property datasets means defeating
   bot detection. I will not do that. homedy.com and guland.vn are open and are
   the workable price sources.
3. **Google Street View forbids this use case four times over.** Its terms bar
   bulk download, storage or caching, creating derivative content, and — most
   directly — using Maps content to train, test, validate or fine-tune machine
   learning models. There is no research exemption. Its Vietnam imagery is also
   2017–2021 and so five to nine years stale. The free metadata endpoint is a
   published interface and panorama ids may be stored indefinitely, so it is
   usable as an independent coverage cross-check and nothing else. Mapillary
   carries no such restriction and has fresher, denser alley coverage.

## What this changes in the plan

- **A1's specification table needs rewriting against Overture**, not OSM. The
  sparsity column on the POI rows is wrong by an order of magnitude, and open
  question 4 (does `building:levels` support a floor-area row) is answered twice
  over: no at 3.9 %, and irrelevant because 3D-GloBFP supplies height directly.
- **A2 is answered.** The human-side layers are LandScan, VIIRS, gridded GDP,
  Mapillary-derived perception, and ShopeeFood. All are conditions or derived
  fields rather than the building-scale activity data A2 hoped for, so the
  supply-side-only scope holds, but now on measured grounds.
- **B1's tile-count argument should be restated.** 1 023 trainable tiles rather
  than about 600 changes the expected width of the generalization window.
- **B6 should be re-run and its conclusion reversed** on the record, since the
  300 m ring it forced is no longer necessary on the core.
- **The already-rasterized channels need rebuilding on Overture footprints.**

## What is on disk now

Everything that needed no credential and no judgement call has been fetched.

| layer | serves | size |
|---|---|---|
| `data/osm/vietnam-latest.osm.pbf` | all OSM layers, no Overpass limit | 314 MB |
| `hcmc/raw/osm/*.gpkg` | roads, buildings, transit, water, green | 81 MB |
| `hcmc/raw/overture/*.parquet` | places, buildings, land use, water | 142 MB |
| **`hcmc/raw/heights_3dglobfp.parquet`** | **building height → FAR** | 109 MB |
| `external/ghsl/GHS_BUILT_S_E{1975..2020}` | building age, five epochs, 100 m | 62 MB |
| `external/ghsl/GHS_BUILT_H_AGBH_E2018` | coarse height fallback, 100 m | 38 MB |
| `external/worldcover/…N09E105` | land cover 10 m | 92 MB |
| `external/worldpop/vnm_ppp_2020_constrained` | population 100 m | 27 MB |

### Floor-area ratio is now computable

3D-GloBFP gives **1 120 320 buildings inside the study area with height on
100 % of them**, against OSM's 3.9 % and Overture's 0.02 %. It carries its own
footprints, so FAR needs no cross-source join.

| | metres |
|---|---:|
| median height | 11.5 |
| mean | 14.2 |
| 99th percentile | 50.2 |
| maximum | 181.6 |

A median of 11.5 m is three to four storeys, which is the right answer for the
tube-house fabric. **The maximum is the caveat.** HCMC's tallest building is
461 m and its second is 262 m, so a dataset maximum of 181.6 m confirms the
known underestimation of tall buildings in Southeast Asia. FAR will be
systematically low in the high-rise core — District 1, Thủ Thiêm, Bình Thạnh —
and that should be stated wherever a FAR figure is reported. The 100 m GHSL
height raster is the independent cross-check for exactly this.

### Still outstanding, each blocked on a credential or a decision

- **LandScan Global** — free, but registration at `landscan.ornl.gov`
- **Mapillary** — free, but register your own API token first
- **ShopeeFood and Cốc Cốc** — open APIs, but the terms call is yours
- **SDGSAT-1** — portal registration plus a written proposal
- **Sentinel-2 dry-season composite and VIIRS** — Earth Engine computations, not downloads

## Reproducing

```bash
cd data-processing
V=../ddpm-win-master/.venv/bin/python
$V scripts/extract_hcmc_from_pbf.py
$V scripts/build_osm_layers.py
$V scripts/fetch_overture_hcmc.py
$V scripts/report_overture_poi_coverage.py
$V scripts/report_overture_ring_coverage.py
$V scripts/report_osm_tag_coverage.py
$V scripts/fetch_external_layers.py
$V scripts/prepare_building_heights.py
```
