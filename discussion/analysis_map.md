# Analysis map · within and between layers, before and after training

**Issue [#2](https://github.com/wafo210715/ddpm-generalization-theory/issues/2),
part 3 · 23 September 2026.** Every analysis the plan
([`pipeline_v4_map.html`](pipeline_v4_map.html)) lists on the data, sorted on
the axes that were mixed in one list, with the question each one answers, the
decision it fed or will feed, and where its code stands. No decision is taken
here. One probe was run for the page (§4); everything else is quoted from the
place that measured it, with the subset named.

This page takes over the "before and after training" section of the deprecated
[`hcmc_channels.html`](hcmc_channels.html).

---

## 1. Why it read as unclear: three axes in one list

The plan lists the analyses in pipeline order (B10, B10b, B11, C1 … F1). That
order is right for running them and wrong for understanding them, because
three independent choices are folded into each step:

| axis | values |
|---|---|
| **what is compared** | one layer against itself (*within*) · two or more layers against each other (*between*) · the whole tensor at once (*whole cell*) |
| **when** | *before* training, on real cells · *after* training, on generated cells, at every checkpoint of every mutant |
| **at what level** | *unit* — one pixel · *cell* — one tile, one training example · *dataset* — all cells together |

A fourth choice is smaller but real: data space (metres, counts, per km²) or
standardized space (after B11). Statistics that describe meaning are read in
data space; anything that uses a distance between cells is read in
standardized space, so the widest layer does not decide what "near" means
(B10b, E2).

**The level changes the answer, not just the precision.** On the 80 tiles of
`tiles.json` (≥ 50 buildings, 128 px, preview code), building footprint and
building density correlate at Spearman **0.36 per unit** and **0.99 per cell**.
Both are right. Per unit, a pixel on a building and a pixel on the alley beside
it have nearly the same density; per cell, the mean of the footprint mask *is*
the footprint coverage, and so is the mean of a disc average of it. Height and
FAR go from 0.41 to 1.00 the same way. A per-cell number says how much of a
thing a tile holds; a per-unit number says where it sits. The study's claim
lives in the gap between the two.

---

## 2. The rule that connects before to after

Every analysis before training does one of two jobs:

- **decide** something about the encoding, once — a transform, a ring radius,
  grid or condition, which tiles; or
- **fix a reference** — a description of the real cells that generated cells
  will later be held against.

Every analysis after training does one job: **compare** generated cells to a
reference. So each after-training number needs a before-training twin,
computed by the same function, at the same level, in the same space. That is
what "one function for real and generated cells" (C2) protects. The tables
below pair the twins; the blanks are where one side is missing.

---

## 3. The map

State as of 23 Sep. *Preview* means the numbers come from `layers.py` via
`build_layer_tiles.py` or `report_variance_split.py` on a tile sample, with the
square-box filters and the 3D-GloBFP height burn that A1 §5 lists as defects.
None of it has run on a B8 store, because no store exists.

### 3.1 Within one layer

| | analysis | level | question | job | after-training twin | code · state |
|---|---|---|---|---|---|---|
| B6 | ring coverage: share of units whose ring holds ≥ 2 places in ≥ 2 classes | unit | where is entropy defined at all? | **decided** the 300 m ring (A1 §3.2) | none — the ring is a definition | `report_ring_coverage_full.py` · done on all 3 020 urban tiles |
| — | source presence per tile | cell | is the source there at all in this cell? | **decided** the tile set, places with absence as a value, perception as a condition (A1 §3.1–3.3) | none — a property of the source, not of a grid | `report_tile_sources.py` · done on all 3 020 |
| B10 | per-unit marginal: zero fraction, skew, tail | unit, pooled | which transform? is the layer one value most of the time? | **decides** B11's transform | pooled per-unit histogram, real vs generated, as a Wasserstein distance. **No node owns it.** The Wasserstein distances E3b's note mentions are on descriptor columns, which are per cell | preview: `tiles.json`, 80 tiles ≥ 50 bld, 128 px. Store runner: `report_channel_marginals.py`, waits on B8 |
| — | within-tile share of variance | unit vs cell | does the layer vary inside a tile, or only between tiles? | **decided** grid vs condition (A1 §3.4) | **none listed.** Computable from the descriptor: mean of `std²` over cells against the variance of `mean` | preview: `variance_split.json`, 40 tiles ≥ 100 bld, at 2 … 1024 px (256 is not among them) |
| B10b, first half | per-cell distribution of tile means | cell → dataset | what does a typical training example look like on this layer? | **reference** | E3 | preview: `tiles.json` `cell_marginals`, same 80 tiles |
| C2 | descriptor per layer: mean, std, zero fraction, Moran's I, skew | cell | the same, with shape and clumping | **reference** | E3 | `analysis/cell_descriptor.py`, tested (the descriptor, spatial, embedding, gap-ratio and transform suites: 114 pass on 23 Sep); never run on v4 data. **Moran's I is uninformative on 13 of 16 layers, §4.1** |
| C4b | 2-D map of the dataset, one per grid | dataset | where do cells sit relative to each other on this grid? | **reference picture** | E3b | `analysis/embedding.py`, tested; not run |

After training, per checkpoint and mutant, the twins are:

- **E1** — undo the transform and standardization. A precondition, not an
  analysis; `datasets/channel_transforms.py` makes the inverse exact.
- **E3** — the descriptor on generated cells, each column compared with its
  real distribution. No v4 runner exists; the pilot's `channel_statistics.py`
  (April, three-channel RGB) is superseded.
- **E3b** — generated cells projected onto the C4b map for each grid.

### 3.2 Between layers

| | analysis | level | question | job | after-training twin | code · state |
|---|---|---|---|---|---|---|
| C1 | Spearman correlation among layers, per unit and per cell | unit; cell | which layers carry the same information? what would a driving-factor regression see? | **flags** near-duplicates; **is** the correlationist baseline F1 argues against | E3's correlation columns; F1 | preview: `tiles.json`, 80 tiles. A1 flagged count–entropy (0.90 unit, 0.96 cell). **Density–FAR is higher and was not flagged, §5** |
| C2 | within-cell correlation for every pair of layers | cell | inside one tile, do two layers rise and fall together? | **reference** | E3 | in the descriptor. **Pearson**, where C1 uses Spearman |
| B10b, second half | the joint occupied region: a unit's 13 values as one point, the city as the set of points it actually fills | unit | which combinations does the city never produce? | **reference** | the share of generated units that land outside the real region. No node defines the measure | **not measured, no code** |
| — | rules that hold by construction (height only on a building; entropy only where the ring holds two places …) | unit | can this combination exist at all? | **acceptance test** for B8, then **reference** | violation rate on generated cells — an exact error count, not a distance | **new.** No node lists it. The real data already breaks it, §4.2 |
| E3c | spatial relations between grids: bivariate Moran's I, cross-correlogram by distance, co-location of unit classes | cell | do buildings sit *on* accessible units, or only share a tile with them? | **reference and comparison** — the node asks for both | the same, on generated cells | `analysis/spatial.py` has `bivariate_morans_i_matrix`, tested; not in the descriptor, not run |
| C3 | typology: HDBSCAN on unit vectors, then each cell's class composition | unit → cell | what kinds of place exist, and how does each cell mix them? | **reference** for the descriptor's `unitclass_*` columns; input to C5 and C6 | composition of generated cells, through `describe_cells(unit_labeller=…)` | **no v4 code.** The pilot's k-means in `pre_encoding_analysis.py` (June, RGB) clustered cell summaries, not units |
| C4, C5 | intrinsic dimension (PCA k95, Levina–Bickel MLE, PCA per typology); do typologies sit in low-dimensional subspaces? | dataset | how many examples does each typology need before the model can generalize on it? | **decides** C6's minimal set and C7's mutant sizes | E5, the denoiser's Jacobian rank | pilot code on RGB images, `pre_encoding_analysis.py`. **The level is unstated, §5** |

After training, the twins are E3's correlation columns, E3c on generated cells,
the class composition of generated cells, the rule violation rate, the share of
generated units outside the real joint region, **E5** (future;
`jacobian.py` in the EDM repo, not adapted) and **F1**, which sets the mutant
invariants against C1.

### 3.3 The whole cell at once

These look at the full tensor and belong to neither side of the within/between
split.

| | analysis | when | question | code · state |
|---|---|---|---|---|
| D2 | F_mem of an untrained model | before, per mutant | what does the memorization metric score on nonsense? | `scripts/fmem_noise_floor.py` |
| E2 | k-NN gap ratio on standardized tensors | after | did the model copy a specific training cell? | `evaluation/gap_ratio.py`, tested |
| E4 | the window: structure onset (E3, E3b, E3c) to memorization onset (E2 above D2) | after | when does the model generalize, and for how long? | pilot `multi_signal_window.py` (April, RGB); no v4 runner |

E2 is the only after-training analysis that compares a generated cell with
*individual* training cells. Everything else compares distributions.

---

## 4. What the probe measured

`data-processing/scripts/report_descriptor_probe.py` →
`reports/descriptor_probe_256.json`. **40 tiles drawn at random (seed 0) from
the 3 010 cells** (median 401 buildings, range 115 – 4 321), each layer
computed by the preview code at **256 px** with a 400 m buffer, the degree
junction rule included. These are preview numbers; B8 will change some of them,
and the table says which.

### 4.1 Moran's I at one pixel measures the interpolation, not the city

Each smooth layer is computed on a coarse native grid and read onto 256 by
bilinear interpolation (B7). Neighbouring pixels then share most of their value
whatever the city does. To see how much, the same statistic was run on white
noise laid on each layer's native grid and read onto 256 the same way.

| layer | native samples across a tile | Moran's I, real (median of 40) | Moran's I, white noise read the same way |
|---|---:|---:|---:|
| dist_centre | 5 | 1.000 | 0.999 |
| transit_access | 10 | 1.000 | 0.997 |
| poi_count, poi_entropy | 16 | 0.999 | 0.992 |
| dist_park, severance | 16 | 1.000 | 0.992 |
| building_form | 32 | 0.998 | 0.968 |
| building_density, far, road_density | 49 | 0.998 | 0.928 |
| intersection | 49 | 0.996 | 0.928 |
| dist_water | 49 | 1.000 | 0.928 |
| population | 49 | 0.994 | 0.928 |
| **building_binary** | 256 | **0.490** | 0.000 |
| **building_height** | 256 | **0.595** | 0.000 |
| **green** | 256 | **0.904** | 0.000 |

On the 13 smooth layers the real value and the noise value both sit at the
ceiling, so the column cannot tell the city from noise. And because a constant
grid scores 0.0 by design (`spatial.morans_i`), on those layers the column is
in effect a flag for "varies inside this cell or not". Its 10th percentile is
0.000 for transit, entropy and population, so at least 4 of the 40 tiles score
exactly what a constant grid returns on each of those layers. Only the three
sharp layers read a real pattern.

Native samples are counted per 1 201 m of tile, on the grid computed over the
tile plus its 400 m buffer; A1's "native grid" column counts over the tile
alone, hence its 17 where this says 16.

Two things follow. The descriptor needs a spatial statistic at each layer's own
scale: Moran's I on the native grid, or a correlogram with lags in metres. And
the resolution comparison A1 §3.4 planned ("Moran's I on the descriptor", 256
against 512) would, on the smooth layers, compare two interpolations. At 512
lag one is half as far, so it sits closer still to 1.

### 4.2 The real data already breaks rules that hold by construction

| rule | share of units breaking it | tiles | cause | after B8 |
|---|---:|---:|---|---|
| height > 0 → a building is here | **21.3 %** of units with a height | 40 of 40 | height is burned from 3D-GloBFP outlines, footprint from Overture | holds exactly: height joins onto Overture footprints (A1 §5.2) |
| a building → height > 0 | **19.2 %** of building units | 40 of 40 | buildings with no 3D-GloBFP match | **persists.** These are the unmatched buildings, 30 % by count and 18 % by area in A1's sample. Zero height then means either "no building" or "no height known"; the footprint grid tells them apart, the way count does for entropy |
| FAR > 0 → density > 0 | 2.4 % of FAR units | 35 of 40 | the same two sources | holds exactly |
| density > 0 → FAR > 0 | 8.8 % of density units | 38 of 40 | unmatched buildings add no floor area | shrinks, does not vanish |
| entropy > 0 → ≥ 2 places in the ring (≥ 7.07 per km²) | **11.5 %** of entropy units | 28 of 40 | the code guarantees the rule on the native grid; every breach comes from the bilinear read onto 256. Between a native point with places and one without, both values fall linearly, and the count drops below two places before entropy reaches zero | **persists** unless read with nearest-neighbour or checked on the native grid |
| entropy ≤ ln 11 | 0 | 0 | — | holds |

This test has to pass on the real store before it can score a generated
cell. If it doesn't, the model is charged for what the encoding did. So the
check belongs in B8's acceptance, and the two rules that persist need either a
different read or a stated tolerance first. It also refines B10b's example:
"high entropy with no POIs is impossible by construction" is true on the native
grid and false at 256.

---

## 5. Gaps and open questions, in the order they bite

1. **Every before-training number is a preview number.** B10, the variance
   split, C1 and B10b's first half all re-run on the B8 store, on all 3 010
   cells. The variance split should add 256, which it skipped.
2. **Rule check** (§4.2): new. Write it once, run it as a B8 acceptance test,
   then on every checkpoint.
3. **A spatial statistic at each layer's scale** in the descriptor (§4.1),
   before C2's output is used as a reference, and before the 256-vs-512
   comparison.
4. **B10b's joint half has no code.** It needs a measure, such as a density
   estimate on standardized unit vectors or the share of generated units whose
   nearest real neighbour is farther than the real units' own spacing, and a
   twin in E3.
5. **Within-tile share after training** is listed nowhere, though the
   descriptor already carries its inputs. A1 §3.4 relies on it: "E3 will show
   whether the model learns anything inside a tile from" dist_water and
   transit.
6. **Pearson in C2, Spearman in C1.** On zero-inflated, heavy-tailed layers in
   data space, Pearson is set by the tail. Pick per question and say which.
7. **The level of C3–C5 is unstated.** C3 clusters *units*; Wang et al.'s
   subspaces are subspaces of *training examples*, which here are cells; C4
   does not say which it measures. The pilot's k95 of 33 was on cells, since
   `pre_encoding_analysis.py` flattens whole PNGs and the pilot's unit vector
   held three numbers, one per RGB channel. Minimal-set sizing (C6) needs the cell
   level; typology (C3) is a unit-level idea whose cell-level trace is the
   composition vector.
8. **Density and FAR correlate at 0.93** per unit and per cell (80 tiles,
   preview), higher than the count–entropy pair A1 flagged. With 3D-GloBFP
   heights mostly three to four storeys, FAR is close to density times a
   constant. C1 re-measures on B8. A1 §3.2's caution applies: part of it may be
   the shared zeros.
9. **No v4 runner for E3, E3c or E4.** The April scripts
   (`channel_statistics.py`, `synergy_analysis.py`, `multi_signal_window.py`)
   were written for three-channel RGB cells and compared cell averages, which
   is what E3c's note says cannot tell co-location from co-occurrence.

## 6. Running order once B8 exists

Before training: B8 acceptance (rules) → B10 → B11 → C1, variance split, both
halves of B10b → C2 reference descriptors → C3 → C4, C5 → C4b · D2 per mutant.

After training, per checkpoint and mutant: E1 → rules, E2, E3, E3b, E3c → E4 →
across mutants, F1.
