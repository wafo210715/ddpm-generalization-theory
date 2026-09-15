# Indicator specification table — draft for review

> [!warning] Superseded in part, 15 September 2026
> This page predates the move to the current city and to Overture. The study
> area, tile count, source data and channel set have all changed; the figures
> below are the OpenStreetMap-era values and several are wrong by an order of
> magnitude. Read [`v4_corrections_260915.md`](v4_corrections_260915.md) first.
> What survives is the method and the reasoning, not the numbers.

**Status: draft. No indicator on this page is chosen.**

A1 of the [v4 pipeline map](pipeline_v4_map.html) is the first task in the whole
plan, and it is a decision task, not a coding one:

> A footprint is a data source. Binary raster, density, and floor-area ratio are
> three different indicators of it. Each row fixes subject, source, formula,
> measurement scale, ring scale, and expected sparsity. The row count sets the
> channel count, so this comes before any loader change.

This page is the table's structure, its column definitions, and a set of
candidate rows assembled from what the plan and the pilot already establish —
so that the decision has something concrete to be made against. Selecting the
rows, and settling the questions listed at the bottom, is yours.

Once the rows are fixed, the table is the source of truth for three things
downstream: `img_channels` in the training config, `channel_names` in the
`CellTensorStore` manifest, and the per-channel `transforms` list handed to
`ChannelPipeline.fit`.

---

## What is already fixed, and so is not a column

From B1, B2 and B7, the same for every row:

| | |
|---|---|
| Cell | one zoom-15 web-Mercator tile, 1 201 m at HCMC's latitude |
| Grid | 1024 × 1024 units per cell |
| Unit | 1.17 m of ground per pixel |
| Computation | each channel at its own natural scale first, resampled to the common grid last (B7) |
| Storage | one plane of a multi-channel tensor, in physical units (B8) |
| Standardization | fitted and stored per channel after the transform (B11) |

The unit is not a per-indicator choice. B2 settles it once, at the floor set by
the finest indicator, and every row is *read* there regardless of where it is
*measured*.

---

## Column definitions

**Indicator** — the name the channel carries in the store manifest and in every
descriptor key thereafter. Snake case, no channel-position prefix: nothing
downstream should have to know that a channel is third.

**Subject** — what about the city the row is a statement about. Two rows may
share a source and still be different indicators; that is the distinction A1
opens with. Footprints as a source support at least three subjects: *is there a
building here*, *how much building is near here*, and *how much floor area is
near here*.

**Source** — the dataset the values come from, specific enough to re-fetch:
which OSM tags, which product, which release.

**Formula** — what is computed at each measurement location, written so that
two people implementing it would produce the same raster.

**Measurement scale** — the distance over which the value is computed: a decay
radius, a ring radius, or the polygon itself. This is where the indicator's
meaning lives.

**Ring scale** — the neighbourhood a ring-based indicator aggregates over, when
the formula uses one. Written separately from measurement scale because for
entropy and count the ring radius *is* the parameter that decides whether the
channel has any signal at all — a 50 m ring made the pilot's diversity channel
97.5 % zeros.

**Expected sparsity** — the anticipated share of units at or near zero, with the
evidence for the estimate. B10's marginals checkpoint measures this for real;
what belongs here is the prediction and its basis, so that a surprise at B10 is
visible as a surprise.

**Kind** — grid or condition. The companion page's three-way sort: a field that
varies within a tile becomes a grid; a value constant across a tile becomes a
condition attached to the cell; a near-duplicate of a row already present is
dropped. Only the grids count towards the channel count.

---

## Candidate rows

Assembled from what the pipeline map already commits to (B3, B4, B5) and from
the pilot's three channels. **Not a selection.** Several of these are
alternatives to each other, and the table deliberately shows the alternatives
side by side rather than resolving them.

### Subject: built form

| Indicator | Subject | Source | Formula | Measurement scale | Ring scale | Expected sparsity | Kind |
|---|---|---|---|---|---|---|---|
| `building_binary` | Is there a building on this unit | OSM `way["building"]`, HCMC extract | 1 if the unit centre falls inside a footprint polygon, else 0 | the polygon itself | none | 60–80 % zeros; HCMC blocks are dense but alleys, yards and water are not built | grid |
| `building_density` | How much building is near this unit | same | footprint area ∩ disc(r) ÷ area(disc(r)) | r, a small disc | r, open | low; a smooth field by construction | grid |
| `building_far` | How much floor area is near this unit | same, plus `building:levels` | Σ(footprint area × levels) ∩ disc(r) ÷ area(disc(r)) | r | r, open | depends entirely on `building:levels` coverage in HCMC OSM, which is unmeasured — see open question 4 | grid |
| `building_height` | How tall the building on this unit is | OSM `height` / `building:levels` | levels × assumed storey height, or `height` where tagged | the polygon | none | very high; the tag is sparse in most OSM extracts | grid |

B3 fixes the subject — "Buildings at 2 to 3 m: rasterize footprints, binary and
density" — but not the density radius, and not whether floor area enters at all.
The pilot's failure here is the sharpest evidence on the page: a 150 m buffer
coverage ratio flattened the built form to a scalar with Moran's I near 0. The
radius is the parameter that broke it.

### Subject: accessibility

| Indicator | Subject | Source | Formula | Measurement scale | Ring scale | Expected sparsity | Kind |
|---|---|---|---|---|---|---|---|
| `accessibility_gravity` | How reachable this unit is from destinations | OSM road network (`osmnx`, drive or walk graph) + POI set | Σ_j w_j · exp(−d_ij / r), d over the network | r, a few hundred metres | none | ~0 % zeros; the only pilot channel that kept spatial structure, Moran's I 0.251 | grid |
| `road_density` | How much street is near this unit | OSM `highway=*` centrelines | street length ∩ disc(r) ÷ area(disc(r)) | r | r, open | low | grid |
| `intersection_density` | How fine the street grid is here | OSM road network nodes of degree ≥ 3 | node count ∩ disc(r) ÷ area(disc(r)) | r | r, open | moderate; intersections are point events | grid |

B4 fixes the formula and the decay family. The decay radius and the destination
weighting `w_j` are open. Note that a gravity field is smooth by construction,
so reading it at 1.17 m costs nothing and adds nothing — its meaning is entirely
in `r`.

### Subject: activity mix

| Indicator | Subject | Source | Formula | Measurement scale | Ring scale | Expected sparsity | Kind |
|---|---|---|---|---|---|---|---|
| `poi_count` | How many destinations are near this unit | OSM `amenity`, `shop`, `office`, `tourism`, `leisure` | count of POIs within the ring | ring radius | open | high at small radii; the count is non-zero wherever any POI is near, so less sparse than entropy | grid |
| `poi_entropy` | How mixed the destinations near this unit are | same, with a category scheme | −Σ_k p_k log p_k over categories in the ring | ring radius | open | **97.5 % zeros at a 50 m ring in the pilot.** Entropy needs ≥ 2 POIs and ≥ 2 categories; below that it is undefined and was written as zero | grid |
| `poi_density_kernel` | How intense the destination field is here | same | Σ_j K((d_ij) / h) for a kernel K | bandwidth h | none | low; a kernel density is smooth | grid |

B5 pairs entropy with count deliberately — "Count travels as its own channel, so
diversity does not collapse into density" — and defers the category question to
B6's coverage checkpoint: "Semantic categories only if OSM coverage holds up."

### Subject: activity level (A2, and the F2 future run)

| Indicator | Subject | Source | Formula | Measurement scale | Ring scale | Expected sparsity | Kind |
|---|---|---|---|---|---|---|---|
| `night_lights` | How lit this area is at night | VIIRS DNB monthly composite | radiance, resampled | ~500 m native | none | 0 % zeros, but **~2.4 VIIRS cells across a whole 1 201 m tile** | condition, not grid |
| `green_cover` | How vegetated this unit is | Sentinel-2 derived NDVI, or OSM `landuse=grass\|forest`, `leisure=park` | NDVI threshold, or polygon rasterization | 10 m native (Sentinel-2) or the polygon | none | moderate | grid |
| `water` | Is this unit water | OSM `natural=water`, `waterway` | polygon rasterization | the polygon | none | high; HCMC has substantial water but most units are dry | grid |

A2 is explicitly unresolved in the plan — whether activity enters this paper or
the next — and the companion page's caution is a resolution caution, not a
conceptual one: a layer that is nearly constant across a tile is a cell-level
condition, not a grid.

---

## Decisions this table is waiting on

These are the ones I have deliberately not made. Each changes what gets built.

1. **Which rows.** The row count sets the channel count, and the channel count
   sets the training cost measured in
   [`benchmarks/unet_resolution.md`](../ddpm-win-master/benchmarks/unet_resolution.md).
   The plan's working number is six; the companion page's stress case is
   fourteen candidates collapsing to five to eight grids plus a few conditions.

2. **Semantic categories or plain counts** for the POI channels. B5 defers this
   to B6's coverage report and B6 has not run. If categories are used, the
   scheme itself is a further decision: OSM's raw tag values, or a grouping into
   a fixed set of functional classes.

3. **Ring and decay radii**, per row. The pilot's evidence is that this is the
   parameter that decides whether a channel carries signal: 150 m flattened
   built form, 50 m emptied diversity. Nothing here proposes replacements,
   because the right radii are an argument about HCMC's ground conditions
   rather than about the code.

4. **Whether `building:levels` coverage in HCMC OSM supports a floor-area
   row at all.** This is measurable rather than arguable, and step 6's
   re-download will produce the number; the row stays a candidate until it does.

5. **Whether binary and density both ship**, given they share a source. The
   correlationist baseline in C1 is where near-duplicate channels show up, but
   C1 runs after the rasterization, so a decision to include both is a decision
   to pay for both first.

6. **Which rows are conditions rather than grids.** Only VIIRS is obviously one.
   Any indicator whose within-tile variance turns out negligible at B10 joins it.

## Fields to add once the rows are chosen

Each selected row will also need, before B11 can run:

- **Transform** — `identity`, `log1p` or `quantile`, from
  `datasets/channel_transforms.py`. B10's marginals choose it. Note the tested
  limit: no monotone transform spreads an atom, so a channel that is 97 % zeros
  stays 97 % zeros whichever is picked.
- **Physical unit** — what one unit of the stored value means, so that an
  unpacked sample can be read.
- **Valid range** — for the store's dtype check, and for spotting a generated
  cell that has left the physical range entirely.
