# Prompt for the next session

> **Superseded, 29 Sep.** Two standing handoffs replace this file. Each is updated
> at the end of every session:
> [`HANDOFF_factor_loop.md`](HANDOFF_factor_loop.md) for the factor loops (issue #2
> parts 1–2, urgent) and [`HANDOFF_diffusion_reading.md`](HANDOFF_diffusion_reading.md)
> for reading against the argument's two threats. What follows is kept for the
> record.

> **Done, 15 Sep (later):** A1 is written and decided in `indicator_specification.md`.
> The next job is B8 at 256 px: 13 grids, 3 010 cells, following that page's §5
> list of what the store must do differently from `layers.py`. The prompt below
> is the one that produced A1, kept for the record.

Copy everything below the line into a new session in this repo.

---

We are at stage A1 of `discussion/pipeline_v4_map.html`. The job this session is
**a layer-by-layer quality check that ends in a decided A1 table** — not more
acquisition, not training.

## Read these three first, in order

1. `discussion/v4_corrections_260915.md` — every corrected number with the script
   that produces it. Several claims in older docs are wrong by an order of
   magnitude; this page says which.
2. `discussion/pipeline_v4_map.html` — the plan. It carries a banner and in-place
   corrections at B3–B6, B9 and D3. **Read B3–B5 as history**: they are one step
   per channel of the pilot's RGB file, and the "six channels" you may see
   elsewhere is that triple doubled plus one activity layer. Six was never chosen
   on the merits.
3. `data-processing/scripts/layers.py` — the 16 layers as actually computed, each
   with its measurement scale separate from its sampling unit.

Do not trust figures in `hcmc_data_status.md`, `indicator_specification.md`,
`cells_channels_units.html` or `hcmc_data_landscape.html` without checking them
against the corrections page. They carry supersession banners for this reason.

## Where things stand

- **Study area**: the 2025 city, 6 718.7 km², **5 170 zoom-15 tiles**, 3 020 of
  them holding 100+ buildings. The boundary only *chooses* tiles; data is fetched
  over the tile union (7 414 km²) plus a 1 km buffer, so no cell has a false edge.
- **Sources on disk**, all clipped to that union: Overture buildings 3 037 029
  and places 411 279, 3D-GloBFP heights 2 233 344 at 100 % coverage, the OSM
  layers, WorldCover, WorldPop, and a Mapillary survey of 10 464 658 image
  locations per cell.
- **Done**: A2, A3, B1–B7, B9, B10, B10b, C2 (`analysis/cell_descriptor.py`, with
  tests). B11's transform code exists but is unapplied.
- **Not done and blocking everything**: **B8**. The only tensor store on disk is
  ten cells of six stale channels at 1024 px, built on the old boundary from OSM.
  It is worthless; treat it as absent.

## What to actually do

Go through the layers **one at a time** and decide each. For each layer, look at
it (`hcmc_wards.html` renders all 16 per ward at a fixed 3.75 m unit), check its
marginal and its zero fraction, then write the A1 row: source, formula,
measurement scale, expected sparsity, and **grid or condition**.

Three decisions are already settled by measurement and should be carried, not
re-litigated:

- **Grid vs condition** follows the variance split in `variance_split.json`:
  share of variance living inside a tile is 89 % for footprints, 83 % footprint
  grain, 69 % junctions, 68 % height, down to 0 % for distance-to-centre and 1 %
  for distance-to-park and severance. The bottom three are conditions.
- **Resolution cannot be settled by variance.** The share is flat from 128 px to
  1024 px. What changes with resolution is arrangement, where edge density falls
  about fivefold. Any 128-vs-512 comparison needs a spatial statistic.
- **Per unit ≠ per cell.** A unit is one pixel, a cell is one tile and one
  training example. B10 needs the per-unit marginal; B10b needs the per-cell
  distribution, which is what E3 compares against generated cells.

Two decisions are **open and are the point of the session**:

- **Places and perception are rich in the core and absent elsewhere.** 524 of
  2 968 built-up tiles have no Overture place; 1 260 have no street view imagery
  at all. See `hcmc_coverage.html` and `b6_ring_coverage.html`. Decide per layer:
  keep as a grid and let absence be a value, demote to a condition, or restrict
  the tile set. This belongs in A1, not discovered at B8.
- **The diversity ring radius is not settled.** B6 reversed — the old 1.1 %
  coverage was OSM's sparsity, not the city's — but a re-run on the full urban
  set is thinner than the headline suggests. `b6_ring_coverage.html` has it.

## After A1 is written

Build **B8 at 256 px** (13 grids, 17.6 GB) before anything larger. It gives a
complete dataset to debug C3–C7 and D1 against, and 512 px at 70 GB is only worth
the disk and the days once a short run trains cleanly. That also sets the
resolution comparison up properly, with two real stores instead of an argument.

## How to work

Measure before asserting; several claims in this project failed exactly there.
Check numbers against the data on disk rather than against older documents. When
a figure contradicts a document, the figure wins and the document gets corrected
in place with the old text quoted.
