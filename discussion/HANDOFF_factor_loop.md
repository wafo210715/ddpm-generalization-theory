# Handoff · factor loop (issue #2, parts 1–2)

**Start a session with:**

> Read `discussion/HANDOFF_factor_loop.md` and do the next paper: `references/papers/<file>`.

**Next up:** loop A, concentration as built intensity. First paper:
`references/papers/urban-vitality/Gómez-Varo-2022-Jane-Jacobs-reloaded-A-contemporary-operationalization-of-urban-vitali.pdf`.
Jacobs 1961 isn't in the folder; ask the user whether they have the chapter on
concentration. *(Update this line at the end of every session.)*

This is the urgent one of two reading tasks. The other is
[`HANDOFF_diffusion_reading.md`](HANDOFF_diffusion_reading.md). Keep them in
separate sessions.

---

## What this task is

GitHub issue #2 asks for three things. This handoff covers the first two:

1. go through the data factor by factor: its source, its limits, what we compute
   from it and into it;
2. for every factor, including those with only a substitute or no source at all,
   look for a better or new measure.

The unit of work is a **factor**, not a raster. `building_binary` is data. The
factor it serves is Jacobs' **concentration**, which the vitality literature
measures in competing ways: coverage per street block, GSI, FSI, entropy of
building size, morphological types. Each measure has a bias. Reading those
measures critically, and finding where a better one sits, is the point.

## Read first, in this order

1. [`layer_cards.md`](layer_cards.md). It holds the plan (the loop, the seven
   factor loops, loop A's paper list) and the finished cards: the base map, and
   loop A's data source, `building_binary`.
2. [`../references/annotated_bibliography.md`](../references/annotated_bibliography.md):
   the index, and the entry template every paper follows.
3. [`indicator_specification.md`](indicator_specification.md), called **A1**:
   the store's current specification. 13 grids, 3 conditions, 3 010 cells,
   256 px. A card changes it only where a decision says so.
4. [`vitality_provenance.md`](vitality_provenance.md): how 32 papers measured
   each factor, extracted earlier on the Mac, in Chinese. §1 compares the
   same-named factors across papers. Its `L#` point to text files on the Mac, so
   re-cite from `references/papers/`.
5. [`vitality_indicators_review.md`](vitality_indicators_review.md) §4–§5: the
   deduplicated factor tables, with which papers used each factor and the effect
   each found.

Only when needed: [`hcmc_data_landscape.html`](hcmc_data_landscape.html) is the
factor-versus-source page, and its numbers are stale (old city, before
Overture). The measured corrections are in
[`v4_corrections_260915.md`](v4_corrections_260915.md).

## The papers on disk

`references/papers/urban-vitality/` holds **49 PDFs**, each with a text layer.
All extracted cleanly; none needs OCR. VitalVizor (Zeng 2018) is a slide deck,
and Maas 1984 is a 267-page thesis. The folder holds 29 of the 32 papers in
`vitality_provenance.md`, plus 20 new ones.

**Missing:** Jacobs 1961 (a book, not among the 32), and three of the 32:
Collins 2024, Williams 2019, Shi 2023. Ask for them when their loop comes up.

**Loop A, in reading order** (file names start with these prefixes). The order
was set on 29 Sep by what each paper measures, checked against how often its
text uses GSI, FSI/FAR, coverage, Spacematrix and typology:

| # | file prefix | why |
|---:|---|---|
| 1 | `Gómez-Varo-2022` | Jacobs made measurable. Concentration is measured as **population** density, not buildings |
| 2 | `Ye-2018` | Spacematrix per street block (GSI, FSI, L), with block types defined by GSI; Shenzhen. It uses these terms more than any other paper in the folder. Watch L341: the text calls FSI "gross floor area to building footprint", while its formula divides by block area |
| 3 | `Xia-2020-Analyzing` | BCR and FAR per street block |
| 4 | `Jiang-2024` | GSI and FSI with a morphology set on traffic zones; GSI comes out ↓ |
| 5 | `Dogan-2024` | entropy of building coverage size; 22 complexity measures from footprints |
| 6 | `Tan-2026` | FAR threshold at 0.25 from random forest and SHAP |
| 7 | `Doan-2025-Nonlinear` | non-linear and threshold effects; HKU authors, study area to check. With Tan, this is the non-linear correlationist method the paper must set itself apart from |
| 8 | `Wu-2023` | GSI and FSI with street-view perception |

Then, as time allows: `Garau-2022` (vitality potential of urban form, with
Spacematrix, Cagliari), `Kim-2020`, `Qin-2025`, `Liu-2025`, `Chen-2021` and
`Li-2022` (morphological types), `Xia-2022-The-Varying` (multidimensional
urban form), `Zhang-2021` (landscape per street block), `Yue-2021` (HCMC).

**Read across all loops:** theory in `Maas-1984` (a thesis defining vitality),
`Montgomery-1998` (urbanity and vitality) and `Sena-Işiklar-2017`. Reviews in
`Doan-2025-A-systematic-review`, `Jiang-2023` (the Chinese literature) and
`Amer-2025` (what deep learning can and cannot see). Amer and `Jiang-2025` (a
deep network predicting vitality) also matter to the argument in the other
handoff.

**Other loops, a first sort; refine when each loop starts:**
- **B** short blocks: `Scepanovic-2021`, `Kim-2020`, `Wu-2023`, `Tan-2026`,
  `Li-2022`, `Gómez-Varo-2022`, `Dogan-2024`, `Li-2021`, `Cai-2025`, `Garau-2022`,
  `Xia-2022-The-Varying`
- **C** mixed uses: `Liu-2019`, `Li-2021`, `Jiang-2024`, `Yue-2021`, `Cai-2025`,
  `Gómez-Varo-2022`, `Liu-2022`, `Wu-2023`, `Tan-2026`, `Dogan-2024`,
  `Chen-2024`, `Liu-2020`
- **D** accessibility: `Wang-2025`, `Lee-2022` (= Lee & Kang), `Mouratidis-2020`,
  `Qin-2025`, `Li-2022`, `Scepanovic-2021`, `Doan-2025-Nonlinear`, `Fan-2023`
  (waterfront)
- **E** green: `Li-2022`, `Wu-2023`, `Chen-2024`, `Mouratidis-2020`, `Zhang-2021`
- **F** people: `Chen-2021`, `Lee-2022`, `Pan-2021`, `Liu-2025`, `Tan-2026`,
  `Lan-2020`, `Jin-2017`, `Yang-2020`, `Yue-2019`
- **G** aged buildings: `Gómez-Varo-2022`, `Yue-2021`, `Cai-2025`, `Xia-2020-Analyzing`, `Li-2022`
- **Low relevance** for the factor loops: `Zhang-2025-Digital` (entrepreneurial
  vitality), `Zhang-2025-Spatially` (vitality as a cause of emissions),
  `Xia-2022-Simulation` (growth boundaries), `Zeng-2018-VitalVizor` (slides),
  and the qualitative `Nathansohn-2022`, `Ruszczyk-2023`, `Paköz-2022`.

## The loop

| step | who | what | when |
|---:|---|---|---|
| 1 | user | brings a paper into `references/papers/` | each paper session |
| 2 | Claude | extracts how it measures the loop's factor, with `L#` | each paper session |
| 3 | together | reads each measure critically; writes the bibliography entry | each paper session |
| 4 | Claude | computes the candidate measures on HCMC data, with figures | once per loop, after its papers |
| 5 | user | decides which measures become grids or conditions, and any new one | once per loop |
| 6 | Claude | builds them in B8, checks them, closes the card, updates A1 | once per loop |

Loops, from `layer_cards.md`: **A** concentration as built intensity · **B** short
blocks · **C** mixed primary uses · **D** accessibility · **E** green · **F**
concentration as people · **G** aged buildings.

## A paper session, step by step

1. Find the paper in `references/papers/urban-vitality/`. Every PDF there
   already has a text layer beside it (`<same name>.txt`, from `pdftotext -layout`,
   29 Sep). Cite lines as `L#` in that file. For a new PDF, make one the same way.
2. Add or update its row in the bibliography index: file name, status *read*.
3. Read what bears on the **current loop's factor**. Note in passing, in the
   entry, anything relevant to other loops; don't work it now.
4. Extract every measure of the factor into the entry's table: formula,
   denominator or areal unit, scale, data source, effect found, method (linear,
   spatial regression, or non-linear such as random forest), `L#`.
5. Compare with the paper's rows in `vitality_provenance.md`. Where they
   disagree, the new reading of the text wins. Note the old wording in the
   entry.
6. Critique. What the measure captures and hides. What its data source biases.
   Whether our HCMC data could reproduce it, and what that would cost.
7. Add the paper's measures to the loop's section in `layer_cards.md`. Create
   the section on the loop's first paper: `## Loop A · how the literature
   measures concentration`, holding a table of all measures across papers plus
   a list of open questions. Keep it one table, so the comparison across papers
   is readable.
8. Update this file: the **Next up** line at the top, and one line in the
   progress log at the bottom.

A paper session doesn't compute measures on the data (step 4) unless the user
asks. That happens once the loop's papers are read, so candidates are compared
side by side.

## Loop A: what to carry into every paper

- **Is coverage concentration at all?** Gómez-Varo 2022's JANE index measures
  Jacobs' concentration with population density (DENS, residents per km² in a
  100 m radius; provenance L738), not with buildings. Check what Jacobs herself
  counts.
- **The denominator is a hidden choice.** Street block (Xia's BCR), grid cell
  (Wu, 500 m), traffic zone (Jiang, Qin), buffer, and our 100 m disc (A1's
  `building_density` is in effect GSI over a disc). Each changes what the
  number means.
- **The vertical dimension.** BCR and GSI ignore it; FAR and FSI need height.
  Our height is 3D-GloBFP. It matches 70 % of Overture buildings by count and
  82 % by area (A1 row 3), and it is biased low in high-rises. Over the tile
  union's 2 233 344 polygons the maximum is 181.6 m, the median 9.9 m, and only
  229 exceed 100 m; the city's tallest building is 461 m.
- **Names are unreliable.** Qin 2025 calls FSI "building density" and GSI
  "volume rate" (provenance §1.4). Align measures by formula, never by name.
- **Non-linearity.** Tan 2026 finds a FAR threshold at 0.25 with random forest
  and SHAP. Dogan & Lee 2024 find building count rising and then falling. These
  are the non-linear correlationist methods the paper has to set itself apart
  from.
- **Our footprints have a provider seam.** Overture's footprints are 57 %
  Microsoft, 39 % Google, 4 % OSM (3 010 cells). Inside the same cell, Google
  draws buildings at 0.45× Microsoft's median size
  (`report_building_sources.py`). Any measure built on building size or count
  (grain, entropy of coverage size, building count, morphological types) will
  carry this. Each cell's shares are in `b8_256/cells.parquet`. **Whether to
  rebuild footprints from a single provider is decided in loop A.**

## State of the data, 29 Sep

- **Cells.** 3 010 training cells: 3 020 tiles with ≥ 100 Overture buildings,
  minus 10 on Côn Đảo. Each is rasterized on its exact web-Mercator tile at
  256 px, 4.69 × 4.66 m per pixel. The list is `b8_256/cells.parquet`: cell id,
  z/x/y, EPSG:3857 bounds, building count, density band, provider shares.
- **Built.** `b8_256/layers/building_binary.npy`: 3 010 × 256 × 256, uint8,
  pixel-centre rule, checked against the vectors on every cell. Mean cover is
  9.09 % in both.
- **Queued, not decided:** single provider or Overture's mix (loop A);
  mangrove wetland counted as water, 396 km² (loop D).
- **Parked on purpose** (see the other handoff): the analysis design in
  `analysis_map.md` and the correlation twin. Don't rewrite them here.

Paths are under `data-processing/data/hcmc/`: `raw/` holds the sources
(`overture_tu/`, `osm_tu/`, `heights_3dglobfp_tu.parquet`),
`processed/b8_256/` the store, and `processed/reports/` the report outputs.

## Code

- Python: `ddpm-win-master/.venv/bin/python`. The system python has no numpy.
- The B8 builder is `data-processing/scripts/build_b8.py`:
  `python build_b8.py cells | <layer> | check <layer>`. Each decided layer adds
  one function plus one check.
- Round-1 reports: `report_building_footprints.py`, `report_building_sources.py`.
  Figures: `figure_base_map.py`, `figure_building_binary.py`,
  `figure_b8_mosaic.py`, all writing to `discussion/layer_cards_fig/`.
- `layers.py` and `build_layer_preview.py` are the **preview**. B8 must fix
  their defects (A1 §5): square boxes instead of discs, height burned from
  3D-GloBFP outlines, coverage-weighted grain, vertex distances.

## Rules the user set

- **Measure before asserting.** Name the script and the tile subset for every
  number. When a number contradicts a document, the number wins, and the
  document is corrected in place with the old text quoted.
- **Plain language first.** Say what is compared and what question it answers;
  pipeline codes (B10, E3…) go in brackets. State methodology problems outright.
- **Don't write a factor's literature section from memory.** Ground each
  measure in a paper the user brought, with `L#`.
- **Look at the data.** A loop's decision session shows figures.
- **Don't commit without asking.** Papers and their text never go into git
  (`references/papers/` is ignored).

## Progress log

| date | session | paper | what changed | next |
|---|---|---|---|---|
| 2026-09-23 | base map, loop A data source | — | 3 010 cells; exact tile grid; `building_binary` built and checked; provider seam found | loop A papers |
| 2026-09-28 | plan | — | cards restructured into factor loops; loop A reading list | Jacobs 1961 |
| 2026-09-29 | papers arrive | — | 49 vitality PDFs in `references/papers/urban-vitality/`, text layers made; loop A order revised with Ye 2018 and Doan 2025 added; Jacobs 1961 missing | Gómez-Varo 2022 |
