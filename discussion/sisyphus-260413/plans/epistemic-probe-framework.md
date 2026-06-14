# Epistemic Probe Framework: DDPMs as Analytical Instruments for Urban Vitality

## TL;DR

> **Quick Summary**: Develop and demonstrate a methodological framework that uses DDPM training dynamics (generalization window detection, F_mem tracking, latent space probing) as analytical instruments for urban vitality research — shifting from correlationist paradigm to generative/connectionist paradigm.
> 
> **Deliverables**:
> - Redesigned HCMC data pipeline (full municipality boundary, 50m calculation units, global-normalized RGB fusion)
> - Pre-encoding analysis suite (intrinsic dimension via PCA, typology classification)
> - DDPM training infrastructure with logarithmic checkpointing and F_mem tracking
> - Pilot experiment validating generalization window exists for urban raster data
> - Minimum viable Section 2.4: window detection + channel synergy analysis protocol
> - Methodological framework documentation (transferable to other cities)
> 
> **Estimated Effort**: Large (3-4 weeks of focused work)
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Boundary → Pipeline → Pilot (GATE) → Full Training → Analysis

---

## Context

### Original Request
User wants to develop a methodological framework for using DDPMs as "epistemic probes" in urban vitality research, demonstrating this on HCMC data. The framework shifts vitality study from correlationist (regress vitality against factors) to generative (learn distribution, probe what model internalized), and shifts generative AI from black-box generation to connectionist paradigm (understand latent representations).

### Interview Summary
**Key Discussions**:
- Current pipeline has boundary misalignment (urban core bbox → black cells in training data). User wants full HCMC municipality.
- Core framing: DDPMs are NOT generators but "epistemic probes" — the generalization window is the methodological opening.
- Supply (physical BE: density/diversity/accessibility) vs Demand (human-centric: population/OD/activity). Model learns supply only.
- RGB encoding is explicitly lossy — value is in JOINT distribution synergies between channels, not individual channel fidelity.
- Section 2.4 (sampling/analysis) is the critical gap. Minimum viable: detect window first, then design full synergy analysis.
- Building representation: accepted that lossy encoding is inevitable in Stage 1.
- Core thesis contribution: the methodological framework itself, not empirical findings.
- User provided detailed methodology draft (sections 2.1-2.4) with specific design decisions.

**Research Findings**:
- Bonnaire et al. (2025): Generalization window detection via F_mem (k-NN gap ratio, threshold=0.333). Code in repo.
- Wang et al. (2025): Diffusion training ≡ subspace clustering. Jacobian rank analysis proves low-rank manifold learning. Code in repo.
- Non-linear thresholds (Jinan 2026, Beijing 2025) provide validation targets for latent space analysis.
- COHO (He & Aliaga 2024): Latent vectors predict socio-economic metrics >75% — proves latent representations encode real urban structure.
- AI4US (Zhang et al. 2025): GenAI oversimplifies — poor diversity, loss of outliers. Validates need for careful probing.

### Metis Review
**Identified Gaps** (addressed):
- **Per-cell normalization bug**: Current pipeline normalizes per-cell, destroying inter-cell comparability. RESOLVED: global normalization in redesign.
- **HCMC_BBOX undefined**: Pipeline can't re-run from scratch. RESOLVED: define via OSMnx geocode.
- **Small dataset risk**: ~700-2000 samples at 256×256 may memorize immediately. RESOLVED: pilot training gate (1000 steps, check F_mem before committing).
- **Resolution mismatch**: 50m calculation units = 20×20 pixels, too small for DDPM. RESOLVED: rasterize buildings at native resolution, compute diversity/accessibility at 50m then upsample.
- **Pre-encoding analysis scope**: Full subspace clustering is too complex for Stage 1. RESOLVED: PCA + k-means clustering only.
- **Circular reasoning risk**: Using PCA to CURATE training data would make DDPM's "discovery" tautological. RESOLVED: pre-encoding analysis is DIAGNOSTIC only.

---

## Work Objectives

### Core Objective
Demonstrate that DDPM training on urban raster data produces a detectable generalization window, and that the model's learned representations capture meaningful urban structure (density-diversity-accessibility synergies).

### Concrete Deliverables
- Fishnet grid covering full HCMC municipality (1km cells, data collected per-grid, not per-boundary)
- Redesigned data pipeline producing per-channel-balanced RGB PNGs per 1km cell (50m calculation units, global per-channel percentile normalization)
- Pre-encoding analysis report (PCA scree plot, intrinsic dimension, typology classification)
- DDPM training with logarithmic checkpointing (100→50K steps)
- Pilot training report with F_mem curve + resolution comparison (GATE decision: proceed/iterate)
- FID computation included as experimental metric (validates or refutes applicability to urban rasters)
- Minimum viable Section 2.4: window detection + per-channel statistics + joint distribution analysis

### Definition of Done
- [x] Pipeline produces ≥500 valid (non-empty) cells from full HCMC municipality
- [x] Pre-encoding analysis report shows intrinsic dimension estimate
- [x] Pilot training (1000 steps) produces F_mem curve
- [x] GATE decision: F_mem < 0.3 at step 1000 → proceed; else → iterate
- [ ] Full training (50K steps) with checkpoints at logarithmic intervals
- [ ] Section 2.4 analysis: F_mem evolution, channel statistics, joint distribution plots
- [ ] Methodological framework documented and transferable

### Must Have
- Fishnet grid covering full HCMC municipality (data collected per-grid extent, not boundary)
- Per-channel global percentile normalization (balances B/G/R dynamic ranges, preserves inter-cell differences)
- Logarithmic checkpointing following Bonnaire et al. protocol
- F_mem computation adapted from compute_fmem.py for ddpm-win-master format
- Pilot training as GATE decision before full training (with resolution comparison: 64/128/256)
- FID computation included (experimental — validates or refutes applicability to urban rasters)
- Pre-encoding PCA analysis (diagnostic, NOT data curation)

### Must NOT Have (Guardrails)
- NO per-cell max-stretching (old pipeline bug: each cell's max→255 destroys inter-cell magnitude information)
- NO pre-computed vitality scores baked into training data (three channels are physical BE elements commonly associated with vitality, not vitality scores)
- NO full subspace clustering implementation (Stage 2 territory)
- NO multi-resolution experiments beyond pilot comparison (500m/200m are Stage 2)
- NO dataset augmentation by default (conditional only if pilot shows immediate memorization)
- NO interleaving pipeline work with training (sequential: pipeline → validate → pilot → train)
- NO circular reasoning: pre-encoding analysis is diagnostic only, not data curation

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (ddpm-win-master has test suite)
- **Automated tests**: Tests-after (validation scripts after each pipeline stage)
- **Framework**: pytest for unit tests, Python scripts for data validation

### QA Policy
Every task includes agent-executed QA scenarios with evidence saved to `.sisyphus/evidence/`.
- **Pipeline outputs**: Python scripts checking file counts, pixel ranges, normalization correctness
- **Training**: Loss curves, checkpoint verification, F_mem computation
- **Analysis**: Statistical tests, plot generation, report files

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — START IMMEDIATELY):
├── Task 1: HCMC full municipality boundary [quick]
├── Task 2: Data pipeline redesign (50m calc units, global norm, RGB fusion) [deep]
└── Task 3: Pre-encoding analysis scripts (PCA, typology classification) [quick]
    ↳ Task 3 can start in parallel with Task 2 (uses placeholder data for testing)

Wave 2 (Validation — AFTER Wave 1):
├── Task 4: Dataset characterization + pre-encoding analysis report [unspecified-high]
└── Task 5: DDPM config + dataset loader for HCMC RGB images [quick]

Wave 3 (PILOT GATE — AFTER Wave 2):
├── Task 6: Pilot training (1000 steps, 5 checkpoints, F_mem) [deep] *** GATE ***
└── Task 7: F_mem computation pipeline (adapt compute_fmem.py) [unspecified-high]
    ↳ Task 7 can start in parallel with Task 6 (prepare scripts while GPU trains)

Wave 4 (CONDITIONAL on Gate passing — AFTER Wave 3):
├── Task 8: Full DDPM training (50K steps, log checkpoints) [deep]
├── Task 9: Section 2.4 minimum viable analysis (window detection + channel stats) [unspecified-high]
└── Task 10: Section 2.4 synergy analysis (joint B/G/R distribution evolution) [deep]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real QA — run pipeline end-to-end, verify outputs (unspecified-high)
└── F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: Task 1 → Task 2 → Task 4 → Task 5 → Task 6 → (GATE) → Task 8 → Task 9 → Task 10 → FINAL
Parallel Speedup: ~40% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 2, 3 | 1 |
| 2 | 1 | 4, 5 | 1 |
| 3 | — | 4 | 1 |
| 4 | 2, 3 | 6, 7 | 2 |
| 5 | 2 | 6, 7 | 2 |
| 6 | 4, 5 | 8, 9, 10 | 3 |
| 7 | 5 | 9 (uses F_mem results) | 3 |
| 8 | 6 (GATE PASS) | 9, 10 | 4 |
| 9 | 7, 8 | 10, FINAL | 4 |
| 10 | 9 | FINAL | 4 |

### Agent Dispatch Summary

- **Wave 1**: 3 tasks — T1 → `quick`, T2 → `deep`, T3 → `quick`
- **Wave 2**: 2 tasks — T4 → `unspecified-high`, T5 → `quick`
- **Wave 3**: 2 tasks — T6 → `deep`, T7 → `unspecified-high`
- **Wave 4**: 3 tasks — T8 → `deep`, T9 → `unspecified-high`, T10 → `deep`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. HCMC Full Municipality Boundary

  **What to do**:
  - Get HCMC municipality boundary using `ox.geocode_to_gdf("Ho Chi Minh City, Vietnam")` (OSM Relation 37650) — for reference only
  - Generate a **fishnet grid** (1km cells) that COVERS the entire HCMC municipality boundary
    - Fishnet may extend slightly beyond HCMC boundary (some cells partially outside) — this is fine
    - Data is collected based on FISHNET extent, not boundary extent
  - Replace the undefined `HCMC_BBOX` constant in `data-processing/scripts/build_hcmc_pipeline.py` (currently crashes — `HCMC_BBOX` referenced on line 96 but never defined)
  - Grid-aligned at 1km resolution (snap to UTM coordinates divisible by 1000m in EPSG:32648)
  - Also generate grid templates for 500m and 200m resolutions (snap at finest resolution, coarser grids naturally align)
  - Save fishnet grid GeoPackage to `data-processing/data/hcmc/raw/`
  - Validate OSM data completeness per grid cell: count buildings, POIs, and road network km
  - Cells completely outside HCMC with no data are excluded during pipeline processing
  - Rural/peri-urban cells with sparse data are KEPT — useful for urban-rural contrast

  **Must NOT do**:
  - Do NOT clip data collection to HCMC boundary — collect data for the full fishnet extent
  - Do NOT exclude rural/peri-urban cells (they provide useful contrast)
  - Do NOT delete existing cached `hcmc_boundary.gpkg` until new one is validated

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 3)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 2, Task 3 (needs boundary before pipeline redesign)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `data-processing/scripts/build_hcmc_pipeline.py:86-104` — Current boundary function (`get_hcmc_boundary()`) using undefined `HCMC_BBOX`
  - `data-processing/scripts/plot_poi_distribution.py:84` — Current bbox `(106.58, 10.70) to (106.85, 10.90)` for reference

  **API/Type References**:
  - OSMnx API: `ox.geocode_to_gdf("Ho Chi Minh City, Vietnam")` returns GeoDataFrame with administrative boundary
  - OSM Relation ID: 37650 (Ho Chi Minh City municipality)

  **External References**:
  - OSMnx docs: geocoding and boundary retrieval

  **Acceptance Criteria**:

  - [ ] `data-processing/data/hcmc/raw/hcmc_boundary_full.gpkg` exists and contains 1 polygon
  - [ ] `data-processing/data/hcmc/raw/hcmc_grid_1km.gpkg` exists with grid cells
  - [ ] Boundary area is approximately 2,000-2,100 km² (full municipality)
  - [ ] Grid is aligned: all grid lines are at integer UTM coordinates divisible by 1000m
  - [ ] Data completeness report saved showing per-cell building/POI/road counts
  - [ ] Rural cells (<10% building coverage) are flagged but not deleted

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Boundary is full municipality, not urban core
    Tool: Bash (python)
    Preconditions: boundary file exists
    Steps:
      1. python -c "import geopandas as gpd; gdf = gpd.read_file('data-processing/data/hcmc/raw/hcmc_boundary_full.gpkg'); area = gdf.to_crs(32648).area.iloc[0] / 1e6; print(f'Area: {area:.1f} km²'); assert area > 1500, f'Too small: {area}'"
    Expected Result: Area > 1500 km²
    Failure Indicators: Area < 1000 km² (still urban core) or crash
    Evidence: .sisyphus/evidence/task-1-boundary-area.txt

  Scenario: Grid alignment verification
    Tool: Bash (python)
    Preconditions: grid file exists
    Steps:
      1. python -c "import geopandas as gpd; grid = gpd.read_file('data-processing/data/hcmc/raw/hcmc_grid_1km.gpkg').to_crs(32648); bounds = grid.total_bounds; xmin, ymin = bounds[0], bounds[1]; assert xmin % 1000 == 0, f'Grid not aligned: xmin={xmin}'; assert ymin % 1000 == 0, f'Grid not aligned: ymin={ymin}'; print(f'Grid aligned: {len(grid)} cells')"
    Expected Result: Grid origin at integer km, no fractional coordinates
    Failure Indicators: Coordinates not divisible by 1000
    Evidence: .sisyphus/evidence/task-1-grid-alignment.txt
  ```

  **Commit**: YES
  - Message: `feat(boundary): add full HCMC municipality boundary via OSMnx`
  - Files: `data-processing/scripts/build_hcmc_pipeline.py`, `data-processing/data/hcmc/raw/hcmc_boundary_full.gpkg`, `data-processing/data/hcmc/raw/hcmc_grid_1km.gpkg`
  - Pre-commit: verify boundary loads and area > 1500 km²

- [x] 2. Data Pipeline Redesign (50m Calculation Units, Global Normalization, RGB Fusion)

  **What to do**:
  - Redesign `build_hcmc_pipeline.py` (or create new `build_hcmc_pipeline_v2.py`) with:
    - **Calculation unit**: 50m × 50m squares within each 1km cell (20×20 units per cell)
    - **B channel (Building density)**: For each 50m unit, compute building footprint coverage within a buffer ring (user suggested buffer approach). Result: 20×20 density map per cell → bilinear upsample to 128×128 or 256×256
    - **G channel (POI diversity)**: For each 50m unit, compute Shannon entropy of POI categories within a buffer ring. Result: 20×20 entropy map per cell → bilinear upsample to 128×128 or 256×256
    - **R channel (Accessibility)**: For each 50m unit, compute gravity-model distance decay to road network. Result: 20×20 accessibility map per cell → bilinear upsample to 128×128 or 256×256
    - **Buffer ring size**: Determine empirically (100m? 200m?) — must be larger than 50m unit, smaller than 1km cell
    - **Channel-balanced normalization**: For each channel, compute global 95th percentile across ALL cells. Normalize each channel by its global percentile (not per-cell max-stretching). This balances B/G/R dynamic ranges so no single channel dominates the fused image, while preserving inter-cell magnitude differences. Map to [0, 255].
    - **Why not per-cell max-stretch**: Per-cell normalization (old pipeline bug) would make a cell with accessibility [0, 0.1] look identical to one with [0, 1.0], destroying the actual magnitude information the DDPM needs to learn.
    - **Why not simple divide-by-255**: The raw channel values may have very different scales (B: 0-0.8, G: 0-3.5, R: 0-1.0). Simply stacking them would make one channel dominate. Per-channel percentile normalization brings them to comparable ranges.
    - **RGB fusion**: Stack B+G+R into single PNG per cell
    - **Cell filtering**: Exclude cells with >90% zero pixels across all channels
  - The output resolution for DDPM training: 128×128 (start conservative — smaller images = less memorization risk per Metis analysis)
  - Save per-cell PNGs to `data-processing/data/hcmc/processed/cells/`
  - Save per-channel B/W images to `data-processing/data/hcmc/processed/channels/` for visualization
  - Generate data quality report: per-channel pixel statistics, cross-channel correlations, Moran's I

  **Must NOT do**:
  - Do NOT apply per-cell max-stretching (each cell's max→255 destroys inter-cell magnitude information — this was the old pipeline bug)
  - Do NOT pre-compute vitality scores or bake human-behavioral data into images
  - Do NOT delete old pipeline files — create v2 alongside
  - Note: Resolution (128×128 vs 256×256) will be tested in pilot — not fixed yet

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Multi-file pipeline rewrite requiring understanding of geospatial processing, rasterization, normalization

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1 boundary)
  - **Parallel Group**: Wave 1 (starts after Task 1)
  - **Blocks**: Task 4, Task 5
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `data-processing/scripts/build_hcmc_pipeline.py:255-283` — Current `compute_building_channel()` returns scalar, not raster (this is what we're replacing)
  - `data-processing/scripts/build_hcmc_pipeline.py` — `rasterize_buildings()`, `rasterize_poi_entropy()`, `rasterize_accessibility()` in `main_rework()` — existing rasterization functions to build upon
  - `data-processing/scripts/fix_cell_images.py` — Contains the per-cell normalization bug to AVOID

  **API/Type References**:
  - `rasterio.features.rasterize()` — for building footprint rasterization
  - `scipy.ndimage.zoom` — for bilinear upsampling from 20×20 to 128×128
  - `pandana.Network` — for accessibility computation
  - `scipy.stats.entropy` — for Shannon index

  **External References**:
  - User's methodology section 2.2.1: "for each unit, we provide it with a buffer ring to calculate the Shannon index"
  - User's methodology section 2.2.2: "each urban space unit doesn't need to normalize across the whole study area, but within each unit, the three channels should be normalized to avoid one channel being too dominant"
  - **NOTE**: User said "within each unit" for channel balance but the previous per-cell normalization was a BUG. The correct interpretation: normalize each channel GLOBALLY (so inter-cell differences are preserved), but ensure the three channels have comparable dynamic ranges (so one doesn't dominate).

  **Acceptance Criteria**:

  - [ ] `data-processing/data/hcmc/processed/cells/` contains ≥500 PNG files (128×128×3)
  - [ ] Each PNG has 3 channels with values in [0, 255]
  - [ ] Per-channel B/W images saved to `data-processing/data/hcmc/processed/channels/`
  - [ ] Global normalization verified: pixel max varies across cells (not all 255)
  - [ ] Data quality report with per-channel statistics, cross-channel correlations
  - [ ] ≤5% of cells are >90% empty (black)
  - [ ] Cross-channel correlations reported (B-G, B-R, G-R)
  - [ ] ⚠️ **GAP-3 FIXED**: Cell naming uses `{row:03d}_{col:03d}.png` (not sequential `{label:04d}.png`)
  - [ ] ⚠️ **GAP-4 FIXED**: Building channel is continuous density [0,1] (not binary 0/255)
  - [ ] ⚠️ **GAP-6 FIXED**: Normalization uses per-channel global 95th percentile (not per-canvas max)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Global normalization preserves inter-cell differences
    Tool: Bash (python)
    Preconditions: cells directory populated
    Steps:
      1. python -c "import numpy as np; from PIL import Image; from pathlib import Path; cells = sorted(Path('data-processing/data/hcmc/processed/cells').glob('*.png')); vals = [np.array(Image.open(c)).astype(float) for c in cells[:20]]; maxes = [v.max() for v in vals]; mins = [v.min() for v in vals]; print(f'Max across cells: min={min(maxes):.0f}, max={max(maxes):.0f}'); print(f'Min across cells: min={min(mins):.0f}, max={max(mins):.0f}'); assert max(maxes) > 200, 'Max values too low'; assert min(maxes) < 250, 'All cells maxed out — likely per-cell norm'"
    Expected Result: Cell max values vary (not all 255). Some cells have low max (rural), some high (urban core).
    Failure Indicators: All cells have identical max values → per-cell normalization applied
    Evidence: .sisyphus/evidence/task-2-global-norm.txt

  Scenario: No black-void cells in training set
    Tool: Bash (python)
    Preconditions: cells directory populated
    Steps:
      1. python -c "from pathlib import Path; from PIL import Image; import numpy as np; cells = sorted(Path('data-processing/data/hcmc/processed/cells').glob('*.png')); empty = [c for c in cells if np.array(Image.open(c)).max() < 10]; print(f'Total: {len(cells)}, Empty: {len(empty)}, Valid: {len(cells)-len(empty)}'); assert len(empty) <= len(cells)*0.05, f'Too many empty: {len(empty)}/{len(cells)}'"
    Expected Result: ≤5% empty cells
    Failure Indicators: >10% empty cells → boundary or filtering issue
    Evidence: .sisyphus/evidence/task-2-cell-quality.txt
  ```

  **Commit**: YES
  - Message: `feat(pipeline): redesign data pipeline with 50m calculation units and global normalization`
  - Files: `data-processing/scripts/build_hcmc_pipeline_v2.py`, processed data files
  - Pre-commit: verify cell count ≥ 500 and global normalization

- [x] 3. Pre-encoding Analysis Scripts (PCA, Typology Classification)

  **What to do**:
  - Create `data-processing/scripts/pre_encoding_analysis.py` with:
    - **PCA on flattened images**: Load all cell PNGs → flatten to 1D vectors → fit PCA → scree plot → report variance explained by top-k components → estimate intrinsic dimension (where cumulative variance reaches 95%)
    - **Typology classification**: Extract per-cell scalar features (B_mean, G_mean, R_mean, B_std, G_std, R_std, Moran's I per channel) → k-means clustering (k=3 to 7) → silhouette score → identify optimal k → label each cell with typology
    - **Distribution analysis**: Check if data fits single Gaussian or mixture → report (if mixture, note that subspace clustering would be appropriate per Wang et al. 2025, but is deferred to Stage 2)
    - **Minimal dataset identification**: Based on PCA, identify how many images capture 95% of variance → report as "effective dataset size"
  - Scripts should work with placeholder data (random images) for testing during Wave 1, then run on real data in Wave 2
  - Save outputs to `data-processing/data/hcmc/reports/pre_encoding/`

  **Must NOT do**:
  - Do NOT implement full subspace clustering (Stage 2 territory)
  - Do NOT use PCA to CURATE or FILTER the training data (circular reasoning — diagnostic only)
  - Do NOT remove images based on PCA results

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1 — uses placeholder data for testing)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 4
  - **Blocked By**: None (for script creation; actual analysis in Task 4 depends on Task 2)

  **References**:

  **Pattern References**:
  - `Diffusion-Model-Generalizability-main/edm/jacobian.py` — Jacobian rank analysis (reference for manifold analysis methodology, not direct port)
  - `Diffusion-Model-Generalizability-main/edm/trainMoLRG.py` — Mixture of Low-Rank Gaussians (Wang et al.'s synthetic data for subspace clustering validation)

  **External References**:
  - Wang et al. (2025): "Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering" — the subspace clustering theory motivating intrinsic dimension analysis
  - User's methodology section 2.2.3: "PCA to identify the minimal dataset that represents the 95% variance"
  - User's methodology section 2.2.3: "classify the typology of our dataset using the three-dimensional information"

  **Acceptance Criteria**:

  - [ ] `data-processing/scripts/pre_encoding_analysis.py` exists and runs without error on test data
  - [ ] PCA produces scree plot saved to `data-processing/data/hcmc/reports/pre_encoding/pca_scree.png`
  - [ ] Typology classification produces labeled dataset saved to `data-processing/data/hcmc/reports/pre_encoding/typology_labels.csv`
  - [ ] Intrinsic dimension estimate reported (number of components for 95% variance)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: PCA runs on random test data
    Tool: Bash (python)
    Preconditions: Script exists
    Steps:
      1. Generate 50 random 128x128x3 images as test data in temp directory
      2. Run: python data-processing/scripts/pre_encoding_analysis.py --input <test_dir> --output data-processing/data/hcmc/reports/pre_encoding/
      3. Check that pca_scree.png and typology_labels.csv exist
    Expected Result: Script completes without error, produces output files
    Failure Indicators: Script crashes or produces empty files
    Evidence: .sisyphus/evidence/task-3-pca-test.txt
  ```

  **Commit**: YES
  - Message: `feat(analysis): add pre-encoding PCA and typology classification scripts`
  - Files: `data-processing/scripts/pre_encoding_analysis.py`
  - Pre-commit: verify script runs on test data

- [x] 4. Dataset Characterization + Pre-encoding Analysis Report

  **What to do**:
  - Run `pre_encoding_analysis.py` on the REAL cell images from Task 2
  - Generate comprehensive report including:
    - Intrinsic dimension estimate (PCA scree plot, cumulative variance)
    - Typology classification results (optimal k, cluster sizes, silhouette score)
    - Distribution analysis: single Gaussian vs mixture assessment
    - Cross-channel correlation analysis (B-G, B-R, G-R — mutual information if >0 means synergies are learnable)
    - Per-channel marginal histograms
    - Sample images from each typology cluster
  - If mutual information between channels is near zero, flag as risk (synergy analysis may be meaningless)
  - Save full report to `data-processing/data/hcmc/reports/pre_encoding/full_report.md`
  - This report INFORMS interpretation but does NOT guide data selection (diagnostic only)

  **Must NOT do**:
  - Do NOT filter or curate training data based on analysis results
  - Do NOT implement subspace clustering

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 2 and 3)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6
  - **Blocked By**: Task 2, Task 3

  **References**:

  **Pattern References**:
  - `data-processing/scripts/pre_encoding_analysis.py` — Created in Task 3
  - `data-processing/data/hcmc/reports/` — Existing quality reports from old pipeline

  **Acceptance Criteria**:

  - [ ] Full report saved to `data-processing/data/hcmc/reports/pre_encoding/full_report.md`
  - [ ] Intrinsic dimension estimate stated with specific number
  - [ ] Typology classification with optimal k and cluster sizes
  - [ ] Cross-channel MI or correlation values reported
  - [ ] Sample images from each cluster saved

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Report contains all required sections
    Tool: Bash (python)
    Preconditions: Report exists
    Steps:
      1. python -c "from pathlib import Path; report = Path('data-processing/data/hcmc/reports/pre_encoding/full_report.md'); content = report.read_text(); required = ['intrinsic dimension', 'typology', 'correlation', 'mutual information']; for r in required: assert r.lower() in content.lower(), f'Missing: {r}'; print('All sections present')"
    Expected Result: Report contains all required analysis sections
    Evidence: .sisyphus/evidence/task-4-report-check.txt
  ```

  **Commit**: YES (groups with Task 3)
  - Message: `feat(analysis): generate pre-encoding analysis report for HCMC dataset`
  - Files: report files in `data-processing/data/hcmc/reports/pre_encoding/`

- [x] 5. DDPM Config + Dataset Loader for HCMC RGB Images

  **What to do**:
  - Create `ddpm-win-master/configs/hcmc_vitality_v2.yaml` with:
    - Image size: 128×128×3 (initial; pilot Task 6 will test 64/128/256 and select best)
    - Schedule type: linear (same as Bonnaire et al.)
    - Training steps: 50,000 (with pilot override to 1,000)
    - Log checkpointing: steps [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 50000]
    - U-Net: base_channels=32 (conservative for memorization risk; dual RTX 3090 can handle 64 if pilot shows it's needed)
    - Batch size: 16 (fitting dual RTX 3090 memory at 128×128×3)
    - Learning rate: 0.0001 (Bonnaire et al. default)
    - Data path: `data-processing/data/hcmc/processed/cells/`
  - Create/update `ddpm-win-master/datasets/dataset.py` with:
    - `GeoSpatialDataset` class: loads PNG images from directory
    - Supports train/val/test split (80/10/10, stratified by typology from Task 4)
    - Returns tensor in [-1, 1] range (standard DDPM normalization)
    - Preserves cell ID in metadata for later analysis
  - Validate: DataLoader returns correct shapes, values in [-1, 1]

  **Must NOT do**:
  - Do NOT fix resolution — pilot (Task 6) will compare 64/128/256 and select best

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 4)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6, Task 7
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `ddpm-win-master/configs/default.yaml` — MNIST config for reference
  - `ddpm-win-master/configs/geospatial.yaml` — Existing geospatial config (512×512, needs update)
  - `ddpm-win-master/config_loader.py` — `TrainingConfig` class with `_generate_log_schedule()`
  - `ddpm-win-master/train.py:153-158` — Checkpoint saving logic

  **API/Type References**:
  - `ddpm-win-master/model/ddpm.py` — DDPM class expecting specific config format
  - `ddpm-win-master/model/unet.py` — DDPMUnet with `model_channels` parameter (NOTE: `base_channels` is NOT an actual parameter name — it's `model_channels`)
  - ⚠️ **GAP-1**: train.py:80 HARDCODES `DDPMUnet(config.img_channels, 64)` — `base_channels` config field is NOT wired through. Task 5 must fix this: read from config or keep hardcoded and document.

  **Acceptance Criteria**:

  - [ ] `configs/hcmc_vitality_v2.yaml` exists with correct parameters
  - [ ] `datasets/dataset.py` has `GeoSpatialDataset` class
  - [ ] DataLoader returns batch shape (B, 3, 128, 128) with values in [-1, 1]
  - [ ] Train/val/test split files saved (cell IDs per split)
  - [ ] Config loads without error in train.py
  - [ ] ⚠️ **GAP-1 FIXED**: train.py reads model_channels from config (or documents hardcoded value)
  - [ ] ⚠️ **GAP-2 FIXED**: GeoSpatialDataset returns dict `{'image': Tensor, 'cell_id': str}` (not plain Tensor)
  - [ ] evaluation/memorization.py:218 also uses config-based model_channels (not hardcoded 64)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Dataset loader returns correct shapes
    Tool: Bash (python)
    Preconditions: cells directory populated, dataset.py exists
    Steps:
      1. python -c "import sys; sys.path.insert(0, 'ddpm-win-master'); from datasets.dataset import GeoSpatialDataset; from torch.utils.data import DataLoader; ds = GeoSpatialDataset(root='data-processing/data/hcmc/processed/cells', split='train'); dl = DataLoader(ds, batch_size=4); batch = next(iter(dl)); print(f'Shape: {batch[\"image\"].shape}'); print(f'Range: [{batch[\"image\"].min():.2f}, {batch[\"image\"].max():.2f}]'); assert batch['image'].shape == (4, 3, 128, 128); assert batch['image'].min() >= -1.0; assert batch['image'].max() <= 1.0"
    Expected Result: Shape (4, 3, 128, 128), values in [-1, 1]
    Failure Indicators: Wrong shape, values outside range
    Evidence: .sisyphus/evidence/task-5-dataloader.txt
  ```

  **Commit**: YES
  - Message: `feat(config): add HCMC vitality v2 config and GeoSpatialDataset loader`
  - Files: `ddpm-win-master/configs/hcmc_vitality_v2.yaml`, `ddpm-win-master/datasets/dataset.py`

- [x] 6. Pilot Training — GATE DECISION (1000 steps, 5 checkpoints, F_mem)

  **What to do**:
  - Run DDPM training for 1000 steps at MULTIPLE resolutions to compare memorization dynamics:
    - 64×64 (5 checkpoints)
    - 128×128 (5 checkpoints)
    - 256×256 (5 checkpoints)
  - Checkpointing at [100, 250, 500, 750, 1000] for each resolution
  - After training completes, for each checkpoint at each resolution:
    - Load checkpoint
    - Generate 50 sample images
    - Compute F_mem using adapted memorization.py (Task 7)
    - Compute FID against training set (experimental — validate applicability to urban rasters)
  - Plot F_mem vs training steps for all three resolutions (overlay)
  - **GATE DECISION**:
    - **PASS**: At least one resolution shows F_mem < 0.3 at step 1000 AND loss still decreasing → select that resolution for full training (Task 8)
    - **ITERATE**: F_mem transitions between step 100-500 at all resolutions → try smaller U-Net or controlled augmentation
    - **FAIL**: F_mem > 0.5 at step 500 at all resolutions → dataset insufficient → report, discuss alternatives
  - Also report FID results: if FID shows meaningful quality tracking → include in full analysis. If FID is meaningless for urban rasters → document as finding (validates hypothesis).
  - Save gate decision to `ddpm-win-master/logs/hcmc_vitality_pilot/gate_decision.md`

  **Must NOT do**:
  - Do NOT skip the pilot and go straight to 50K steps
  - Do NOT use augmentation during pilot (we need to see natural behavior first)
  - Do NOT proceed past this task without documenting the gate decision

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Requires understanding of training dynamics, F_mem interpretation, and gate decision criteria

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 7 — scripts prepare while GPU trains)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 8, Task 9, Task 10
  - **Blocked By**: Task 4, Task 5

  **References**:

  **Pattern References**:
  - `ddpm-win-master/train.py` — Training loop with checkpoint saving
  - `ddpm-win-master/sample.py` — Sample generation from checkpoints
  - `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Training/run_Unet.py` — Bonnaire et al.'s training script with log checkpointing
  - `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Generation/generate.py` — DDIM sampling protocol

  **External References**:
  - Bonnaire et al. (2025): F_mem goes from ~0 to ~1 during training. Generalization window is the sigmoid transition region.
  - Wang et al. (2025): In generalization regime, model learns low-dimensional manifold. In memorization regime, it memorizes individual samples.

  **Acceptance Criteria**:

  - [ ] Pilot training completed: 1000 steps, 5 checkpoints saved
  - [ ] F_mem computed at each checkpoint
  - [ ] F_mem vs steps plot saved to `ddpm-win-master/logs/hcmc_vitality_pilot/fmem_curve.png`
  - [ ] Gate decision documented in `gate_decision.md` with PASS/ITERATE/FAIL
  - [ ] Loss curve plotted and saved

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Pilot training produces valid checkpoints
    Tool: Bash (python)
    Preconditions: Training completed
    Steps:
      1. python -c "from pathlib import Path; ckpts = sorted(Path('ddpm-win-master/model_ckpts/hcmc_vitality_pilot').glob('*.pth')); print(f'Checkpoints: {len(ckpts)}'); assert len(ckpts) >= 5, f'Need >= 5 checkpoints, got {len(ckpts)}'; [print(f'  {c.name}') for c in ckpts]"
    Expected Result: ≥5 checkpoint files
    Failure Indicators: Fewer than 5 → training didn't complete
    Evidence: .sisyphus/evidence/task-6-pilot-checkpoints.txt

  Scenario: F_mem curve shows training dynamics
    Tool: Bash (python)
    Preconditions: F_mem computed
    Steps:
      1. python -c "from pathlib import Path; report = Path('ddpm-win-master/logs/hcmc_vitality_pilot/gate_decision.md'); content = report.read_text(); print(content); assert 'PASS' in content or 'ITERATE' in content or 'FAIL' in content, 'No gate decision'"
    Expected Result: Gate decision clearly stated
    Evidence: .sisyphus/evidence/task-6-gate-decision.txt
  ```

  **Commit**: YES
  - Message: `feat(training): pilot DDPM training with F_mem gate analysis`
  - Files: checkpoints, logs, gate decision report

- [x] 7. F_mem Computation Pipeline (Adapt compute_fmem.py for ddpm-win-master)

  **What to do**:
  - Adapt `ddpm-win-master/evaluation/memorization.py` (already partially ported) for integration with:
    - ddpm-win-master's checkpoint format (state_dict with `eps_model` key)
    - ddpm-win-master's sample.py output format
    - Batch processing across multiple checkpoints
  - Create `ddpm-win-master/evaluation/compute_fmem_batch.py` that:
    - Takes a directory of checkpoints
    - For each checkpoint: load model → generate N samples → compute F_mem against training set
    - Outputs F_mem curve as CSV and plot
  - The F_mem computation uses k-NN gap ratio (d₁/d₂ < 1/3 = memorized), following Bonnaire et al.
  - Ensure computation works on GPU (distance matrix can be large)
  - Reference implementation: `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation/compute_fmem.py`

  **Must NOT do**:
  - Do NOT couple to Bonnaire et al.'s file structure (make it work with ddpm-win-master format)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 6 — prepare scripts while GPU trains)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 9
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `ddpm-win-master/evaluation/memorization.py` — Already ported F_mem computation (lines 61-80: `compute_memorization_fraction()`)
  - `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation/compute_fmem.py` — Original implementation with batch processing and diagnostic plots

  **API/Type References**:
  - `ddpm-win-master/evaluation/memorization.py:61` — `compute_memorization_fraction(train_images, generated_images, gap_threshold=1/3, k=2)`
  - `ddpm-win-master/sample.py` — Sample generation script (loads checkpoint, generates images)

  **Acceptance Criteria**:

  - [ ] `evaluation/compute_fmem_batch.py` exists and accepts checkpoint directory as input
  - [ ] Outputs CSV with columns: checkpoint_step, f_mem, f_mem_ci_lower, f_mem_ci_upper
  - [ ] Outputs F_mem curve plot (fmem_curve.png)
  - [ ] Tested on at least 2 checkpoints from pilot training

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: F_mem batch computation produces valid output
    Tool: Bash (python)
    Preconditions: At least 2 pilot checkpoints exist
    Steps:
      1. Run compute_fmem_batch.py on pilot checkpoint directory
      2. Verify CSV exists with ≥2 rows
      3. Verify F_mem values are in [0, 1] range
    Expected Result: CSV with F_mem values between 0 and 1
    Failure Indicators: F_mem > 1 or < 0, or script crashes
    Evidence: .sisyphus/evidence/task-7-fmem-batch.txt
  ```

  **Commit**: YES
  - Message: `feat(evaluation): batch F_mem computation across checkpoints`
  - Files: `ddpm-win-master/evaluation/compute_fmem_batch.py`

- [ ] 8. Full DDPM Training (50K steps, Log Checkpoints) — CONDITIONAL ON TASK 6 GATE PASS

  **What to do**:
  - Only execute if Task 6 gate decision is PASS
  - Run full DDPM training for 50,000 steps
  - Checkpoint at logarithmic intervals: [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 50000]
  - Save training loss log (JSONL format with step, loss per entry)
  - Monitor GPU usage, save training time per 1K steps
  - Use dual GPU if available (DataParallel)
  - Training config from `hcmc_vitality_v2.yaml`
  - After training completes, generate 50 samples from each checkpoint for Section 2.4

  **Must NOT do**:
  - Do NOT train if pilot gate was FAIL (report and discuss alternatives instead)
  - Do NOT skip checkpoint saving at any log interval
  - Do NOT use augmentation unless pilot specifically showed it was needed

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (GPU-bound)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 9, Task 10
  - **Blocked By**: Task 6 (GATE PASS required)

  **References**:

  **Pattern References**:
  - `ddpm-win-master/train.py` — Main training loop
  - `ddpm-win-master/config_loader.py:_generate_log_schedule()` — Log checkpoint schedule generation
  - `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Training/run_Unet.py` — Reference training protocol

  **Acceptance Criteria**:

  - [ ] Training completed: 50K steps
  - [ ] ≥8 checkpoints saved at log intervals
  - [ ] Training loss log saved as JSONL
  - [ ] 50 samples generated from each checkpoint
  - [ ] Total training time recorded

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All checkpoints present and loadable
    Tool: Bash (python)
    Preconditions: Training completed
    Steps:
      1. python -c "from pathlib import Path; import torch; ckpts = sorted(Path('ddpm-win-master/model_ckpts/hcmc_vitality').glob('*.pth')); print(f'Checkpoints: {len(ckpts)}'); assert len(ckpts) >= 8; [torch.load(str(c), map_location='cpu') for c in ckpts]; print('All loadable')"
    Expected Result: ≥8 checkpoints, all loadable without error
    Evidence: .sisyphus/evidence/task-8-full-training.txt
  ```

  **Commit**: YES
  - Message: `feat(training): full DDPM training on HCMC vitality dataset`
  - Files: checkpoints, logs, samples

- [ ] 9. Section 2.4 Minimum Viable: Generalization Window Detection + Channel Statistics

  **What to do**:
  - Run F_mem batch computation on ALL checkpoints from full training (Task 8)
  - Plot F_mem vs training steps — identify generalization window boundaries
  - For each checkpoint within the generalization window:
    - Dissect each sample into B/G/R channels
    - Compute per-channel statistics: mean, std, histogram, spatial autocorrelation (Moran's I)
    - Compare sampled channel statistics with training data channel statistics
    - Compute Wasserstein distance per channel between samples and training set
  - Track evolution of channel statistics across checkpoints:
    - What's stable across checkpoints = "irreducible characteristics"
    - What changes = learned detail or memorization
  - Generate report:
    - F_mem curve with annotated window boundaries
    - Per-channel statistic evolution plots
    - Training vs sample distribution comparison at key checkpoints
  - Save all analysis to `ddpm-win-master/analysis/section_2_4/`

  **Must NOT do**:
  - Do NOT use SHAP or mapping model (that's full 2.4, not minimum viable)
  - Do NOT proceed to Task 10 if generalization window is NOT detectable

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 8 if samples available)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 10
  - **Blocked By**: Task 7 (F_mem pipeline), Task 8 (checkpoints + samples)

  **References**:

  **Pattern References**:
  - `ddpm-win-master/evaluation/memorization.py` — F_mem computation
  - `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation/compute_fmem.py` — Original F_mem with diagnostic plots
  - User's methodology section 2.4: "Sample from all models during generalization window, dissect into B/G/R"

  **External References**:
  - Bonnaire et al. (2025): Generalization window dynamics — structural concepts learned before noisy details
  - Wang et al. (2025): Low-rank Jacobian in generalization regime → manifold learning

  **Acceptance Criteria**:

  - [ ] F_mem curve plotted with window boundaries annotated
  - [ ] Per-channel statistics computed at each checkpoint within window
  - [ ] Channel evolution plots saved
  - [ ] Report saved to `ddpm-win-master/analysis/section_2_4/minimum_viable_report.md`
  - [ ] "Irreducible characteristics" identified (features stable across window checkpoints)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: F_mem curve shows generalization window
    Tool: Bash (python)
    Preconditions: Analysis completed
    Steps:
      1. python -c "from pathlib import Path; report = Path('ddpm-win-master/analysis/section_2_4/minimum_viable_report.md'); content = report.read_text(); assert 'generalization window' in content.lower(); assert 'F_mem' in content; print('Report valid')"
    Expected Result: Report identifies generalization window boundaries
    Evidence: .sisyphus/evidence/task-9-window-report.txt

  Scenario: Channel statistics evolution plotted
    Tool: Bash (python)
    Preconditions: Analysis completed
    Steps:
      1. Check that channel evolution plots exist in analysis directory
      2. Verify each plot covers at least 3 checkpoints within window
    Expected Result: Plots exist showing B/G/R mean/std evolution
    Evidence: .sisyphus/evidence/task-9-channel-evolution.png
  ```

  **Commit**: YES
  - Message: `feat(analysis): Section 2.4 generalization window detection and channel statistics`
  - Files: analysis outputs in `ddpm-win-master/analysis/section_2_4/`

- [ ] 10. Section 2.4 Synergy Analysis: Joint B/G/R Distribution Evolution

  **What to do**:
  - This is the FULL Section 2.4 analysis, building on Task 9's minimum viable results
  - For each checkpoint within the generalization window:
    - Compute JOINT distribution of (B_mean, G_mean, R_mean) across 50 samples
    - Compare with training data joint distribution
    - Analyze synergies:
      - Density-Diversity synergy: when model generates high-B, does it also generate high-G?
      - Density-Accessibility synergy: B vs R relationship in samples
      - Diversity-Accessibility synergy: G vs R relationship in samples
      - Vitality sweet spot: cells where all three are high
    - Check if synergies match empirical thresholds (Jinan: blending >1.5, density <4000)
  - Track how synergies EVOLVE across checkpoints:
    - Early checkpoints: synergies emerge (structural learning)
    - Window middle: synergies stabilize (irreducible characteristics)
    - Late checkpoints: synergies degrade (memorization of individual cells)
  - Generate final report:
    - Joint distribution scatter plots colored by checkpoint
    - Synergy strength metrics (correlation, mutual information) vs training step
    - Comparison with empirical thresholds from literature
    - Methodological framework documentation: how to replicate this analysis on other cities
  - Save to `ddpm-win-master/analysis/section_2_4/synergy/`

  **Must NOT do**:
  - Do NOT claim causal relationships (correlation ≠ causation, even in generative models)
  - Do NOT train a mapping model in this task (that's future work)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 9 results)
  - **Parallel Group**: Wave 4
  - **Blocks**: FINAL
  - **Blocked By**: Task 9

  **References**:

  **Pattern References**:
  - `ddpm-win-master/analysis/section_2_4/minimum_viable_report.md` — From Task 9

  **External References**:
  - Jinan (2026): POI blending degree threshold at 1.5, density diminishing returns at 4000/km²
  - Beijing (2025): Intersection density optimal 22-28/km²
  - User's methodology section 2.4: "understanding the sampling from models in generalization window, what's the irreducible characteristics"

  **Acceptance Criteria**:

  - [ ] Joint distribution scatter plots exist for at least 3 checkpoints within window
  - [ ] Synergy metrics (B-G, B-R, G-R correlation/MI) computed and plotted vs step
  - [ ] Comparison with at least 2 empirical thresholds from literature
  - [ ] Methodological framework section documented (how to replicate)
  - [ ] Full report saved to `ddpm-win-master/analysis/section_2_4/synergy/full_report.md`

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Synergy analysis produces valid metrics
    Tool: Bash (python)
    Preconditions: Analysis completed
    Steps:
      1. python -c "from pathlib import Path; report = Path('ddpm-win-master/analysis/section_2_4/synergy/full_report.md'); content = report.read_text(); required = ['synergy', 'threshold', 'framework', 'irreducible']; for r in required: assert r.lower() in content.lower(), f'Missing: {r}'; print('Report valid')"
    Expected Result: Full report with synergy analysis, threshold comparison, and framework documentation
    Evidence: .sisyphus/evidence/task-10-synergy-report.txt
  ```

  **Commit**: YES
  - Message: `feat(analysis): Section 2.4 synergy analysis with joint distribution evolution`
  - Files: analysis outputs in `ddpm-win-master/analysis/section_2_4/synergy/`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` + linter + `bun test`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty state, invalid input, rapid actions. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(boundary): add full HCMC municipality boundary via OSMnx` — boundary files
- **Wave 1**: `feat(pipeline): redesign data pipeline with 50m calculation units and global normalization` — pipeline scripts
- **Wave 2**: `feat(analysis): add pre-encoding PCA and typology classification` — analysis scripts
- **Wave 2**: `feat(config): add HCMC vitality config and dataset loader` — config + dataset files
- **Wave 3**: `feat(evaluation): adapt F_mem computation for ddpm-win-master` — evaluation scripts
- **Wave 4**: `feat(training): pilot and full DDPM training with log checkpointing` — checkpoints + logs
- **Wave 4**: `feat(analysis): Section 2.4 generalization window and synergy analysis` — analysis results

---

## Success Criteria

### Verification Commands
```bash
# Pipeline output validation
python -c "from pathlib import Path; from PIL import Image; import numpy as np; cells = sorted(Path('data-processing/data/hcmc/processed/cells').glob('*.png')); print(f'Total: {len(cells)}'); empty = [c for c in cells if np.array(Image.open(c)).max() == 0]; print(f'Empty: {len(empty)}'); print(f'Valid: {len(cells) - empty}')"
# Expected: Total >= 500, Empty <= 5%

# Global normalization verification
python -c "import numpy as np; from pathlib import Path; from PIL import Image; cells = sorted(Path('data-processing/data/hcmc/processed/cells').glob('*.png')); vals = [np.array(Image.open(c)) for c in cells[:10]]; print(f'Pixel range: [{min(v.min() for v in vals)}, {max(v.max() for v in vals)}]')"
# Expected: range covers [0, 255] or [0, 1] across cells (NOT each cell independently)

# Training checkpoint verification
python -c "from pathlib import Path; ckpts = sorted(Path('ddpm-win-master/model_ckpts/hcmc_vitality').glob('*.pth')); print(f'Checkpoints: {len(ckpts)}'); [print(f'  {c.name}') for c in ckpts]"
# Expected: >= 8 checkpoints at log intervals

# F_mem computation
python -c "from pathlib import Path; report = Path('ddpm-win-master/logs/hcmc_vitality/fmem_report.txt'); print(report.read_text() if report.exists() else 'NO REPORT')"
# Expected: F_mem values at each checkpoint
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Pilot training GATE decision documented (pass/fail/iterate)
- [ ] Generalization window detected (or documented why not)
- [ ] Section 2.4 analysis completed (or documented as blocked)
- [ ] Methodological framework documented
