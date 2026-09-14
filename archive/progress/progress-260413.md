# Progress Report — 2026-04-13

**Session type**: Multi-session execution sprint (April 10-13, 4 days)
**Outcome**: Complete Stage 1 infrastructure — data pipeline, DDPM configs, pilot GATE validation, analysis scripts. 7/10 tasks done, 3 GPU-blocked.

---

## Day 1: April 10 — Pipeline Rework + Sprint Kickoff

**Session**: ses_287e99693ffew3qxnK3fN66JSK (Sisyphus-Ultraworker, spanned April 10-11)

### What happened

User reviewed the existing pipeline output (22x30x3 float32 grid with scalar values per cell) and identified the fundamental flaw: the pipeline produced single scalars per cell, but each cell should be a 512x512 pixel image with spatial gradients. This was a pivot from scalar representation to spatial raster encoding.

Three channels were redesigned from scratch:
- **B (Buildings)**: `rasterio.features.rasterize` building polygons into binary raster (white=building, black=empty). Not density scalar.
- **G (POI entropy)**: User pushed back on KDE (density heatmap) — "how do you calculate POI diversity at 2m resolution?" Settled on coarse 20x20 sub-cell (50m units) Shannon entropy within each 1km cell, then bilinear interpolate to 512x512. This preserves actual diversity information at a meaningful scale.
- **R (Accessibility)**: pandana `type="sum" decay="exponential"` at all network nodes, then `scipy.interpolate.griddata` to regular pixel grid. Multi-bin exponential decay at [300, 600, 1000, 2000, 3000]m.

Architecture: Full-city raster canvas → crop 512x512 per cell → stack as RGB PNG.

User raised the boundary alignment problem: current HCMC urban core boundary doesn't tile cleanly to 1km grid lines. Decided to use full HCMC municipality boundary and snap grid to UTM 1000m multiples.

4 parallel exploration agents launched to map codebase, data sources, grid geometry, and rasterization approaches.

### Key decisions
- 512x512 target resolution (later scaled to 128x128 for memory efficiency)
- Coarse-grid entropy over KDE for diversity channel
- Full municipality boundary instead of urban core bbox

---

## Day 2: April 11 — Deep Methodology Discussion + Plan Creation + Wave 1 Execution

**Session**: ses_284a0e09cffe2oYCdQ616onw0b (Prometheus → Atlas, spanned April 11-13)

### Morning: Prometheus Deep Discussion

User provided a long methodological critique: "conflicting image size is not a core problem, that's childish. let's just stick to 512x512px size." This triggered a fundamental re-examination of the entire approach.

Key conceptual tensions explored:
1. **RGB channel fusion is "merely for convenience"** — the analysis ceiling is channel-level patterns. You cannot recover building footprints, road networks, or POI locations from sampled RGB images.
2. **Stage 1 vs Stage 2**: Stage 1 (image encoding) learns distribution patterns but hits an analysis wall. Stage 2 (vector embedding + decode) could recover actual geospatial data, but the decoder architecture is unsolved.
3. **Supply-demand gap**: DDPM learns P(physical configuration). Vitality requires V(physical config) → vitality score. These are separate problems — the generative model doesn't make vitality judgments.
4. **Non-linear thresholds**: "the higher the better" is wrong for density, diversity, accessibility. Jacobs argued for thresholds. Whether generative models capture them is an open question.
5. **Operating size**: 200m? 100m? 500m? The scale should match Jacobs' neighborhood theory (~200m-800m).

3 background research agents launched: vector encoding decode-back approaches, physical-to-vitality mapping models, HCMC boundary data source (found pre-2025 merger boundary, OSM Relation 1973756).

### Afternoon: Transition to Atlas Orchestration

Created the epistemic-probe-framework plan (`/`.sisyphus/plans/epistemic-probe-framework.md`) — 10 implementation tasks, 4 execution waves, 4 final verification reviewers.

**Wave 1 launched (Tasks 1, 2, 3 in parallel)**:
- **Task 1**: HCMC full municipality boundary + fishnet grid
  - `build_boundary_grid.py` (339 lines) — uses Overpass API for historical boundary (pre-April 2025 merger)
  - Vietnam merged HCMC with Binh Duong and Ba Ria-Vung Tau in April 2025 — current OSM boundary is 36,423 km2 (megacity). We use pre-merger boundary.
  - Result: 2406.5 km2, 8436 cells (1km), UTM-aligned EPSG:32648
  - HCMC_BBOX: (106.357, 10.140, 107.028, 11.160) WGS84
- **Task 2**: Data pipeline v2
  - `build_hcmc_pipeline_v2.py` (757 lines)
  - 732 non-empty cells saved as 128x128x3 uint8 RGB PNGs (resolution changed from 512 to 128 for memory efficiency during pilot)
  - Per-channel global 95th percentile normalization: B_p95=0.25, G_p95=0.46, R_p95=48.4
  - Cross-channel correlations: B-G=0.44, B-R=0.35, G-R=0.91 (diversity+accessibility co-occur)
  - Moran's I: B=0.60, G=0.84, R=0.92 (strong spatial autocorrelation)
  - Fixed 7 codebase gaps: per-cell max-stretching (GAP-6), binary buildings (GAP-4), sequential naming (GAP-3), undefined HCMC_BBOX (GAP-5), missing model_channels (GAP-1), plain Tensor returns (GAP-2)
- **Task 3**: Pre-encoding analysis script
  - `pre_encoding_analysis.py` (353 lines) — PCA, k-means typology, distribution analysis, test mode

---

## Day 3: April 12 — Tasks 4-5 Completion + Bug Discovery

**Session**: Continuation of ses_284a0e09cffe2oYCdQ616onw0b

### What happened

- **Task 4**: Pre-encoding analysis report on real data
  - Intrinsic dimension: 40 @95%, 137 @99% (from 49152 features = 1230:1 compression)
  - Typology: k=6 optimal, silhouette=0.3972 — 6 urban typologies identified
  - Cluster 2 (well-connected residential, n=213) is the dominant urban form
  - Cell-level MI(G;R) = 0.485 nats — moderate, meaning synergy analysis is meaningful
  - Pixel-level MI is much lower (0.05-0.13) due to 97% sparsity
  - Memorization risk: HIGH (small N=731) but mitigated by low d=40

- **Task 5**: HCMC Vitality v2 config + dataset loader
  - `hcmc_vitality_v2.yaml`: 50K steps, 128x128, batch 16, linear schedule (b0=0.0001, bT=0.02)
  - `HCMCVitalityV2Dataset`: 584 train / 73 val / 74 test = 731 usable cells
  - model_channels wired through config → TrainingConfig → train.py → DDPMUnet (GAP-1 fix)
  - Dataset returns dict `{'image': Tensor, 'cell_id': str}` (GAP-2 fix)

- **Critical bug discovered in train.py**: Step loop used `range(config.n_steps)` (0-indexed), causing the final checkpoint to be silently dropped. Fixed to `range(1, config.n_steps + 1)` (1-indexed). Verified with 5-step E2E test.

---

## Day 4: April 13 — Wave 3 Completion + GPU Validation + Commits

**Session**: New Atlas session

### What happened

- **Bug fix**: train.py GPU auto-detect — hardcoded `device_ids=[0, 1]` changed to `device_ids=list(range(torch.cuda.device_count()))` with fallback warning for single-GPU machines.

- **Task 6 (Pilot GATE)**:
  - 3 pilot configs created: `hcmc_pilot_64/128/256.yaml` (1000 steps, 5 checkpoints at [100, 250, 500, 750, 1000])
  - `run_pilot_gate.py` (717 lines) — orchestrates training + F_mem evaluation + report generation
  - Pilot training completed on RTX 4060 Laptop (8.6GB VRAM) for ALL 3 resolutions:
    - 64x64: Loss 1.11 → 0.034, ~83s, 0.17 GB VRAM
    - 128x128: Loss 0.42 → 0.026, ~3.5min, 0.18 GB VRAM
    - 256x256: Loss 0.42 → 0.020, ~6.3min, 0.20 GB VRAM
  - F_mem = 0.0000 at step 1000 for ALL resolutions — **CONDITIONAL PASS** (model learning without memorizing)

- **Task 7**: F_mem batch computation pipeline
  - `compute_fmem_batch.py` (279 lines) — discovers checkpoints via regex, generates samples via DDIM, computes k-NN gap ratio

- **Task 9a**: Channel statistics script
  - `channel_statistics.py` (1100 lines) — Wasserstein distance, Moran's I, generalization window detection, irreducible characteristics

- **Task 10a**: Synergy analysis script
  - `synergy_analysis.py` (1554 lines) — joint distributions, regime classification, irreducible synergies

- **E2E GPU validation on RTX 4060**: Training → checkpoint → F_mem → channel statistics → synergy analysis — full pipeline working end-to-end in ~2 minutes.

- **Pre-commit cleanup**: Deleted 10 junk files, added `references/RegionDCL/` to `.gitignore`.

- **7 atomic commits created** on `wafo/dev` branch (not yet pushed):
  1. `add HCMC full municipality boundary and 1km fishnet grid generation`
  2. `add HCMC data pipeline v2 with global normalization and pre-encoding analysis`
  3. `add HCMC vitality DDPM configs, dataset loader, and pilot GATE configs`
  4. `add batch F_mem evaluation and pilot GATE orchestration`
  5. `add Section 2.4 analysis scripts for channel statistics and synergy`
  6. `fix train.py step indexing and GPU auto-detect, remove stale progress files`
  7. `restore progress reports 260409 and 260410 mistakenly deleted during cleanup`

### Early verification reviews launched
- **F1 (Plan Compliance)**: CONDITIONAL APPROVE — Must Have 6/7 (FID computation missing), Must NOT Have 6/6, Tasks 7/10 (3 GPU-blocked)
- **F2 (Code Quality)**: APPROVE — 0 critical, 6 medium (3 missing encoding="utf-8", 1 Moran's I doc label, 2 duplicate discover_checkpoints), 7 low. Cross-cutting finding: memorization.py:235 tuple unpacking risk.
- **F4 (Scope Fidelity)**: Pending (still running as background task)

---

## Current State

### Completed (7/10 tasks)
1. HCMC boundary + grid (8436 cells, 1km, pre-2025 merger)
2. Data pipeline v2 (732 cells, 128x128x3, global p95 normalization)
3. Pre-encoding analysis script (PCA, k-means, distribution)
4. Pre-encoding analysis report (intrinsic dim=40, k=6 typology, MI=0.485)
5. DDPM config + dataset loader (584/73/74 split, model_channels wired)
6. Pilot training GATE (3 resolutions, F_mem=0.0000, CONDITIONAL PASS)
7. F_mem batch computation pipeline (279 lines)

### Blocked on 3090 (3/10 tasks)
8. Full 50K-step training (~14 hours dual RTX 3090)
9. Section 2.4 channel statistics on full checkpoints
10. Section 2.4 synergy analysis on full checkpoints

### Key Technical Decisions
- Resolution: 128x128 (not 512x512) — pilot showed 64/128/256 all work; 128 for balance of detail and memorization risk
- Buffer ring size: 150m for building density and POI entropy computation
- Normalization: per-channel global 95th percentile (not per-cell, not per-canvas)
- U-Net: model_channels=32 (conservative), no attention (is_attn all False), 9.88M params
- Batch size: 16 (fits dual RTX 3090 at 128x128x3)
- HCMC boundary: pre-2025 merger (2406.5 km2), OSM Relation 1973756, date="2024-06-01"

### Known Issues (from early verification)
1. **FID computation missing** — plan lists as "Must Have" but no implementation exists. Need to add or explicitly waive.
2. **memorization.py:235 tuple unpacking** — `generated, _ = sample_ddim(...)` could fail if function returns 3 values. Analysis scripts correctly use `result[0]`.
3. **Duplicate discover_checkpoints** — synergy_analysis.py and run_pilot_gate.py reimplement locally instead of importing from compute_fmem_batch.py.
4. **Missing encoding="utf-8"** — 3 file I/O locations in config_loader.py, dataset.py, run_pilot_gate.py.

### Next Steps
1. `git push origin wafo/dev` (from 4060 laptop)
2. `git pull origin wafo/dev` (on 3090 machine)
3. `python train.py --config configs/hcmc_vitality_v2.yaml` (~14 hours)
4. Run channel statistics and synergy analysis on full training checkpoints
5. Final verification wave (F1-F4)
