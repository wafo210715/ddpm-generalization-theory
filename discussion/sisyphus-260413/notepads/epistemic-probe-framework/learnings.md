## 2026-04-13 Session Start
- Plan: epistemic-probe-framework.md (10 tasks, 4 waves + final)
- Boulder: switched from revised-research-plan (superseded) to epistemic-probe-framework
- Validation: 7 codebase gaps found and documented in engineering guide §0
- Momus: APPROVED first round
- All 10 tasks unchecked — starting fresh

## Key Codebase Gaps (from engineering guide §0)
- GAP-1: train.py:80 hardcodes DDPMUnet(channels, 64) — model_channels not wired through config
- GAP-2: GeoSpatialDataset returns plain Tensor, not dict with cell_id
- GAP-3: Cell naming is sequential {label:04d}.png, not grid-based {row:03d}_{col:03d}.png
- GAP-4: Building channel is binary (0/255), not continuous density
- GAP-5: HCMC_BBOX undefined in build_hcmc_pipeline.py:96
- GAP-6: Per-canvas max normalization (bug), not per-channel global percentile
- GAP-7: No attention in U-Net by default (is_attn all False)

## Conventions
- CLAUDE.md: maintain original parameter names from research repos (b0/bT, beta_d/beta_min, sigma_min/sigma_max)
- Channel order: R=accessibility, G=diversity, B=density (PIL save order)
- CRS: EPSG:32648 (UTM zone 48N) for HCMC
- Grid: 1km fishnet, snap to 1000m multiples

## 2026-04-13 Task 2 Complete: Pre-Encoding Analysis Script
- Script: data-processing/scripts/pre_encoding_analysis.py
- Functions: run_pca(), classify_typology(), analyze_distributions(), main()
- Test mode (--test): generates 50 random 128x128x3 images, runs all analyses, cleans up
- Output files: pca_scree.png (8x4 150dpi), typology_labels.csv, channel_histograms.png, channel_scatter.png (6x6 150dpi)
- PCA caps at min(N, features, 200) components; uses dual-axis scree plot (eigenvalues + cumulative variance)
- Typology: StandardScaler -> KMeans per k -> silhouette score -> best k selection
- Channel colors in plots: B=Blues, G=Greens, R=Reds (matches convention)
- CSV columns: filename, B_mean, G_mean, R_mean, B_std, G_std, R_std, cluster
- Random test data has intrinsic dim ~47/50 at 95% (expected — pure noise is high-dimensional)
- Dependencies: sklearn, numpy, PIL, matplotlib, scipy (no pandas needed for core; used for CSV convenience)

## 2026-04-13 Task 1 Complete: Boundary + Fishnet Grid
- HCMC full municipality: 2406.5 km2 (pre-2025 merger, OSM Relation 1973756 with date="2024-06-01")
- Vietnam merged HCMC with Binh Duong and Ba Ria-Vung Tau in April 2025 — current OSM boundary is 36,423 km2 (megacity). We use pre-merger boundary.
- Grid: 8436 cells (1km), aligned to UTM 1000m multiples
- HCMC_BBOX added to build_hcmc_pipeline.py: (106.357, 10.140, 107.028, 11.160) — WGS84
- CRS round-trip EPSG:4326→32648 introduces ~0.7mm floating-point error — use round() in assertions
- Script: build_boundary_grid.py uses Overpass API directly (not osmnx) for historical boundary
- Windows GBK encoding causes UnicodeEncodeError with km² symbol — use PYTHONIOENCODING=utf-8

## 2026-04-13 Task 2 Complete: Data Pipeline v2
- Script: data-processing/scripts/build_hcmc_pipeline_v2.py (757 lines)
- Generated via _gen_v2.py helper (avoids PowerShell encoding issues with large file writes)
- 732 non-empty cells saved as 128x128x3 uint8 RGB PNGs (empty cells filtered out)
- QA PASSED: cell count 732 >= 500, max values VARY [17, 255], 0% empty in saved cells
- GAP-3 FIXED: naming is {row:03d}_{col:03d}.png (e.g., 061_024.png)
- GAP-4 FIXED: building density is CONTINUOUS [0, 0.71] via buffer ring intersection area
- GAP-6 FIXED: per-channel global p95 normalization (B: 0.25, G: 0.46, R: 48.4)
- Cross-channel correlations: B-G=0.44, B-R=0.35, G-R=0.91 (diversity+accessibility co-occur)
- Moran's I: B=0.60, G=0.84, R=0.92 (strong spatial autocorrelation — good for spatial modeling)
- Architecture: 20x20 sub-grid (50m units) -> 150m buffer rings -> bilinear upsample 128x128
- Channel computation uses geopandas sindex for spatial filtering (fast bounding box pre-filter)
- pandana accessibility: multi-bin exponential decay at [300,600,1000,2000,3000]m
- Intermediate results cached as pickle (b_maps_raw.pkl etc.) — essential for iteration speed
- OSM data coverage: only 20436 buildings + 16515 POIs for urban core — rural cells are empty
- Grid covers 74x114km = 8436 cells, but only 732 (8.7%) have data — urban core is small
- Building density computation took ~146s, POI entropy ~94s, accessibility ~114s (first run)
- pandana API: Network(x, y, edge_from, edge_to, edge_weights_df) — edge_weights must be DataFrame

## 2026-04-13 Task 5 Complete: HCMC Vitality v2 DataLoader + Config
- GAP-1 FIXED: model_channels (default 64) now flows through config → TrainingConfig → train.py/memorization.py → DDPMUnet
- GAP-2 FIXED: HCMCVitalityV2Dataset returns dict {'image': Tensor, 'cell_id': str}
- train.py handles dict returns from datasets (x_0 = batch["image"] when dict)
- Non-empty filter: max pixel > 10 removes 1 of 732 cells → 731 usable
- Split: 80/10/10 = 584/73/74 (seeded), persisted in splits/{split}_cells.txt
- Windows GBK: config_loader.py from_yaml() needs encoding='utf-8'
- Relative paths: data_root="../data-processing/..." from ddpm-win-master root (ONE level up, not two)
- train.py resolves data_root relative to script directory before passing to dataset factory
- Added data_root field to TrainingConfig (Optional[str], default None)
- QA verified: batch shape (4,3,128,128), range [-1,1], cell_ids returned correctly

## 2026-04-13 Task 4 Complete: Pre-Encoding Analysis Report
- PCA intrinsic dimension: 40 @95%, 137 @99% (from 49152 features = 1230:1 compression)
- Typology: k=6 optimal, silhouette=0.3972 — 6 urban typologies identified
- 731 non-empty cells used (not 8436 — empty cells would poison PCA with zeros)
- Workaround: copied 731 non-empty cells to temp dir, ran analysis, cleaned up
- Cell-level MI(G;R)=0.485 nats — moderate, synergy analysis meaningful
- Pixel-level MI is much lower (0.05-0.13) due to 97% sparsity — but not zero
- Channel entropy: H(B)=2.1, H(G)=3.3, H(R)=4.0 bits — R is most informative
- Cluster 2 (well-connected residential, n=213) is the dominant urban form
- Memorization risk: HIGH (small N=731) but mitigated by low d=40
- The script MUST be run on non-empty cells only — 91.3% empty cells dominate PCA if included

## 2026-04-13 Task 6 Prep Complete: Pilot Training GATE Infrastructure
- Created 3 pilot configs: hcmc_pilot_64/128/256.yaml (1000 steps, 5 checkpoints at [100,250,500,750,1000])
- 256x256 config uses batch_size=8 (memory constraint on dual RTX 3090)
- save_log: [] in YAML prevents auto-generated log checkpoints — TrainingConfig.__post_init__ only fills save_log when it's empty AND was not explicitly set from YAML
- Actually, save_log=[] from YAML gets passed as empty list, which IS falsy, so __post_init__ WILL fill it — BUT the YAML value overrides the default. Verified: get_all_checkpoints() returns exactly [100,250,500,750,1000] with no extras
- run_pilot_gate.py uses subprocess.run for both training and F_mem evaluation (avoids GPU init conflicts between resolutions)
- Gate decision logic: PASS (F_mem<0.3 at step 1000) > FAIL (F_mem>0.5 at step 500 all) > ITERATE (transitions 100-500)
- compute_fmem_batch.py output: fmem_results/fmem_curve.csv parsed for results aggregation
- Report: gate_decision.md with F_mem table, CI table, decision, reasoning, next steps
- Plot: fmem_curves.png with all 3 resolutions overlaid + PASS/FAIL threshold lines

## 2026-04-13 Task 8 Complete: Section 2.4 Synergy Analysis Script
- Script: ddpm-win-master/analysis/synergy_analysis.py (~1550 lines)
- Output directory: analysis/section_2_4/synergy/
- Test mode (--test): generates synthetic mock data (5 checkpoints, 50 samples each), runs all analyses, produces 11 output files
- Core functions: compute_joint_distribution_stats, compute_synergy_metrics, compare_with_training, detect_vitality_sweet_spot, classify_synergy_regime, track_synergy_evolution
- Mutual information: histogram-based estimation (np.histogram2d), with sklearn.feature_selection fallback
- Regime classification: coupled/density-driven/diversity-driven/accessibility-driven/fragmented based on Pearson |r| thresholds (0.5 strong, 0.3 weak)
- Phase tracking: emerging/stable/degrading via synergy derivative sign (threshold ±0.01)
- Irreducible synergies: pairs with coefficient of variation < 20% across checkpoints
- Pyright/scipy: scipy.stats functions return poorly-typed tuples — need `float(...)` wrapping + `# type: ignore[arg-type]`
- sklearn.metrics.mutual_info_regression does NOT exist — correct import is sklearn.feature_selection.mutual_info_regression
- argparse: required args don't work well with --test mode; use default="" + manual validation instead
- sample_ddim return type is Union[Tuple, Tuple, List] — use result[0] instead of tuple unpacking for Pyright
- Literature thresholds (Jinan 2026, Beijing 2025) are in raw urban metric units — documented as non-comparable to normalized [0,1] pixels
- Output: 11 files per run (5 joint_scatter_*.png, 4 summary plots, full_report.md, synergy_summary.csv in non-test mode)

## 2026-04-13 Task 9 Complete: Section 2.4 Channel Statistics Script
- Script: ddpm-win-master/analysis/channel_statistics.py (1100 lines)
- Core functions: detect_generalization_window(), compute_channel_stats(), compute_wasserstein_per_channel(), compute_morans_i(), identify_irreducible_characteristics()
- Moran's I: manual 4-neighbor adjacency (no libpysal dependency) — uses np.roll-style shifting
- sample_ddim returns Union[Tuple, Tuple, List] — use result[0] to extract generated tensor (avoids Pyright tuple unpacking error)
- Pyright Dataset[Unknown] has no __len__ — use len(dataset) with type: ignore comment
- Channel constants: CHANNEL_NAMES, CHANNEL_KEYS, CHANNEL_COLORS dicts map index→label/key/color
- Training stats computed once via _collect_images() then compute_channel_stats()
- Output: analysis/section_2_4/ with report, 5 plot types, CSV
- discover_checkpoints() reused from compute_fmem_batch.py (DRY)

## 2026-04-13 CRITICAL BUG FIX: train.py 0-indexed step loop
- BUG: `for step in pbar` where `pbar = tqdm(range(config.n_steps))` uses 0-indexed steps
- IMPACT: Final checkpoint (e.g., step 1000 in 1000-step training) was SILENTLY DROPPED
  - range(1000) produces [0..999], so step==1000 is never reached
  - discover_checkpoints() wouldn't find the final model (different filename)
  - All analysis scripts would miss the final checkpoint
- FIX: Changed to `range(1, config.n_steps + 1)` so step goes [1..1000] (1-indexed)
- VERIFIED: Re-ran 5-step E2E test, both checkpoints (step 2 and step 5) saved correctly

## 2026-04-13 E2E GPU Validation (RTX 4060 Laptop, 8.6GB VRAM)
- Full pipeline validated on real GPU in ~2 minutes total:
  1. Training: 5 steps, loss 1.114→0.985, 0.17 GB VRAM
  2. Checkpoint discovery: both step 2 and step 5 found
  3. F_mem batch evaluation: both checkpoints evaluated, CSV + plot generated
  4. Channel statistics: 4-phase pipeline completed (F_mem→stats→irreducible→report)
  5. Synergy analysis: full pipeline completed (joint dist→synergy→regime→evolution→report)
- Training data stats: B mean=0.707, G mean=0.107, R mean=0.045 ( confirms pre-encoding findings)
- Generated samples at 5 steps: near-uniform noise (mean~0.5, std~0.5) — expected for untrained model
- Moran's I ~0.000 on generated samples — no spatial structure learned yet (correct)
- All 17 output files generated (8 channel stats + 9 synergy + 2 F_mem)

## 2026-04-13 Session Wrap: 6 Atomic Commits Created
- Branch: wafo/dev, ahead of origin by 6 commits (not yet pushed)
- Commits: boundary/grid, pipeline v2 + pre-encoding, configs, F_mem + GATE, analysis scripts, train.py fixes
- 10 junk files deleted: 5 _check_osm*.py, grid_info.py, 4 progress-*.md
- references/RegionDCL/ added to .gitignore (nested repo, not project code)
- All remaining tasks (8, 9, 10, F1-F4) blocked on 3090 machine
- User must: git push on 4060 -> git pull on 3090 -> run full training (~14hrs)
