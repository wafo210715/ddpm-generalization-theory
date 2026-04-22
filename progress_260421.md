# Progress Report — 27 April 2026 (continued from 26 April)

## Session: F_mem Investigation Deepening, Pre-encoding Analysis, and F_mem 1000-Sample Run

### Summary
Continued the F_mem ≈ 0 investigation from the previous session. Discovered the gap ratio distribution at 1000 samples reveals genuine memorization onset at step ~240K, overturning the initial "curse of dimensionality" hypothesis. Ran first-ever pre-encoding analysis on real data. Audited the complete data pipeline interim files.

---

## Key Findings

### 1. F_mem at 1000 Samples — Memorization IS Detected

| Step | F_mem | Memorized | Min gap ratio | Median gap ratio |
|------|-------|-----------|---------------|-----------------|
| 200K | 0.000 | 0/1000 | 0.41 | 0.97 |
| 240K | **0.004** | **4/1000** | **0.12** | 0.95 |
| 300K | **0.006** | **6/1000** | **0.20** | 0.95 |
| 400K | **0.011** | **11/1000** | **0.14** | 0.94 |
| 500K | **0.012** | **12/1000** | **0.12** | 0.95 |

**Memorization starts at ~240K steps and monotonically increases.** The prior 50-sample and 100-sample runs showed F_mem ≈ 0 because the sampling probability of catching one of the ~12 memorized images was too low.

### 2. Gap Ratio Distribution Is a Richer Signal Than F_mem

- **88.5%** of all 19,000 gap ratios are in [0.9, 1.0] — model generates mostly novel samples
- **p5 gap ratio drops from 0.80 (step 200K) to 0.55 (step 500K)** — 30% shift in the tail
- **Median stays at ~0.95** throughout — confirming memorization is a minority phenomenon
- The percentile curves are a **leading indicator** that detects the model converging toward training points well before any sample crosses the 1/3 threshold

### 3. Gap Ratio Distribution Tracking Is Genuinely Novel

Literature review (librarian agent, 10+ papers checked): every paper uses gap ratio as a **binary classifier** (below/above 1/3 = memorized/not). **Zero papers track the gap ratio distribution** (histograms, percentiles, shape evolution). This is a potential methodological contribution.

Papers using gap ratio: Yoon et al. (2023) [originator], Gu et al. (2023) [adopter], Bonnaire et al. (2025), Buchanan et al. (2025), Favero et al. (2025), Kaiser et al. (2025), Fang et al. (2025), Chen et al. (2024), Li et al. (2025).

### 4. Pre-encoding Analysis — First Real Run

**Prior claims were WRONG**: "intrinsic dim ~40, k=6 typologies" — never computed on real data.

| Metric | Claimed | Actual |
|--------|---------|--------|
| Intrinsic dim (PCA 95%) | ~40 | **33** |
| MLE intrinsic dim | — | 3.93 |
| Best k (k-means) | 6 | **3** (silhouette=0.574) |

**3 typologies** (heavily imbalanced):
| Cluster | Count | % | Character |
|---------|-------|---|-----------|
| 0 | 873 | 20.9% | Urban residential (moderate density + accessibility) |
| 1 | 3,069 | **73.4%** | Rural/suburban/empty (very low all channels) |
| 2 | 241 | 5.8% | Dense urban core (high density + diversity + accessibility) |

### 5. Channel Distribution Imbalance

| Channel | Mean intensity | Near-empty cells | Issue |
|---------|---------------|-----------------|-------|
| B (density) | 8.1 ± 21.9 | 82% | Most informative, but still 82% empty |
| G (diversity) | 2.2 ± 10.2 | **95%** | Nearly dead — POI entropy is extremely sparse |
| R (accessibility) | 22.3 ± 46.8 | 57% | Moderate spread — best signal after density |

**The G channel (POI diversity) is the bottleneck** — 95% of cells have near-zero diversity. The model cannot learn POI patterns because there's almost no signal.

### 6. Dataset Size Clarification

- **8,436** total fishnet grid cells
- **4,183** non-empty cells → PNGs produced by pipeline
- **4,074** in train/val/test splits (3,259/407/408)
- **109** cells in cells/ but NOT in any split (edge case)
- The "731" number in README was from an earlier pipeline version — **stale and wrong**

---

## Work Completed

### F_mem Investigation
- [x] Code walkthrough: memorization.py line-by-line, verified identical to Bonnaire et al. reference
- [x] Gap ratio formula confirmed: `d₁/d₂` (nearest/second-nearest), threshold 1/3
- [x] Bonnaire paper analysis: 10K samples, 32×32 grayscale, threshold is empirical (k∈{1/4,1/3,1/2} insensitive)
- [x] Sanity check: 5/5 injected training images detected (gap_ratios: 0.000, 0.001, 0.000, 0.063, 0.000)
- [x] F_mem 50 samples: max F_mem = 0.021, inconsistent detection
- [x] F_mem 100 samples: max F_mem = 0.020, still inconsistent
- [x] **F_mem 1000 samples: F_mem monotonically increasing from step 240K, memorization confirmed**

### Pre-encoding Analysis
- [x] First-ever run on real HCMC data (4,183 PNGs)
- [x] PCA intrinsic dimension: k95=33, k99=149, MLE=3.93
- [x] K-means clustering: best k=3 (not 6)
- [x] UMAP projection, MI heatmap, geographic cluster map, representative examples
- [x] Curation guide (54.8 KB)

### Data Pipeline Audit
- [x] Complete interim file inventory: raw/ (5 files, 546 MB), processed/ (4,183 PNGs + 12,549 channel PNGs)
- [x] No GeoJSON or TIFF files exist — pipeline uses GPKG → PNG directly
- [x] No minimal dataset folder exists
- [x] 109 cells unassigned (in cells/ but not in splits)

### Scripts Created / Modified
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/compute_fmem_100samples.py` | F_mem with gap ratio stats | Created |
| `scripts/compute_fmem_1000samples.py` | F_mem 1000 samples, raw CSV output | Created |
| `scripts/plot_fmem_results.py` | Separate visualization script (reads CSVs) | Created, produces 5 plots |
| `scripts/debug_fmem.py` | Sanity check + gap ratio distribution | Created |
| `scripts/generate_grids.py` | Publication grid generator | Created |
| `analysis/section_2_4/fmem_curve_1000samples.csv` | Summary: 19 rows × 15 columns | Generated |
| `analysis/section_2_4/gap_ratios_1000samples.csv` | Raw: 19,000 rows | Generated |
| `analysis/section_2_4/plots/` | 5 PNG plots (comparison, histograms, percentiles) | Generated |

### Pre-encoding Outputs
- `data-processing/data/hcmc/reports/pre_encoding/` — PCA plots, UMAP, MI heatmap, geographic clusters, channel histograms, channel scatter, typology_labels.csv, CURATION_GUIDE.md, representative_examples/cluster_{0,1,2}/

---

## User Observations (from visual inspection)

- Generated samples at step 500K "look good" — similar color palette (red, pink, black, blue, some green) to training data
- Cannot visually distinguish generated from training — positive signal that model learned marginal distribution
- Channel histograms show only density (B) has meaningful spread; diversity (G) and accessibility (R) distributions look weird
- This suggests normalization or rasterization may not work well for diversity/accessibility channels

---

## Open Questions & Discussion

### Dataset Curation
- Current k=3 typologies don't feel distinctive enough for comparative experiments
- Two clustering approaches proposed by user:
  1. **Method 1**: Per-image 3D features (B_mean, G_mean, R_mean) → k-means or DBSCAN based on 3D scatter
  2. **Method 2**: Per-pixel 3D features → DBSCAN → image typology = composition of pixel clusters
- Blocking issue: G channel is 95% near-zero, making any clustering on diversity unreliable

### Training Configuration TODOs (User-Mentioned)
- U-Net depth adjustment
- Resolution increase to 512×512 (batch size will decrease, training less smooth)
- Need proper train/val loss logging (currently logs were lost for v3 run)

### Resume Training
- **Checkpoint continuation works**: skips already-saved checkpoints, continues logarithmic schedule
- **Log continuation has bug**: opens with `"w"` (overwrite) instead of `"a"` (append) — old logs are lost

### Train/Val Loss Divergence vs Memorization Gap
- Train and val loss diverge well before step 240K (where memorization starts)
- User wants to investigate WHY the divergence is so far from memorization onset
- This is an open theoretical question

### 10,000 Samples
- Not urgent — should wait for dataset curation and adequate model training
- Current 1000 samples sufficient to establish the trend

### Latent Walk
- User mentioned this as a future evaluation technique — details TBD

---

## Files Reference

### Data Pipeline
| Path | Content |
|------|---------|
| `data-processing/data/hcmc/raw/` | 5 files (boundary, grid, buildings, POIs, network) — 546 MB |
| `data-processing/data/hcmc/processed/*.pkl` | Raw channel arrays (b/g/r_maps_raw.pkl) — 78 MB |
| `data-processing/data/hcmc/processed/cells/` | 4,183 RGB PNGs (128×128×3) |
| `data-processing/data/hcmc/processed/cells/splits/` | train_cells.txt (3,259), val_cells.txt (407), test_cells.txt (408) |
| `data-processing/data/hcmc/processed/channels/` | 12,549 per-channel grayscale PNGs |
| `data-processing/data/hcmc/reports/pre_encoding/` | Pre-encoding analysis outputs |

### Evaluation Outputs
| Path | Content |
|------|---------|
| `ddpm-win-master/analysis/section_2_4/fmem_curve.csv` | F_mem 50 samples (19 rows) |
| `ddpm-win-master/analysis/section_2_4/fmem_curve_100samples.csv` | F_mem 100 samples (19 rows) |
| `ddpm-win-master/analysis/section_2_4/fmem_curve_1000samples.csv` | F_mem 1000 samples (19 rows) |
| `ddpm-win-master/analysis/section_2_4/gap_ratios_1000samples.csv` | Raw per-sample gap ratios (19,000 rows) |
| `ddpm-win-master/analysis/section_2_4/plots/` | 5 visualization PNGs |

### Scripts
| Path | Purpose |
|------|---------|
| `ddpm-win-master/scripts/compute_fmem_100samples.py` | F_mem 100-sample computation |
| `ddpm-win-master/scripts/compute_fmem_1000samples.py` | F_mem 1000-sample computation (CSV only) |
| `ddpm-win-master/scripts/plot_fmem_results.py` | Visualization (reads CSVs, no GPU) |
| `ddpm-win-master/scripts/debug_fmem.py` | Sanity check + gap ratio distribution |
| `ddpm-win-master/scripts/generate_grids.py` | Publication grid generator |

### Session Metadata
- Pre-encoding analysis: installed scikit-learn 1.8.0 + umap-learn 0.5.12 in data-processing venv
- F_mem 1000 samples: ~55 min on RTX 5090 GPU (100 samples per batch × 10 batches × 19 checkpoints)
- Git commit: BLOCKED — no git config user.name/user.email set

---

## Roadmap: What's Needed to Move On

### Track 1: Fix the Data (CRITICAL PATH — blocks everything else)

1. **Fix channel data quality** — G (diversity) is 95% near-zero.
   - Channel stats evidence: B mean=8.1 (82% empty), G mean=2.2 (95% empty), R mean=22.3 (57% empty)
   - Histogram: `data-processing/data/hcmc/reports/pre_encoding/channel_histograms.png`
   - Rasterization code: `data-processing/scripts/build_hcmc_pipeline_v2.py`
     - POI buffer radius hardcoded at ~line where `buffer` is set for POI entropy computation (search `"buffer"`, `"150"`, or `"distance"` in the G channel section)
     - G channel = Shannon entropy of POI categories within 150m buffer rings, computed on 20×20 subcell grid, then upsampled to 128×128
   - Normalization: global p95 normalization (search `"p95"` or `"percentile"` in the same file) — may suppress diversity signal
   - Raw channel arrays before normalization: `data-processing/data/hcmc/processed/g_maps_raw.pkl` (26.1 MB, 4183 × 20×20 float arrays)
   - Per-channel PNGs: `data-processing/data/hcmc/processed/channels/diversity/*.png` (4183 files, 128×128 grayscale)
   - Options: increase POI buffer (150m→300-500m), log-scale entropy, per-channel loss weighting, merge diversity into combined metric

2. **Re-cluster with better features** — Current k=3 on (B_mean, G_mean, R_mean) not distinctive enough.
   - Current clustering script: `data-processing/scripts/pre_encoding_analysis.py` (840 lines)
     - `classify_typology()` function: computes per-image (B_mean, G_mean, R_mean) then k-means with silhouette selection over k_range=[3,4,5,6,7]
     - `cluster_stability()` function: plots silhouette score vs k
   - Current results: `data-processing/data/hcmc/reports/pre_encoding/typology_labels.csv` (4183 rows, columns: filename, B_mean, G_mean, R_mean, cluster)
   - Current plots: `data-processing/data/hcmc/reports/pre_encoding/cluster_stability.png`, `umap_typology.png`
   - Representative examples: `data-processing/data/hcmc/reports/pre_encoding/representative_examples/cluster_{0,1,2}/`
   - **Method 1**: Per-image 3D features → visualize 3D scatter (`channel_scatter.png` already exists but may not show clusters clearly) → decide k-means vs DBSCAN
   - **Method 2**: Per-pixel clustering — no script exists yet. Raw data at `g_maps_raw.pkl` / `b_maps_raw.pkl` / `r_maps_raw.pkl` (each is 4183 arrays of shape 20×20). Each subcell has (b, g, r) features. DBSCAN on these 4183×20×20×3 = 5M points.
   - Extended features to consider: not just (B_mean, G_mean, R_mean), but also per-channel std, spatial autocorrelation, edge density

3. **Curate comparative datasets** from typology_labels.csv.
   - Source CSV: `data-processing/data/hcmc/reports/pre_encoding/typology_labels.csv` (4183 rows)
   - Intrinsic dim analysis: `data-processing/data/hcmc/reports/pre_encoding/intrinsic_dimension.txt` (PCA k95=33)
   - PCA eigenvalues saved in `pca_scree.png` — exact component variances need to be re-extracted from the script's output or re-run
   - Split files: `data-processing/data/hcmc/processed/cells/splits/train_cells.txt` (3259 filenames)
   - All PNGs: `data-processing/data/hcmc/processed/cells/*.png` (4183 files, 128×128×3)
   - Datasets to curate:
     - Minimal dataset (~33 images spanning intrinsic dim) — copy to a new folder
     - Cluster-balanced dataset (equal from clusters 0/1/2) — subsample cluster 1 (3069) down to ~241
     - Dense-core-only dataset (241 images, cluster 2 only)

4. **Consider 512×512 resolution** — tradeoff: higher resolution vs smaller batch size.
   - Current resolution: 128×128, set in `ddpm-win-master/configs/hcmc_vitality_v3_5090.yaml` → `img_size: 128`
   - Pipeline upsamples from 20×20 to 128×128: `build_hcmc_pipeline_v2.py` → search `"zoom"` or `"128"` or `"upsample"`
   - U-Net: `ddpm-win-master/model/unet.py` — DDPMUnet class, `channel_mults=[1,2,2,2]`, 4 resolution levels
   - At 512×512: 4 levels → 32×32 bottleneck. May need 5th level (`channel_mults=[1,2,2,2,4]`) or attention at deeper levels
   - Memory: batch_size=16 at 128×128 uses ~20GB VRAM. At 512×512: 16× more pixels per image → batch_size=1-2 likely

5. **Consider smaller model** — base_ch=32 (~10M params) so memorization is actually reachable.
   - Current model: `ddpm-win-master/model/unet.py` → `DDPMUnet(in_channels=3, model_channels=128)` → 157.8M params
   - Param formula: `p(d,W) = W(2d+13) + d + 6W²` where d=img_size=128, W=model_channels=128 → ~158M
   - At W=32: p(128,32) = 32(2×128+13) + 128 + 6×32² = 32×269 + 128 + 6144 = 14,864 params (tiny)
   - More realistic: `model_channels=64` → ~39M params. Or keep ch=128 but reduce depth (`channel_mults=[1,2]` → 2 levels) → ~17M
   - v3 config: `ddpm-win-master/configs/hcmc_vitality_v3_5090.yaml` — change `model_channels: 128` → `32` or `64`
   - Data: 3259 training images. Bonnaire et al. used n up to 32768 with p up to 16M. Our p/n ratio at 10M params = ~3000, which is in their "memorization regime"
   - Bonnaire's scaling law: τ_mem ∝ n (memorization time linear in dataset size)

### Track 2: Train and Evaluate (after Track 1)

6. **Train on curated datasets** — compare memorization dynamics across dataset compositions.
   - Training script: `ddpm-win-master/train.py` (411 lines)
   - Config template: `ddpm-win-master/configs/hcmc_vitality_v3_5090.yaml`
   - Config loader: `ddpm-win-master/config_loader.py` → `TrainingConfig.from_yaml()`
   - Dataset loader: `ddpm-win-master/datasets/dataset.py` → `get_dataset('hcmc_vitality_v2', img_size=128, data_dir=..., split='train')`
   - To train on curated subset: either modify split files or create new config pointing to subset directory
   - Resume: `python train.py --config ... --resume model_ckpts/.../xxx_step500000.pth`

7. **Resume training or retrain** — with fixed data + possibly smaller model + longer (1M-2M steps).
   - Resume checkpoint loading: `ddpm-win-master/train.py` line 215-238 — loads step, optimizer, skips already-saved checkpoints
   - Checkpoint schedule: `ddpm-win-master/config_loader.py` → `_generate_log_schedule()` — doubling until 100K, then ×1.5
   - Log continuation bug: line 207 — `open(log_csv_path, "w")` should be `"a"` on resume
   - All v3 checkpoints: `ddpm-win-master/model_ckpts/hcmc_vitality_v3/*.pth` (20 files, 19 with step numbers)
   - VRAM: dual RTX 5090 (32GB each, NVLink) — batch_size=16 at 128×128 uses ~20GB

8. **F_mem 10,000 samples** — for publication quality, once model has actually memorized enough.
   - Computation script: `ddpm-win-master/scripts/compute_fmem_1000samples.py` — modify `N_SAMPLES = 1000` → `10000`, `GEN_BATCH = 100` → may need `50` for GPU memory
   - Will produce: `fmem_curve_10000samples.csv` + `gap_ratios_10000samples.csv` (190,000 rows)
   - Estimated runtime: ~9 hours on GPU (10× the 1000-sample run of 55 min)
   - Not urgent — wait for curated dataset and model that actually memorizes

9. **Feature-space F_mem** — compute gap ratios in U-Net bottleneck embeddings instead of raw 49K-dim pixels.
   - Current F_mem: `ddpm-win-master/evaluation/memorization.py` line 100-113 — flattens images to 49152-dim vectors, computes `torch.cdist`
   - U-Net bottleneck: `ddpm-win-master/model/unet.py` → `MiddleBlock` (line ~237) — outputs feature map at resolution bottleneck (128→64→32→16 = 16×16 spatial, 256 channels = 65536-dim)
   - Need new script: extract bottleneck features for all training images and all generated images, then compute gap ratios in that space
   - Potential issue: 65536-dim still high, but bottleneck features are more semantically meaningful than raw pixels

10. **Investigate train/val loss divergence vs memorization gap** — divergence well before step 240K, why?
    - Bonnaire et al. theoretical framework: memorization timescale τ_mem ∝ n (linear in dataset size), generalization timescale τ_gen = O(1) (architecture-dependent)
    - Paper: arXiv:2505.17638v2 at `references/2505.17638v2.pdf` — Section 3 (theory) and Section 2 (dynamics)
    - Key concept: "implicit dynamical regularization" — diffusion training naturally prevents memorization through the stochastic dynamics
    - Our observation: train/val loss diverge early (maybe step ~50K-100K?) but memorization only starts at 240K — large gap
    - Oracle consultation recommended for this theoretical question

### Track 3: Methodology & Paper (ongoing, not blocking)

11. **Gap ratio distribution as novel contribution** — percentile curves, histogram evolution, leading indicators.
    - Data source: `ddpm-win-master/analysis/section_2_4/gap_ratios_1000samples.csv` (19,000 rows, columns: step, sample_idx, gap_ratio, is_memorized)
    - Summary: `ddpm-win-master/analysis/section_2_4/fmem_curve_1000samples.csv` (19 rows with percentiles)
    - Visualization: `ddpm-win-master/scripts/plot_fmem_results.py` — produces 5 plots from CSVs
    - Current plots: `ddpm-win-master/analysis/section_2_4/plots/*.png` (5 files)
    - Key finding: p5 drops from 0.80→0.55 while F_mem stays near 0 — leading indicator
    - Literature: Yoon et al. (2023) [originator of gap ratio], Gu et al. (2023) [adopter], Bonnaire et al. (2025), Buchanan et al. (2025), Kaiser et al. (2025), Fang et al. (2025), Favero et al. (2025) — ALL use binary F_mem only, zero track distribution

12. **Latent walk evaluation** — details TBD from user

13. **Visualization at every step** — computation scripts produce CSVs, visualization scripts read CSVs.
    - Computation scripts (generate CSVs, no plots): `scripts/compute_fmem_1000samples.py`, `analysis/channel_statistics.py --save_samples`, `analysis/synergy_analysis.py --save_samples`
    - Visualization scripts (read CSVs, produce plots, no GPU): `scripts/plot_fmem_results.py`
    - Outputs: `analysis/section_2_4/plots/*.png`
    - Future: R or Python for publication-ready figures — CSVs serve as the canonical data interface

14. **Fix resume training log bug** — `train.py` line 207.
    - File: `ddpm-win-master/train.py`
    - Line 207: `log_file = open(log_csv_path, "w", newline="")` → should be `"a"` when resuming
    - Resume check at line 215: `if resume_from:` — add conditional to choose `"w"` vs `"a"`
    - Log path pattern: `logs/{run_name}/training_log.csv`

### Priority Order

Track 1 is the critical path. Without better data quality and curated datasets, training more models or generating more samples won't produce meaningful results. The model is currently learning mostly "generate empty/rural cells" because that's 73% of the dataset.

### Cross-Reference: Key File Inventory

| Purpose | Path |
|---------|------|
| Data pipeline (rasterization) | `data-processing/scripts/build_hcmc_pipeline_v2.py` (821 lines) |
| Raw channel arrays | `data-processing/data/hcmc/processed/{b,g,r}_maps_raw.pkl` (26 MB each) |
| Fused RGB PNGs | `data-processing/data/hcmc/processed/cells/*.png` (4183 files) |
| Per-channel PNGs | `data-processing/data/hcmc/processed/channels/{density,diversity,accessibility}/*.png` (4183 each) |
| Splits | `data-processing/data/hcmc/processed/cells/splits/{train,val,test}_cells.txt` |
| Pre-encoding analysis script | `data-processing/scripts/pre_encoding_analysis.py` (840 lines) |
| Pre-encoding results | `data-processing/data/hcmc/reports/pre_encoding/` (all outputs) |
| Typology labels | `data-processing/data/hcmc/reports/pre_encoding/typology_labels.csv` |
| Model definition | `ddpm-win-master/model/unet.py` (DDPMUnet) |
| DDPM core | `ddpm-win-master/model/ddpm.py` |
| Diffusion config | `ddpm-win-master/model/diffusion_config.py` |
| Training script | `ddpm-win-master/train.py` (411 lines) |
| Active config | `ddpm-win-master/configs/hcmc_vitality_v3_5090.yaml` |
| Config loader | `ddpm-win-master/config_loader.py` |
| Dataset loader | `ddpm-win-master/datasets/dataset.py` |
| F_mem evaluation | `ddpm-win-master/evaluation/memorization.py` |
| DDIM sampling | `ddpm-win-master/sampling/ddim.py` |
| DDPM sampling | `ddpm-win-master/sampling/ddpm_sample.py` |
| F_mem 1000-sample script | `ddpm-win-master/scripts/compute_fmem_1000samples.py` |
| F_mem viz script | `ddpm-win-master/scripts/plot_fmem_results.py` |
| Gap ratio raw data | `ddpm-win-master/analysis/section_2_4/gap_ratios_1000samples.csv` (19K rows) |
| F_mem summary CSVs | `ddpm-win-master/analysis/section_2_4/fmem_curve_{50,100,1000}samples.csv` |
| Visualization plots | `ddpm-win-master/analysis/section_2_4/plots/*.png` (5 files) |
| All v3 checkpoints | `ddpm-win-master/model_ckpts/hcmc_vitality_v3/*.pth` (20 files) |
| Bonnaire reference | `Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Evaluation/compute_fmem.py` |
| Bonnaire paper | `references/2505.17638v2.pdf` (also at `discussion/2505.17638v2.pdf`) |
| Progress reports | `progress_260420.md`, `progress_260421.md` (this file) |
| TODO list | `TODO.md` |
| Pre-encoding venv | `data-processing/.venv/` (sklearn 1.8.0, umap-learn 0.5.12, pandas, matplotlib) |
| DDPM venv | `ddpm-win-master/.venv/` (torch 2.11.0+cu128, CUDA) |
