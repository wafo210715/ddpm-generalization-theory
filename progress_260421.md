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

1. **Fix channel data quality** — G (diversity) is 95% near-zero. Options:
   - Increase POI buffer radius (150m → 300-500m)
   - Log-scale entropy instead of linear
   - Per-channel loss weighting to compensate imbalance
   - Merge diversity into a combined "urban activity" metric
2. **Re-cluster with better features** — Current k=3 on (B_mean, G_mean, R_mean) not distinctive enough:
   - **Method 1**: Per-image 3D features → visualize 3D scatter → choose k-means vs DBSCAN
   - **Method 2**: Per-pixel 3D features → DBSCAN → image typology = composition of pixel clusters
   - Consider extended features: B_std, G_std, R_std, spatial autocorrelation (not just mean)
3. **Curate comparative datasets** from `typology_labels.csv` (4,183 rows):
   - Minimal dataset (~33 images spanning intrinsic dimension)
   - Cluster-balanced dataset (equal samples from clusters 0, 1, 2)
   - Dense-core-only dataset (241 images, cluster 2)
4. **Consider 512×512 resolution** — tradeoff: higher resolution vs smaller batch size (less smooth convergence)
5. **Consider smaller model** — base_ch=32 (~10M params) so memorization is actually reachable within training budget

### Track 2: Train and Evaluate (after Track 1)

6. **Train on curated datasets** — compare memorization dynamics across dataset compositions
7. **Resume training or retrain** — with fixed data + possibly smaller model + longer (1M-2M steps)
8. **F_mem 10,000 samples** — for publication quality, once model has actually memorized enough
9. **Feature-space F_mem** — compute gap ratios in U-Net bottleneck embeddings instead of raw 49K-dim pixels
10. **Investigate train/val loss divergence vs memorization gap** — divergence occurs well before step 240K where memorization starts. Why? Theoretical question related to implicit regularization in diffusion training.

### Track 3: Methodology & Paper (ongoing, not blocking)

11. **Gap ratio distribution as novel contribution** — percentile curves, histogram evolution, leading indicators for memorization onset. Zero literature precedent (10+ papers checked — all use binary F_mem). Data source already exists: `gap_ratios_1000samples.csv` (19,000 rows).
12. **Latent walk evaluation** — details TBD
13. **Visualization at every step** — computation scripts produce CSVs, visualization scripts read CSVs and produce plots (already separated for future R/Python publication figures)
14. **Fix resume training log bug** — `train.py` opens log with `"w"` (overwrite) instead of `"a"` (append) when resuming

### Priority Order

Track 1 is the critical path. Without better data quality and curated datasets, training more models or generating more samples won't produce meaningful results. The model is currently learning mostly "generate empty/rural cells" because that's 73% of the dataset.
