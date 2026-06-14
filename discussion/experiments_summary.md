# Experiment Outputs Summary — ISM Report Section 4 Reference

**Generated**: 2026-04-20
**Updated**: 2026-06-14 (Section 4 re-evaluated with 1000-sample F_mem data)
**Training run**: hcmc_vitality_v3, 500K steps, 157.8M params (base_ch=128), linear schedule
**Checkpoints**: 20 logarithmic checkpoints (steps 100–500000)
**Evaluation**: DDIM 50-step; F_mem originally 50 samples/checkpoint, re-evaluated at 1000 samples on 5/23

> **6/14 changelog**: Section 4 (Generalization Window) reversed from "no window" → "window = 160K–240K (80K wide)" based on 1000-sample re-evaluation. Other sections unchanged.

---

## 1. Mapping Table: ISM Section → Output → Key Finding

| ISM Section | Output File | Key Finding |
|-------------|-------------|-------------|
| 3.2.3 Pre-encoding analysis | `data-processing/scripts/pre_encoding_analysis.py` (extended, NOT yet run) | Script extended with MLE intrinsic dim, UMAP, MI heatmap, geographic clusters, curation guide. **Not yet executed on real data.** |
| 3.6 Memorization (F_mem) | `analysis/section_2_4/fmem_curve_1000samples.csv` (1000-sample, supersedes 50-sample `fmem_curve.csv`) | **F_mem onset at 240K** (0.004, CI=[0.001,0.008]), rising to 0.012 at 500K. **Generalization window = 160K→240K** (80K wide). 50-sample data missed this due to statistical noise. |
| 3.7.1 Distributional fidelity | `analysis/section_2_4/minimum_viable_report.md` | Wasserstein distances decrease from ~0.5 (early) to ~0.1 (late), showing distribution learning. All channel statistics still evolving (CV 0.12–0.90). |
| 3.7.2 Channel statistics evolution | `analysis/section_2_4/channel_mean_evolution.png`, `channel_std_evolution.png` | Channel means oscillate early, stabilize around steps 80K–200K. Channel stds converge more slowly. |
| 3.7.3 Irreducible characteristics | `analysis/section_2_4/minimum_viable_report.md` §4 | **No stable characteristics** — all 6 metrics (B/G/R mean and std) are still evolving at 500K steps (CV range 0.12–0.90). |
| 3.7.4 Inter-channel synergy | `analysis/section_2_4/synergy/full_report.md`, `synergy_metrics.csv` | Synergy evolves through regimes: fragmented → diversity-driven → **coupled** (160K–200K, 300K, 500K). 11 phase transitions detected. |
| 3.7.5 Synergy evolution | `analysis/section_2_4/synergy/regime_evolution.png`, `synergy_correlation_evolution.png` | Final state (500K) is **coupled** with GR r=0.87, BR r=0.75, BG r=0.65. Model learns joint structure without memorizing. |
| Multi-signal triangulation | `analysis/section_2_4/multi_signal_window.png` | Combined view: F_mem onset at 240K (1000-sample), Wasserstein decreasing, synergy coupled at 160K. **Generalization window = 160K→240K** (80K wide). |

---

## 2. Figures Inventory

### F_mem and Distributional Fidelity (Task 3)

| File | Description | Proposed Caption |
|------|-------------|------------------|
| `analysis/section_2_4/fmem_curve_window.png` | F_mem vs training step with generalization window shading | "Memorization fraction (F_mem) across training (1000 samples/checkpoint). Onset at step 240K (F_mem=0.004), rising to 0.012 at 500K. Generalization window: 160K–240K." |
| `analysis/section_2_4/channel_mean_evolution.png` | B/G/R spatial mean across checkpoints | "Evolution of per-channel spatial means (density, diversity, accessibility) across training checkpoints." |
| `analysis/section_2_4/channel_std_evolution.png` | B/G/R spatial std across checkpoints | "Evolution of per-channel spatial variability across training checkpoints." |
| `analysis/section_2_4/wasserstein_evolution.png` | Wasserstein-1 distance per channel vs step | "Wasserstein distance (generated vs. training) per channel, showing distributional convergence." |
| `analysis/section_2_4/channel_histograms_step{N}.png` (×19) | Per-channel intensity histograms at each checkpoint | "Channel intensity distributions at training step {N}." |

### Inter-Channel Synergy (Task 4)

| File | Description | Proposed Caption |
|------|-------------|------------------|
| `analysis/section_2_4/synergy/synergy_correlation_evolution.png` | Pearson r for BG/BR/GR over training | "Inter-channel Pearson correlation evolution: density-diversity (BG), density-accessibility (BR), diversity-accessibility (GR)." |
| `analysis/section_2_4/synergy/synergy_mi_evolution.png` | Mutual information for each pair over training | "Mutual information (histogram-based) between channel pairs across training." |
| `analysis/section_2_4/synergy/regime_evolution.png` | Color-coded regime timeline | "Synergy regime classification across training: fragmented → diversity-driven → coupled." |
| `analysis/section_2_4/synergy/vitality_sweet_spot.png` | Cells in top 25th percentile for all channels | "Vitality sweet spot proportion: generated samples achieving high density, diversity, and accessibility simultaneously." |
| `analysis/section_2_4/synergy/training_vs_sample_comparison.png` | Training vs generated joint distributions | "Joint distribution comparison between training data and generated samples at best checkpoint." |
| `analysis/section_2_4/synergy/joint_scatter_step{N}.png` (×19) | Pairwise scatter plots at each checkpoint | "Inter-channel scatter at step {N}: BG (density-diversity), BR (density-accessibility), GR (diversity-accessibility)." |

### Multi-Signal Visualization (Task 6)

| File | Description | Proposed Caption |
|------|-------------|------------------|
| `analysis/section_2_4/multi_signal_window.png` | 2×2 combined: F_mem + Wasserstein + synergy regime + normalized | "Multi-signal analysis: F_mem (top-left), Wasserstein distances (top-right), synergy regime evolution (bottom-left), and normalized combined view (bottom-right)." |

---

## 3. Tables Inventory

| File | Description | Key Columns |
|------|-------------|-------------|
| `analysis/section_2_4/fmem_curve.csv` | F_mem per checkpoint (19 rows) | step, f_mem, std, lower_ci, upper_ci, n_memorized, n_total |
| `analysis/section_2_4/synergy/synergy_metrics.csv` | Synergy metrics per pair per checkpoint (57 rows) | step, pair, pearson_r, pearson_p, spearman_r, spearman_p, mi_histogram, regime |
| `analysis/section_2_4/synergy/synergy_summary.csv` | Per-checkpoint regime summary | step, regime, avg_pearson, avg_spearman, avg_mi, sweet_spot_n, sweet_spot_prop |
| `analysis/section_2_4/minimum_viable_report.md` | Full evaluation report | Sections 1–5: F_mem, channel stats, Wasserstein, Moran's I, irreducible characteristics |
| `analysis/section_2_4/synergy/full_report.md` | Full synergy report | Sections 1–7: methodology, per-checkpoint analysis, evolution, comparison, thresholds |

---

## 4. Generalization Window Assessment

> **6/14 update**: This section originally (4/20) was based on 50-sample F_mem data and concluded "no window." The 5/23 1000-sample re-evaluation (`fmem_curve_1000samples.csv`) reverses that conclusion. Numbers below are from the 1000-sample evaluation unless noted.

### F_mem Signal (1000-sample)
**Generalization window = 160K → 240K (~80K wide).**

| Step | F_mem | n_memorized | 95% CI |
|------|-------|-------------|--------|
| 100–200K | 0.000 | 0/1000 | — |
| 240K | 0.00394 | 4/1000 | [0.001, 0.008] |
| 300K | 0.00601 | 6/1000 | [0.002, 0.011] |
| 360K | 0.00516 | 5/1000 | [0.001, 0.010] |
| 400K | 0.01104 | 11/1000 | [0.005, 0.018] |
| 500K | 0.01203 | 12/1000 | [0.006, 0.019] |

**Onset**: F_mem first > 0 at step 240K (CI excludes 0 → statistically robust onset).
**Plateau**: F_mem rises to ~0.012 by 500K; 360K shows small dip (0.006 → 0.005).
**Open question**: Absolute memorization level (< 1.2%) is low — untrained-baseline comparison needed (Phase 0 Task 0.8 in progress_260611.md).

### Wasserstein Signal
Wasserstein distances decrease from early checkpoints (~0.5–1.4) to late checkpoints (~0.05–0.28), indicating the model IS learning the data distribution. The steepest decrease occurs between steps 10K–80K. The lowest combined Wasserstein is at step 200K (B: 0.090, G: 0.080, R: 0.088).

### Synergy Signal
Synergy regime evolves through a clear trajectory:
1. **Steps 100–2500**: Fragmented → diversity-driven (early learning, GR pair first to correlate)
2. **Steps 5000–40000**: Fragmented (regression — model loses inter-channel structure)
3. **Steps 50000–100000**: Diversity-driven (GR pair re-emerges as anchor)
4. **Steps 160000–200000**: **Coupled** (all pairs r > 0.5 — model captures full joint structure)
5. **Step 240000**: **Density-driven** (transient shift — coincides with F_mem onset)
6. **Step 300000**: Coupled
7. **Steps 360000–400000**: Diversity-driven
8. **Step 500000**: **Coupled** (final convergence, strongest correlations: GR r=0.87)

**Threshold disclosure**: Coupled regime requires all pairwise |r| > 0.5 (`synergy_analysis.py:351-356`). Changing this threshold shifts window boundaries.

### Multi-Signal Conclusion (updated)
**A generalization window exists, bounded by two independent signals:**
- **Window start** (structure learning onset): synergy first reaches coupled regime at **160K**
- **Window end** (memorization onset): F_mem first > 0 at **240K**
- **Window width**: 80K steps (160K → 240K)

**Key tension**: At step 240K (F_mem onset), synergy has already regressed from coupled to density-driven. Memorization onset coincides with structural coupling degradation — not with stable coupled state. This departs from Bonnaire et al.'s framework where memorization is expected within coupled state. Two interpretations:
- (a) Density-driven at 240K is a precursor signal of impending memorization
- (b) Synergy and F_mem are more weakly coupled than theory predicts for this dataset

Future work should investigate whether the 80K window width matches Bonnaire et al.'s theoretical predictions for n=4,182 samples and 157.8M parameters.

**Task 9 (window refinement)**: Unblocked — the window is now well-defined. Next refinement: untrained baseline (Task 0.8) + per-channel Moran's I on training data (Task 0.7) to diagnose whether B/G channel degradation (97.5% zeros on G, 92.7% zeros on B) limits what the model can learn.

---

## 5. Irreducible Characteristics Summary

From the minimum viable report §4, **no channel statistics have converged** by step 500K:

| Metric | Mean Value | CV | Status |
|--------|-----------|-----|--------|
| B_mean (density level) | 0.3600 | 0.6444 | Evolving |
| B_std (density variability) | 0.2935 | 0.2847 | Evolving |
| G_mean (diversity level) | 0.2583 | 0.9040 | Evolving (highest CV) |
| G_std (diversity variability) | 0.2438 | 0.3823 | Evolving |
| R_mean (accessibility level) | 0.3703 | 0.4266 | Evolving |
| R_std (accessibility variability) | 0.3846 | 0.1173 | Most stable (lowest CV) |

**Interpretation**: The diversity channel (G) is most volatile (CV=0.90 for G_mean), suggesting the model has not settled on a consistent diversity representation. The accessibility channel (R) is most stable. Training beyond 500K steps may be needed for convergence.

---

## 6. Synergy Regime Summary

### Regime Frequency

| Regime | Checkpoints | Steps |
|--------|-------------|-------|
| Fragmented | 5 | 100, 5000, 10000, 20000, 40000 |
| Diversity-driven | 7 | 250, 500, 2500, 50000, 80000, 360000, 400000 |
| Accessibility-driven | 1 | 1000 |
| Density-driven | 1 | 240000 |
| Coupled | 5 | 160000, 200000, 300000, 500000 |

### Phase Transitions (11 total)
The model undergoes 11 regime transitions across 19 checkpoints, indicating unstable inter-channel learning. The "coupled" regime (the target state where all channels are well-correlated) appears at:
- Steps 160K, 200K (first emergence)
- Step 300K (re-emergence after density-driven dip)
- Step 500K (final convergence)

---

## 7. Pre-Encoding Status

The pre-encoding analysis script (`data-processing/scripts/pre_encoding_analysis.py`) has been extended from 353 to 840 lines with the following new capabilities:
- MLE intrinsic dimension estimation
- UMAP visualization colored by k-means cluster
- Mutual information heatmap (6×6 feature matrix)
- Geographic cluster map overlaid on HCMC boundary
- Representative examples per typology cluster
- Cluster stability analysis (silhouette scores for k=3..8)
- Curation guide for controlled experiment design

**Status**: Script is code-complete but **NOT yet executed on real HCMC data**. Running it requires ~30 min CPU time on 4183 PNGs. Expected outputs would populate `data-processing/data/hcmc/reports/pre_encoding/`.

**ISNY claims to verify** (currently unsubstantiated):
- Intrinsic dimension ~40 (PCA 95% variance)
- k=6 optimal clusters
- MI between channels ≈ 0.485

---

## 8. Tools and Scripts Created/Modified

### New Scripts
| File | Purpose | Lines |
|------|---------|-------|
| `sampling/ddpm_sample.py` | DDPM 1000-step sampling (Ho et al. Algorithm 2) | 139 |
| `analysis/multi_signal_window.py` | Multi-signal generalization window visualization | ~200 |
| `scripts/generate_grids.py` | Publication-quality sample grid generator (3 modes) | ~200 |

### Modified Scripts
| File | Changes |
|------|---------|
| `analysis/channel_statistics.py` | Added `--save_samples` flag + `_save_samples()` helper |
| `analysis/synergy_analysis.py` | Added `--save_samples` flag + `_save_samples()` + `synergy_metrics.csv` export |
| `evaluation/memorization.py` | Fixed v3 checkpoint format handling |
| `sample_v3.py` | Added `--sampling_method {ddim,ddpm}` + `diff_cfg` fix |
| `train.py` | Added CSV logging (`training_log.csv`) + `training_summary.txt` |
| `sampling/__init__.py` | Export `sample_ddpm` |
| `data-processing/scripts/pre_encoding_analysis.py` | Extended 353→840 lines (7 new analysis functions) |

---

## 9. Outstanding Items

1. **Pre-encoding on real data** — Script extended but never run on actual HCMC PNGs. Requires ~30 min CPU.
2. **DDPM 1000-step sampling** — Implementation complete (`ddpm_sample.py`) but not run on v3 checkpoints. DDPM sampling would produce higher-quality samples for final evaluation.
3. **Publication grids** — `generate_grids.py` created but not run on checkpoints. Requires GPU.
4. **100-sample refinement (Task 9)** — Blocked by no-window finding. Could be adapted to re-run on "coupled regime" checkpoints (160K, 200K, 300K, 500K) for higher precision synergy analysis.
5. **Training loss curve** — `training_log.csv` exists and runs through step 500K (2653 rows). However `training_summary.txt` shows "Total steps: 260000" — a mid-training snapshot that was never updated. The full loss curve is recoverable from `training_log.csv`.
6. **Missing checkpoint** — Only 19 of 20 expected checkpoints discovered. Step 240000 may have naming mismatch.
