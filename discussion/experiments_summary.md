# Experiment Outputs Summary — ISM Report Section 4 Reference

**Generated**: 2026-04-20
**Training run**: hcmc_vitality_v3, 500K steps, 157.8M params (base_ch=128), linear schedule
**Checkpoints**: 20 logarithmic checkpoints (steps 100–500000)
**Evaluation**: DDIM 50-step, 50 samples per checkpoint

---

## 1. Mapping Table: ISM Section → Output → Key Finding

| ISM Section | Output File | Key Finding |
|-------------|-------------|-------------|
| 3.2.3 Pre-encoding analysis | `data-processing/scripts/pre_encoding_analysis.py` (extended, NOT yet run) | Script extended with MLE intrinsic dim, UMAP, MI heatmap, geographic clusters, curation guide. **Not yet executed on real data.** |
| 3.6 Memorization (F_mem) | `analysis/section_2_4/fmem_curve.csv` | **F_mem ≈ 0 at all checkpoints** (max 0.021 at step 360K). No memorization detected. No generalization window. |
| 3.7.1 Distributional fidelity | `analysis/section_2_4/minimum_viable_report.md` | Wasserstein distances decrease from ~0.5 (early) to ~0.1 (late), showing distribution learning. All channel statistics still evolving (CV 0.12–0.90). |
| 3.7.2 Channel statistics evolution | `analysis/section_2_4/channel_mean_evolution.png`, `channel_std_evolution.png` | Channel means oscillate early, stabilize around steps 80K–200K. Channel stds converge more slowly. |
| 3.7.3 Irreducible characteristics | `analysis/section_2_4/minimum_viable_report.md` §4 | **No stable characteristics** — all 6 metrics (B/G/R mean and std) are still evolving at 500K steps (CV range 0.12–0.90). |
| 3.7.4 Inter-channel synergy | `analysis/section_2_4/synergy/full_report.md`, `synergy_metrics.csv` | Synergy evolves through regimes: fragmented → diversity-driven → **coupled** (160K–200K, 300K, 500K). 11 phase transitions detected. |
| 3.7.5 Synergy evolution | `analysis/section_2_4/synergy/regime_evolution.png`, `synergy_correlation_evolution.png` | Final state (500K) is **coupled** with GR r=0.87, BR r=0.75, BG r=0.65. Model learns joint structure without memorizing. |
| Multi-signal triangulation | `analysis/section_2_4/multi_signal_window.png` | Combined view: F_mem flat ≈ 0, Wasserstein decreasing, synergy strengthening. **No generalization window via F_mem, but distributional learning is ongoing.** |

---

## 2. Figures Inventory

### F_mem and Distributional Fidelity (Task 3)

| File | Description | Proposed Caption |
|------|-------------|------------------|
| `analysis/section_2_4/fmem_curve_window.png` | F_mem vs training step with generalization window shading | "Memorization fraction (F_mem) across training. No memorization detected (F_mem < 0.05) at any checkpoint." |
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

### F_mem Signal
**No generalization window detected.** F_mem = 0.0000 at 17/19 checkpoints. Maximum F_mem = 0.0209 at step 360K (well below the 0.05 threshold). This means the model does not memorize individual training samples at any point during 500K steps of training.

### Wasserstein Signal
Wasserstein distances decrease from early checkpoints (~0.5–1.4) to late checkpoints (~0.05–0.28), indicating the model IS learning the data distribution. The steepest decrease occurs between steps 10K–80K. The lowest combined Wasserstein is at step 200K (B: 0.090, G: 0.080, R: 0.088).

### Synergy Signal
Synergy regime evolves through a clear trajectory:
1. **Steps 100–2500**: Fragmented → diversity-driven (early learning, GR pair first to correlate)
2. **Steps 5000–40000**: Fragmented (regression — model loses inter-channel structure)
3. **Steps 50000–100000**: Diversity-driven (GR pair re-emerges as anchor)
4. **Steps 160000–200000**: **Coupled** (all pairs r > 0.5 — model captures full joint structure)
5. **Step 240000**: Density-driven (transient shift)
6. **Step 300000**: Coupled
7. **Steps 360000–400000**: Diversity-driven
8. **Step 500000**: **Coupled** (final convergence, strongest correlations: GR r=0.87)

### Multi-Signal Conclusion
The model learns the data distribution without memorizing individual samples. The "generalization window" concept — a phase where the model generalizes before memorizing — does not manifest for this dataset/model combination. Instead, the model appears to generalize throughout training, with synergy structure emerging gradually and stabilizing at the coupled regime by step 160K.

**Task 9 (window refinement with 100 samples) is BLOCKED** — there is no window to refine since F_mem never exceeds threshold. Future work should investigate whether this is a property of the dataset (731 samples, low intrinsic dimension), the model capacity (157.8M params), or the evaluation methodology (DDIM 50-step sampling with k-NN gap ratio).

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
5. **Training loss curve** — Training logs were lost (train.py never wrote to file). train.py now has logging capability for future retraining.
6. **Missing checkpoint** — Only 19 of 20 expected checkpoints discovered. Step 240000 may have naming mismatch.
