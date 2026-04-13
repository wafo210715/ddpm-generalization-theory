## Issues Log

### 2026-04-13 GPU Blockers (updated after pilot completion)

**RESOLVED: Task 6 (Pilot GATE) — Completed on 4060 laptop**
- Pilot ran successfully on RTX 4060 (8.6GB) for all 3 resolutions
- GATE decision: CONDITIONAL PASS (F_mem=0.0000 at step 1000, all resolutions)

**ALL NON-GPU WORK COMPLETE — BLOCKED ON 3090 MACHINE (as of 2026-04-13 evening)**

Completed this session:
- F1+F2 code review fixes (11 items, committed 3596876)
- .gitignore bug fix: datasets/dataset.py was never committed (committed 3596876)
- FID evaluation script compute_fid.py (committed e3b2713)
- R channel "super white" analysis (documented below)

Branch: wafo/dev, 9 commits ahead of origin. User must `git push origin wafo/dev` from 4060, then `git pull origin wafo/dev` on 3090.

Remaining tasks ALL require 3090 dual-GPU:
- Task 8: 50K-step training (~14hrs)
- Task 9: Section 2.4 channel statistics
- Task 10: Section 2.4 synergy analysis
- F1-F4: Final verification wave

**BLOCKED: Task 8 (Full 50K-step Training)**
- Requires dual RTX 3090 machine (14+ hours GPU time)
- 6 commits need to be pushed from 4060, then pulled on 3090
- Command on 3090: `cd ddpm-win-master && python train.py --config configs/hcmc_vitality_v2.yaml`
- Config: hcmc_vitality_v2.yaml — 50K steps, 128x128, batch 16, dual GPU

**Task 9b: Channel Statistics**
- Requires Task 8 checkpoints + GPU for sample generation
- Command: `python analysis/channel_statistics.py --checkpoint_dir model_ckpts/hcmc_vitality_v2 --config configs/hcmc_vitality_v2.yaml`

**Task 10b: Synergy Analysis**
- Requires Task 8 checkpoints + GPU for sample generation
- Command: `python analysis/synergy_analysis.py --checkpoint_dir model_ckpts/hcmc_vitality_v2 --config configs/hcmc_vitality_v2.yaml`

**F1-F4: Final Verification Wave**
- Requires all implementation tasks complete

### 2026-04-13 Integration Validation PASSED
- ALL 4 configs load: hcmc_vitality_v2, hcmc_pilot_64/128/256
- Dataset: 584 train + 73 val + 74 test = 731 usable cells
- All 12 analysis functions import and work with synthetic data
- Pipeline wiring verified: config -> DDPMUnet -> DDPM -> train.py -> checkpoint regex
- DDPMUnet: 9,881,699 params (model_channels=32, 3 input channels)
- Moran's I: ~0.000 for random data (correct)

---

## 2026-04-13 Deep Analysis: Accessibility (R) Channel "Super White" Problem

### Executive Summary

**The R channel is NOT actually "super white" — B (density) is.** The user's visual perception likely stems from viewing individual channel PNGs or the RGB composite where the spatial homogeneity of R (Moran's I=0.92, the highest) creates a visually uniform wash that reads as "too bright" relative to its information content. However, the real quantitative concern is **B's dominance** (mean pixel ≈218/255) and the **high G-R collinearity** (r=0.91), not R being "too white" per se.

### 1. Numerical Ground Truth

#### 1a. Training Statistics ([-1,1] space after ToTensor + Normalize(0.5,0.5))

| Channel | Mean [-1,1] | Mean [0,1] | Mean [0,255] | Relative Brightness |
|---------|-------------|-------------|--------------|---------------------|
| B (density) | **+0.707** | **0.854** | **218** | ★ WHITEST by far |
| G (diversity) | +0.107 | 0.554 | 141 | Medium |
| R (accessibility) | +0.045 | 0.523 | 133 | Medium |

**Conclusion**: B is the dominant channel at 85% brightness. R is at 52% — middle gray. The "super white" perception cannot be about R's mean brightness.

#### 1b. Why B is So Bright

B (building density) has raw range [0, 0.71] and p95 = 0.25. After p95 normalization:
- A cell with density 0.25 → 255 (full white)
- A cell with density 0.15 → 153
- The urban core has relatively uniform building coverage, so most non-zero B values are 0.10–0.30, mapping to pixels 102–255
- Mean B pixel ≈ 218 confirms most cells have high normalized density

This makes sense: HCMC's urban core has dense building coverage, and p95=0.25 is a low normalization ceiling.

#### 1c. Why R Appears "Super White" Visually (Despite Not Being the Brightest)

The key insight is **spatial homogeneity** vs. **mean brightness**:

- **Moran's I (spatial autocorrelation)**: R=0.92, G=0.84, B=0.60
- R has the HIGHEST spatial autocorrelation — values vary very little WITHIN each cell
- Each 1km cell in the urban core is well-connected by road network to nearby POIs
- The pandana accessibility metric produces smooth, spatially uniform values
- After bilinear upsampling 20×20→128×128, this smoothness is amplified

**Visual effect**: An R channel image looks like a nearly uniform gray wash with little internal texture. Human vision perceives "lacks detail" as "too white/washed out" — not because the brightness is high, but because there's no contrast within each cell image.

In contrast, B (density) has lower spatial autocorrelation (0.60) — there's more variation within cells (some sub-cells have buildings, some don't), creating texture that reads as "information" even at similar brightness levels.

### 2. Root Cause Analysis: The Pandana Accessibility Metric

#### 2a. How the Metric Works (lines 462-476)

```
bins = [300, 600, 1000, 2000, 3000] meters
bin_centers_km = [0.15, 0.30, 0.50, 1.00, 1.50]
weights = exp(-0.3 × bin_centers_km) = [0.956, 0.914, 0.861, 0.741, 0.638]
```

This computes: accessibility(node) = Σ_bins [marginal_POI_count_in_bin × weight]

**Critical observation**: ALL weights are high (>0.6). The exponential decay with β=0.3 is very gentle:
- POIs within 300m get weight 0.956 (96%)
- POIs at 3000m still get weight 0.638 (64%)
- The effective "half-life" is at distance d where exp(-0.3 × d/2) = 0.5 → d ≈ 4.6 km

For a 1km urban cell in HCMC's core:
- Within 3000m of any sub-cell center, there are MANY POIs (HCMC has 16,515 POIs total)
- The accessibility score accumulates as a SUM over bins
- Even peripheral sub-cells within the same 1km cell see similar numbers of POIs at the 300-3000m scale

#### 2b. Why This Produces Spatial Homogeneity

The accessibility metric is fundamentally a LOW-PASS FILTER over the POI distribution:
1. The 3000m cutoff means each sub-cell "sees" POIs from a 3km radius (π×3²≈28 km²)
2. A 1km cell (1 km²) is tiny relative to this search radius
3. All 20×20 sub-cell centers within a cell have overlapping 3km neighborhoods
4. The marginal differences between adjacent sub-cells are tiny

This is BY DESIGN in accessibility analysis — accessibility IS supposed to be spatially smooth. It measures systemic connectivity, not local variations. But for a DDPM learning per-cell spatial patterns, this smoothness means the within-cell variance of R is minimal.

#### 2c. The p95=48.4 Is Not the Problem

The p95 value of 48.4 is a consequence of the metric, not its cause. Even with a different normalization:
- The issue is the DISTRIBUTION SHAPE, not the scale
- R values are concentrated in a narrow range within each cell
- A different percentile (p50, p99) would rescale but not change the within-cell variance

### 3. Impact Assessment for DDPM Training

#### 3a. Signal-to-Noise per Channel

| Channel | Spatial Autocorrelation | Within-Cell Variance | Information Content |
|---------|------------------------|---------------------|---------------------|
| B (density) | 0.60 (moderate) | **HIGH** — building footprints are local | HIGH — local urban form |
| G (diversity) | 0.84 (high) | Moderate — POI categories vary locally | MODERATE |
| R (accessibility) | 0.92 (very high) | **LOW** — smooth network metric | LOW (per pixel) |

R has the HIGHEST Shannon entropy (H(R)=4.0 bits from pre-encoding analysis), but this entropy is **between cells**, not within cells. For a DDPM generating 128×128 images, the model needs per-pixel variation to learn meaningful spatial patterns.

#### 3b. Channel Imbalance and Learning Dynamics

The DDPM loss is: `L = E[||ε - ε_θ(x_t, t)||²]` summed over all 3 channels.

If R is spatially uniform within each cell:
- The model quickly learns R ≈ constant per cell (low-frequency component)
- The remaining capacity focuses on B and G
- This is actually EFFICIENT — the model allocates capacity proportional to information

**Risk**: The G-R correlation (r=0.91) might be inflated because:
1. Both are high in urban core, low in periphery → between-cell correlation
2. Within-cell, they may have very different spatial patterns
3. The 0.91 correlation is computed on flattened pixel values across all cells, which confounds between-cell and within-cell effects

#### 3c. Is This a Problem for the Research?

**Mild concern, not critical.** Here's why:

1. **The DDPM should still learn**: B has high within-cell variance. G has moderate. The model has plenty of signal in 2 of 3 channels.

2. **Synergy analysis remains meaningful**: Even if R is spatially smooth, the BETWEEN-cell variation in R is informative. The synergy analysis (Section 2.4) studies how channels co-vary ACROSS cells, not within cells. High G-R correlation means the model should learn this relationship.

3. **Generalization window detection is not affected**: This depends on F_mem curves and loss dynamics, not individual channel quality.

4. **Potential issue — "trivial synergy"**: If R is near-constant within cells, the model might learn a trivial mapping: "given B and G, R ≈ f(G) ≈ constant" rather than a genuine tripartite interaction. This would make the synergy decomposition (Section 2.4) show G-R pair synergy as dominant but mechanistically uninteresting.

5. **Memorization dimensionality**: Low within-cell variance in R effectively reduces the per-cell information dimension. This HELPS generalization (less to memorize) but REDUCES the richness of what the model can generate for R.

### 4. Recommendations (Ranked by Impact/Effort)

#### Tier 1: Do Now (Before Full Training) — HIGH IMPACT, LOW EFFORT

**(a) Document and proceed (RECOMMENDED)**
- Effort: 0 | Impact: Baseline
- The current data is scientifically defensible. R's spatial smoothness is a PROPERTY of accessibility, not a bug.
- The DDPM will learn the joint distribution correctly, including the smooth R channel.
- Section 2.4 analysis can note the low within-cell variance of R as a limitation.

**(b) Log-transform R before normalization**
- Effort: 1 line change | Impact: MODERATE
- Change: `r_values = np.log1p(r_values)` before `_normalize(r_maps, "R")`
- Effect: Compresses the high end, spreads the low end, increases within-cell contrast
- Risk: Changes the metric's semantics (log of accessibility is not accessibility)
- Note: This respects the constraint "within each unit, the three channels should be normalized to avoid one channel being too dominant" — log transform reduces dominance before normalization

#### Tier 2: Consider for Pipeline v3 — MODERATE IMPACT, MODERATE EFFORT

**(c) Per-channel standardization (z-score) instead of percentile clipping**
- Effort: ~10 lines | Impact: HIGH for channel balance
- Method: For each channel, compute mean μ and std σ across all cells, then normalize to (x - μ) / σ, clip to [-3, 3], rescale to [0, 255]
- Effect: All channels have equal mean and variance → no channel dominance
- Risk: Different absolute scales; DDPM's [-1,1] normalization will produce different ranges per channel; may need to adjust the Normalize() step
- Concern: The user's constraint says "within each unit, the three channels should be normalized" — z-score is a form of this

**(d) Rank-based normalization (CDF matching / quantile transform)**
- Effort: ~15 lines | Impact: HIGH for uniformity
- Method: Map each channel's CDF to a uniform [0, 1] distribution using empirical quantiles
- Effect: All channels have identical distributions (uniform) → perfect balance
- Risk: Destroys the absolute scale of each channel; changes the correlation structure (quantile transform preserves monotonic relationships but changes linear correlations)
- Note: This is the most aggressive transformation and would need careful justification

#### Tier 3: Research Design Changes — HIGH IMPACT, HIGH EFFORT

**(e) Redesign the accessibility metric**
- Effort: Significant (recompute pandana) | Impact: FUNDAMENTAL
- Options:
  - **Inverse distance weighting** (β=1.0 or 2.0 instead of 0.3): Sharper distance decay → more local variation
  - **Single-bin approach**: Count only POIs within 500m, not multi-bin cumulative → less spatial smoothing
  - **Network distance percentiles**: Use rank of accessibility within study area, not raw gravity score
  - **Local accessibility**: Count POIs within walking distance (400m) only → more variation between sub-cells
- Concern: Changes the theoretical interpretation. Multi-bin exponential decay with β=0.3 is standard in transport geography (Geurs & van Wee, 2004).

**(f) Replace R with a different third channel**
- Effort: Very high (conceptual redesign) | Impact: Depends on alternative
- Options: Street interface density (Jacobs), land use mix index, block size variation, intersection density
- Concern: Would require rebuilding the entire theoretical framework for channel selection

### 5. Quantitative Projections for Log-Transform Option (b)

If we apply `log1p` to R before p95 normalization:
- Raw R range: [0, ~max]. Let's say typical urban core values are 20-60.
- log1p(20) = 3.04, log1p(48.4) = 3.91, log1p(60) = 4.11
- After normalization by p95 of log1p values:
  - Previous p95=48.4 → log1p(48.4) = 3.91 → new p95 ≈ 3.91
  - Value 20 → log1p(20)/3.91 = 0.78 → pixel 199
  - Value 5 → log1p(5)/3.91 = 0.42 → pixel 107
  - Value 1 → log1p(1)/3.91 = 0.27 → pixel 68
- **Effect**: Low-accessibility areas (R=1-5) get boosted from 5-26 pixels to 68-107 pixels. High areas (R=20-60) compress from 105-255 to 178-212. This increases within-cell contrast significantly.

### 6. Critical Resolution: What the User Is Actually Seeing

The user's perception of "super white R" is most likely one of:

1. **Viewing the RGB composite**: R=R, G=G, B=B. Since B (density) is at 218/255, the composite looks bright overall. The R component adds a reddish tint but isn't the cause of overall brightness.

2. **Viewing individual channel PNGs**: The accessibility PNGs (`channels/accessibility/*.png`) show smooth, uniform gray values. Human vision interprets "no texture/detail" as "washed out" or "too bright" even at moderate gray levels (133/255 = 52%).

3. **Comparing across cells**: Within the urban core, R values are HIGH and similar across cells, so all cells look similar — the eye perceives this homogeneity as "everything is white."

**The real problem is not brightness but information density per pixel.** R conveys information primarily BETWEEN cells, not WITHIN cells. This is a fundamental property of network-based accessibility metrics at 1km scale.

### 7. Bottom Line

| Question | Answer |
|----------|--------|
| Is R actually "super white"? | **No.** B is the brightest channel (218/255). R averages 133/255. |
| Why does R LOOK super white? | **Spatial homogeneity** (Moran's I=0.92). Within-cell variance is minimal → no visual texture → perceived as washed out. |
| Is this a DDPM training problem? | **Mild concern.** R provides little per-pixel learning signal. The model will quickly learn R as a smooth low-frequency component. This is efficient but reduces R's contribution to synergy analysis. |
| Should we fix it before training? | **No urgency.** The current data is scientifically defensible. Log-transform (option b) is a safe, easy improvement for v2.1 or v3. |
| Does it undermine Section 2.4? | **Partially.** G-R correlation (0.91) is likely dominated by between-cell co-occurrence. Within-cell G-R synergy is weaker than the aggregate number suggests. Document this limitation. |
| What about the user's constraint? | "Within each unit, channels should be normalized to avoid dominance" — **B is the dominant channel**, not R. If the user wants balance, z-score (option c) or log-transform (option b) should target B first, then R. |

### 8. Recommended Action

1. **Proceed with current data for 50K-step training** — the pipeline is valid
2. **Add a note in Section 2.4** acknowledging R's low within-cell variance and its implications for synergy interpretation
3. **For pipeline v3**: Implement log-transform for R (1-line change) and consider z-score normalization for all channels to equalize variance
4. **During analysis**: When interpreting G-R synergy, distinguish between between-cell correlation (high, r=0.91) and within-cell pixel-level synergy (likely much lower due to R's spatial smoothness)
