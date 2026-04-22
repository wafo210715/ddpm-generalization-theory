# Section 2.4 — Synergy Analysis: Joint B/G/R Distribution Evolution

**Generated**: 2026-04-20 22:25:10
**Checkpoints analyzed**: 19
**Step range**: 100 – 500000

## 1. Introduction

This analysis examines how a diffusion model learns **inter-channel synergies** —
the statistical relationships between density (B), diversity (G), and
accessibility (R) channels across training. Understanding when and how these
synergies emerge tells us whether the model captures the joint structure of
urban vitality or merely learns marginal distributions independently.

**Key question**: Does the model learn that density, diversity, and
accessibility co-vary in structured ways, and at what training stage
does this joint structure emerge?

**Important caveat**: All correlations reported here are *associational*.
Correlation does not imply causation. The synergies reflect co-occurrence
patterns in urban spatial structure, not causal mechanisms.

## 2. Methodology

### 2.1 Per-Channel Spatial Means
For each generated image, we compute the spatial mean of each channel
(after converting from [-1, 1] to [0, 1]). This gives three scalar values
per image: B_mean, G_mean, R_mean.

### 2.2 Synergy Metrics
For each channel pair (B-G, B-R, G-R), we compute:
- **Pearson correlation** (linear relationship)
- **Spearman rank correlation** (monotonic relationship)
- **Mutual information** (non-linear dependency, histogram-based)

### 2.3 Distribution Comparison
- **Wasserstein distance** (Earth Mover's Distance) on marginal distributions
- **Correlation structure comparison** between samples and training data

### 2.4 Regime Classification
Based on pairwise Pearson |r| thresholds:
| Regime | Criteria |
|--------|----------|
| coupled | All pairs r > 0.5 |
| density-driven | B correlates with G and R, but G-R weak |
| diversity-driven | G correlates with B and R, but B-R weak |
| accessibility-driven | R correlates with B and G, but B-G weak |
| fragmented | All pairs r < 0.3 |

## 3. Joint Distribution Analysis Per Checkpoint

### Step 100

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.0982 | 4.97e-01 | 0.2570 | 1.3534 |
| BR | 0.0977 | 5.00e-01 | 0.0638 | 1.6678 |
| GR | 0.4127 | 2.90e-03 | 0.5281 | 1.5789 |

**Vitality sweet spot**: 3/50 (0.060)
- Training proportion: 0.141
- Ratio (sample/training): 0.425

### Step 250

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.2285 | 1.11e-01 | 0.2974 | 1.9704 |
| BR | 0.2118 | 1.40e-01 | 0.2065 | 2.0127 |
| GR | 0.6538 | 2.63e-07 | 0.7001 | 2.2969 |

**Vitality sweet spot**: 3/50 (0.060)
- Training proportion: 0.141
- Ratio (sample/training): 0.425

### Step 500

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.2673 | 6.06e-02 | 0.3685 | 0.9373 |
| BR | 0.1848 | 1.99e-01 | 0.1280 | 2.0096 |
| GR | 0.4004 | 3.96e-03 | 0.5242 | 0.9537 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.141
- Ratio (sample/training): 0.708

### Step 1000

**Regime**: `accessibility-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.1770 | 2.19e-01 | 0.1476 | 1.1307 |
| BR | 0.2988 | 3.51e-02 | 0.4117 | 1.1567 |
| GR | 0.4206 | 2.36e-03 | 0.4145 | 2.1587 |

**Vitality sweet spot**: 2/50 (0.040)
- Training proportion: 0.141
- Ratio (sample/training): 0.283

### Step 2500

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.2497 | 8.03e-02 | 0.3579 | 0.5832 |
| BR | 0.1218 | 4.00e-01 | 0.1955 | 0.3199 |
| GR | 0.7080 | 8.91e-09 | 0.5926 | 1.4726 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.141
- Ratio (sample/training): 0.708

### Step 5000

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | -0.1420 | 3.25e-01 | -0.0985 | 0.8822 |
| BR | -0.1436 | 3.20e-01 | -0.0289 | 1.5722 |
| GR | 0.1291 | 3.72e-01 | 0.2826 | 1.2391 |

**Vitality sweet spot**: 0/50 (0.000)
- Training proportion: 0.141
- Ratio (sample/training): 0.000

### Step 10000

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.1005 | 4.87e-01 | 0.2841 | 0.8045 |
| BR | 0.0373 | 7.97e-01 | -0.0624 | 0.8927 |
| GR | 0.0266 | 8.54e-01 | 0.0641 | 1.6860 |

**Vitality sweet spot**: 2/50 (0.040)
- Training proportion: 0.141
- Ratio (sample/training): 0.283

### Step 20000

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | -0.0500 | 7.30e-01 | -0.0239 | 0.8277 |
| BR | 0.1923 | 1.81e-01 | 0.2009 | 0.8156 |
| GR | 0.2477 | 8.28e-02 | 0.2409 | 1.6723 |

**Vitality sweet spot**: 1/50 (0.020)
- Training proportion: 0.141
- Ratio (sample/training): 0.142

### Step 40000

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | -0.0937 | 5.17e-01 | 0.0344 | 1.2955 |
| BR | 0.1316 | 3.62e-01 | 0.0431 | 1.4105 |
| GR | 0.2875 | 4.29e-02 | 0.2570 | 2.1637 |

**Vitality sweet spot**: 1/50 (0.020)
- Training proportion: 0.141
- Ratio (sample/training): 0.142

### Step 50000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | -0.2169 | 1.30e-01 | -0.2499 | 0.9193 |
| BR | 0.1344 | 3.52e-01 | 0.0284 | 1.3345 |
| GR | 0.4692 | 5.89e-04 | 0.6551 | 1.2258 |

**Vitality sweet spot**: 3/50 (0.060)
- Training proportion: 0.141
- Ratio (sample/training): 0.425

### Step 80000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.2528 | 7.65e-02 | 0.3271 | 1.1899 |
| BR | 0.2152 | 1.33e-01 | 0.3772 | 0.7234 |
| GR | 0.5283 | 8.03e-05 | 0.5614 | 1.3196 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.141
- Ratio (sample/training): 0.708

### Step 100000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.4706 | 5.63e-04 | 0.5295 | 1.8914 |
| BR | 0.4330 | 1.69e-03 | 0.4939 | 1.7390 |
| GR | 0.6530 | 2.74e-07 | 0.7431 | 1.9516 |

**Vitality sweet spot**: 4/50 (0.080)
- Training proportion: 0.141
- Ratio (sample/training): 0.567

### Step 160000

**Regime**: `coupled`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.5797 | 1.03e-05 | 0.6831 | 1.3820 |
| BR | 0.6084 | 2.77e-06 | 0.7077 | 1.4361 |
| GR | 0.6743 | 7.95e-08 | 0.6766 | 1.6978 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.141
- Ratio (sample/training): 0.708

### Step 200000

**Regime**: `coupled`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.6948 | 2.17e-08 | 0.5630 | 1.4208 |
| BR | 0.6573 | 2.16e-07 | 0.5482 | 1.7289 |
| GR | 0.7392 | 8.77e-10 | 0.6347 | 1.4961 |

**Vitality sweet spot**: 9/50 (0.180)
- Training proportion: 0.141
- Ratio (sample/training): 1.275

### Step 240000

**Regime**: `density-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.5346 | 6.37e-05 | 0.1353 | 0.9347 |
| BR | 0.4031 | 3.70e-03 | 0.3752 | 1.9648 |
| GR | 0.2326 | 1.04e-01 | 0.0677 | 0.8971 |

**Vitality sweet spot**: 2/50 (0.040)
- Training proportion: 0.141
- Ratio (sample/training): 0.283

### Step 300000

**Regime**: `coupled`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.6448 | 4.32e-07 | 0.7185 | 1.4545 |
| BR | 0.5186 | 1.14e-04 | 0.6312 | 1.8276 |
| GR | 0.6470 | 3.85e-07 | 0.6903 | 1.3817 |

**Vitality sweet spot**: 9/50 (0.180)
- Training proportion: 0.141
- Ratio (sample/training): 1.275

### Step 360000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.4651 | 6.65e-04 | 0.5764 | 1.3868 |
| BR | 0.4509 | 1.02e-03 | 0.4716 | 1.5889 |
| GR | 0.8243 | 1.91e-13 | 0.7516 | 1.3400 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.141
- Ratio (sample/training): 0.708

### Step 400000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.5679 | 1.70e-05 | 0.5337 | 0.8719 |
| BR | 0.4680 | 6.11e-04 | 0.4087 | 1.9136 |
| GR | 0.4956 | 2.53e-04 | 0.4976 | 0.8416 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.141
- Ratio (sample/training): 0.708

### Step 500000

**Regime**: `coupled`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.6507 | 3.12e-07 | 0.7783 | 1.3264 |
| BR | 0.7457 | 5.17e-10 | 0.6696 | 1.4635 |
| GR | 0.8690 | 2.81e-16 | 0.7780 | 1.5407 |

**Vitality sweet spot**: 8/50 (0.160)
- Training proportion: 0.141
- Ratio (sample/training): 1.134

## 4. Synergy Evolution Across Checkpoints

### Phase Identification

| Phase | Step Range | Avg Synergy Strength |
|-------|------------|----------------------|
| emerging | 100 – 250 | 0.2838 |
| degrading | 250 – 500 | 0.3244 |
| emerging | 500 – 2500 | 0.3143 |
| degrading | 2500 – 10000 | 0.1843 |
| emerging | 10000 – 20000 | 0.1091 |
| stable | 20000 – 40000 | 0.1671 |
| emerging | 40000 – 200000 | 0.4355 |
| degrading | 200000 – 240000 | 0.5436 |
| emerging | 240000 – 300000 | 0.4968 |
| degrading | 300000 – 400000 | 0.5647 |
| emerging | 400000 – 500000 | 0.6328 |

**Dominant regime**: `diversity-driven`

## 5. Comparison with Training Data

### Wasserstein Distances (Best Checkpoint)

| Metric | Value |
|--------|-------|
| W₁ marginal B | 0.2433 |
| W₁ marginal G | 0.1737 |
| W₁ marginal R | 0.2167 |
| W₁ avg BG | 0.2085 |
| W₁ avg BR | 0.2300 |
| W₁ avg GR | 0.1952 |

## 6. Comparison with Empirical Thresholds

### Literature Reference Points

| Source | Metric | Threshold | Notes |
|--------|--------|-----------|-------|
| Jinan (2026) | POI blending degree | 1.5 | Functional mix threshold |
| Jinan (2026) | Building density | 4000 buildings/km² | Diminishing returns |
| Beijing (2025) | Intersection density | 22–28 | Optimal walkability range |

**Important limitation**: These thresholds are in raw urban metric units
(buildings/km², intersection counts, etc.) and do **not** directly map to
our normalized [0, 1] pixel values. Our preprocessing pipeline applies
per-channel global percentile normalization, so pixel intensity 0.5
corresponds to approximately the median value of each channel in the
training data, not to any specific raw urban metric value.

Future work should calibrate these thresholds against the normalization
parameters (e.g., the p95 values used during data processing) to enable
direct comparison.

## 7. Methodological Framework for Replication

### 7.1 Required Data
- RGB cell images where channels encode distinct urban vitality dimensions
- Channel convention: B=density, G=diversity, R=accessibility
- Images normalized to [-1, 1] range (standard DDPM preprocessing)
- Multiple training checkpoints at known step counts

### 7.2 Metrics to Compute
1. Per-image spatial means per channel → joint distribution
2. Pairwise Pearson and Spearman correlations → synergy strength
3. Mutual information (histogram-based) → non-linear dependencies
4. Wasserstein distance → distribution shift between samples and training
5. Regime classification → qualitative synergy pattern
6. Sweet spot detection → multi-dimensional vitality convergence

### 7.3 Interpretation Guide
- **Coupled regime**: Model has learned strong inter-channel dependency — 
  generation respects the joint distribution
- **{X}-driven regime**: One channel dominates — model may be learning 
  marginal structure before joint structure
- **Fragmented regime**: No synergy — model treats channels independently
- **Emerging phase**: Synergy strength increasing — joint structure being learned
- **Stable phase**: Synergy plateaued — model has captured the joint structure
- **Degrading phase**: Synergy decreasing — possible onset of memorization

### 7.4 Limitations
- Spatial means lose within-cell spatial patterns
- Histogram-based MI estimation is sensitive to bin count
- Small sample sizes increase variance in correlation estimates
- Normalized pixel values cannot be directly compared to raw urban metrics
- Regime classification uses fixed thresholds that may need calibration
