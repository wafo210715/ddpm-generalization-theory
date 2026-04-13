# Section 2.4 — Synergy Analysis: Joint B/G/R Distribution Evolution

**Generated**: 2026-04-13 19:05:37
**Checkpoints analyzed**: 5
**Step range**: 100 – 5000

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

**Regime**: `accessibility-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.0541 | 7.09e-01 | 0.0033 | 1.1671 |
| BR | -0.1336 | 3.55e-01 | -0.0863 | 1.3162 |
| GR | 0.5977 | 4.58e-06 | 0.7325 | 1.2063 |

**Vitality sweet spot**: 3/50 (0.060)
- Training proportion: 0.100
- Ratio (sample/training): 0.600

### Step 500

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.3357 | 1.72e-02 | 0.3566 | 1.5755 |
| BR | 0.0331 | 8.19e-01 | 0.1129 | 1.6491 |
| GR | 0.6021 | 3.73e-06 | 0.5838 | 1.4838 |

**Vitality sweet spot**: 3/50 (0.060)
- Training proportion: 0.100
- Ratio (sample/training): 0.600

### Step 1000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.3185 | 2.42e-02 | 0.2458 | 1.6137 |
| BR | 0.2962 | 3.67e-02 | 0.2392 | 1.4524 |
| GR | 0.8220 | 2.54e-13 | 0.7516 | 1.5952 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.100
- Ratio (sample/training): 1.000

### Step 2000

**Regime**: `diversity-driven`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.5436 | 4.51e-05 | 0.4522 | 1.6887 |
| BR | 0.4290 | 1.88e-03 | 0.3553 | 1.5926 |
| GR | 0.9028 | 3.26e-19 | 0.8902 | 1.8920 |

**Vitality sweet spot**: 5/50 (0.100)
- Training proportion: 0.100
- Ratio (sample/training): 1.000

### Step 5000

**Regime**: `coupled`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.6551 | 2.45e-07 | 0.6453 | 2.1073 |
| BR | 0.6229 | 1.36e-06 | 0.6273 | 2.1699 |
| GR | 0.9291 | 2.24e-22 | 0.8933 | 2.1029 |

**Vitality sweet spot**: 7/50 (0.140)
- Training proportion: 0.100
- Ratio (sample/training): 1.400

## 4. Synergy Evolution Across Checkpoints

### Phase Identification

| Phase | Step Range | Avg Synergy Strength |
|-------|------------|----------------------|
| emerging | 100 – 5000 | 0.4850 |

**Dominant regime**: `diversity-driven`

### Irreducible Synergies

These channel pairs maintain stable correlation across the entire
generalization window, suggesting the model has learned fundamental
structural relationships:

- **GR**: avg |r| = 0.7707

## 5. Comparison with Training Data

### Wasserstein Distances (Best Checkpoint)

| Metric | Value |
|--------|-------|
| W₁ marginal B | 0.0339 |
| W₁ marginal G | 0.1201 |
| W₁ marginal R | 0.0935 |
| W₁ avg BG | 0.0770 |
| W₁ avg BR | 0.0637 |
| W₁ avg GR | 0.1068 |

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
