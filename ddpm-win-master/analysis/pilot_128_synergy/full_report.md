# Section 2.4 — Synergy Analysis: Joint B/G/R Distribution Evolution

**Generated**: 2026-04-13 19:40:07
**Checkpoints analyzed**: 5
**Step range**: 100 – 1000

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
| BG | 0.0895 | 6.38e-01 | 0.1666 | 2.3610 |
| BR | -0.2452 | 1.92e-01 | -0.2721 | 2.3148 |
| GR | -0.1549 | 4.14e-01 | -0.1493 | 2.3148 |

**Vitality sweet spot**: 1/30 (0.033)
- Training proportion: 0.072
- Ratio (sample/training): 0.463

### Step 250

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.0532 | 7.80e-01 | 0.0532 | 1.9277 |
| BR | -0.1494 | 4.31e-01 | -0.0647 | 2.0838 |
| GR | -0.0027 | 9.89e-01 | 0.0198 | 2.2050 |

**Vitality sweet spot**: 1/30 (0.033)
- Training proportion: 0.072
- Ratio (sample/training): 0.463

### Step 500

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.3041 | 1.02e-01 | 0.3780 | 2.0117 |
| BR | 0.0258 | 8.92e-01 | -0.0848 | 2.1678 |
| GR | 0.1694 | 3.71e-01 | 0.1511 | 2.1587 |

**Vitality sweet spot**: 0/30 (0.000)
- Training proportion: 0.072
- Ratio (sample/training): 0.000

### Step 750

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | 0.0914 | 6.31e-01 | 0.1097 | 2.3785 |
| BR | -0.2030 | 2.82e-01 | -0.1960 | 2.1649 |
| GR | 0.0360 | 8.50e-01 | 0.0665 | 2.3785 |

**Vitality sweet spot**: 1/30 (0.033)
- Training proportion: 0.072
- Ratio (sample/training): 0.463

### Step 1000

**Regime**: `fragmented`

| Pair | Pearson r | p-value | Spearman r | MI (nats) |
|------|-----------|---------|------------|-----------|
| BG | -0.1063 | 5.76e-01 | -0.0883 | 2.2573 |
| BR | 0.0521 | 7.85e-01 | -0.0394 | 2.6270 |
| GR | -0.1031 | 5.88e-01 | -0.0456 | 2.4421 |

**Vitality sweet spot**: 0/30 (0.000)
- Training proportion: 0.072
- Ratio (sample/training): 0.000

## 4. Synergy Evolution Across Checkpoints

### Phase Identification

| Phase | Step Range | Avg Synergy Strength |
|-------|------------|----------------------|
| degrading | 100 – 250 | 0.1158 |
| emerging | 250 – 500 | 0.1174 |
| degrading | 500 – 1000 | 0.1212 |

**Dominant regime**: `fragmented`

## 5. Comparison with Training Data

### Wasserstein Distances (Best Checkpoint)

| Metric | Value |
|--------|-------|
| W₁ marginal B | 0.2903 |
| W₁ marginal G | 0.3291 |
| W₁ marginal R | 0.4307 |
| W₁ avg BG | 0.3097 |
| W₁ avg BR | 0.3605 |
| W₁ avg GR | 0.3799 |

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
