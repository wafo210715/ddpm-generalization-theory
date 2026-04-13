# Section 2.4: Minimum Viable Analysis Report

## 1. F_mem Curve and Generalization Window

**No generalization window was detected.** F_mem remained near zero throughout training, or the data does not exhibit the expected transition pattern.

### F_mem Values at Each Checkpoint

| Step | F_mem | 95% CI | Window |
|------|-------|--------|--------|
| 100 | 0.0000 | [0.0000, 0.0000] |  |
| 250 | 0.0000 | [0.0000, 0.0000] |  |
| 500 | 0.0000 | [0.0000, 0.0000] |  |
| 750 | 0.0000 | [0.0000, 0.0000] |  |
| 1000 | 0.0000 | [0.0000, 0.0000] |  |

## 2. Per-Channel Statistics (Window Checkpoints)

| Step | B mean | B std | G mean | G std | R mean | R std |
|------|--------|-------|--------|-------|--------|-------|
| 100 | 0.4965 | 0.4969 | 0.5019 | 0.4970 | 0.4736 | 0.4963 |
| 250 | 0.5024 | 0.4910 | 0.5012 | 0.4915 | 0.4761 | 0.4906 |
| 500 | 0.5358 | 0.4784 | 0.4974 | 0.4800 | 0.4818 | 0.4786 |
| 750 | 0.5139 | 0.4663 | 0.4842 | 0.4657 | 0.4253 | 0.4594 |
| 1000 | 0.4861 | 0.4500 | 0.4332 | 0.4439 | 0.4426 | 0.4430 |

## 3. Wasserstein Distance per Channel (Window Checkpoints)

| Step | B EMD | G EMD | R EMD |
|------|-------|-------|-------|
| 100 | 17.4289 | 18.6224 | 17.8481 |
| 250 | 6.2530 | 6.8534 | 6.5453 |
| 500 | 2.6998 | 2.9682 | 2.8714 |
| 750 | 1.5581 | 1.7353 | 1.6731 |
| 1000 | 0.9924 | 1.1010 | 1.0928 |

### Spatial Autocorrelation (Moran's I) of Generated Samples

| Step | B Moran's I | G Moran's I | R Moran's I |
|------|-------------|-------------|-------------|
| 100 | 0.0023 | 0.0160 | 0.0201 |
| 250 | 0.0097 | 0.0260 | 0.0214 |
| 500 | 0.0138 | 0.0437 | 0.0175 |
| 750 | 0.0179 | 0.0400 | 0.0379 |
| 1000 | 0.0429 | 0.0368 | 0.0411 |

## 4. Irreducible Characteristics

### Stable (Irreducible) Characteristics

These statistics remain constant across the generalization window — the model has captured them as irreducible properties.

| Metric | Mean | CV |
|--------|------|----|
| b_mean | 0.5069 | 0.0375 |
| b_std | 0.4765 | 0.0397 |
| g_std | 0.4756 | 0.0451 |
| r_std | 0.4736 | 0.0469 |

### Evolving Characteristics

These statistics are still changing — the model has not converged on these aspects.

| Metric | Mean | CV |
|--------|------|----|
| g_mean | 0.4836 | 0.0601 |
| r_mean | 0.4599 | 0.0535 |

## 5. Summary

### What the Model Learned (Within Window)
- **b_mean**: stable at 0.5069
- **b_std**: stable at 0.4765
- **g_std**: stable at 0.4756
- **r_std**: stable at 0.4736

### What Remains Noise
- **g_mean**: still evolving (CV=0.0601)
- **r_mean**: still evolving (CV=0.0535)
