# Section 2.4: Minimum Viable Analysis Report

## 1. F_mem Curve and Generalization Window

**No generalization window was detected.** F_mem remained near zero throughout training, or the data does not exhibit the expected transition pattern.

### F_mem Values at Each Checkpoint

| Step | F_mem | 95% CI | Window |
|------|-------|--------|--------|
| 100 | 0.0000 | [0.0000, 0.0000] |  |
| 250 | 0.0000 | [0.0000, 0.0000] |  |
| 500 | 0.0000 | [0.0000, 0.0000] |  |
| 1000 | 0.0000 | [0.0000, 0.0000] |  |
| 2500 | 0.0000 | [0.0000, 0.0000] |  |
| 5000 | 0.0000 | [0.0000, 0.0000] |  |
| 10000 | 0.0000 | [0.0000, 0.0000] |  |
| 20000 | 0.0000 | [0.0000, 0.0000] |  |
| 40000 | 0.0000 | [0.0000, 0.0000] |  |
| 50000 | 0.0000 | [0.0000, 0.0000] |  |

## 2. Per-Channel Statistics (Window Checkpoints)

| Step | B mean | B std | G mean | G std | R mean | R std |
|------|--------|-------|--------|-------|--------|-------|
| 100 | 0.4970 | 0.4970 | 0.5282 | 0.4961 | 0.5102 | 0.4968 |
| 250 | 0.4895 | 0.4906 | 0.5043 | 0.4904 | 0.5051 | 0.4905 |
| 500 | 0.5043 | 0.4788 | 0.4702 | 0.4775 | 0.4863 | 0.4784 |
| 1000 | 0.4398 | 0.4409 | 0.4624 | 0.4397 | 0.4331 | 0.4462 |
| 2500 | 0.3475 | 0.3709 | 0.3760 | 0.3756 | 0.3559 | 0.3849 |
| 5000 | 0.5736 | 0.3820 | 0.3486 | 0.3711 | 0.3713 | 0.4016 |
| 10000 | 0.4534 | 0.3847 | 0.4962 | 0.3942 | 0.3750 | 0.4316 |
| 20000 | 0.4794 | 0.3862 | 0.6085 | 0.3816 | 0.4101 | 0.4030 |
| 40000 | 0.5697 | 0.3592 | 0.2708 | 0.3447 | 0.5750 | 0.4373 |
| 50000 | 0.5100 | 0.4126 | 0.3911 | 0.4026 | 0.5382 | 0.4254 |

## 3. Wasserstein Distance per Channel (Window Checkpoints)

| Step | B EMD | G EMD | R EMD |
|------|-------|-------|-------|
| 100 | 18.2917 | 17.8377 | 17.9123 |
| 250 | 6.2268 | 6.0907 | 6.1145 |
| 500 | 2.6908 | 2.7611 | 2.7357 |
| 1000 | 0.8781 | 1.0336 | 1.0712 |
| 2500 | 0.3256 | 0.5540 | 0.4790 |
| 5000 | 0.5587 | 0.5293 | 0.5018 |
| 10000 | 0.3887 | 0.6452 | 0.6840 |
| 20000 | 0.4255 | 0.7865 | 0.4246 |
| 40000 | 0.4814 | 0.4962 | 0.6588 |
| 50000 | 0.4408 | 0.5572 | 0.5630 |

### Spatial Autocorrelation (Moran's I) of Generated Samples

| Step | B Moran's I | G Moran's I | R Moran's I |
|------|-------------|-------------|-------------|
| 100 | 0.0027 | 0.0027 | 0.0090 |
| 250 | 0.0088 | -0.0008 | 0.0186 |
| 500 | 0.0063 | -0.0036 | 0.0072 |
| 1000 | 0.0057 | 0.0053 | 0.0465 |
| 2500 | 0.2436 | 0.2486 | 0.2469 |
| 5000 | 0.2426 | 0.2510 | 0.2516 |
| 10000 | 0.1176 | 0.0726 | 0.2217 |
| 20000 | 0.2491 | 0.1029 | 0.2456 |
| 40000 | 0.2434 | 0.2372 | 0.2489 |
| 50000 | 0.1366 | 0.2501 | 0.2499 |

## 4. Irreducible Characteristics

### Evolving Characteristics

These statistics are still changing — the model has not converged on these aspects.

| Metric | Mean | CV |
|--------|------|----|
| b_mean | 0.4864 | 0.1338 |
| b_std | 0.4203 | 0.1249 |
| g_mean | 0.4456 | 0.2220 |
| g_std | 0.4174 | 0.1309 |
| r_mean | 0.4560 | 0.1692 |
| r_std | 0.4396 | 0.0880 |

## 5. Summary

### What the Model Learned (Within Window)
- No stable characteristics identified (all metrics evolving)

### What Remains Noise
- **b_mean**: still evolving (CV=0.1338)
- **b_std**: still evolving (CV=0.1249)
- **g_mean**: still evolving (CV=0.2220)
- **g_std**: still evolving (CV=0.1309)
- **r_mean**: still evolving (CV=0.1692)
- **r_std**: still evolving (CV=0.0880)
