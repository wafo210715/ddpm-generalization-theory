## Synergy Analysis (Task 5) — 2026-04-20

### Checkpoint Loading Fix
- synergy_analysis.py had the SAME checkpoint format bug as memorization.py and channel_statistics.py
- v3 format: {"step": int, "eps_model": OrderedDict, "optimizer": dict}
- Original code checked "eps_model" in checkpoint first, which matched v3 but then called ddpm.load_state_dict(checkpoint) — wrong because v3 has extra keys
- Fix: check "optimizer" in checkpoint FIRST to detect the v3 wrapper, then extract checkpoint["eps_model"]

### CSV Export Added
- Added synergy_metrics.csv (long format) with columns: step, pair, pearson_r, pearson_p, spearman_r, spearman_p, mi_histogram, regime
- Existing synergy_summary.csv (wide format) was kept intact
- Both CSVs are generated in the main pipeline

### Results Summary (19 checkpoints)
- Dominant regime: **diversity-driven** (G-R correlation consistently strongest)
- Synergy progression: fragmented early → diversity-driven → coupled at late stages
- Key phase transitions:
  - Steps 100-10000: mostly fragmented/weak synergy (avg |r| < 0.3)
  - Steps 40000-200000: emerging synergy (avg |r| → 0.43)
  - Steps 200000+: coupled regime appears at 160K, 200K, 300K, 500K
- Best checkpoint by Wasserstein distance: step 50000
- Final checkpoint (500K): strong coupled regime (BG r=0.65, BR r=0.75, GR r=0.87)
- step 240000 shows a temporary regression (density-driven instead of coupled)
