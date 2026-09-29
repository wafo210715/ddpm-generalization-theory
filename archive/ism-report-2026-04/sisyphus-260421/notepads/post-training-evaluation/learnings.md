# Post-Training Evaluation Learnings

## Key Findings

### F_mem ≈ 0 Across All Checkpoints (Critical)
- Maximum F_mem = 0.0209 at step 360000 (19 out of 19 checkpoints)
- No generalization window detected by the F_mem metric
- This means the model does NOT memorize training data at any point during training
- Implication: The model learned to generalize throughout, or the sample size (N=50) is too small to detect weak memorization
- **Task 9 (window refinement) is BLOCKED** — no window exists to refine

### Synergy Analysis: Fragmented Regime Dominant
- 11 phase transitions across 19 checkpoints
- Fragmented regime (all |r| < 0.3) is most common
- Diversity-driven regime appears occasionally
- No stable "coupled" regime (all |r| > 0.5) at any checkpoint

### Model Convergence
- v3 (ch=128) converged at ~86K steps
- v2 (ch=32) failed to converge at 50K steps
- 157.8M parameters, linear schedule, b0=0.0001, bT=0.02

### Checkpoint Format Issues
- v3 format: `{"step": int, "eps_model": OrderedDict, "optimizer": dict}`
- Required fixes in memorization.py, channel_statistics.py, synergy_analysis.py
- DDPM model does NOT store diffusion_config as attribute — unpacks to individual buffers

### Data Characteristics
- 4183 total PNGs in cells/ directory
- ~731 non-empty cells after filtering
- Channel convention: B=0 (density), G=1 (diversity), R=2 (accessibility)

## Technical Notes
- Only 19 of 20 checkpoints discovered by discover_checkpoints() — step 240000 may have naming mismatch
- multi_signal_window.py reads from CSVs, doesn't re-compute metrics
- generate_grids.py has 3 modes: grid, comparison, channels
- --save_samples flag is additive-only, doesn't change evaluation logic
