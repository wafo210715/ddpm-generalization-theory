# Progress Report — 2026-04-14

**Session type**: Implementation sprint — v3 training preparation and launch
**Outcome**: Diagnosed v2 model undertraining, scaled model 16x (ch=32→128), benchmarked batch sizes, added resume support, launched 500K-step training.

---

## Session Summary

### Problem Diagnosis

Analyzed v2 training results (50K steps, ch=32, ~9.88M params):
- **Loss still decreasing** at step 50K (0.001021) — no convergence plateau
- **Generated samples don't match training distribution**: training data is dark red (dominated by accessibility channel), but v2 samples are multi-colored (red, blue, orange, pink)
- **Memorization stage not reached** — model hasn't even converged to the data distribution, so the generalization window framework can't be applied
- **Full loss curve extracted** from stderr log: 50,401 data points across 50K steps

Conclusion: 50K steps with a 9.88M param model is insufficient for convergence on 3,263 training samples. Need significantly more model capacity and training time.

### Benchmark: VRAM + Throughput (ch=128, dual RTX 3090)

Comprehensive benchmark across batch sizes 8–96 on dual GPU (DataParallel, gloo backend — no NCCL on Windows):

| BS | Per GPU | Peak VRAM | it/s | 500K Time | Status |
|----|---------|-----------|------|-----------|--------|
| 8  | 4       | 4.4 GB    | 4.78 | ~29h      | OK     |
| 16 | 8       | 6.4 GB    | 3.17 | ~44h      | OK     |
| 24 | 12      | 8.4 GB    | 2.38 | ~58h      | OK     |
| 32 | 16      | 10.3 GB   | 1.90 | ~73h      | OK     |
| 48 | 24      | 14.3 GB   | 1.41 | ~98h      | OK     |
| 64 | 32      | 18.3 GB   | 1.13 | ~123h     | OK     |
| 80 | 40      | OOM       | —    | —         | OOM    |

**Key finding**: Larger batches are *slower* due to DataParallel overhead without NCCL (Windows limitation). bs=8 is optimal at 4.78 it/s.

### v3 Training Configuration

Created `configs/hcmc_vitality_v3.yaml`:
- **Model**: ch=128 (157.8M params, 16x larger than v2's 9.88M)
- **Batch size**: 8 (benchmark-confirmed fastest)
- **Steps**: 500,000 (~29 hours estimated)
- **Validation loss**: tracked every 5,000 steps (new feature)
- **Checkpoints**: 19 total, logarithmic schedule dense in early phase for generalization window analysis

### Infrastructure Changes

1. **Validation loss tracking** added to `train.py` and `config_loader.py`:
   - `val_every: 5000` — compute val loss every N steps
   - `val_steps: 50` — number of val batches to average over
   - Uses `split="val"` from existing dataset loader (73 val samples)
   - `torch.no_grad()` evaluation, model toggles eval/train

2. **Resume support** added to `train.py`:
   - `--resume <checkpoint.pth>` CLI flag
   - Handles 3 checkpoint formats: new (with optimizer/step), legacy full DDPM, raw state_dict
   - Infers step from filename if old format has no `step` key
   - Skips already-saved checkpoints
   - New checkpoints save optimizer state for seamless resumes

3. **VRAM + throughput benchmark tool**: `benchmark_vram_throughput.py`
4. **Quick VRAM test**: `test_vram.py`

### Training Incident

- First launch crashed at ~step 1000 due to SSD thermal shutdown
- Corrupted the step 1000 checkpoint (deleted)
- Steps 100/250/500 survived
- Resume support saved us — restarted cleanly from step 500
- Training resumed, both GPUs stable at ~89% utilization, 6.5 GB each

---

## Files Changed

| File | Change |
|------|--------|
| `ddpm-win-master/train.py` | Added val loss tracking, resume support (184 lines added) |
| `ddpm-win-master/config_loader.py` | Added `val_every` and `val_steps` fields |
| `ddpm-win-master/configs/hcmc_vitality_v3.yaml` | New v3 config (ch=128, bs=8, 500K steps) |
| `ddpm-win-master/benchmark_vram_throughput.py` | New: comprehensive VRAM + it/s benchmark tool |
| `ddpm-win-master/test_vram.py` | New: quick VRAM test script |
| `ddpm-win-master/test_samples/v2_checkpoint_comparison/generate_v2_samples.py` | New (gitignored): v2 checkpoint sampling + grid comparison |
| `data-processing/scripts/build_hcmc_pipeline_v2.py` | Modified (from earlier session, carried over) |

---

## Next Steps

1. **Monitor v3 training** (~29h, resume if crashes with `--resume`)
2. **Address SSD thermals** — likely root cause of shutdown; consider SSD cooling
3. **Sample from v3 checkpoints** at milestones (1K, 5K, 10K, 50K, 100K) to track convergence
4. **Compare v3 samples vs v2** to confirm the larger model is learning faster
5. **Track train/val loss divergence** as signal of memorization onset
6. **Stage 1 metrics**: compute F_mem on v3 checkpoints to find the generalization window

---

## Key Insights

- **50K steps was never enough** — Bonnaire et al. train 500K+; the v2 loss curve confirms no convergence
- **DataParallel on Windows (gloo) has a sweet spot at small batches** — NCCL would fix this but requires Linux/WSL2
- **bs=8 gives more gradient updates per epoch** (408 vs 51 with bs=64) — arguably better for learning dynamics
- **Resume support is essential** for long-running GPU jobs on hardware that may overheat
- **Old checkpoint format (raw state_dict) needs careful handling** — new format saves step + optimizer for robust resumes
