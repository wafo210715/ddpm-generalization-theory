# Progress Report — 2026-05-21

**Session type**: v3 training resume + Pro 6000 Blackwell benchmark
**Status**: Benchmark complete, Pro 6000 config ready, training command prepared

---

## Summary

After confirming dual RTX 3090 stability (step 5000→10000), switched to Pro 6000 Blackwell for benchmarking. Set up new CUDA 12.8 environment on `wafo/5090-setup` branch, ran comprehensive batch size benchmark, and discovered that **bs=8 remains optimal** even with 96GB VRAM — the model is compute-bound, not memory-bound.

## Part 1: Dual RTX 3090 — Stability Verification

| Item | Value |
|------|-------|
| Config | `configs/hcmc_vitality_v3.yaml` |
| Resume from | step 5000 checkpoint |
| Reached | step 10000 |
| Duration | ~30 min (5000 steps) |
| Speed | ~4.78 it/s (consistent with benchmark) |
| Crash | None — outlet fix confirmed stable |

**Checkpoints** (all in `model_ckpts/hcmc_vitality_v3/`): 100, 250, 500, 1K, 2.5K, 5K, **10K** (1.76 GB each)

## Part 2: Pro 6000 Blackwell — Environment Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB VRAM, 300W) |
| Driver | 591.86 / CUDA 13.1 |
| Branch | `wafo/5090-setup` |
| venv | `.venv-cu128` (separate from 3090's `.venv` cu118) |
| PyTorch | 2.11.0+cu128 (sm_120 support) |

Setup steps:
1. `git stash` on `wafo/v3-training` → `git checkout wafo/5090-setup` → `git stash pop`
2. Created `.venv-cu128` with `python -m venv .venv-cu128`
3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
4. `pip install tqdm matplotlib scipy pyyaml safetensors pytorch-fid natsort`

## Part 3: Batch Size Benchmark — Key Discovery

**Method**: Modified `benchmark_vram_throughput.py` to auto-scale batch sizes by GPU VRAM (96GB → test up to bs=160). Each config: 2 warmup + 20 timed iterations with full forward/backward/optimizer step. Used `python -u` for unbuffered output.

**Results**:

| BS | Peak VRAM | it/s | samples/s | 500K time |
|----|-----------|------|-----------|-----------|
| **8** | **6.4 GB** | **10.98** | **87.8** | **12.7 h** |
| 16 | 10.4 GB | 5.67 | 90.7 | 24.5 h |
| 32 | 18.4 GB | 2.97 | 95.0 | 46.7 h |
| 64 | 34.8 GB | 1.47 | 94.1 | 94.3 h |
| 96 | 51.4 GB | 0.97 | 93.1 | 143.9 h |
| 128 | 67.9 GB | 0.72 | 92.2 | 192.3 h |
| 160 | 84.4 GB | 0.06 | — | thrashing |

**Finding: throughput is compute-bound, not memory-bound.**
- samples/s ≈ 88–95 **regardless of batch size** — the GPU processes the same number of samples per second no matter the batch
- Larger batch = each iteration takes proportionally longer, zero parallelism gain
- bs=160 at 84GB caused memory thrashing (0.06 it/s)
- **Conclusion: bs=8 is optimal** — fastest wall time (12.7h) + most gradient updates (best for diffusion convergence)

**Why this method works**: The benchmark measures real training iterations (forward + backward + grad clip + optimizer step), not just inference. The `samples/s` metric reveals whether the GPU is compute-bound (constant) or has parallelism headroom (increasing). Constant samples/s = compute-bound = no benefit from larger batches.

## Pro 6000 vs Dual RTX 3090

| Metric | Dual RTX 3090 | Pro 6000 Blackwell | Speedup |
|--------|--------------|-------------------|---------|
| it/s (bs=8) | 4.78 | 10.98 | **2.3x** |
| 500K time | ~29 h | ~13 h | **2.3x** |
| VRAM used | 4.4 GB/GPU (×2) | 6.4 GB (×1) | — |
| Power | ~700W (2×350W) | 300W (Max-Q) | **2.3x efficient** |

## Pro 6000 Training Command

**cmd:**
```cmd
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
.venv-cu128\Scripts\activate.bat
python train.py --config configs/hcmc_vitality_v3_pro6k.yaml
```

**PowerShell:**
```powershell
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
.venv-cu128\Scripts\Activate.ps1
python train.py --config configs/hcmc_vitality_v3_pro6k.yaml
```

> **注意**: cmd 用 `.bat`，PowerShell 用 `.ps1`，混用会导致 venv 不激活、torch 找不到。如 PowerShell 报执行策略错误，先运行 `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`。

Config: `configs/hcmc_vitality_v3_pro6k.yaml` (bs=8, ch=128, 500K steps, fresh run)

## Dual RTX 3090 Resume Command (for when GPUs are swapped back)

**cmd:**
```cmd
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
.venv\Scripts\activate.bat
python train.py --config configs/hcmc_vitality_v3.yaml --resume model_ckpts/hcmc_vitality_v3/hcmc_vitality_v2_128x128_steps500000_step10000.pth
```

**PowerShell:**
```powershell
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
.venv\Scripts\Activate.ps1
python train.py --config configs/hcmc_vitality_v3.yaml --resume model_ckpts/hcmc_vitality_v3/hcmc_vitality_v2_128x128_steps500000_step10000.pth
```

## Next Steps

1. [ ] **Start Pro 6000 training** — run the command above, verify ~11 it/s and loss convergence
2. [ ] **Run to 500K steps** — ~13h on Pro 6000 (vs ~28h remaining on dual 3090)
3. [ ] **Resume v3 训练**（双卡，从 step 10000 继续）— 等双卡空出来再跑
4. [ ] **Post-training analysis**: channel statistics, synergy, F_mem memorization detection

## Files Referenced

- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k.yaml` — new Pro 6000 config (bs=8, fresh run)
- `ddpm-win-master/configs/hcmc_vitality_v3.yaml` — v3 config (dual 3090, resume from 10K)
- `ddpm-win-master/configs/hcmc_vitality_v3_5090.yaml` — old 5090 config (bs=16, superseded by pro6k)
- `ddpm-win-master/benchmark_vram_throughput.py` — updated to auto-scale by GPU VRAM
- `ddpm-win-master/model_ckpts/hcmc_vitality_v3/` — dual 3090 checkpoints (up to 10K)
- `progress_260519.md` — previous session (outlet diagnosis)
