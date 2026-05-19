# Progress Report — 2026-05-19

**Session type**: Crash diagnosis — v3 training black screen investigation
**Status**: Root cause identified (outlet voltage), pending user verification

---

## Problem

v3 training (ch=128, 157.8M params, 500K steps) crashes to black screen after ~8 minutes.
Previous v2 training (ch=32, 9.88M params, 50K steps) completed without issues.

## GPU Log Analysis (gpu_log.csv)

- Time range: 12:13:54 → 12:21:20 (~7.5 min, log stops abruptly)
- GPU temps: normal (GPU0 max 57°C, GPU1 max 67°C)
- VRAM: normal (~7 GB / 24 GB per card)
- Power: GPU0 295-305W (limit 370W), GPU1 277-301W (limit 390W)
- No anomaly before crash — data simply stops

## History

| Date | Event |
|------|-------|
| Apr 14 | v2 training completed smoothly (50K steps, ch=32) |
| Apr 14 | v3 training launched, crashed at step ~1000 (originally misdiagnosed as SSD thermal) |
| Apr 14-22 | v3 completed 500K steps through multiple crash-resume cycles |
| Apr 22 | Post-training analysis: F_mem memorization onset at ~step 240K |
| May 19 | v3 re-run from scratch, crashes again after ~8 min (up to step ~2500) |

## Investigation Timeline

### Ruled out

- **PSU capacity**: 1200W is sufficient for dual RTX 3090 (~600W combined GPU load)
- **SSD thermal**: Previously suspected, but actual SSD temp is ~60°C (normal). The earlier diagnosis was based on fan control software reading factory warning thresholds, not actual overheating.
- **GPU die temperature**: Max 57°C / 67°C, well within limits
- **VRAM capacity**: Only 7 GB / 24 GB used

### SSD health

- C: drive 92% full (157 GB free out of 1.9 TB)
- Two NVMe SSDs: Kingston SNVS2000G + 致钛 TiPlus7100 2TB
- Cleaning up C: drive space is advisable regardless

### Root cause: outlet voltage drop

User reported a Xiaomi fan on the same outlet can only run at speed 2 (out of 4). Speeds 3-4 work on other outlets. This proves the outlet has low voltage or significant voltage sag under load.

**Mechanism**:
1. Dual GPU sustained draw ~600W + CPU/system = ~800-900W total
2. Low outlet voltage → PSU draws more current → further voltage sag
3. Sustained load drains PSU capacitor buffer (~7-8 min)
4. RTX 3090 transient power spike exceeds what PSU can deliver at reduced voltage
5. PSU undervoltage protection triggers or GPU power delivery fails → black screen

**Why v2 worked but v3 doesn't**:
- v2 (ch=32): ~200W GPU load, outlet voltage can sustain it
- v3 (ch=128): ~600W GPU load, exceeds what the weakened outlet can deliver

## Updates

### 2026-05-19 (续): 插口已更换，准备恢复训练

- **已换插口**：将 PC 移至新插座，等待验证
- **C: 盘已清理**：释放至 588GB 可用空间
  - 训练需要约 99GB（56 个 checkpoint × 1.76GB each），空间充足
- **GPU 功耗回顾**（来自 gpu_log.csv）：
  - GPU0: ~295-305W, GPU1: ~277-301W
  - RTX 3090 TDP = 350W，双卡合计 ~600W 属正常范围
- **GPU logging 不再需要**：monitor_gpu.py 是诊断黑屏用的，训练脚本自带 loss logging（每 100 步）+ validation（每 5000 步）

### 存储估算

| 项目 | 数量 | 单个大小 | 总计 |
|------|------|----------|------|
| Checkpoint | 56 个 | 1.76 GB | ~99 GB |
| 可用空间 | — | — | 588 GB |

## Next Steps

1. [x] **换插口** — 已完成
2. [x] **清理 C: 盘** — 588GB 可用
3. [ ] **验证新插口**：用小米风扇测试能否跑满速度 4
4. [ ] **恢复 v3 训练**：
   ```bash
   cd ddpm-win-master
   python train.py --config configs/hcmc_vitality_v3.yaml --resume model_ckpts/hcmc_vitality_v3/hcmc_vitality_v2_128x128_steps500000_step1000.pth
   ```
5. [ ] **观察 30 分钟**确认稳定后再放跑全程
6. [ ] **如果又黑屏**：查 `eventvwr.msc → System → Event ID 41 (Kernel-Power)`，若确认断电则考虑电源老化

## Files Referenced

- `gpu_log.csv` — GPU telemetry from crashed run
- `ddpm-win-master/configs/hcmc_vitality_v3.yaml` — v3 config (ch=128, bs=8, 500K steps)
- `ddpm-win-master/train.py` — training script with resume support
- `ddpm-win-master/model_ckpts/hcmc_vitality_v3/` — today's checkpoints (steps 100-1000)
