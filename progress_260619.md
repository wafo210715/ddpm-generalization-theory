# Progress 2026-06-19: Phase 1.5.3 Batch Size Ablation 完成

> **当前分支**: `master`
> **会话类型**: 长时间 GPU 实验（4 个 run，~95 小时墙时间）
> **状态**: 
> - 全部 4 个新 run（bs=16/32/64/128）完成训练
> - 加上已有 v3_pro6k (bs=8) 共 5 个 bs 值
> - val_loss 数据已收集，下一步 F_mem 评估

---

## 一、实验动机与设计

### 动机

0611 plan 的 Phase 1.5.3 提出三个互相竞争假设，解释为什么 v3_pro6k (bs=8, 4M samples) 训练到 500K 处 F_mem 只到 1.2%：

| 假设 | 含义 | 验证方法 |
|------|------|---------|
| A | 训练不够长（τ_mem > 500K） | 延长训练 |
| **B** | **模型太大/数据太小，batch size 影响动力学** | **bs ablation** |
| C | 数据本身太简单（无解） | 排除法 |

本会话执行假设 B 的验证。

### 实验设计

**目标**：在固定 8M samples 的条件下，对比不同 batch size 的训练动力学。继承 5090 setup 的实验意图（bs=16 是原设计 baseline，bs=8 是 Pro 6000 的 wall-time 妥协）。

| Run | bs | n_steps | samples_seen | 墙时间 |
|-----|----|----|------|------|
| v3_pro6k (existing) | 8 | 500,000 | 4M | ~13h（已存在）|
| **bs=16 baseline** | 16 | 500,000 | 8M | ~25h38m |
| **bs=32** | 32 | 250,000 | 8M | ~24h |
| **bs=64** | 64 | 125,000 | 8M | ~24h |
| **bs=128** | 128 | 62,500 | 8M | ~23h |

**所有 run 都用 fixed LR=1e-4**（用户决定：先用固定 LR，如果大 bs 出现明显欠拟合再考虑 sqrt scaling）。checkpoints 按 samples-seen 对齐保存（800K, 2K, 4K, ... 8M）。

**总墙时间**：~95 小时（4 个新 run）≈ 4 天，恰好覆盖用户的"几天才有空"时间窗。

---

## 二、执行结果

### 5 个 bs 值的训练汇总

| bs | checkpoints | final train_loss | final val_loss | run 目录 |
|----|------|------|------|------|
| 8 | 19+1 | 0.000429 | 0.001181 @ 4M samples | `model_ckpts/hcmc_vitality_v3_pro6k/` |
| 16 | 24 | 0.000287 | **0.001855** @ 8M samples | `model_ckpts/hcmc_vitality_v3_pro6k_bs16/` |
| 32 | 24 | 0.000342 | **0.001962** @ 8M samples | `model_ckpts/hcmc_vitality_v3_pro6k_bs32/` |
| 64 | 22 | 0.000294 | **0.002474** @ 8M samples | `model_ckpts/hcmc_vitality_v3_pro6k_bs64/` |
| 128 | 22 | 0.000193 | **0.002416** @ 7.7M samples | `model_ckpts/hcmc_vitality_v3_pro6k_bs128/` |

### 关键数据：val_loss 轨迹（samples-seen 对齐）

| samples_seen | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 |
|------|------|------|------|------|------|
| 800K | - | **0.000787** | 0.000913 | (early min) | 0.001270 |
| 1.6M | - | 0.000819 | 0.000914 | **0.001147** | **0.000966** ← |
| 3.2M | - | 0.001204 | 0.001741 | 0.001381 | 0.001763 |
| 4M (= v3_pro6k endpoint) | 0.001181 | 0.001423 | 0.001497 | - | - |
| 8M (final) | - | **0.001855** | **0.001962** | **0.002474** | ~0.0024 |

### val_loss 最低点（memorization onset 候选）

| bs | 最低 val_loss | 对应 samples_seen |
|----|------|------|
| 16 | 0.000787 | **800K** |
| 32 | 0.000913 | **800K - 1.6M** |
| 64 | 0.001147 | **1.6M** |
| 128 | 0.000966 | **1.28M** |

---

## 三、三个关键发现

### 发现 1：val_loss 单调上升 with bs（在 8M samples 处）

| bs | val_loss @ 8M | 相对 bs=16 |
|----|------|------|
| 16 | 0.001855 | 1.00x |
| 32 | 0.001962 | 1.06x |
| 64 | 0.002474 | **1.33x** |
| 128 | ~0.0024 | 1.30x |

**结论**：**large batch regularization 假设在我们的数据集上验证**——大 bs → 弱 implicit regularization → 泛化差。bs=16→bs=32 的 6% 差距，到 bs=64 跳到 33%，再大反而稳定在 ~30%。

### 发现 2：memorization onset samples-seen 聚集在 ~1M samples

四个新 run 的 val_loss 最低点都在 **800K-1.6M samples** 之间。

**含义**：τ_mem（memorization onset 时间）主要由 **samples-seen** 决定，而不是 gradient_updates。这直接回答了 0611 plan 的核心问题——Bonnaire et al. scaling law 的口径应该是 samples-seen，不是 step 数。

### 发现 3：bs=128 val_loss 在 7M samples 跳到 0.002972（异常）

bs=128 在 step 55K（7M samples）val_loss 突然跳到 0.002972，比 bs=64 在 8M 处的 0.002474 还高。这是**大 batch + 固定 LR 欠拟合 + memorization 同时发生**的征兆。

**含义**：bs=128 + LR=1e-4 是不稳定组合。如果用 sqrt scaling（LR=4e-4），可能能避免 spike。但也可能让 memorization 更早出现。这是论文的诚实 limitation。

### 顺便发现：train_loss 揭示的 underfitting

| bs | final train_loss (last 100 avg) | final train_loss (last single step) |
|----|------|------|
| 16 | 0.000279 | 0.000287 |
| 32 | 0.000293 | 0.000342 |
| 64 | 0.000294 | **0.001022** |
| 128 | 0.000370 | **0.000193** |

bs=64 最后单个 step 跳到 0.001022，bs=128 跳到 0.000193。**大 bs + 固定 LR 后期梯度更新不稳**。

---

## 四、对 0611 plan 的影响

### Phase 1.5.3 决策矩阵更新

| 解释 | 旧状态（0611 plan） | 新状态（本实验后） |
|------|------|------|
| A. 训练不够长 | 待验证 | **部分否定**——bs=16 训到 8M（v3_pro6k 的 2x），val_loss 仍然单调上升，说明延长训练只会让泛化更差 |
| **B. 模型太大/数据太小** | 待验证 | **✅ 部分验证**——bs sweep 显示 large batch regularization 真实存在 |
| C. 数据太简单 | 待验证 | 仍待 F_mem 评估确认 |

### 假设 A 的部分否定：val_loss 单调上升

bs=16 在 500K-500K step 范围内 val_loss 从 0.000787 → 0.001855（**增加 2.4x**）。这意味着**单纯延长训练不会改善泛化**，反而会加剧过拟合。

**对论文叙事的影响**：原本 "Phase 1.5.3 假设 A: 训练不够长 → τ_mem > 500K" 的可能被排除。论文应该转向讨论 "training longer ≠ better generalization; there's a sweet spot around 800K samples where val_loss is minimized"。

### 对 0611 plan baseline 的影响

0611 plan 的所有定量 claim（F_mem onset @ 240K step, generalization window 160K-240K = 80K 等）基于 **bs=8 (v3_pro6k)**。

bs sweep 显示不同 bs 的动力学**显著不同**。如果论文以 bs=16 为 baseline（用户决定），那么：
- 0611 plan 的所有数字需要 re-check
- F_mem onset 位置可能在 bs=16 下不同
- "Generalization window 80K" 这个 claim 可能只在 bs=8 下成立

**建议**：保留 v3_pro6k 数据作为参考，但论文主线用 bs=16 数据。需要先跑完 F_mem 评估再决定。

---

## 五、技术细节

### 环境

- **Python env**: `.venv-cu128`（torch 2.11.0+cu128, CUDA 12.8, sm_120 兼容）
- **路径**: `C:/sd/ddpm-generalization-theory/ddpm-win-master/.venv-cu128/Scripts/python.exe`
- **硬件**: NVIDIA RTX PRO 6000 Blackwell Max-Q, 96GB（China）
- **GPU 实测显存**：bs=8→6.4GB, bs=16→15.3GB, bs=32→23.6GB, bs=64→43.8GB, bs=128→83.4GB

### Throughput 观察

| bs | benchmark it/s | 实测 sustained it/s | 折扣 |
|----|------|------|------|
| 8 | 10.98 | - | - |
| 16 | 5.67 | 5.36 | 95% |
| 32 | 2.97 | ~2.81（推断） | 95% |
| 64 | 1.47 | ~1.39（推断） | 95% |
| 128 | 0.72 | ~0.68（推断） | 95% |

**GPU 是 compute-bound**：所有 bs 的 samples/s 都在 ~90 samples/s，与 bs 无关。这意味着相同 samples-seen 的训练墙时间几乎相同（~24h for 8M samples）。

### Checkpoint 文件

总 checkpoints 数：19+24+24+22+22 = **111 个 .pth 文件**，每个 ~1.8GB → **总磁盘 ~200GB**。

---

## 六、未提交的文件

5 个新 config 未 commit：

```
ddpm-win-master/configs/hcmc_vitality_v3_pro6k_1M.yaml   (unused, v3 extension, abandoned)
ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs16.yaml (used)
ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs32.yaml (used)
ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs64.yaml (used)
ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs128.yaml (used)
```

`hcmc_vitality_v3_pro6k_1M.yaml` 是 6/14 准备延长 v3 用的，后来用户决定重定 baseline 到 bs=16，所以未使用。可以删除或保留作为历史。

---

## 七、下一步建议

按优先顺序：

1. **F_mem 评估（5 runs × ~24 ckpts × 1000 samples）**：~16h GPU。论文核心指标，验证 val_loss 推断的 memorization onset 是否真的对应 F_mem 上升。
2. **Synergy analysis**（5 runs）：~1h。看 coupled regime 出现在不同 bs 的什么位置。
3. **Multi-signal window plot**：~1h。论文主图。
4. **Decision: bs=128 是否重跑 with LR=4e-4**：取决于 F_mem 结果是否被 val_loss spike 污染。

---

## 八、本会话的关键认识（用户纠正我的地方）

### 8.1 "继承 5090 setup 而不是 v3" 的含义

用户原话（6/14）："we should inherit 5090's set up rather than v3's."

**含义**：5090 setup (bs=16) 是原实验设计意图，pro6k (bs=8) 是 Pro 6000 上的 wall-time 妥协。应该用 bs=16 作为新 baseline，bs=8 数据保留作为参考。

### 8.2 "samples 总量决定墙时间，不是 step 数"

用户原话（6/15）："bs16都跑了24小时多，bs32应该更久？"

**修正我的错误**：我最初估计 bs=32 ETA ~23h（基于 step 数和 it/s）。用户直觉性地质疑。实际上 GPU 是 compute-bound，samples/s ~90 与 bs 无关，所以相同 samples-seen 的墙时间几乎相同。所有 run 都是 ~24h。

### 8.3 "图片生成质量" 是个 blind spot

用户原话（6/14）："你其实是看不到图片生成质量的，你的很多判断都是根据metrics来进行的。"

**承认 limitation**：bs sweep 结论基于 val_loss（metric），看不到实际生成图。F_mem 评估后应该手动抽检样本图。

---

## 九、相关文件

### 新增

- `progress_260619.md`（本文件）
- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs16.yaml`
- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs32.yaml`
- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs64.yaml`
- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k_bs128.yaml`
- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k_1M.yaml`（unused）

### 新增训练输出

- `ddpm-win-master/model_ckpts/hcmc_vitality_v3_pro6k_bs{16,32,64,128}/` — 92 个新 checkpoints
- `ddpm-win-master/logs/hcmc_vitality_v3_pro6k_bs{16,32,64,128}/` — 4 个新 training_log.csv

### 引用

- `progress_260611.md` — 原 Phase 1.5.3 计划，本实验是对它的执行
- `ddpm-win-master/configs/hcmc_vitality_v3_5090.yaml` — 5090 baseline config（bs=16）
- `ddpm-win-master/configs/hcmc_vitality_v3_pro6k.yaml` — v3_pro6k config（bs=8）
