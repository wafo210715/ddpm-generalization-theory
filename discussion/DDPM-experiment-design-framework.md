# DDPM实验设计框架
**研究主题**: 通过Diffusion Models发现胡志明市城市活力的内在本质结构

**日期**: 2026-03-17
**状态**: Conceptual Framework (待与导师讨论)

---

## 核心研究问题的演进

### 初始问题（已放弃）
❌ "DDPM能否发现超越西方理论的活力模式？"
→ 问题：这是对抗性框架，且预设了"西方 vs 非西方"的二元对立

### 修正后的核心问题
✅ **"DDPM如何作为认识论工具，帮助我们发现胡志明市城市活力的内在本质结构？"**

**关键转变**：
- 从"对抗西方理论" → "发现本质结构"
- 从"验证假设" → "探索性发现"
- 从"预测活力" → "理解活力的生成机制"

---

## 理论定位：DDPM作为"结构发现器"

### Jane Jacobs的方法 vs DDPM的方法

| 维度 | Jane Jacobs (1961) | DDPM (2026) |
|------|-------------------|-------------|
| **观察对象** | 街道现象（人流、混合使用、建筑形态） | 城市数据的联合分布 |
| **方法** | 定性观察 → 归纳理论 | 统计学习 → 发现manifold结构 |
| **规模** | 几条街道（纽约） | 整个城市（胡志明市） |
| **输出** | "Four Generators"理论 | Latent space中的稳定结构 |
| **验证** | 案例论证 | Generalization window稳定性 |

### 核心类比

**物理学类比**：
- 牛顿定律 = 人类对物理世界的近似推断
- AI天气预测 = 神经网络对物理世界的近似推断（黑盒）

**城市活力类比**：
- Jacobs的Four Generators = 人类对城市活力的近似推断
- DDPM的Latent Manifold = 神经网络对城市活力的近似推断（黑盒）

**关键洞察**：
> Jacobs关注的是**联合分布**——"eyes on street" + "mixed uses" + "small blocks" **同时出现**时产生活力。DDPM学习的正是这种联合分布 P(eyes, mixed uses, small blocks, aged buildings)。

---

## 实验设计：两阶段策略

### 阶段一：简单方法（MVP，3-4周）

**目标**: 验证方法可行性

**Input编码**：
```
2D RGB图像（256×256 pixels = 1km × 1km）
- R通道: 人流量密度 (LBS/mobile signal heatmap)
- G通道: POI功能混合度 (Shannon entropy of categories)
- B通道: 街道网络密度 (intersection density per km²)
```

**数据来源**：
- 人流量：Viettel Mobile数据 ✅
- POI：Google Places + OSM ✅
- 街道网络：OSM路网数据 ✅

**模型配置**：
- 架构：纯DDPM（无条件生成）
- 参数：参考昨天的`ddpm-experiment-design.md`
- 训练策略：逐步增加数据量，识别intrinsic dimension

**预期产出**：
1. 确认DDPM能否学习胡志明市的城市数据分布
2. 初步识别generalization window
3. Latent space的初步可视化（UMAP）

---

### 阶段二：多模态方法（如果阶段一成功）

**Input编码**（更复杂）：
```
多模态融合
- 街景图像 → encode人流视觉线索（人数、活动类型）
- POI数据 → encode类别 + 地理位置（需要geospatial encoder）
- 建筑数据 → encode高度/年龄分布
- 时间维度 → encode昼夜/季节变化
```

**技术栈**（待调研）：
- Geospatial Encoder: Space2Vec / Tile2Vec
- 街景编码: 预训练的视觉模型（如CLIP）
- 时序编码: Transformer-based temporal encoder

**与发小合作**：
- 发小专长：感知压缩（perceptual compression）
- 合作点：设计高效的多模态encoder

---

## Generalization Window方法论

### 什么是Generalization Window？

**定义**：神经网络在训练过程中，先学习数据的本质结构（低频信号），后记忆噪声细节（高频信号）的阶段性窗口。

**物理类比**：
- 早期训练 = 学习"城市活力的物理定律"（结构性规律）
- 后期训练 = 记忆"每条街道的具体样子"（过拟合细节）

**关键洞察**：
> 在generalization window内，DDPM学到的latent manifold反映的是**城市活力的本质维度**，而非数据集的偶然特征。

### 检测方法

#### 方法1：训练曲线分析
```python
# 监控指标
metrics = {
    'train_loss': [],      # 训练损失
    'val_loss': [],        # 验证损失
    'fid_score': [],       # 生成质量
    'latent_rank': []      # Latent空间有效秩
}

# Generalization window = val_loss最低点附近
# 过拟合信号：train_loss持续下降但val_loss上升
```

#### 方法2：Intrinsic Dimension识别
```python
# 逐步增加训练数据量，观察性能饱和点
data_sizes = [100, 500, 1000, 2000, 5000, 10000]
for n in data_sizes:
    model = train_ddpm(data[:n])
    fid = evaluate_fid(model)
    # Intrinsic dimension = FID不再显著改善的最小n
```

#### 方法3：Latent Space稳定性
```python
# 在不同训练阶段提取latent表征
checkpoints = [epoch_100, epoch_200, epoch_300, ...]
for ckpt in checkpoints:
    latent = extract_latent(ckpt, test_data)
    structure_score = measure_manifold_stability(latent)
# Generalization window = 结构稳定但未过拟合的阶段
```

---

## 数据策展实验（Data Curation Experiments）

### 实验目的

通过控制训练数据的组成，观察DDPM学习到的结构如何变化，从而识别：
1. 哪些是**普遍的活力结构**（在不同区域都出现）
2. 哪些是**特定语境的活力模式**（仅在某类区域出现）

### 实验设计

| 实验组 | 训练数据来源 | 预期学习到的结构 | 验证方法 |
|--------|-------------|-----------------|---------|
| **Control** | 全胡志明市随机采样 | 城市活力的通用manifold | Baseline FID + latent分布 |
| **Exp A** | 仅District 1（殖民区） | 西方规划下的活力模式 | 与Jacobs理论对比 |
| **Exp B** | 仅District 7（城中村） | 自发生长的活力模式 | 发现非西方维度 |
| **Exp C** | District 1 + District 7混合 | 两种模式的联合分布 | 观察是否形成双峰分布 |

### 关键对比分析

#### 对比1：Exp A vs Exp B的Latent Space
```python
# 提取两个模型的latent表征
latent_A = extract_latent(model_A, district1_data)
latent_B = extract_latent(model_B, district7_data)

# 计算分布差异
wasserstein_dist = compute_wasserstein(latent_A, latent_B)
# 如果距离很大 → 两种活力模式本质不同
```

#### 对比2：跨区域泛化能力
```python
# 用District 1训练的模型，在District 7上测试
fid_A_on_D7 = evaluate_fid(model_A, district7_test)
fid_B_on_D1 = evaluate_fid(model_B, district1_test)

# 如果FID很高 → 两种活力模式不可互相泛化
```

#### 对比3：发现独特维度
```python
# 识别仅在Exp B中出现的latent维度
unique_dims_B = find_unique_dimensions(latent_B, latent_A)
# 这些维度可能对应"非西方活力要素"
```

---

## Latent Space分析方法

### 表征提取策略

#### 从DDPM中提取Latent
```python
def extract_latent_from_ddpm(model, x, timestep=500):
    """
    在扩散过程的中间时刻提取表征
    timestep=500: 平衡语义信息与噪声
    """
    # 前向扩散到t=500
    t = torch.full((x.shape[0],), timestep)
    noise = torch.randn_like(x)
    x_t = sqrt_alphas_cumprod[t] * x + sqrt_one_minus_alphas_cumprod[t] * noise

    # 通过U-Net编码器
    with torch.no_grad():
        latent = model.encoder(x_t, t)  # 提取bottleneck层

    return latent.flatten(start_dim=1)  # [B, latent_dim]
```

### 可视化方法

#### UMAP降维
```python
import umap

# 降维到2D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
latent_2d = reducer.fit_transform(latent_features)

# 按不同维度着色
plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
            c=vitality_scores, cmap='viridis')
plt.title('Latent Space colored by Vitality Score')
```

### 定量分析指标

#### 指标1：表征-活力相关性
```python
from scipy.stats import spearmanr

# 计算每个latent维度与活力分数的相关性
correlations = []
for dim in range(latent.shape[1]):
    corr, pval = spearmanr(latent[:, dim], vitality_scores)
    if pval < 0.05:
        correlations.append((dim, corr))

# 报告：显著相关的维度数量和强度
```

#### 指标2：聚类质量
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 基于latent进行聚类
kmeans = KMeans(n_clusters=3)  # 高/中/低活力
clusters = kmeans.fit_predict(latent)

# 评估聚类质量
silhouette = silhouette_score(latent, clusters)
# 高分 → latent空间有清晰的活力分层结构
```

#### 指标3：反事实生成（Counterfactual Generation）
```python
# 在latent空间中插值，观察活力如何变化
def interpolate_vitality(model, low_vitality_sample, high_vitality_sample):
    # 提取两个样本的latent
    z_low = extract_latent(model, low_vitality_sample)
    z_high = extract_latent(model, high_vitality_sample)

    # 线性插值
    alphas = np.linspace(0, 1, 10)
    interpolated_samples = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z_low + alpha * z_high
        x_interp = model.decode(z_interp)  # 从latent生成图像
        interpolated_samples.append(x_interp)

    return interpolated_samples
# 分析：中间态揭示了"从低活力到高活力"的转变机制
```

---

## 定性分析框架

### 与Jacobs理论对比

#### 对比维度
| Jacobs的Four Generators | DDPM可能学到的对应结构 | 验证方法 |
|------------------------|---------------------|---------|
| Mixed primary uses | POI功能混合度的latent表征 | 相关性分析 |
| Small blocks | 街道网络密度的latent表征 | 相关性分析 |
| Aged buildings | （阶段一无此数据） | 阶段二补充 |
| Concentration | 人流密度的latent表征 | 相关性分析 |

#### 关键问题
1. DDPM是否自发学习到了Jacobs的"联合分布"？
2. 是否存在Jacobs未提及但DDPM认为重要的维度？

### 发现新维度的方法

#### 步骤1：识别"异常"样本
```python
# 在latent空间中，找到与西方活力定义不一致的样本
# 例如：西方活力分数低，但latent表征显示"高活力"
anomalies = find_samples_where(
    vitality_score < threshold_low,
    latent_vitality_proxy > threshold_high
)
```

#### 步骤2：人工观察异常样本
- 查看这些样本的RGB图像（人流、POI、街道网络）
- 识别共同特征（如：夜市、宗教场所、非正规市场）

#### 步骤3：命名新维度
- 基于观察，提出假设性的"非西方活力维度"
- 例如："临时性聚集"、"文化仪式空间"、"非正规经济活力"

### 理论生成流程

```
DDPM学习 → Latent结构发现 → 与现有理论对比 → 识别差异 →
人工解释 → 提出新概念 → 回到数据验证 → 形成新理论
```

**关键原则**：
> DDPM不直接告诉我们"什么是活力"，但它揭示了"哪些特征组合在一起"。我们的任务是解释这些组合的意义。

---

## 时间线与下一步

### 短期计划（3-4周）

**Week 1**: 数据准备
- 收集胡志明市LBS数据、POI数据、OSM路网
- 完成RGB编码和空间切片
- 划分District 1和District 7数据集

**Week 2-3**: 模型训练
- 训练Control、Exp A、Exp B三个模型
- 监控generalization window
- 保存多个checkpoint

**Week 4**: 初步分析
- Latent space可视化
- 跨区域泛化测试
- 撰写初步发现报告

### 与ISM Proposal的整合

**ISM截止日期**: 2026-03-31

**整合策略**：
1. 将本框架作为ISM proposal的"详细实验设计"附录
2. 在proposal正文中强调：
   - DDPM作为"结构发现器"的认识论定位
   - District 7案例的重要性（非西方语境）
   - Generalization window作为核心方法论
3. 明确声明：这是探索性研究，不预设"对抗西方理论"的立场

### 长期展望

**如果阶段一成功**：
- 扩展到多模态输入（街景图像、建筑数据、时序数据）
- 与发小合作开发高效encoder
- 扩展到其他东南亚城市（曼谷、雅加达）

**如果阶段一失败**：
- 分析失败原因（数据质量？模型容量？编码方式？）
- 调整方法（如：改用conditional DDPM）
- 或转向其他生成模型（VAE、GAN）

---

## 总结

本实验设计的核心贡献：

1. **认识论创新**：将DDPM定位为"城市活力的结构发现器"，而非预测工具
2. **方法论创新**：使用generalization window识别本质维度，避免过拟合
3. **理论创新**：通过对比District 1和District 7，探索非西方活力模式
4. **实践价值**：为城市规划提供数据驱动但不预设理论的分析工具

**下一步行动**：
- [ ] 与导师讨论本框架的可行性
- [ ] 确认数据源的可获取性
- [ ] 开始编写数据处理pipeline
- [ ] 更新ISM proposal以整合本框架

---

**文档状态**: 完整框架 v1.0
**创建日期**: 2026-03-17
**待讨论**: 与导师确认实验设计的合理性
