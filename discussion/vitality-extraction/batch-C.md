# Batch C 逐篇提取记录（8 篇）

> 提取层：`/Users/guanhongli/Documents/wafo-base/literature/papers/` 下对应 txt。
> 分类约定（2026-09-10 用户裁定）：POI 计数/密度/熵 → BE；静态人口存量 → Human；夜间灯光 → Human（provisional）；主观感知分数 → Human；房价/社区年龄 → 待裁定。

---

### Wu 2023

**城市 / 尺度**：深圳 / 500 m grid（4526 个网格，L196、L201）

**vitality 表征**：flat
- **定义**：腾讯 EasyGo（微信 LBS）实时人口密度，2019-05-04 至 17 两周数据，聚合到 500 m 网格、2 小时间隔，以 20:00 划分昼/夜两个活力因变量（L238 "real-time density data are aggregated to a 500 m grid at two-hour intervals"，L242 "8 p.m. as the time node to define daytime and nighttime"）
- **构成**：n/a

**因子表**（RFR 重要性 + SLR 系数；SLR R² 昼 0.797 / 夜 0.786，L625-628 附近）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Choice（空间句法） | BE | IV（SLR 自变量） | 不显著（昼夜均不显著） | L676 "choice, lively, GPOI and RPOI are not significant in the daytime"；L678-679 夜间仅 choice、lively 不显著 |
| Integration（空间句法整合度） | BE | IV | ↑ | L680 "integration, road area ratio and road density exhibit significant positive effects" |
| Road area ratio | BE | IV | ↑ | 同上，L680 |
| Road density | BE | IV | ↑ | 同上，L680 |
| POI diversity（Shannon 熵） | BE | IV | ↑（城市功能类中影响最大） | L637 "variable exhibits the greatest influence among the categories of urban function"（指 POI diversity） |
| BPOI（商业 POI 数） | BE | IV | ↑（夜间最重要因子，RFR 重要性 0.160） | L640 "BPOI and IPOI also have notable impacts on urban vitality" |
| IPOI（工业 POI 数） | BE | IV | ↑（昼间最重要因子，0.163） | 同上，L640 |
| GPOI（绿地 POI 数） | BE | IV | 昼不显著 / 夜 ↑ | L634-635 "GPOI and RPOI have no significant effects on urban vitality in the daytime but have significant positive effects at night" |
| RPOI（居住 POI 数） | BE | IV | 昼不显著 / 夜 ↑ | 同上，L634-635 |
| PPOI（公共设施 POI 数） | BE | IV | 未报告方向（RFR 重要性低，昼 0.030） | SLR 叙述未单列方向；L589-590 RFR 排名 |
| TPOI（交通 POI 数） | BE | IV | 未报告方向 | 同上；L560-561 "GPOI, RPOI and TPOI exhibit less important effects" |
| Floor space index（FSI/容积率） | BE | IV | ↓ | L730-732 "floor space index variable has a negative effect on urban vitality"（叙述在 L726-736） |
| Ground space index（GSI/建筑覆盖率） | BE | IV | ↑ | L736-738 "ground space index variable has a significant positive effect on vitality" |
| Building continuity（建筑连续性，SVI 目标测度） | BE | IV | ↑（昼夜均显著） | L742-744 "building continuity, greenness, openness and walkability have significant effects on both daytime and nighttime vitality" + L750 "high values of building continuity, sky openness and walkability ... conducive to boosting urban vitality" |
| Greenness（绿视率，SVI 目标测度） | BE | IV | ↓ | L743 "greenness, representing space quality, displays a negative effect on vitality" |
| Openness（天空开敞度） | BE | IV | ↑ | L750（同上引文） |
| Walkability（人行道占比） | BE | IV | ↑ | L750（同上引文） |
| Beautiful（主观感知） | Human | IV | ↓（反直觉，作者归因于功能单一） | L752 "Beautiful has a negative effect on urban vitality" |
| Boring（主观感知） | Human | IV | ↓ | L691-692 "significant negative relationship between boring and vitality" |
| Lively（主观感知） | Human | IV | 不显著（昼夜均不显著） | L676、L678-679 |
| Safe（主观感知） | Human | IV | ↑（夜间效应更大） | L693-695 "safe variable shows a positive relation with the vitality, and the effect is greater at night" |
| Wealthy（主观感知） | Human | IV | ↑ | L695-698 "wealthy variable has a positive effect on urban vitality" |

## 附注：主观感知（Place Pulse 类）与客观 SVI 测度被原文混在 "human perception" 一类；按约定拆分后 beautiful 的负效应是少数显著的感知因子。FSI↓ 与 Jiang 2024 的 FSI↑ 直接冲突，值得在组装时对照。

---

### Jiang 2024

**城市 / 尺度**：北京（六环内七个区）/ TAZ（L55 "Taking the Beijing Sixth Ring"，L456 "normalized by dividing them by the area of TAZ"）

**vitality 表征**：composite index
- **定义**：三类数据源各自除以 TAZ 面积归一后做 PCA（特征值>1、累计贡献 ≥90% 的成分），得综合活力得分；分工作日/周末两套（L455-458 "we applied the PCA method to quantify urban vitality. Composite indicators with eigenvalues exceeding 1..."）
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 珞珈一号夜间灯光（夜经济代理） | Human（provisional） | "nocturnal and economic vitality employing LUOJIA-01 nighttime lighting data"（L440-441） |
| 手机信令人口数据（24h 聚合） | Human | "demographic data ... derived from cell phone signaling"（L442-443） |
| OD 出行流数据 | Human | "OD flow data"（L441-442） |

**因子表**（OLS + MGWR；形态变量为 IV，活力综合得分为 DV；工作日/周末分别建模）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| MBH（平均建筑高度） | BE | IV | ↑ | L1309-1312 "higher building height ... correspond to heightened urban vitality" |
| MBA（平均建筑面积） | BE | IV | ↑ | L1309-1312（同上句，"larger building area"） |
| MBD（平均建筑间距） | BE | IV | ↓ | L1326-1328 "longer building distances ... associated with lower urban vitality"；L1369-1372 |
| GSI（建筑覆盖率） | BE | IV | ↓（100% TAZ，周末/工作日一致；与 Ye 2018 相反） | L1377-1379 "impact of GSI on weekend and workday vitality is consistently and significantly negative" |
| FSI（容积率） | BE | IV | ↑（100% TAZ） | L1398-1400 "FSI exhibits a positive influence on urban vitality" |
| RD（路网密度） | BE | IV | ↑（99.6% TAZ 正，0.4% 负） | L1314-1317 "99.6% of TAZs experience a positive impact, 0.4% ... negative" |
| TFD（公交设施密度） | BE | IV | ↑ | L1309-1312（"increased transportation facility density"）；L1422-1428 |
| IFD（街道基础设施密度） | BE | IV | ↑（88.3%/63.1% TAZ 为正） | L1318-1320 "impacts of MBD, IFD, LUM, and MGD ... positive for 66.0%, 88.3%..."；L1431-1436 |
| SHD（商业设施密度） | BE | IV | 工作日 ↓（100% TAZ 负）/ 周末不显著 | L1334-1338 "SHD and GED ... significant influence on weekday vitality ... consistent negative impact"；L1438-1440 |
| LUM（土地利用混合） | BE | IV | ↑（100% TAZ，周末/工作日均正） | L1455-1458 "parameter estimates of LUM exert a positive impact on urban vitality, influencing 100% of the TAZs" |
| MGD（平均绿地距离） | BE | IV | ↑ 为主（周末 75.7% 正，工作日全部 TAZ 正；越远活力越高） | L1318-1320；L1483-1486 "The farther the average green distance, the higher the urban vitality" |
| LGPI（最大绿地斑块占比） | BE | IV | ↓（100% TAZ） | L1487-1490 "significant negative influence of the proportion of the largest green patch on all TAZs" |
| GED（绿地边缘密度） | BE | IV | 工作日 ↓ / 周末不显著 | L1334-1338（同 SHD 引文）；L1326-1328 "more intricate urban green space boundaries are associated with lower urban vitality" |

## 附注：活力指数三个成分按约定全部落 Human 侧，本质上是「三个 Human 代理的 PCA」去解释 BE 形态；GSI↓ 与 FSI↑ 并存（Spacematrix 两轴方向相反）是本文与 Ye 2018 争议点（L1379-1381）。

---

### Chen 2024

**城市 / 尺度**：旧金山 / census tract（人口变量，N=241，L293）+ block group（Flickr 照片与 UGS 可达性聚合，L509-510、L558 "distribution of Flickr images based on the blocks unit"）

**vitality 表征**：flat
- **定义**：IUNV（Implied Urban Nature Vitality）= 2013–2022 年 Flickr 地理标记照片（清洗后 3434 张）按空间单元的照片计数；另用 Google Vision API 标签层次聚类得到 12 个主题类并人工归为活力类型作辅助解读（L547-552 "conclusions based on the quantity of photographs associated with different parks"，L758-768）
- **构成**：n/a

**因子表**（Pearson 相关 + 三层 OLS + SEM；Moran's I = −0.0136 近随机，L793-795）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| UGS 可达性（1000 m 服务覆盖） | BE | IV（宏观层） | ↑（宏观层中最显著） | L983-984 "accessibility demonstrated the most significant positive association with IUNV"；L1006-1010 |
| Land use entropy | BE | IV（宏观层） | ↓（与前人文献相反） | L984-985 "land use entropy, which exhibited a negative relationship"；L994 "negative impact of land use entropy" |
| Road density | BE | IV（宏观层） | 未报告方向 | 摘要说其为 dominant 变量之一（L58-59）但未给方向；Table 3 系数列在 txt 中打散不可对位（L1100-1130） |
| Population density | Human | IV（宏观层） | 未报告方向 | 同上；注：L748-749 "population density and road density were found to have a positive association" 是 IV–IV 相关，不是与 IUNV 的效应 |
| Age<17（17 岁以下人口占比） | Human | IV（宏观层） | ↑ | L987-989 "areas with more teenagers tended to have higher IUNV levels" |
| Age>60 / 65+（老龄人口） | Human | IV | 未报告方向（仅暗示老龄化社区 IUNV 较低） | L987-988 "Compared with aging communities..."（对照措辞，无系数方向） |
| 教育（25+ 无高中文凭人口） | Human | IV（census） | 未报告（表述模糊） | L730-734 "regions with lower high school diploma achievement correlate with IUNV"——低教育人口占比与 IUNV 相关但未给明确方向 |
| 贫困（贫困线 150% 以下人口） | Human | IV（census） | 未报告 | 仅列入变量（L299-301），结果未单独叙述 |
| 族裔构成（黑人/西裔/亚裔/原住民） | Human | IV（census） | 未报告 | 同上（L301-303） |
| Tree（街景树木占比） | BE | IV（街景微观层） | ↑（与 sculpture 并为最强街景因子） | L778-780 "sculptures and trees were the two most powerful street view features affecting IUNV"；"IUNV tended to be higher in places with higher proportions of greenery" |
| Sculpture（街景雕塑） | BE | IV | ↑ | L784-786 "sculptures ... significantly increased IUNV in the areas where they appeared" |
| Person（街景中行人占比） | BE | IV | ↓（拥挤降低感知活力） | L788-789 "presence of people negatively correlated with IUNV across all models" |
| Wall / Fence（街景墙/围栏） | BE | IV | ↓ | L790-791 "Walls and fences negatively influenced activity willingness" |
| Chair（街景座椅） | BE | IV | ↑ | L791-792 "chairs had a positive correlation" |
| 其余街景要素（sidewalk, street light, grass, sky, water, earth, fountain, mountain, bridge） | BE | IV | 未报告/不显著 | Table 3 该列 p 值大多 >0.05（L1100-1160，txt 打散），叙述未单列方向 |

## 附注：DV 是 Flickr 照片计数（用户生成内容活跃度），本质是 Human 侧信号被用来当「自然空间活力」；人口/贫困/教育等 census 变量只做进模型未逐一报告方向。Table 3 在 txt 里严重打散，未对位的系数一律记「未报告」。

---

### Collins 2024

**城市 / 尺度**：意大利 7 城（Milan, Rome, Turin, Naples, Venice, Palermo, Bari）/ TIM 运营商不规则网格（cell 大小随基站密度变化，L145-147、L151-156）

**vitality 表征**：flat
- **定义**：CDR 通话详单（TIM Big Data Challenge 2015，2015-02-28 至 04-30）的用户活动量作为 vibrancy 代理，15 分钟粒度；分性别计算，核心 DV 是**男女活力差 = 女性 − 男性**（L191 "We calculate gender differences by subtracting the male value from the female"，L189-190 "As a proxy measure for urban vibrancy, we use CDR data"）
- **构成**：n/a

**因子表**（Kendall τ 相关 + SAR 空间滞后模型；注意：效应对象是**男女活力差**，不是活力水平本身）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Third place density（第三场所密度） | BE | IV | ↑（最强且最一致的效应，正间接效应） | L880-881 "the strongest effect ... was the positive indirect effect of the third place density" |
| POI density | BE | IV | ↓（显著负间接效应，四大城市一致） | L871-872 "significant negative indirect effect between male-female differences and the density of 'Points of Interest'" |
| Building density | BE | IV | ↑（Table 1 τ 大多为显著正） | L408-409 "except for a few instances, all of our results were positive"；Table 1 τ 值 L430-450 |
| Highway density | BE | IV | 未报告方向（仅聚合模型显著） | L876-877 "highway density was only significant in the aggregate model"（方向未述） |
| Road/cycling/walking network length（密度） | BE | IV | ↑（Table 1 大多显著正） | L408-409；Table 1 |
| Cycling & walking intersection density | BE | IV | ↑（显著正间接效应） | L875-876 "Both cycling and walking intersection density were significant with positive indirect effects" |
| Road intersection density | BE | IV | ↓（不显著） | L876 "for road intersections, though there was a negative indirect effect, this was not significant" |
| POI / building / highway / third-place diversity（Shannon） | BE | IV | 不显著（diversity 类整体不显著） | L877-878 "diversity metrics generally were not significant" |
| Organized activity & outdoor third place diversity | BE | IV | 显著但方向未报告 | L878-879 "'organised activity' and 'outdoor' third place categories were significant" |
| 其余 third place 子类（eating-drinking / commercial venue / commercial service）密度与多样性 | BE | IV | ↑ 为主（Table 1 大多显著正） | L408-409；Table 1（L430-470，含少量负 τ 如 Milano 聚合 −0.10/−0.04） |

## 附注：效应的 DV 是**性别活力差**而非活力水平——「POI 密度 ↑ → 男女差 ↓」不等于「POI 密度与活力负相关」；组装时需决定这篇的因子是按「与活力水平的关联」还是「与性别差的关联」入表，本篇 txt 未报告对活力水平本身的回归。

---

### Dogan & Lee 2024

**城市 / 尺度**：首尔 / 500 m × 500 m grid（1881 个单元，L511-512）

**vitality 表征**：composite index
- **定义**：10 个活力变量先归一到 0–100，再用因子分析（FA）按载荷合成为 3 个复合指标——intensity（强度）、diversity（多样性）、consistency（时间一致性）（L548-551 "created 10 vitality variables"，L560-562 "used Factor Analysis (FA) to reduce 10 variables into three indicators"）
- **构成**（Table 5，L952-975）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 人口流入密度（总量） | Human | Population inflow density (total) |
| 经济活动密度（信用卡交易额） | Human | Economic activity density (amount) |
| POI 密度 | **BE**（约定） | POI density |
| 博客点评密度 | Human | Blog review density |
| Airbnb 房源密度 | Human | Airbnb listing density |
| 人口流入熵（年龄） | Human | Entropy of population flow (age) |
| 人口流入熵（来源地） | Human | Entropy of population flow (space-origin) |
| POI 多样性（Simpson，13 类） | **BE**（约定） | POI diversity |
| 人口流入熵（日内时段） | Human | Entropy of population flow (daily hours) |
| 人口流入熵（周内日） | Human | Entropy of population flow (week days) |

**因子表**（22 个城市物理复杂度 IV → 3 个活力指标分别做线性回归 + MINE 非线性统计 MIC-R²）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Ent_BCS（建筑覆盖尺寸熵） | BE | IV | ↑（三个指标均正） | L1072-1075 "entropy of building coverage size (Ent_BCS) has positive associations with urban vitality in all three models" |
| Tot_NB（街区总数/小街区） | BE | IV | ↑ intensity & consistency + **非线性**（先增后衰减，MIC-R² 高） | L1065-1067 "total number of blocks (Tot_NB) has positive associations with intensity and consistency"；L1439-1441 "Increasing the number of blocks initially enhances vitality, but beyond a certain point, the influence diminishes" |
| Ent_BH（建筑高度熵） | BE | IV | ↑ diversity | L1067-1068 "entropy of building height (Ent_BH) significantly boosts the diversity of urban vitality" |
| Var_ED（高差方差/坡度） | BE | IV | ↓（三个模型均负） | L1069-1070 "elevation difference (Var_ED) has a negative impact in all models" |
| Ent_SO（街道朝向熵） | BE | IV | ↑ diversity；intensity/consistency 不显著 | L1080-1082 "street orientation entropy (Ent_SO) increases vitality's diversity but does not significantly impact vitality's intensity or consistency" |
| Hetr_BO（建筑-开敞异质性，3D isovist） | BE | IV | ↑ intensity & diversity，↓ consistency | L1082-1084 "Hetr_BO ... increases intensity and diversity while decreasing consistency" |
| Access（全城可达性） | BE | IV | ↑ intensity & diversity，↓ consistency | L1455-1459 "accessibility (Access) ... a positive indicator of the intensity and diversity ... However, high Access has a negative association with the consistency" |
| Mean_SL（平均街段长度） | BE | IV | ↓ diversity（街段越短多样性越高） | L1463-1465 "Neighborhoods with shorter streets ... have more diverse vitality" |
| Var_SL（街段长度方差） | BE | IV | ↑ diversity（层级化街长多样性） | L1465-1468 "not only shorter streets but also hierarchical diversity of street lengths helps urban vitality" |
| Mean_BC（平均建筑凸度） | BE | IV | ↑ consistency + 非线性；其余不显著 | L1471-1474 "only Mean_BC demonstrates a positive association with vitality's consistency"；L1444-1445 "Mean_BC has a more nonlinear association with vitality's consistency" |
| Mean_BSq（平均建筑方正度） | BE | IV | 不显著 | L1471-1474 "neither significantly influences other dimensions"（Mean_BSq 句） |
| Tot_CS（尽端路数量） | BE | IV | ↓ intensity | L1477-1478 "number of cul-de-sacs (Tot_CS) decreases the intensity" |
| Tot_FW（十字交叉口数量） | BE | IV | ↑ intensity | L1478-1479 "number of four-way street intersections (Tot_FW) increases the intensity" |
| Mean_BE（平均建筑边数/形态不规则度） | BE | IV | ↓ intensity | L1416-1418 "Mean_BE can sometimes decrease urban vitality, with lower vitality intensity in areas with more abundant irregular building surfaces" |
| 其余（Uni_DB、Uni_DI、Var_BS、Tot_BC、Area_BC、Mean_SS 等） | BE | IV | 混合/未逐一报告 | Table 7（L1076-1120，txt 系数列打散）；叙述只给出上述变量的方向；Table 7 含各变量 MIC-R²（非线性强度）但未全部解读 |

## 附注：活力指数把 POI 密度/多样性（按约定 BE）与 Human 流量类变量混进同一 FA 复合指标——指数本身是 BE+Human 混合体；本文是 batch 中少数明确做非线性识别（MINE/MIC-R²）并发现 Tot_NB 递减型非线性的论文。

---

### Wang 2025

**城市 / 尺度**：深圳 / 300 m × 300 m grid（L297 "employs a 300 m × 300 m grid as the basic spatial unit"），结果汇总到 community

**vitality 表征**：composite index
- **定义**：改进高斯 2SFCA 可达性指数——供给端 Sj = 大众点评 POI 评论数，需求端 Dk = 居住区人口，d0 = 1000 m 路网距离；9 类设施分别算 Ai 后等权聚合，自然断点分 4 级活力（L399-404 "Sj denotes the number of POI reviews; d0 represents 1000 meters"，L440-441 "results for each POI category are aggregated using an equal weighting approach"）
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 9 类大众点评设施 POI 评论数（供给端 Sj：交通、酒店、餐饮、生活服务、商场、教育、医疗、美容、景点） | BE（POI 派生，约定） | Table 2 九类（L251-284）；评论数作供给权重 |
| 居住区人口（需求端 Dk，含人口加权的距离衰减） | Human | "Dk denotes the population number of each demand unit k"（L399） |

**因子表**：

n/a（仅制图/描述）——全文只做可达性指数计算 + Moran's I / Getis-Ord G 空间自相关与聚类验证（L450-465），识别 99 个高活力社区（L20-21），未对任何外部因子做关联/回归分析（grep "regression|correlat|independent variable" 仅命中空间自相关与文献综述）。

## 附注：指数供给端用「评论数」而非 POI 计数，属于 POI 派生的 BE 代理 + Human 需求人口的供需比；零评论 POI 被剔除（L350-353），可能系统性低估低评论区的活力。

---

### Cai 2025

**城市 / 尺度**：北京（中心城区）/ block（街区；人口密度先 100 m 网格再聚合到街区，L1036-1038 "aggregate the population density at the block scale"）

**vitality 表征**：composite index
- **定义**：6 维度 14 指标按预设方向做 z-score 加权合成维度值（公式 6-11，L1201-1213），再用 SOFM 神经网络把街区聚成 4 类活力簇；维度权重为作者设定的固定分数（如 C = 0.4·zPD + 0.3·zCD + 0.3·zPFD）
- **构成**（Table 1，L900-930；"Impact on Vitality" 列为原文构建指数时的**预设方向**，非估计结果）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 人口密度（WorldPop 类，2021） | Human（静态人口存量） | Population density (PD)，预设 + |
| 商业密度（高德 POI） | BE | Commercial density (CD)，预设 + |
| 公共设施密度（高德 POI） | BE | Public facilities density (PFD)，预设 + |
| 建筑功能混合 | BE | Building facilities mix (BFM)，预设 + |
| 居住-非居住平衡 | BE | Residential–non-residential balance (RNR)，预设 + |
| 商业设施混合 | BE | Commercial facilities mix (CFM)，预设 + |
| 基本-非基本商业平衡 | BE | Basic–nonbasic commercial facilities balance (BNB)，预设 + |
| 交叉口密度 | BE | Intersection density (ID)，预设 + |
| 开敞度（百度街景照片） | BE | Openness (O)，预设 − |
| 建筑平均年龄（链家） | **待裁定**（建筑年龄） | Building average age (BAA)，预设 − |
| 建筑年龄多样性（链家） | **待裁定**（建筑年龄派生） | Diversity of building age (DBA)，预设 + |
| 平均房价（链家） | **待裁定**（房价） | Average room rate (ARR)，预设 − |
| 距公共交通距离 | BE | Distance to public transportation (DPT)，预设 − |
| 距铁路与大型公园距离（边界真空） | BE | Distance to railroads and large parks (DRLP)，预设 + |

**因子表**：

n/a（仅制图/描述/聚类）——SOFM 只对上述 6 个维度做聚类与空间格局描述，无外部自变量关联分析。仅有的「效应形态」信息是空间格局描述：(1) 五个维度从核心向边缘递减、建筑多样性反向（L29-31 "decreased from the core area to the periphery, except for building diversity"）；(2) **15 km 半径阈值**——簇 0/1/2 集中于距市中心 15 km 内，与北京总规圈层一致（L33-34 "a 15 km radius serves as a significant threshold"，L2585-2589）。

## 附注：指数构建时已把方向当先验写进权重公式（ARR、BAA 取负号，O 取负号），所以本文不存在「估计出的效应」；房价（ARR）、建筑年龄（BAA/DBA）按约定列待裁定。15 km 是空间阈值描述，非剂量-反应阈值。

---

### Tan 2026

**城市 / 尺度**：深圳 / 500 m grid（6714 个有效网格，L486；行政区尺度另做描述，L431）

**vitality 表征**：flat
- **定义**：百度慧眼人口热力图，2024-03-11 至 03-24 共 14 天（排除节假日），逐小时采样调用百度地图 SDK 的去重设备数，14 天平均后按 500 m 网格聚合为活力强度（L466-469 "derived from the sum of unique device counts invoking the Baidu Maps Location SDK"；"Average activity records across all 14 days per 500m grid cell"）
- **构成**：n/a

**因子表**（RF + SHAP 全局/局部重要性 + 双因子 PDP + ISM/BN；全样本 RF R² = 70.7%，L1118-1119）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| FAR（容积率） | BE | IV | ↑（SHAP=104.89，第一重要）+ **阈值[0.25]**（局部重要性分析显示低-中 FAR 不增效甚至抑制；其他设施因子需 FAR>0.25 才激活协同增益） | L1100-1102 "the three most influential predictors ... FAR (SHAP = 104.89)"；L1357-1363 "relationship between FAR and urban vitality revealed a distinct threshold effect"；L1411-1413 "require FAR beyond 0.25 to trigger significant vitality gains" |
| Global integration（全局整合度） | BE | IV | ↑（30.26，第二） | L1101-1102；L1320-1322 "global integration (6.17 %) and global choice ... positively correlated" |
| Local integration / Global choice / Local choice | BE | IV | ↑（可达性类全部为正） | L1323-1325 "All four factors within the accessibility category exert a positive influence"（四因子指 integration/choice 组） |
| Business residence（商住设施数） | BE | IV | ↑（10.03，第三） | L1102-1103 "the number of business residence facilities (10.03), indicating their enhancement roles" |
| Life service（生活服务设施数） | BE | IV | ↓（42.12%，累计最大但方向为负；作者归因于过饱和/同质化） | L1104-1105 "number of life service facilities (206.44) ... showed inverse associations with vitality"；L1526-1527 "oversaturation and homogeneity" |
| Food（餐饮设施数） | BE | IV | ↓（15.91） | L1104-1105（同上句 "food facilities (15.91)"） |
| Road density | BE | IV | ↓ | L1107-1108 "Additional negatively correlated variables comprised road density and distance to the nearest subway station" |
| Distance to subway | BE | IV | ↓（离地铁越远活力越低——负系数即近地铁活力高） | L1107-1108（同上句）；Table 3 方向列 Negative |
| Distance to medical facility / school | BE | IV | ↑（距离越远活力越高的正贡献，作者列为 positive contributors） | L1103-1104 "Other notable positive contributors included distance to the nearest medical facilities and schools" |
| Work place / Leisure / School / Tourist attraction（设施数） | BE | IV | ↑ | Table 3 方向列（L1231-1249，txt 抽取形，依次为 Positive/Positive/Positive/Positive） |
| Lodging / Shopping / Medical facility / Bus stop / Subway station（设施数） | BE | IV | ↓ | Table 3 方向列（同上，Negative） |
| Mixing degree（功能混合度） | BE | IV | ↓ | Table 3 方向列（Negative）；注：BN 叙述称活力主要由 functional mixing 塑造（L41-43），方向列与因果叙述存在张力 |
| Housing price（房价） | **待裁定** | IV | ↓ | L1344-1345 "both factors in the community attributes category demonstrate negative effects on urban vitality" |
| Community age（社区年龄） | **待裁定** | IV | ↓ | 同上，L1344-1345 |
| Walkability（视觉感知） | BE | IV | ↑（0.70%） | L1352-1353 "walkability (0.70 %) ... positively affect urban vitality" |
| Sky（视觉感知-天空占比） | BE | IV | ↓（0.94%） | L1353-1354 "sky (0.94 %) ... exert negative influences" |
| Vegetation / Road（视觉感知） | BE | IV | ↑ | Table 3 方向列（Positive） |
| Enclosure degree / Facility visibility（视觉感知） | BE | IV | ↓ | Table 3 方向列（Negative） |
| Safe（情绪感知） | Human | IV | ↑（0.40%） | L1352-1353 "safe (0.40 %) positively affect urban vitality" |
| Wealthy（情绪感知） | Human | IV | ↓（0.31%） | L1353-1354 "wealthy (0.31 %) exert negative influences" |
| Beautiful / Boring / Depressing / Lively（情绪感知） | Human | IV | ↑（方向列均 Positive；boring/depressing 为正反直觉） | Table 3 方向列（L1303-1324，txt 抽取形） |

## 附注：(1) 全文的亮点是 **FAR 阈值[0.25]**——多因子的活力增益需 FAR 越过 0.25 才激活（PDP 交互），直接对应 DDPM 想看的阈值效应。(2) 方向列在 txt 中与变量名跨行打散，Beautiful/Boring/Depressing/Lively 四个的正方向依据列对位、未经叙述复核（叙述只确认 safe+、wealthy−、walkability+、sky−），组装时如需引用这四个建议回 PDF 核对。(3) Mixing degree 方向列记负，但 BN 因果叙述以 functional mixing 为主要驱动，二者口径不同（SHAP 平均方向 vs 因果路径）。
