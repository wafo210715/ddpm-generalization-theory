# batch-B：8 篇逐篇指标提取（2026-09-12）

提取人：协作子 Agent（research 角色）。所有行号指向 `/Users/guanhongli/Documents/wafo-base/literature/papers/` 下同名 txt 提取层。分类遵循 2026-09-10 五条裁定约定。

---

### Pan 2021

**城市 / 尺度**：成渝地区双城经济圈（CCATEC）/ 140 个区县单元（"140 districts and counties" L781）

**vitality 表征**：composite index
- **定义**：区域活力（UV）潜变量，由夜间灯光（Luojia 1-01，2019 年 8 月，DN 均值，L759）与高德 POI 计数（15 个子类求和，L765、L1108-1112）两个观测指标构成；PLS-SEM 框架下由 EnV/EcV/SV/CV 四维潜变量驱动。
- **构成**（仅 composite）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 夜间灯光 DN 均值 | Human（约定：经济活动代理，provisional） | Nighttime light data (UV1, L759) |
| POI 计数 | BE（约定） | POI data (pieces) (UV2, L765) |

**因子表**（composite 论文，关联分析 = PLS-SEM 潜变量路径，对象为四个潜构念）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| 社会活力 SV（构念：城市化率、人均社零、人均可支配收入） | 混合（Human 为主） | 内生潜变量，H3 | ↑ 直接 0.402 | "path coefficients of SV to UV (0.402)" (L1997)；Table 8 "0.402 ***" (L2033) |
| 文化活力 CV（构念：公共文化服务、中小学教师数） | 混合 | 内生潜变量，H4 | ↑ 直接 0.602 | "CV to UV (0.602)" (L1998)；Table 8 "0.602 ***" (L2031) |
| 经济活力 EcV（构念：第三产业占比、人均 GDP、建筑企业数、娱乐设施数） | 混合（含 POI 计数 EcV3） | 内生潜变量，H2 | 直接不显著；间接 ↑，总效应 0.909 | "H1, H2, H6, and H10 are statistically insignificant" (L2000-2001)；"namely 0.909" (L2009)；"total effect of 0.909" (L2044) |
| 环境活力 EnV（构念：常住人口密度、路网密度、GDP/km²、交通设施数） | 混合（EnV1=Human，EnV2/4=BE） | 外生潜变量，H1 | 直接不显著；间接 ↑，总效应 0.444；另 EnV→CV 为 ↓ −0.286 | "a total effect of 0.444 on UV" (L2057)；"EnV to CV (−0.286) was negative" (L1999)；Table 8 "0.444 ***" (L2032) |

## 附注：活力指数 UV 的 POI 成分（UV2）与自变量侧的娱乐设施（EcV3）、交通设施（EnV4）同为高德 POI 计数，存在测度同源；txt L2053 处 "CV ... 0.402" 与 Table 7/8 的 0.602 不一致，以表格为准（txt 抽取疑似笔误）。

---

### Li 2021（期刊页码为 2022 年 1 月，Published 28 December 2021）

**城市 / 尺度**：成都三环路以内 / 4434 个街道段（"street unit is defined as a segment interrupted by two consecutive intersections" L421-422；"4434" L498）

**vitality 表征**：flat（社会/经济/文化三维各自独立测度，未合成单一指数）
- **定义**：社会活力 = 出租车+共享单车 OD 点均值（"average number of OD points on weekdays and holidays" L843）；经济活力 = 大众点评商户评分（人气/质量/规模 PCA 综合分 F）加权的商户数和（"with F as the weight" L999）；文化活力 = 街道 300 m 缓冲区内五类文化设施 POI 计数（"300 m buffer zone around the street line feature" L1157-1158）。

**因子表**（flat 论文，11 个建成环境指标，多元回归 Tables 6-8，L2211/L2344/L2474；三模型 R²=0.590/0.423/0.409，L2185-2197）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Sky rate（天空可视率） | BE | IV | 社↑ / 经↓ / 文↓ | Table 6 B=516.24, t=3.88 (L2246)；Table 7 B=−3.10 (L2376)；Table 8 B=−6.41 (L2588) |
| Green rate（绿视率） | BE | IV | 社不显著 / 经↓ / 文↓ | "did not pass the house price and greening rate test" (L2172-2173)；社 B=−15.80 (L2247)；"decrease in its sky-view factor and green-view index values" (L2618-2620) |
| Building density（建筑密度） | BE | IV | 社↑ / 经↓ / 文↓ | "significant negative correlation with cultural vitality" (L2616-2617)；Table 6-8 系数区 L2246-2250 |
| Mixture of POI（14 类 POI 熵） | BE（约定） | IV | ↑（三维皆正） | Table 6 B=202.58 (L2272)；Tables 7/8 同向 |
| POI density | BE（约定） | IV | ↑（三维皆正，标准化系数最大之一：社 0.36、经 0.53、文 0.27） | "standardized coefficients of 0.38 and 0.36" (L2178-2179)；"coefficients of 0.53 and 0.14"（经济，L2606）；"0.27 and 0.26"（文化，L2615） |
| House price level（房价） | 待裁定（约定） | IV | 社不显著 / 经↑ / 文↑ | "did not pass the house price ... test"（社，L2172-2173）；Table 7 t=4.01、Table 8 t=3.62 |
| Road sparsity（300 m 内道路数） | BE | IV | 社↓ / 经不显著 / 文↑ | Table 6 B=−36.77 (L2299)；"failed the economic vitality regression model" (L2173-2174)；"an opposing effect on both" (L2669) |
| Bus convenience | BE | IV | ↑（三维皆正） | Table 6 B=20.68 (L2300)；Table 7 p=0.009；Table 8 B=0.53 (L2530) |
| Subway convenience | BE | IV | 社↓（p=0.028）/ 经不显著 / 文↑ | Table 6 B=−36.57 (L2301)；"Subway accessibility ... failed the economic vitality regression model" (L2173-2174)；Table 8 B=0.76 (L2531) |
| Crowd counting（腾讯热力图人流） | Human | IV | ↑（三维皆正；社 β=0.38 最大） | "Crowd counting and POI density have standardized coefficients of 0.38 and 0.36" (L2178-2179) |
| Work convenience（招聘企业核密度分级，就业密度代理） | BE（约定：POI 核密度 → BE） | IV | 社↑ / 经不显著 / 文↑（β=0.26） | "coefficients for POI density and work convenience were 0.27 and 0.26"（文化，L2615-2616）；经济模型 t=1.23 p=0.220（Table 7，L2344 起） |

## 附注：文化活力 DV 本身是文化 POI 计数，与自变量 POI density（全类 POI）存在部分同源；subway convenience 对社会活力显著为负是少数反直觉结果。

---

### Chen 2021

**城市 / 尺度**：9 城（新加坡、上海、北京、纽约、芝加哥、西雅图、伦敦、巴黎、柏林）/ 约 1 km × 1 km 网格（"nine cities spanning multiple continents" L342；"approx. 1 km × 1 km grids" L345）

**vitality 表征**：composite index
- **定义**：vitality score（0–100）= 五个活力指标标准化后求和再 rescale（"standardize the vitality indicators, (2) sum them up, and (3) rescale" L654-657）；tweet density 因 R² 过低被移出最终 score（"exclude tweet density in the calculation of the vitality score" L802）。
- **构成**（仅 composite）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| POI 核密度 | BE（约定） | POI density（Table 2, L683） |
| 推文密度 | Human | Tweet density（L691；最终剔除，L802） |
| 夜间灯光月均亮度（VIIRS DNB） | Human（约定，provisional） | NTL brightness（L697） |
| 人口规模（WorldPop） | Human（约定：静态存量） | Population size（L693） |
| Airbnb 房源核密度 | BE（约定：POI 类计数/核密度） | Airbnb listing density（L694-695） |

**因子表**（composite 论文，关联分析 = LightGBM 对 vitality score 及各指标的预测，形态指数为自变量）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Building density（BuD） | BE | LightGBM 特征 | 重要性最高组（POI density、NTL 模型）；方向未报告 | "indicators regarding the development intensity (e.g. building density and intersection density) play a vital role" (L905-907)；"tied strongly with building density" (L926-927) |
| Intersection density（InD） | BE | LightGBM 特征 | 重要性最高组；方向未报告 | "building density, intersection density, and block density"（POI density 模型，L924-925） |
| Block density（BlD） | BE | LightGBM 特征 | 重要性高；方向未报告 | 同上 L924-925；population/Airbnb 模型亦高（L928-929） |
| Road density（RD） | BE | LightGBM 特征 | NTL 模型重要性高；方向未报告 | "building density, average building footprint area and road density"（NTL，L926-927） |
| Average building footprint area（ABFA） | BE | LightGBM 特征 | 方向未报告 | 同上 L926-927 |
| Average block area（ABA） | BE | LightGBM 特征 | 未报告 | 全文未被单列；查 "ABA"/"average block area" 仅 Table 1/6 出现（L534-535、L1071-1085），无效应叙述 |
| Land use mixture（LUM，Shannon 熵） | BE | LightGBM 特征 | 未报告 | 同上，仅定义处出现（L497-501、L534-535），无效应叙述 |
| Prob. no pattern（路网无形态概率） | BE（形态类） | LightGBM 特征 | ↓ | "a clear gap of average vitality score between no pattern and the other categories" (L937-938)；Table 5：no pattern 均值 10.86 vs 其他 ≈19–21 (L996-1002) |
| Prob. gridiron | BE（形态类） | LightGBM 特征 | 全局不重要；高活力端 ↑（尾部效应，top-10 活力格网 gridiron 均值 89.65 最高） | "probabilities of road network categories except for no pattern are in the opposite" (L907-909)；"the gridirons surpass the counterparts much further" (L960-961)；Table 6 (L1071-1085) |
| Prob. organic | BE（形态类） | LightGBM 特征 | 全局不重要；高活力端次之（77.68） | 同上 L960-961、L1071-1085 |
| Prob. radial | BE（形态类） | LightGBM 特征 | 全局不重要；高活力端最低（57.27，尽管 BuD 最高） | "radials keep the greatest building density whereas only achieve the lowest vitality score" (L961-963) |

## 附注：LightGBM 仅输出 feature importance（分裂次数），无系数符号；方向证据只来自类别均值比较（Table 5）与 top-10 高活力格网对比（Table 6），gridiron/radial 对比直接提示非单调/阈值式尾部效应，与 DDPM 关注点最相关。增强模型 R² 0.5957→0.6227（L887-888），路网分类"small but positive"（L933）。

---

### Yue 2021

**城市 / 尺度**：上海 vs 胡志明市对比 / sub-district（街道办事处/坊级）（"Ho Chi Minh City versus Shanghai"，标题；指标均按 sub-district 计算，L347-408）

**vitality 表征**：composite index
- **定义**：projection pursuit model 将三维度（built environment / human activity / human–environment interaction）9 个指标合成为活力值（"projection pursuit model based on three dimensions" L37-38；"measured urban vitality by using the projection pursuit model" L178）。
- **构成**（仅 composite；Table 1，L329-410）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 混合土地利用（12 类熵指数） | BE | Mixed land use（L329） |
| 路网交叉口密度 | BE | Road junction density（L330-331） |
| 老建筑数量（历史保护名录） | 待裁定（约定：社区/建筑年龄类） | Aged building（L332） |
| 人口密度 | Human（约定：静态存量） | Population density（L334） |
| POI 密度 | BE（约定） | POI density（L389） |
| 居住密度（居住用地面积占比） | BE | Dwelling density（L392-394） |
| 道路连通性（道路总长/km²） | BE | Road connectivity（L396） |
| 物理隔离（高速/铁路/河流缓冲占比） | BE | Physical segregation to urban barriers（L398-403） |
|  Amenities 可用性（公交站/广场/公园密度） | BE（约定：POI 类密度） | Amenity availability（L404-408） |

**因子表**：n/a（仅制图/描述）——全文只做双城活力空间格局制图与规划叙事比较，无任何回归/相关分析；grep "regression|correlation|OLS|GWR" 全文仅出现在文献综述与描述性语句（L101、L127、L474、L944），无实证关联分析。

## 附注：Table 1 的 "Impact on vitality" 列（L338 起：8 正 1 负，physical segregation 为负）是指标设计阶段的先验方向假设，非实测效应，不应作为效应证据引用。

---

### Scepanovic 2021

**城市 / 尺度**：意大利 6 城（米兰、博洛尼亚、佛罗伦萨、都灵、巴勒莫、罗马）/ 人口普查区（district / area di censimento，平均 2.47 km²）（"six Italian cities" L15-16；"district (or area di censimento) as the unit of analysis" L518-522）

**vitality 表征**：flat
- **定义**：activity density = 移动互联网连接数按 Voronoi 多边形面积加权分摊到 district（TIM 数据，2014 年 2–10 月，典型工作日）（"We used mobile Internet activity as a proxy for urban vitality" L625；"February to October 2014" L629）。

**因子表**（flat 论文；两阶段回归第二阶段 = 6 个活力代理 → 活力，ElasticNet 系数 Table 5）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Intersection density | BE | 活力代理（第二阶段自变量） | ↑ 系数 0.497（最大） | Table 5 "intersections density 0.497" (L841-851) |
| Small parks（距小公园距离倒数） | BE | 活力代理 | ↑ 0.124 | Table 5 (L852) |
| Anisotropicity（街区各向异性） | BE | 活力代理 | ↑ 0.063 | Table 5 (L853) |
| Land use mix（Urban Atlas 三分类熵） | BE | 活力代理 | ↑ 0.019（最弱） | Table 5 (L854) |
| Block size | BE | 活力代理 | ↓ −0.118 | Table 5 (L855) |
| Building height（平均楼层数） | BE | 活力代理 | ↓ −0.152 | Table 5 (L856)；"the higher it is, ... the lower the vitality" (L874-876) |
| OSM PoI 三类得分（餐饮/交通/娱乐） | BE（约定） | 解释性 logistic 回归自变量（预测模型推断的高/低活力类别） | ↑（三类均正向预测高活力类） | "all three place categories are positively predictive of vitality" (L1143-1144) |

## 附注：第二阶段自变量是卫星影像**预测**的代理值而非实测值（R²=.37±.07，L866-868），系数方向与 Jacobs/De Nadai 预期一致但属间接证据；卫星特征直接预测活力 R²adj=.55（L886-890），leave-one-city-out 泛化在 Rome 跌至 .25（L910）。

---

### Nathansohn-Lahat 2022

**归档（质性，无量化指标）**：概念性论文，将 urban vitality 重构为"urban vitalisation"过程，案例为以色列海法 Hadar 社区居民用数字平台参与智慧城市倡议；活力仅作概念背景，全文为质性分析（"Our research was based on qualitative" L353；"We did not measure trust/distrust quantitatively" L787），无活力测度、无因子关联分析，不进入因子库。

---

### Gómez-Varo 2022

**城市 / 尺度**：巴塞罗那 Nou Barris 区（外围低收入区）/ 50 m × 50 m 格网（"Nou Barris district in Barcelona" L287-288；"50 × 50 m cell grid" L501）

**vitality 表征**：composite index
- **定义**：JANE Index 更新版——22 个建成与社会环境变量 z-score 化后按专家权重加总成 6 个条件分（concentration / functional diversity / contact opportunity / building diversity / accessibility / border vacuums），测的是"潜在活力条件"而非实测活力（"composite measure of 22 variables" L37；"JANE Index consisting of 22 variables" L388-389）。
- **构成**（仅 composite；Table 1，L551-760）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 人口密度 | Human（约定） | (1) Population density DENS（L565） |
| 商业密度（retail/km²） | BE（约定：POI 类） | (2) Commercial density COM_DENS（L615） |
| 公共设施密度 | BE（约定） | (3) Public facilities density FACIL_DENS（L621） |
| 建筑功能混合熵 | BE | (4) Building-Use Mix Indicator BuMI（L627） |
| 居住-非居住平衡 | BE | (5) Residential-NonResidential balance RNR（L635） |
| 商业与设施混合熵 | BE | (6) Commercial and Facility Mix CFMIX（L643） |
| 基础-非基础商业平衡 | BE | (7) Basic-NonBasic balance BNB（L647） |
| 交叉口密度 | BE | (8) Intersection density INTERSEC（L660） |
| 距广场/口袋公园距离 | BE | (9) PPDIST（L661） |
| 距公共 Wi-Fi 热点距离 | BE（约定：POI 类） | (10) WFDIST（L662） |
| 街道座椅有无 | BE | (11) BENCHES（L663） |
| 介数中心性（600 m） | BE | (12) Betweenness BTW（L664） |
| 建筑平均建造年份 | 待裁定（约定：社区年龄类） | (13) YEARM（L677） |
| 建造年份离散度 | 待裁定 | (14) YEARSD（L690） |
| 社区平均租金 | 待裁定（约定：房价/地价类） | (15) RENT（L698） |
| 加权公共交通距离 | BE | (16) TPDIST（L704） |
| 街道坡度 | 待裁定（地形类，约定未覆盖） | (17) INCLIN（L708） |
| 街道照明有无 | BE | (18) LIGHT（L718） |
| 30 km/h 限速区有无 | BE | (19) 30A（L719） |
| 距人行横道距离 | BE | (20) PVDIST（L720） |
| 人行道有无 | BE | (21) SWALKS（L729） |
| 距 border vacuum（大型单一功能体/大公园/地面干道等）距离 | BE | (22) BVDIST（L730） |

**因子表**：n/a（仅制图/描述）——全文为指数构建 + Nou Barris 案例制图（Results 为各条件分值空间分布描述，L925-1050），未与任何实测活力/人流数据做关联或验证分析（grep "correlation|regression|validation" 无实证分析命中）。

## 附注：测的是"potential vitality"（条件供给）而非实测活力；权重为作者设定（如 DENS 4/10、COM_DENS 与 FACIL_DENS 各 3/10，L517-544），指数含大量供给侧/有无型指标。

---

### Lee-Kang 2022

**城市 / 尺度**：首尔全市 / 424 个行政洞（dong），日频面板 2018-11-01–2019-04-30，取 6:00–21:00 活动时段（"424 administrative dongs" L475；"6 months from November 1, 2018 to April 30" L440）

**vitality 表征**：flat
- **定义**：城市活力 = 访问（活动）人口密度 person/m²，来自 SKT 通信基站浮动人口大数据（50 m 网格聚合到 dong），仅取 visiting population（"measured by the active population density (active population/area)" L455；"the visiting population was used as a surrogate variable" L467-469）。

**因子表**（flat 论文；SAC 空间面板模型，工作日/周末分别估计，Table 3 起于 L884）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| PM10 浓度（log） | 待裁定（大气环境因子，BE/Human 二分未覆盖） | 核心 IV（H1） | 工作日不显著（假设被拒绝）；周末 ↓（PM 每 +1% 活力 −0.066%） | "rejected in the weekday model and accepted in the weekend model" (L739-742)；"decreased by 0.066 %" (L742-744) |
| 地块面积 Land area（log） | BE | 控制 IV（开发程度） | 工作日 ↑（0.0191）；周末不显著（−0.0021, t=−1.01） | Table 3 工作日估计 L955；周末估计 L1038、周末 t 值 L1069 |
| 总建筑面積 Total building area（log） | BE | 控制 IV（开发程度） | ↑（工作日 0.0311 / 周末 0.0561，均显著） | "the total floor area ... positively affected urban vitality for both weekends and weekdays" (L754-756)；Table 3 |
| 居住用地占比（log） | BE | 控制 IV（土地利用） | ↓（双模型皆负显著） | "Areas with large residential and industrial areas had low level of urban vitality" (L757-759)；Table 3 weekday −0.0774 |
| 商业用地占比（log） | BE | 控制 IV | ↑（双模型皆正显著） | "areas with high commercial and business use areas had a high level of urban vitality" (L777-778)；Table 3 weekday 0.0625 / weekend 0.0798 |
| 商务（办公）用地占比（log） | BE | 控制 IV | 工作日 ↑（0.0032）/ 周末不显著（0.0007, t=0.777） | 同上文本 L777-778；Table 3：工作日估计 L963、周末估计 L1042、周末 t 值 L1073 |
| 工业用地占比（log） | BE | 控制 IV | ↓（双模型皆负显著） | L757-759；Table 3 weekday −0.0106 / weekend −0.0038 (t=−5.926) |
| 土地利用混合（熵，log） | BE | 控制 IV | ↑（双模型皆正显著） | "land-use mix ... had a statistically significant positive impact ... for both weekdays and weekends" (L829-832) |
| 地铁可达性（站点/面积，log） | BE（约定：POI 类密度） | 控制 IV | ↑（双模型） | 同上 L829-835；Table 3 weekday 0.1041 |
| 公交站可达性（log） | BE（约定） | 控制 IV | ↑（双模型） | 同上；Table 3 weekday 0.0518 / weekend 0.0815 |
| 步行道路长度/面积（log） | BE | 控制 IV | ↑（双模型） | 同上 L832-835；Table 3 weekday 0.0569 / weekend 0.0737 |
| Walk score（log） | BE | 控制 IV | ↑（每 +1%：工作日活力 +0.34%、周末 +0.50%） | "increased by 0.34 % on weekdays and 0.50 % on weekends" (L846-847) |
| 邻里公园面积占比（log） | BE | 控制 IV（公园） | ↓（显著负向，拒绝原假设） | "neighborhood parks ... were found to have a statistically significant negative impact" (L866-869) |
| 小公园面积占比（log） | BE | 控制 IV | 不显著/方向在工作日与周末间变化 | "small parks and children's parks ... were not statistically significant or had a change in the impact" (L862-865) |
| 儿童公园面积占比（log） | BE | 控制 IV | 不显著/变号 | 同上 L862-865 |
| 主题公园面积占比（log） | BE | 控制 IV | ↑（显著） | "Theme parks and urban nature parks ... had a statistically significant impact on increasing the vitality" (L859-861) |
| 城市自然公园面积占比（log） | BE | 控制 IV | ↑（显著） | 同上 L859-861 |
| 教育机构密度（log） | BE（约定：POI 类） | 控制 IV（生活 SOC） | ↓（唯一负向的 SOC） | "Educational institutions were the only ones that had a negative relationship" (L1158-1159) |
| 公共设施密度（log） | BE（约定） | 控制 IV | ↑（双模型） | "Areas with many public facilities, hospitals, and markets were found to have a high level of urban vitality on both weekends and weekdays" (L1152-1155) |
| 医院密度（log） | BE（约定） | 控制 IV | ↑（双模型） | 同上 L1152-1155 |
| 老年福利设施密度（log） | BE（约定） | 控制 IV | 不一致：工作日 ↑（0.023, t=9.03）/ 周末 ↓（−0.0188, t=−5.028） | Table 3 log(Welfare) 行 L925；工作日估计 L993；周末估计 L1057；正文未单独讨论 |
| 市场设施密度（log） | BE（约定） | 控制 IV | ↑（双模型） | L1152-1155；Table 3 weekday 0.0424 |
| 海拔（log） | 待裁定（地形类，约定未覆盖） | 控制 IV | ↓（每 +1%：工作日 −0.015%、周末 −0.073%） | "average altitude increased by 1 %, urban vitality decreased by 0.015 % on weekdays and by 0.073 % on weekends" (L1163-1166) |
| 常住人口密度（log） | Human（约定：静态存量） | 控制 IV（人口特征） | ↑（双模型） | "Resident population density and working population density ... had a positive relationship with urban vitality both on weekdays and weekends" (L1168-1170) |
| 工作人口密度（log） | Human | 控制 IV | ↑（双模型，系数最大：weekday 0.4079） | 同上 L1168-1170；Table 3 |

## 附注：PM2.5 与坡度因共线性在最终模型中剔除（"the PM2.5 and the slope were excluded from the final model" L702-703）；txt 中 Table 3 的 t 值/显著性列与变量行错位（抽取散乱），weekday PM10 的显著性以正文叙述（假设被拒绝）为准。工作日/周末效应差异（尤其 PM10 与公园）是本批最明确的"条件性效应"案例。
