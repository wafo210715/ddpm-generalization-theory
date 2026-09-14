# Batch D 逐篇提取（8 篇）

提取日期 2026-09-12。行号均为各 txt 提取层（pdftotext）行号。分类遵循 2026-09-10 用户裁定（POI→BE；静态人口存量→Human；夜间灯光→Human provisional；主观感知→Human；房价/地价、社区年龄→待裁定）。

---

### Qin 2025

**城市 / 尺度**：南京市中心城区（江南、江北，撤县设区后期市区除外）/ TAZ（交通分析小区，509 个，L164）

**vitality 表征**：composite index
- **定义**：社会、文化、经济、创新四维各自测度后用熵权法加权求和得 comprehensive vitality index（L424 "entropy weight method to weigh and sum the four vitality indicators"）；数据为 2018 年手机信令（10 月，小时级 OD）、珞珈一号夜光（2018-10-29）、AMap POI（2018 年末）
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 手机信令推算的 TAZ 人口密度（月度平均流动） | Human | Social vitality（L274） |
| 文化设施 POI 密度（科技馆、图书馆、影院等） | BE（POI 约定） | Cultural vitality（L278） |
| 珞珈一号夜间灯光亮度 | Human（夜光约定，provisional） | Economic vitality（L289） |
| 创新机构 POI 密度（高校、科研院所、创新企业） | BE（POI 约定） | Innovative vitality（L297） |

**因子表**（OLS 全局回归，DV = comprehensive vitality；另做 MGWR 看空间异质性；stepwise 从 11 个候选 UBE 指标中保留 7 个）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Number of business districts（商圈数量） | BE | IV（OLS/MGWR 自变量） | ↑ | L684 "0.349"，p=0.000***；L736 "More business districts indicate greater prosperity" |
| Road density（道路密度 km/km²） | BE | IV | ↑ | L692 "0.370"，p=0.000***；L874 "Road density is positively impacted" |
| Average number of building floors（平均楼层数） | BE | IV | ↑ | L693 "0.144"，p=0.000***；L742 "slight positive correlation with the average number of building floors" |
| Building density（原文命名，定义 = Gross floor area/Gross area，即惯例上的 FAR/容积率） | BE | IV | ↑（OLS）；MGWR 大部分区域不显著 | L701 "0.086"，p=0.025**（L705 附近）；L751 "strong positive correlation between building density and vitality"；L849 "density of bus stops and building density estimates are not statistically signiﬁcant"（MGWR） |
| TAZ area（单元面积） | BE | IV | ↑ | L702 "0.068"，p=0.012**；L872-873 "TAZ area … positively impacting comprehensive vitality" |
| Density of bus stop（公交站密度） | BE | IV | ↑（弱显著）；MGWR 不显著 | L703 "0.061"，p=0.072*；L849 同上行 |
| Distance from the city center（距市中心距离） | BE | IV | ↓ | L713 "−0.076"，p=0.010***；L867 "distance to the city center is negatively correlated with comprehensive vitality" |
| Land use mix（ Shannon 熵） | BE | 候选 IV，stepwise 剔除 | 未报告 | L304-305 列入指标系统；L898-900 stepwise 最终清单不含此项 |
| Volume rate（原文命名，定义 = Gross building footprint/Gross area，即惯例上的建筑密度/覆盖率） | BE | 候选 IV，stepwise 剔除 | 未报告 | 定义 L401-404 "Gross building footprint/Gross area"；最终清单 L898-900 不含此项 |
| Distance to the nearest subway station | BE | 候选 IV，stepwise 剔除 | 未报告 | L365/L410 列入表 2；L898-900 不含 |
| Road intersection density | BE | 候选 IV，stepwise 剔除 | 未报告 | L373 列入表 2；L898-900 不含 |
| Distance to the nearest park | BE | 候选 IV，stepwise 剔除 | 未报告 | L383/L420 列入表 2；L898-900 不含 |

## 附注：**命名反转已确认**——原文 "Building density" = 总楼面面积/总面积（L395-398，惯例是 FAR），"Volume rate" = 总建筑基底/总面积（L401-404，惯例是建筑密度/覆盖率），读表时须按公式定义对齐；全部关联均为线性（OLS/MGWR）系数，未报告非线性或阈值；MGWR 显示各因子影响存在空间异质性（带宽 44–507，L835-840）。

---

### Liu 2025

**城市 / 尺度**：南京市主城（2035 总体规划界定主城区）/ 城市道路网围合的 urban block（712 个，L162）

**vitality 表征**：composite index
- **定义**：物理建成特征（P）+ 社会功能特征（S）与主观感知（H）融合：Vblock = (Pj + Sj) × Hj（L380，txt 抽取形，括号丢失）；各维 Min-Max 归一化；以手机信令人口活动强度（GT = Num/Area，2019-04-12/13 两日）作 ground truth 验证
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| POI  Shannon 熵混合度 F1 | BE（POI 约定） | Mixed land use（L276, L288） |
| 建筑基底面积/街区面积 F2 | BE | Building density（L277） |
| Σ(建筑面积×层高)/街区面积 F3（类 FAR） | BE | Built area（L278） |
| POI 类型数 × POI 密度 F4 | BE（POI 约定） | Service perfection（L282） |
| 路网交叉口数/街区面积 F5 | BE | Traffic convenience（L283） |
| Place Pulse 2.0 "Lively" 维度 + DenseNet121 街景预测（采样点 0/1 均值） | Human（主观感知约定） | 主观活力感知 H（L324-330, 公式 2） |

**因子表**：n/a（本文只做活力测度方法构建与精度验证，未做因子→活力的关联分析；验证结果是三种测度法与手机信令 ground truth 的 Pearson 相关：物理维 r=0.8233、主观维 r=0.5434、融合 r=0.8405，L593-595）

## 附注：活力指数以主观感知分数作乘性修正项（H 直接乘在物理+功能得分上），46.35% 街区单用物理指标会错级（L580 "46.35% of the areas were inaccurately assessed"）；属"测量型"论文而非"解释型"。

---

### Liu 2022（Vital Triangle）

**城市 / 尺度**：长三角城市群 27 个城市 / 城市整体（宏观尺度）

**vitality 表征**：composite index
- **定义**：从"城市有机体"视角提出 vital triangle（growth、diversity、mobility），构建人口、经济、社会、政府、环境、外部链接 6 维 28 指标的 Urban Vitality Index（UVI），AHP 主观权重与熵权法客观权重取平均（L252-274），min-max 标准化后加权和；数据以 2019 年为主（ census 2020、建成区 2018）
- **构成**（28 指标，Table 3，L340-440 区域）：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 自然人口增长率 | Human | a1 Natural population growth rate |
| 净流入人口 | Human | a2 Net inflow population |
| 春节后返工人口指数（百度迁徙） | Human | a3 Rework population |
| 14–35 岁人口占比 | Human | a4 Proportion of young people |
| 大专以上人口占比 | Human | a5 College education or above |
| 千人专利申请量 | Human | a6 Patent application quantity |
| GDP 增速（2016–2019） | Human | b1 GDP growth rate |
| 14 行业就业 Shannon 熵 | Human | b2 Economic diversification |
| 人均社会消费品零售额 | Human | b3 Per capita retail sales |
| 单位建设土地平均夜光强度 | Human（夜光约定，provisional） | b4 Night-time economic vitality |
| 实际利用外资/GDP | Human | b5 Foreign capital utilized |
| 千人高新技术企业数 | Human | b6 High-tech enterprises |
| 6 类休闲 POI Shannon 熵 | BE（POI 约定） | c1 Diversity of public leisure space |
| 6 类休闲 POI 密度/建成区 | BE（POI 约定） | c2 Density of public leisure space |
| 23 类 POI Shannon 熵 | BE（POI 约定） | c3 Mixed land use |
| 道路密度（OSM 道路总长/建成区） | BE | c4 Road density |
| 人均财政收入 | Human | d1 Per capita fiscal revenue |
| 科教支出/GDP | Human | d2 Spending on science and education |
| 数字政府服务指数（阿里研究院） | Human | d3 Digital government services index |
| 公共管理/社保就业人员占比 | Human | d4 Proportion of social security staff |
| 年均 PM2.5 浓度（负向指标） | 待裁定（环境质量） | e1 Annual average PM2.5 |
| 空气优良天数比例 | 待裁定（环境质量） | e2 Days with good air quality |
| 建成区绿化覆盖率 | BE | e3 Greenery coverage |
| 人均公园绿地面积 | BE | e4 Per capita park green area |
| 产业链接强度（投入产出+重力模型） | Human | f1 Industrial link |
| 信息链接强度（百度指数） | Human | f2 Information link |
| 交通链接强度（列车/航班/城际巴士） | Human | f3 Transport link |
| 创新链接（合著论文中心性） | Human | f4 Innovation link |

**因子表**（指数成分外的关联分析——仅效标相关与子系统间相关，无因子级回归）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| 人均 GDP | Human | 外部效标（与 UVI 相关分析） | ↑ | L527-528 "UVI is significantly correlated with per capita GDP … R2 … all over 0.5" |
| 第三产业比重 | Human | 外部效标 | ↑ | 同上 L527-528 |
| 人均收入 | Human | 外部效标 | ↑ | 同上 L527-528 |
| 平均预期寿命 | Human | 外部效标 | ↑ | 同上 L527-528 |
| 环境维 vs 社会经济各维 | 待裁定（环境质量） | 子系统间相关 | ↓/不相关 | L448-452 "Every other subsystem, excluding environmental vitality, has a positive correlation"；"relationship between environment and external link has a high negative association" |
| 外部链接维 vs 人口/经济/社会维 | Human | 子系统间相关 | ↑（r>0.5） | L453-455 "positive correlations between external link and subsystems such as population, economy, and society are strong (>0.5)" |

## 附注：概念层明确提出密度阈值假说但未实证检验——"within a specific range, increasing population and building density may contribute to vitality … beyond a certain threshold, overcrowded space is likely to have adverse effects"（L234-242），密度因此被排除在 vital triangle 核心条件之外；28 指标权重在 2.37%–4.32% 之间，高度均质化。

---

### Shi 2023

**城市 / 尺度**：四川绵阳主城区（涪城、游仙两区 22 个街道，L284）/ 栅格-斑块（珞珈一号 NTL 130 m + Baidu 热力图 + POI 叠加，8 个夜间时段 19:00–22:30）

**vitality 表征**：composite index
- **定义**：夜间活力区（VAN）= 两期珞珈一号夜光强度与 Baidu 夜间热力图叠加 POI，自然断点法分 4 级（Grade 4 最强），按 8 个时段等级出现频次划分 VAN-1/2/3 持续型（L445-463）
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 珞珈一号夜间灯光强度（两期融合，4 级） | Human（夜光约定，provisional） | NTL remote sensing（L296-304） |
| Baidu 热力图夜间人口热力（21:00–21:30 等 8 时段） | Human | Baidu map thermal map（Table 1） |
| 8 类夜间经济 POI（餐饮、购物、休闲、娱乐、文艺、体育、住宿、旅游，39,863 条） | BE（POI 约定） | Baidu POI（L309, Table 2） |

**因子表**（多项 logistic 回归，DV = VAN 等级；自变量为距离分类变量 1=很近…5=很远，参照组 = 最远组；另用 SVM 做情景模拟）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| X1 距一级公路（primary highway）远近 | BE | IV（logistic 自变量） | ↑（越近 Level 4 持续概率越高） | L824-825 "Transportation and VAN showed a significant positive correlation (p < 0.01)"；表 4 X1=1..4 Exp(B) 13.318/25.727/13.014/7.391，sig ≤0.009 |
| X2 距二级路网远近 | BE | IV | 未报告（方向不明：表 4 中仅 X2=3/4 显著且 Exp(B)=0.459/0.275 <1，正文未解读） | 表 4（L843 起）X2 行 sig 0.022/0.001 |
| X3 距三级路网（居住区支路）远近 | BE | IV | ↑（最重要因子） | L1055-1056 "Tertiary road networks are the most important spatial elements in urban planning for VAN"；L826-830 距三级路越近 Level 4 概率越大（VAN-2 组 Exp(B) 最高 291,498,309） |
| X4 距水体远近 | 待裁定（自然要素） | IV | ↑（越近越高） | L1031-1036 "Water and VAN showed a significant positive correlation (p < 0.01)… As the distance to the water is shortened, the probability … is greatly increased" |
| X5 距机场远近 | BE | IV | 未报告（正文未讨论；表 4 中 X5 各水平 sig>0.05 居多） | 搜 "airport" 仅出现于表 3（L655 附近）与表 4 行，正文无解读 |
| X6 距火车站远近 | BE | IV | ↓（越近 Level 4 概率越低） | L1047-1053 "Railway stations and VAN showed a significant negative correlation (p < 0.01)… the probability of the duration of the Level 4 VAN is greatly reduced" |
| X7 距铁路线远近 | BE | IV | ↑（越近越高） | L1039-1045 "Railways and VAN showed a significant positive correlation (p < 0.01)" |
| X8 距中央公园（central garden）远近 | BE | IV | 未报告（正文未讨论） | 表 3（L672）与表 4 有行，正文无解读 |
| 土地利用类型（10 类分类变量） | BE | IV | 未报告（txt 表 4 仅列 X1–X8，未见该类结果） | 表 3 L641-651；搜 "land use type" 正文结果段无对应解读 |
| 土地混合度（连续变量） | BE | IV | 未报告 | 表 3 L674-676 "Degree of Land use mix"；结果段未见 |
| 土地利用/覆被变化（LUCC） | BE | IV | 未报告 | 表 3 L677-679 附近；结果段未见 |
| 坡度 | 待裁定（自然地形） | IV | 未报告 | 表 3 L680 "Slope"；结果段未见 |
| 地形起伏度 | 待裁定（自然地形） | IV | 未报告 | 表 3 L681 "Land Relief"；结果段未见 |

## 附注：活力识别成分里 POI 既用于划定 VAN 又用于夜间经济类型分析；logistic 的距离变量方向为" proximity → 活力"，与常规"值越大"编码相反（1=很近），读表时注意；SVM 部分只做更新情景下的 VAN 模拟（三级路网加密后 Level 4 VAN 增 63.99%–73.48%，L1269），不产生因子方向证据。

---

### Ruszczyk 2023

**城市 / 尺度**：孟加拉国西南部 Mongla（10.6 万人）与 Noapara（17 万人）/ 城市整体（质性案例）

**vitality 表征**：无量化表征——混合方法（survey、半结构访谈、焦点小组、storytelling 工作坊、街头剧场等参与式方法，L538-540）以 liveability 为切入点探索小城市当下与未来的 vitality，未构建任何量化活力指标或因子模型（L37-39 "mixed method, participatory exploration of liveability as a stocktaking assessment with projections for urban vitality"）。

## 附注：质性/混合方法论文，无可提取的因子+效应形态，归档跳过（若未来需要"居民感知侧的 vitality 维度"证据可回读）。

---

### Williams 2019

**城市 / 尺度**：中国成都、沈阳（模型应用与实地验证城市）/ 300 m × 300 m 居住网格（L360）

**vitality 表征**：composite index
- **定义**：以"繁荣社区需要基本设施可达"为前提，用 Hansen 重力模型对每个居住网格计算 8 类设施的 amenities score（点评数作需求权重、公共/私人设施差异权重 wk、摩擦距离 Δr=1.5 km），得分低于均值的网格再做 Moran's I 聚类，低值聚类判定为潜在鬼城（L37-41, L368-406, L618-625）；活力在此是**反向使用**（低设施可达 ≈ 低活力 ≈ 空置）
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 8 类设施可达性（餐厅、银行、KTV、杂货店、美容美发、医疗、学校、商场；Dianping/Amap 数据） | BE（POI 约定） | amenities score 成分（L37-38, L406） |
| Landscan 人口栅格（重力模型需求侧权重 Pk） | Human（静态人口存量约定） | Population in zone k（L389, L258-260 区域） |

**因子表**：n/a（本文是识别模型+验证设计：Moran's I 聚类 + 成都/沈阳实地踏勘确认 transitional/underperforming/vacant 三类（L41-42），未做任何因子→活力的关联/回归分析）

## 附注：非典型 vitality 论文——活力的操作化是"设施可达性"的反面（鬼城=低 amenities score 聚类）；房价仅在综述鬼城背景时出现（L138, L199），非变量。

---

### Li 2022（Six Dimensions）

**城市 / 尺度**：武汉三环内七区 / community block（社区/居委会，1025 个，L702）

**vitality 表征**：composite index
- **定义**：深度学习（FCN+ADE20k）分割百度街景（67,761 张，2019-05 采集），计数 6 类街景要素（行人、自行车、机动车、公交车、私家车、货车）标准化求和；与一个月 114,145 条微博签到（标准化）取平均，构成复合活力指标（L708-740）
- **构成**：

| 成分 | BE/Human | 原文名称 |
|---|---|---|
| 街景 6 类移动要素计数（pedestrians/bicycles/motor-vehicles/buses/private cars/trucks） | Human | street-view vitality（L720-726） |
| 微博签到数（一个月） | Human | Weibo check-ins（L735-738） |

**因子表**（空间滞后模型 SLM，Model 1→6 逐维加入，Model 6 adj. R²=0.583；显著变量再做 GWR）：

| 因子 | BE/Human/待裁定 | 原角色 | 效应 | 证据 |
|---|---|---|---|---|
| Population density（社区常住户籍人口/净面积） | Human（静态存量约定） | IV（SLM 自变量） | ↑ | L1447-1448 "population density and community age presented a significantly positive association with urban vitality at the confidence level of p = 0.001" |
| Community age（社区平均年龄，kriging 插值） | 待裁定（社区年龄约定） | IV | ↑ | 同上 L1447-1448 |
| Housing price（社区房价/地价代理，kriging） | 待裁定（房价地价约定） | IV | 不显著 | L1448-1449 "the influence of land price was nonsignificant (p = 0.802)" |
| FAR（容积率） | BE | IV | ↑（仅 Model 2 显著；加入设施因子后不显著） | L1450-1451 "the FAR presented a significantly positive association with urban vitality (p = 0.033)"；L1460-1461 "the influence of the FAR became nonsignificant when facility factors were added (p = 0.733)" |
| Open space rate（开敞空间率） | BE | IV | ↑ | L1403-1405 "significantly positive impacts … open space rate"；L1650 区域 "a higher proportion of open space helps induce human flow" |
| Intersection density（交叉口密度） | BE | IV | 不显著 | L1638-1639 "the effect of the intersection density on urban vitality is nonsignificant" |
| Road density（道路密度） | BE | IV | ↓ | L1451-1452 "the road density influence was significantly negative (p < 0.001)" |
| Sidewalk ratio（人行道比率） | BE | IV | ↑ | L1453-1454 "the sidewalk ratio and number of streetlights significantly promoted the level of urban vitality" |
| Streetlights（路灯道路数） | BE | IV | ↑ | 同上 L1453-1454 |
| Shopping density（购物 POI 密度，500 m 缓冲） | BE（POI 约定） | IV | ↑ | L1456-1458 "shopping density, bus stop density, and leisure density, presented significantly positive associations" |
| Bus stop density（公交站 POI 密度） | BE（POI 约定） | IV | ↑ | 同上 L1456-1458 |
| Leisure density（休闲 POI 密度） | BE（POI 约定） | IV | ↑ | 同上 L1456-1458 |
| 其他设施密度（food/life service/lodging/transit/tourist/workplace） | BE（POI 约定） | IV | 不显著 | L1458-1459 "nonsignificant influences were detected for other facilities (p > 0.1)"；注：food、life service 与 shopping 高相关，仅 shopping 入模（L1431） |
| Land use mix = residential proportion（居住用地占比，越低越混合） | BE | IV | ↓（居住占比越高活力越低） | L1459-1460 "The proportion of residential land had a significantly negative effect on urban vitality (p = 0.017)" |
| Distance to river（距长江及滨水） | BE | IV | 不显著 | L1466-1467 "nonsignificant influences … distance to the Yangtze River (p = 0.559)" |
| Distance to commercial（距购物中心） | BE | IV | 不显著 | L1467 p = 0.916 |
| Distance to park（距公园） | BE | IV | ↓（越近活力越低，即距离系数为正） | L1463 "urban vitality significantly decreased with a shorter distance to parks (p = 0.003)" |
| Distance to bus stop（距公交站） | BE | IV | ↑（越近越高） | L1464-1465 "shorter distances to bus stops (p = 0.001) … significantly lead to a higher volume of urban vitality" |
| Distance to subway（距地铁站） | BE | IV | ↑ | 同上，p = 0.003（L1464） |
| Distance to leisure（距休闲设施） | BE | IV | 不显著 | L1467 p = 0.492 |
| Distance to plaza（距广场） | BE | IV | ↑ | L1465 p = 0.022 |
| NDVI（绿度） | BE | IV | ↓ | L1471 "the NDVI had a significantly negative effect (p = 0.001) on urban vitality" |
| Integration（空间句法整合度） | BE | IV | ↑（半径 ≤1500 m 内显著，2500 m 时不显著，全局尺度复显著） | L1472 "the influence of integration was significantly positive (p = 0.022)"；L1424-1426 "integration's influence on urban vitality became nonsignificant when the analysis include a radius of 2500 m, but eventually, it became significant again for global metrics" |
| Betweenness（穿行度） | BE | IV | 不显著 | L1473-1474 "betweenness had a nonsignificant negative effect on urban vitality (p > 0.1)" |
| Urban fabric type（T-1 历史街区为参照的 7 类 dummy） | BE | IV | T-2/T-4（新建高层）↓，T-7 部分情形 ↓；T-3/T-5/T-6 不显著 | L1543-1547 "newly developed high-rise communities (consistently for T-2 and T-4, as well as for T-7 in some certain cases) significantly inhibit the formation of community vibrancy, whereas the other three types … present nonsignificant differences" |

## 附注：活力复合指标的街景成分含机动车/公交车/货车计数，并非纯"行人活动"；全部关联为线性（SLM/GWR），未报告非线性/阈值；land use mix 故意用"居住占比"替代 Shannon 熵（L887-890），方向解读需反转。

---

### Doan & Zhang 2025（综述归档）

**表征意义**：PRISMA 系统综述（941 篇筛出 191 篇，L33），不给自己的活力操作化定义，而是把文献分为"活力测量（35.6%）"与"影响因素（64.4%）"两大主题（L35），并确认定义层面无共识（L94 "is no consensus on quantifying urban vitality"）——即它对 vitality 的"表征"是文献谱系式的：活力 = 各实证论文各自的操作化的集合。

## 附注：综述不提取因子，归档；对本项目有直接论证价值的一句话：L40 "previous studies of influencing factors on urban vitality often rely on linear models, neglecting nonlinearity, interactions among factors, and causality"——支持 DDPM/connectionist 路线的研究缺口论据。
