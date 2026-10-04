# Future research（基准，持续维护）

与 `vitality_stance.md` 同级。那份记"活力是什么"，这份记"之后要做什么、用什么做、数据从哪来"。
改动规则同 stance：只记我本人说过、确认过的看法，注明日期和原话；改的时候保留旧文和日期。标 **（agent 整理，待确认）** 的段落是对话中 agent 的整理，我确认后才去掉标记。

---

## 1. 技术路线

- **表征时空间 → 理解时空间 → 推理时空间。** 「我主张从'表征时空间'，'理解时空间'，'推理时空间'的技术路线来应对城市这个复杂系统和城市活力这个复杂课题，而不是把城市当做一个动态的多目标优化 - a problem that can be solved」（2026-10-04，Chen 2024 / Gu 2027 的 annotated bibliography）
- **系统视角下，时空分析必不可少；但没有数据就都不行。** 「时空间分析，虽然要做，说是要做，但是没有数据就都是不行的。然后我们能采集到的数据的能力也是有限的」（2026-10-04）
- **问题导向：特定场景需要特定方法。** 「我们特定的场景肯定会需要特定的方法，不可能 anything 就可以解决的」（2026-10-04，谈 JEPA-Anything）
- **LLM / VLM 的 spatial reasoning 是 baseline，不排斥；世界模型去探索更多 intelligence 层面的可能。** 硕士时与导师的 LLM Planning Agent：在地图上选几个点，让大语言模型按地名爬取相关政策和环境保护措施，再给一个城市开发计划，让它评估最好的选址。「这一个复杂任务是可以通过语义层面来解决」，适合平台式、human in the loop、conversation workshop。VLM 看地图、读 vector 和地名、采集数据，可以提高 spatial reasoning，也能用来解决问题；「但正是因为这个东西是工程上好实现的。所以我认为它是工程师会做的事情，而不是科学家会做的事情。我希望用世界模型这个东西来挑战……挑战现有的 Digital Twin 的 Paradigm 之类的」；「大语言模型的那种 Spatial Reasoning 是可以作为我们的 baseline 的，我们并没有排斥这个东西」。硕导写下的终极 research question：有一个东西能做 spatial intelligence（2026-10-04）
  - 论文：Carranza, E. R., Huang, S.-Y., & Li, G. (2025). LLM Planning Agents: Exploring the potential and challenges of large language model agents in urban design and planning. *CAADRIA 2025*, vol. 1, 223–232. https://doi.org/10.52842/conf.caadria.2025.1.223（wafo-base `literature/papers/Carranza-2025-LLM-Planning-Agents.*`；笔记 `research-vault/2-fleeting/papers/Carranza-2025-LLM-Planning-Agents.md`）。摘要自述的局限包括 "limited spatial awareness"。
- **不照搬现有架构，自己研发。** 「我并没有想直接采用之前的架构，而是想自己研发新的。所以我也是我为什么觉得，啊我好想文献综述，去读那些技术论文」（2026-10-04）
- **路线顺序：扩散模型的认知理论 → 世界模型 → 心智模型。** 有几本关于心智模型（human intelligence 是什么）的书还没读，也还没进书单，「因为你本能的会觉得这跟 urban vitality、urban informatics 距离有点远」；希望它们启发 spatial intelligence 的研究（2026-10-04）。不急；心智模型追求的也是 intelligence，暂时的预期是「里面追求的路径可以被世界模型追求 intelligence 所学习」（2026-10-04）
- **当前研究结束后，尽快开始做世界模型（JEPA）。** 「在我们这个任务结束之后，我想赶紧开始做世界模型」；一边做认识论的技术论文，一边把 future research 想清楚，「本该如此」（2026-10-04）

## 1b. 心智模型书单（还没读；记下为什么选它）

追求：城市研究里的 ethnography、urban science、归纳和演绎，都是人的思维方式；希望学习城市研究者的思维方式并用到 spatial intelligence 里。「但是我并不是想用这么一个模拟城市研究者的思维的一个机器或者模型来取代」；具体用途以后再说，这只是一种追求（2026-10-04）。

| 书 | 为什么选（用户，2026-10-04） | agent 补充（待确认） |
|---|---|---|
| 侯世达（Douglas Hofstadter）& 流动性类比研究小组〈概念与类比：模拟人类思维基本机制的灵动计算架构〉（刘林澍、魏军译；原书 *Fluid Concepts and Creative Analogies*，1995——原书名与年份未在页面上核实） | 学习城市研究者的思维方式 | 书的主张是类比即流动概念之间的转换，是创造性的核心机制（微信读书简介）。`AGENTS.md` 记录的用户学习习惯里也有"用自己过去的工作做类比来猜" |
| Shaun Gallagher, *Enactivist Interventions: Rethinking the Mind*（Oxford University Press, 2017），中译《生成主义干预：重新定义心智》 | 同上；注意 enactivism 是认知科学的概念，和生成式（generative）范式不是一回事 | enactivism：心智在身体与环境的行动中生成（4E 认知）。与 §2.1 的"多 actor 在环境中响应"、以及"先研究响应再谈状态"（JEPA 会话 41c7607b）可能相通 |
| Parr, Pezzulo & Friston〈主动推理：心智、大脑与行为的自由能原理〉（原书 *Active Inference*，MIT Press, 2022——作者与出版信息未核实） | 联想到 LeCun 的世界模型推理基于能量：机器人想怎样最省力、避免挨饿、身心放松 | LeCun 里这是两件事：世界模型的 energy F(x,y) 衡量 x 与 y 是否相容（LeCun 2022 L686–688）；"痛、乐、饿"是 Intrinsic Cost 模块（L213–214、L267–272）。LeCun 提到 F "technically ... a zero-temperature free energy"（L800），是与 Friston 自由能的一个接口 |

## 2. 想解决的问题

每个问题同时写：要回答什么、需要什么数据、现在能拿到什么。

### 2.1 政策发布后，不同 actor 如何反应（世界模型）

- 「我最早就是想建模，Policy 发布以后，不同的 actor 是如何反应的。然后这就可以通过世界模型来学习这样的建模的数据。当然数据采集也会是非常严峻的挑战」（2026-10-04）
- **结构：不是第一人称的单个 actor，而是多 actor。** 现有世界模型里的 actor 像机器人，第一人称地观察、预测自己的世界再做出动作。我要的是：**源头 = policy + 环境类数据（世界的表征）**；之后在多少个 time step 里、对哪些 actor 产生了哪些影响，要从 actor 那边采集数据。各 actor 处在不同的时空间维度，这与数据采集（高频 / 低频的 Digital Twin）和数据本身的 scale 有关。「它不是单线的，它不是单独的 actor，它就是多 actor」（2026-10-04）
- **开放：一定需要世界模型吗？** 「一定会需要世界模型吗？我想解决这个问题」（2026-10-04）
- **需要的数据（agent 整理，待确认）**：政策事件的时间与范围；之后各类 actor（居民、商户、摊贩、开发商、政府部门）行为或结果的时间序列；同一地点政策前后的对照。
- **现在能拿到的（agent 整理，待确认）**：HCMC 现有数据基本是单一时间点（`vitality_stance.md` §5）；手机信令、人流没有可用来源（`archive/discussion-2026-09/hcmc_data_acquisition.md` §5）。有时间维度的候选：GAIA 不透水面年份 1985–2018、官方地价表的历次版本（2026 起按合并后的大 HCMC 新表）、Overture / OSM 的历史快照——均未在本机核实能否做成"政策前后"。

### 2.2 Informal vitality（以路边摊为例）

- 问题导向的例子：解决 informal vitality 的问题，而不只是建成环境质量的提升；可以用街景照片加上采集人的主观数据（例如大家喜欢这里有路边摊）来反映（2026-10-04）
- **数据**：路边摊位置可从 Mapillary 识别（台账"候选：每格路边摊计数"，1 293 个格子无覆盖）；YouTube 胡志明市步行视频（Gu 2027 方法）数量与覆盖未核实；人的主观数据需要自己采集。
- **方法线索（agent 整理，待确认）**：摊位是否存在取决于邻近节点送来的人流 → 需要图的视角（§3）。

### 2.3 城市作为图：内循环与外部影响

- 城市可以看成网格（每个 grid 有一组因子参数），也可以看成图（grid 是 node，node 之间有 edge）——「两种你去看这个城市的一个方法」（2026-10-04）
- 传染病 metapopulation 模型的分析逻辑，「有点像内循环跟外部影响的那种分析视角」；**等当前研究结束后再去看**，看能否抽象出原理用到我们的场景（2026-10-04）

## 3. 方法与它适合的任务（agent 整理，待确认）

起因：「不同的算法适合做什么任务？比如说 JEPA 它就适合做预测任务」；「做 GNN 可以启发我们做 JEPA」（2026-10-04）

| 方法 | 设计出来做什么 | 在我们的问题里能给什么 | 给不了什么 |
|---|---|---|---|
| 扩散模型（当前研究） | 学数据分布、生成样本 | 瓦片内部的空间排列与跨尺度结构；泛化窗口里稳定的结构 | 瓦片之间的联系；方向与回路 |
| GNN | 节点 / 边 / 子图的分类或回归（预测） | 显式的节点间边；归因（GNNExplainer）可指出"邻居的什么"起作用 → 跨节点的 CLD 边 | 方向（截面）；节点内部空间排列（常被压成属性向量） |
| JEPA / 世界模型 | 在表示空间里预测被遮住的部分或未来状态；带 action 时可做规划（V-JEPA 2） | 时间上的预测；给定干预（政策）后的状态变化 | 依赖时间序列数据；目标 / 代价怎么定义仍是开放问题 |
| metapopulation 类模型 | 每个节点内部的动力学 + 节点间流动的耦合 | "内循环 + 外部影响"的结构 | 需要节点间流动数据 |

共同点：都是归纳工具，提供候选的边或结构；连成回路仍要演绎（`vitality_stance.md` §3）。

## 4. 已有的材料（别处，不复制）

- **Ch5 世界模型笔记**：`~/Documents/wafo-base/research-vault/1-project/26-chap5-world-model/`（Geo-H-JEPA、LeCun 2022 路线各模块实现状态、IC / TC / Actor 设计、24 个月四关的时间估计）
- **JEPA 文献周扫**：`~/Documents/wafo-base/.context/automation/jepa-worldmodel-paper-sweep/notes.md`（按 A 家族主线 / B 理论 / C 规划控制 / D 批评分线；JEPA-Anything = arXiv 2609.20800，来自周扫记录，agent 未打开原文核实）
- **会话「JEPA建模序列结构探讨」**（工作区外，id 41c7607b，2026-09-14，9 轮）：JEPA 的序列是先 observe 后 action；actor 本体按 institution / community / 非人强迫（季风、水）三类，居民作为 community 响应的见证者；供给侧效应 2–3 年、需求侧可达 10 年（用户原话），由此讨论了双时间尺度的 JEPA 与"评估时间窗决定哪种损害可见"；engagement 仪器（explorable games）产出"强迫–响应"记录。与 §2.1 直接相关，用 `read_task` 加该 id 读取
- **已入库待读**（`literature/papers/`）：LeCun 2022、Dawid & LeCun 2023、I-JEPA（Assran 2023）、V-JEPA（Bardes 2024）、V-JEPA 2（Assran 2025）、LeJEPA（Balestriero & LeCun 2025）、Yan 2026（physical state grounding）；vault 里另有 Zhang 2026（controlled world models identifiability）、Zhao 2026 MiniWorld、WorldExam 等笔记

## 5. 还没想清楚的（待我自己定）

- **活力以什么形式进入世界模型。** 当时判定活力进世界模型的位置是 cost module，但「它不一定以一个可衡量的东西进入」。这个想法的核心不是字面上怎么编码（「我连 cost module 怎么编码我都不知道」），而是：城市活力应该是动态的、可以被更新的，要和数据采集的管线连接，把大众对城市活力的认知接入世界模型（2026-10-04）。怎么编码，未定。
  > 2026-10-04 更正 agent 的旧文本："**旧设计与现立场的冲突。** 6 月的 Geo-H-JEPA 设计里，Intrinsic Cost 是'多指标向量按到活力目标的加权距离融合成一个标量'……这正是把活力当作可优化的数值……cost module 要怎么改？"——把当时的一个想法按字面读成了设计决定。
- **多 actor 怎么建模、数据怎么采。** "actor"的两个意思已厘清（§2.1：不是第一人称的 Actor 模块，而是政策影响到的多个城市主体）。还没定的是：各 actor 的时空尺度、采集频率不同，怎么放进同一个模型。
- **时间尺度。** 政策效应要年到十年才显现（vault `geo-h-jepa-significance.md`），而能拿到的时间序列很短或没有。
- **GNN 在路线里的位置**：作为通向 JEPA 的启发，还是独立的一步？
- **通用框架与特定场景**：JEPA-Anything 类框架能借什么，哪些必须为城市专门设计？
