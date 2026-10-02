# 胡志明市（HCMC）房价与房龄数据核实报告

核实日期：2026-10-02（会话内浏览器实际打开页面；均为本人打开过的 URL，未打开的来源单独标注）
背景：Cai et al. 2025（ISPRS IJGI 14:62，北京）用链家 2022 数据算了每个街区的平均房龄、房龄标准差、平均房价。本报告核实 HCMC 是否有可比数据，并检验原笔记的说法"HCMC 没有链家式逐栋数据；两个较小门户可爬，最大两个有反爬"。

## 一句话结论

原说法**部分错误**：HCMC 有链家式的挂牌数据（batdongsan.com.vn，价格+单价+地址+小区名+站内地图定位，规模与角色都像链家），官方还有**按街道分段**的地价表（公开 PDF，且已被商业网站数字化成 20,871 行）；真正缺的是**逐栋/逐小区的"建成年份"结构化字段**——房龄只能到"正式公寓项目"这一层（项目库+开发商报告有交付年份），临街自建房没有公开年代数据，OSM 的 start_date 覆盖率约 0.02%。

---

## 1. 挂牌门户（逐条核实）

### 1.1 batdongsan.com.vn（越南最大，PropertyGuru 旗下）— 浏览器可正常打开
- HCMC 公寓挂牌列表页（打开）：https://batdongsan.com.vn/ban-can-ho-chung-cu-tp-hcm
  - 当时显示 14,430 条在售公寓；卡片字段：总价（如 3,99 tỷ）、面积（70 m²）、**单价（57 tr/m²，站方直接给出）**、卧室/卫生间数、区（坊）；URL 内嵌街道+项目名（duong-n1-diamond-centery）。
- 公寓挂牌详情页（打开）：https://batdongsan.com.vn/ban-can-ho-chung-cu-duong-n1-diamond-centery/celadon-city-cam-ket-gia-tot-nhat-86-8m-7-8-ty-all-in-2pn-2wc-thanh-toan-gian-pr46343336
  - 字段：总价 7.8 tỷ ≈ 90.07 百万 VND/m²、面积 86.6 m²、卧室/卫生间、产权（Hợp đồng mua bán）、装修、**项目信息块**（Diamond Centery／Celadon City，开发商 Gamuda Land，746 套、6 栋、状态"Đang mở bán"）、地址（Đường N1, Phường Tân Sơn Nhì + 旧区名）、"Xem trên bản đồ"（站内地图）。**没有"năm xây dựng（建成年份）"字段**。
- 独栋房挂牌详情页（打开）：https://batdongsan.com.vn/ban-nha-rieng-duong-nguyen-van-thuong-66/23-ty-ban-toa-5-tang-thang-may-ngay-landmark-phuong-25-binh-thanh-dt-9-04x14-3m-cn-114-2m2-pr46364686
  - 字段：总价、单价 201.05 百万 VND/m²、面积 114.4 m²、面宽、层数、巷宽、产权、地址（街道+坊，标题含门牌）。同样**无建成年份**。
- 地图模式（打开，JS 地图，说明站内对每条挂牌做了定位）：https://batdongsan.com.vn/ban-can-ho-chung-cu-tp-hcm?tpl=map
- robots.txt（打开）：`User-agent: * / Allow: /`，对 ClaudeBot/GPTBot 等给 crawl-delay 10；提供 sitemap.xml。仅禁内部微服务路径。
- 条款（打开）：https://trogiup.batdongsan.com.vn/docs/dieu-khoan-thoa-thuan —— 禁干扰业务、禁"不合理/不相称的流量负担"；未见明确禁止学术抓取的条款。
- 反爬实测：浏览器正常；用 curl 抓同一条挂牌返回 Cloudflare "Just a moment…" 挑战页（6KB）。**无官方公开 API**；第三方爬虫服务存在（Apify 等，仅搜索到、未打开）。
- 自有研究报告（打开）：https://research.batdongsan.com.vn/ —— 基于其挂牌大数据的月度/季度/分省市场报告（免费），另售深度数据服务。

### 1.2 nhatot.com（Chợ Tốt 房产频道，越南第二大）— 确认有反爬
- https://nhatot.com/tp-ho-chi-minh/mua-ban-nha-dat → HTTP 403 "Just a moment…"（Cloudflare，浏览器内也过不去）。www.chotot.com 主站能开，但其房产区重定向到 nhatot 后即被拦。
- 佐证其数据可被获取过：Kaggle 上有 chotot 抓取的 HCMC 公寓数据集（见 §3）。

### 1.3 homedy.com（第三大）— 浏览器与 curl 均可开
- 全越南公寓在售 23,832 条，其中 HCMC 5,138 条（列表页打开：https://homedy.com/ban-can-ho-chung-cu；HCMC 入口 /ban-can-ho-tp-ho-chi-minh）。
- 详情页（打开）：https://homedy.com/ban-can-ho-khu-do-thi-celadon-city/gia-khet-mua-ngay-95m2-3pn-1-kho-5-250-ty-vip-es3219886 —— 总价 5.25 tỷ ≈ 55.26 百万 VND/m²、面积、产权（Sổ hồng riêng）、装修清单、项目块（Celadon City，Gamuda Land，状态"Đã hoàn thiện"）、地址（路+坊+区）、站内地图（homedy.com/maps）。无建成年份字段。
- 项目库字段含"Thời gian khởi công / Thời gian hoàn thành"（开工/竣工时间，常为"Đang cập nhật"未填），正文常给出交付时点，例（打开）：https://homedy.com/astral-city-pj76427740（"Thời gian bàn giao căn hộ: Cuối quý IV/2022"）。
- curl 单页实测：HTTP 200，485KB 完整 HTML，无挑战页。

### 1.4 alonhadat.com.vn — 浏览器与 curl 均可开
- 详情页（打开）：https://alonhadat.com.vn/ban-nha-hxh-nguyen-cuu-van-p17-binh-thanh-dt-5x20m-nh-9m-dtkv-151m2-cn-134m2-16-ty-19116203.html —— 总价、面积、完整地址（路+坊）、进深面宽、层数、卧室、路宽、产权。无地图、无年份。curl 200，128KB。

### 1.5 muaban.net — 浏览器与 curl 均可开
- 房产区（打开）：https://muaban.net/bat-dong-san —— 按"类型×坊×区"组织的挂牌，价格/面积齐全；姊妹站 mogi.vn 有"Giá nhà_dat"价格分析（页脚链接，未打开）。curl 200，264KB。

### 1.6 guland.vn — 地价数字化+规划地图
- （打开）https://guland.vn/gia-dat/tp-ho-chi-minh —— 把官方地价表数字化为 **20,871 行"街道×分段（Từ–Đến）×位置 VT1–VT4"**（例：TRẦN NÃO，XA LỘ HÀ NỘI–LƯƠNG ĐỊNH CỦA 段，VT1=120.7 百万 VND/m²）；另有"房屋价格地图""到 2040 规划地图"。

> 小结：**"两个较小门户可爬"应更正为"homedy、alonhadat、muaban、guland 四个站对普通 HTTP 客户端完全开放；nhatot（Chợ Tốt 房产）浏览器都过不去；batdongsan 浏览器可开但对脚本返回 Cloudflare 挑战"**。

## 2. 官方地价表（bảng giá đất）

- 依据链（PDF 已打开，官方站下载，2.6MB/23 页）：Quyết định **02/2020/QĐ-UBND**（2020-2024 表）→ **79/2024/QĐ-UBND**（2024-10-21 签发，2024-10-31 起用至 2025-12-31）。
  下载地址：https://hochiminhcity.gov.vn/documents/39403/198301/79_2024_Q%C4%90-UBND.pdf
- 空间单元（法条原文核实）：**住宅地价"位置 1"按具名街道逐段给出**（Bảng 8，Phụ lục 2："đường có tên trong Bảng giá đất ở"）；位置 2–4 用系数（0.5 / 0.8×VT2 / 0.8×VT3），临路深度≥100m 再减 10%；农地按区（khu vực I–III）×位置。附录 9–23 页为扫描表格（无文字层）。
- 2026 起新表（合并后大 HCMC=旧 HCMC+平阳+巴地头顿）：**Nghị quyết 87/2025/NQ-HĐND**（2025-12-26），2026-01-01 起施行；旧 HCMC 区住宅最高 687.2 百万 VND/m²（Đồng Khởi/Nguyễn Huệ/Lê Lợi），最低 2.3 百万。来源（政府官网新闻，已打开）：https://xaydungchinhsach.chinhphu.vn/bang-gia-dat-tai-tphcm-ap-dung-tu-1-1-2026-tren-dia-ban-tphcm-119251227190324013.htm
- 官方 GIS/地图版：**没有找到地价的官方地图图层**。官方有的是规划/地籍类地图门户（均打开）：
  - 规划查询（地块级）：https://thongtinquyhoach.hochiminhcity.gov.vn/
  - 建设厅/规划-建筑厅 DPA 图层（含地块规划查询）：https://gisxaydung.tphcm.gov.vn/tracuuttqh
  - 土地信息门户（238 万宗档案，证书二维码查询/办事，非批量数据）：https://hcm.mplis.gov.vn/og
  - 注：thongtinquyhoach.com（无政府后缀的同名商业站）现为待售域名页。

## 3. 项目级数据库 / 数据集

- batdongsan 项目库（打开）：https://batdongsan.com.vn/du-an-can-ho-chung-cu —— 全国 1,743 个公寓项目；项目页含地址、站内地图、状态（Đã bàn giao=已交付/Đang mở bán）、开发商、栋/套数，正文常含建成年代（例，已打开：https://batdongsan.com.vn/du-an-can-ho-chung-cu-quan-5/chung-cu-phan-van-tri-pj6887 ，"2000-2002 年间建设、Block B 2002 年 10 月开工"）。
- homedy 项目库（打开入口 https://homedy.com/tong-quan-du-an ，HCMC 专区 /du-an-tp-ho-chi-minh）：结构化"开工/竣工时间"字段（覆盖率不高）。
- 研究机构季度报告（市场级+新增项目级供应/交付）：
  - DKRA（dkra.vn，已打开首页；季度报告 PDF 公开，如 Q1/2026《TP.HCM 及邻区住宅市场报告》，PDF 链接经搜索获得未逐一打开）
  - Savills（已打开）：https://vn.savills.com.vn/blog/article/221553/vietnam-eng/0425-hcmc-apartment-market-in-q1-2025-limited-new-supply.aspx
  - CBRE（已打开）：https://www.cbrevietnam.com/insights/figures/ho-chi-minh-city-figures-q2-2026
- Kaggle 公开数据集（均打开确认）：
  - https://www.kaggle.com/datasets/lammike/vietnam-housing-hcm —— HCMC 挂牌爬取数据（位置/价格/类型/土地面积/卧卫/层数/朝向/产权，2.93MB CSV）
  - https://www.kaggle.com/datasets/hoandan/apartment-prices-in-the-city-ho-chi-minh-city —— chotot 抓取的约 2,000 套 HCMC 公寓（面积/价格/位置）
  - （搜索还见 trnduythanhkhttt/housepricinghcm 2017–2022 区级均价等，未逐一打开）

## 4. 建筑年代（房龄）

- 门户挂牌：batdongsan/homedy/alonhadat 详情页**均无建成年份结构化字段**（偶尔出现在自由文本里，如"xây mới 100%"）。
- 项目层：正式公寓项目的交付/竣工年份可得（batdongsan 项目页正文、homedy 项目字段、DKRA/Savills/CBRE 报告）。
- OSM start_date 覆盖（Overpass API 实测一次，bbox 10.30–11.15°N, 106.30–107.10°E ≈ 旧 HCMC，略溢出邻省）：**建筑 way 共 116,768 条，带 start_date 的仅 23 条（≈0.02%）**。备用标签（building:year_built 等）查询两次超时未取得数字，但主标签结果已足以判定不可用。
- 地籍/建筑许可：未找到公开的逐栋年代数据集；mplis 门户是办事/证书查询，不提供批量年代数据。
- 学术先例：Chung, Seo & Kim (2018) *Sustainability* 10(12):4720, DOI 10.3390/su10124720（已打开 MDPI 页）：2000 年以来 **384 个公寓项目（HCMC 197 个）**逐项目均价/m²，经 Google Maps 地理化，来源含实地调查、CBRE/Savills、batdongsan.com.vn 与 khudothimoi.com——可作为"项目级价格+坐标"的先例；但其变量里没有建筑年龄。未见 HCMC 逐栋房龄的精细研究（搜索到的是城市形态/结构类型制图类论文，未打开细读）。

## 5. 判定（对应任务第 5 条）

- **(a) 带位置的逐套（逐栋）房价**：**可用（available，挂牌口径）**。路线：batdongsan.com.vn（量最大，浏览器可开；价格+单价 m²+街道坊地址+项目名+站内地图定位）、homedy/alonhadat/muaban（连 curl 都能开）。缺口：无暴露坐标（仅地图）、无建成年份字段、无官方 API、batdongsan 对脚本有 Cloudflare；批量获取需走商用爬虫或其 research 服务询价。
- **(b) 街道分段地价**：**可用（available）**。路线：官方 PDF（02/2020→79/2024→87/2025，按"具名街道分段×位置"）+ guland 已数字化 20,871 行；无官方 GIS 图层，需自行配 geocoder（按街道名+分段匹配路网）。
- **(c) 建筑竣工年份**：**部分可用（partially）**。公寓项目层可得（batdongsan/homedy 项目库+机构报告）；逐栋/自建房层**不可得**（门户无字段、OSM 覆盖 0.02%、无公开地籍年代数据）。做"街区平均房龄+标准差"若限定在有项目的公寓片区可行，链家式的全市逐小区房龄做不了。

## 6. 对原笔记说法的修正

- 原文（大意）："HCMC 没有链家式逐栋数据；两个较小门户可爬，两个最大门户有反爬。"
- 修正：① "没有链家式数据"不成立——batdongsan.com.vn 就是链家角色（挂牌价、单价、地址、小区/项目、地图、自研市场报告一应俱全），homedy 等三个门户还完全无反爬；② "两个最大有反爬"只对了一半：nhatot（Chợ Tốt 房产）确实连浏览器都 403；batdongsan 浏览器完全可用、robots 允许，只是对脚本/爬虫有 Cloudflare；③ 真正的结构性缺口是**逐栋建成年份**，这一点应写进研究限制。

## 7. 未确认/限制

- batdongsan 地图模式下坐标是否可直接从页面 JSON 读取：未验证（页面为 JS 地图；curl 被挑战页拦截，未做绕过尝试）。
- OSM 备用年代标签（building:year_built / year_built / construction_date）数量：两次查询超时，未取得。
- DKRA 季报 PDF 内容细节、CBRE/Cushman 报告内的项目级字段完整性：未逐份打开（仅打开 Savills 博文与 CBRE figures 页）。
- thuvienphapluat.vn 等法律数据库站也返回 Cloudflare 403，但官方 hochiminhcity.gov.vn、chinhphu.vn 均可直连，不影响结论。
- 本核实只做单页浏览/单次 API 查询，未做任何批量抓取、注册或登录。
