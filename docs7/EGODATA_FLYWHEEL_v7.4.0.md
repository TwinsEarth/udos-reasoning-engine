# v7.4.0 第一人称经验数据飞轮（Egocentric Experience Flywheel，CPU 原型）

## 1. 对标概念（外部报道均为 unverified）
egocentric data boom（Maxinsights / Dyna-2 / GENE-26.5）把机器人数据问题从
“采多少小时”推进到“每小时含多少有效新世界”：

- **Experience Scale vs Density**：规模与单位小时信息量并重；
- **Coverage-aware Collection**：Agent 反向诊断数据盲区，定向补采，而非被动上量；
- **Task Coverage → State Coverage**：不只记录“做了什么任务”，而是记录
  （位置×速度×接触×阶段）的细粒度状态覆盖；
- **Yield 良率**：原始采集小时中最终成为 training-ready 数据的比例（报道称 98%）；
- 数据飞轮：主动采集 → 更强理解/重建 → 质检与标注精度提升 → 单位成本下降。

## 2. 实现（`udos7/egodata/`，全部 CPU、零第三方依赖）
- `episodes.py`：在 v7.3.7 具身环境上跑混合策略生成“第一人称片段”（手部轨迹
  代理 + 接触事件 + 子任务边界 + 任务/参数标签）；4 类任务各 12 个参数变体；
  按 25% 比例注入三类真实缺陷：静止无效片段、镜头（SLAM 位姿）漂移、重复摆拍。
- `coverage.py`：离散状态单元 `(任务, x, y, 速度档, 接触档)` 的覆盖图，
  区分 Task Coverage 与 State Coverage。
- `processing.py`：盲检 QC（轨迹总长、相机系轨迹二阶差分抖动、量化签名去重）、
  经验密度（状态单元/接触/子任务切换/手部路程，按分钟归一）、Yield 台账。
- `collection.py`：被动采集（60% 简单 reach 的偏态池、按序选取）vs
  Coverage-aware 主动采集（均匀候选池 + 边际新状态子模贪心，1-1/e 近似）；
  另含“同偏态池只换选择器”的消融。
- `benchmark.py`：160 候选片段、48 片段预算（144 原始分钟）。

## 3. 实测（本机 CPU，可复现，证据 cpu-proto）

| 策略 | Yield 良率 | 状态单元 | 占 oracle | 冗余率 | 经验密度 |
|---|---|---|---|---|---|
| 被动偏态采集 | 0.667 | 157 | 0.695 | 0.599 | 7.80 |
| Coverage-aware | **0.854** | **215** | **0.951** | 0.692 | **11.38** |
| 消融：偏态池+贪心 | 0.792 | 201 | 0.889 | — | — |

解读：
1. 同样 144 原始分钟，主动采集多覆盖约 37% 的状态空间，密度高 46%——
   “第 1000 万小时是早期复刻”的机制在合成世界可复现；
2. 消融显示收益约一半来自**主动选择**（0.695→0.889），另一半来自**采集分布
   多样化**（→0.951）；
3. QC 自动剔除静止/漂移/重复，主动策略因按边际增益选片，良率天然更高。

## 4. 已知限制与闸门
- 无真实第一视角视频、手部追踪、SLAM、MaxVLM；世界为点质量具身环境。
- 缺陷为合成注入、状态单元为手工离散；真实 QC 与状态表征依赖视觉模型。
- 闸门：真实 egocentric 采集网络与手部追踪/SLAM；视频理解大模型与多 LLM
  交叉打分（需 key）；跨本体 Embodiment Gap 真机验证。
- 外部数字（150 万/200 万/98%/1000 万小时、Dyna-2 无饱和、MaxVLM 1/10 成本）
  仅作 unverified 参照，不与本仓数字互证。

## 5. 运行
```bash
python scripts7/egodata_flywheel_demo.py
udos demo flywheel
pytest tests7/test_v740_flywheel.py -q   # 10 项
```
