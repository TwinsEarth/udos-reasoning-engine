# PhysBrain 1.5 深度分析报告

> 生成日期：2026-09-13
> 分析对象：深度机智（DeepCybo）PhysBrain 1.5 物理基座模型
> 用途：为 UDOS（CPU-only、约 52191 参数、合成参数化动力学小模型）提供可落地轻量化能力映射
> 事实纪律：每条事实标注来源与访问状态；用户提供数字一律与官方页面/论文/代码交叉核对；抓不到的资料显式标注。

---

## 0. 执行摘要（30 秒版）

- PhysBrain 1.5 是深度机智于 **2026-09-08/09** 发布的具身视觉-语言模型（VLM），基于 **Qwen3-VL** 扩展词表，提供 **2B（2.2B 参数）与 8B（8.9B 参数）** 两档权重，上下文窗口 262,144 tokens。来源：Project Page、DataNorth 模型卡转述。
- 核心主张是"一个基座、三种能力"：具身理解、动作生成（ActionPiece token）、未来状态预测（RGB+深度+机器人 mask 三模态），全部作为离散 token 在**同一自回归主干**上以 next-token prediction 联合学习，**无任务专用头**。来源：Project Page 原文。
- 官方自评：28 项具身基准综合 **72.5** 分（8B），开源第一；2B 为 **66.6**（参考不计排名）。来源：Project Page 表格 + ZAKER 新闻稿 + DataNorth 三方一致。
- **关键限制**：截至本报告写作，PhysBrain 1.5 的**技术报告原文未公开**——模型卡引用的 GitHub 链接返回 404，arXiv 上无 PhysBrain 1.5 论文。28 项基准**全部测试"理解"**，动作生成与未来预测仅有定性样例，**无任何仿真或真机成功率数据**。来源：DataNorth 报道 + Project Page 实际内容。
- 用户消息中的关键数字（Ego360、Human-as-Humanoid、Prime U 60 自由度、4.8–7.2× 吞吐、未来 1 秒三模态、Physical Loop 五步）**均能在官方新闻稿或 arXiv 论文中找到对应出处**，但分属不同来源：Ego360/五步/1 秒来自量子位/ZAKER 新闻稿（一方陈述）；Prime U 60 自由度与 4.8–7.2× 吞吐来自 **arXiv:2606.32009 Human-as-Humanoid 论文**（经同行评审流程的学术来源）。
- 对 UDOS 的可落地映射：PhysBrain 的 2B/8B VLM、Ego360 全景视频预训练、真机控制**全部不可复现**；但 Physical Loop 五步 runner、统一自回归多任务头、三模态未来预测代理目标、Human-as-Humanoid 形态无关重定向思路、可供性打分、五维评测套件**均可在 CPU 小模型合成数据上做 analogy 级轻量化**。详见第 11 节。

---

## 1. 资料来源与访问状态总览

| # | 资料 | URL | 访问状态 | 角色 |
|---|---|---|---|---|
| S1 | Project Page | https://deepcybo-physai.github.io/PhysBrain-1.5 | ✅ 全文抓取（7775 字符，含架构图、28 基准表、定性样例） | **官方一方陈述，主要事实源** |
| S2 | HuggingFace 集合 | https://huggingface.co/collections/DeepCybo/physbrain-15 | ❌ web.fetch 失败（link fetch error）；raw README 同样失败 | 未直接获取 |
| S3 | HuggingFace 模型卡 8B | https://huggingface.co/DeepCybo/PhysBrain1.5-8B | ❌ web.fetch / API / raw 均失败 | 未直接获取 |
| S4 | HuggingFace 模型卡 2B | https://huggingface.co/DeepCybo/PhysBrain1.5-2B | ❌ 同上 | 未直接获取 |
| S5 | EvalKit 仓库 | https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit | ✅ `git clone --depth 1` 成功（94 文件），通读 README / registry / 评分协议 / 汇总脚本 | **代码级事实源** |
| S6 | 技术报告（1.5） | Project Page citation 指向 https://github.com/DeepCybo-PhysAI/PhysBrain-1.5 | ❌ 该 GitHub URL 被 robots.txt 禁止自动访问；DataNorth 独立核实该链接 **返回 404**；arXiv 搜索无 PhysBrain 1.5 论文 | **[FULL TEXT NEEDED] 未获取** |
| S7 | Human-as-Humanoid 论文 | https://arxiv.org/abs/2606.32009 | ✅ abstract 全文 + HTML 正文 snippet（管线、公式、PrimeU 参数、吞吐数据） | **学术来源，交叉验证** |
| S8 | PhysBrain 1.0 技术报告 | https://arxiv.org/abs/2605.15298 | ✅ abstract 全文（HTML 正文未深读） | 系列背景 |
| S9 | ZAKER/量子位新闻稿 | https://app.myzaker.com/news/article.php?pk=6aa66a7d8e9f094bc570affc | ✅ 全文抓取（16037 字符，三段读完） | **官方新闻稿，一方陈述** |
| S10 | DataNorth 英文报道 | https://datanorth.ai/news/deepcybo-releases-physbrain-1-5-8b | ✅ 全文抓取（2541 字符） | **第三方独立评述，含批评性核对** |
| S11 | HF Demo | https://huggingface.co/spaces/hugging-apps/physbrain1-5-8b-demo | ❌ 未访问（JS 渲染空间） | 未获取 |

> **访问失败说明**：HuggingFace 域名在本环境下对 web.fetch 持续返回 "link fetch error"（含 API 与 raw 路径），GitHub 仓库页被 robots.txt 禁止。已通过 S1（官方 Project Page）、S9（官方新闻稿全文）、S10（第三方转述模型卡细节）、S5（EvalKit 代码）四通道交叉补齐，未绕过访问控制。

---

## 2. 模型定位与发布信息

| 项 | 值 | 来源 |
|---|---|---|
| 模型名 | PhysBrain 1.5 | S1 |
| 厂商 | 深度机智（DeepCybo Team），北京海淀中关村 | S9 |
| 发布日期 | 2026-09-08（权重上线）/ 2026-09-09（官方发布） | S10、S9 |
| 定位 | "From General VLMs to Physical Foundation Model"——统一具身视觉-语言模型 | S1 |
| 主干 | Qwen3-VL（8B 版基于 Qwen3-VL-8B-Instruct；2B 版基于 Qwen3-VL-2B-Instruct） | S1、S10 |
| 实际参数量 | 8B 版 8.9B（17.8 GB 权重）；2B 版 2.2B（4.3 GB） | S10（模型卡转述） |
| 上下文窗口 | 262,144 tokens（两档同） | S10（模型卡转述） |
| 词表扩展 | 在 Qwen3-VL 之上新增 **16,640** 个 token（动作 + 未来视觉状态） | S10（模型卡转述） |
| 结构改动 | 唯一结构变化即词表扩展；输出仍为标准 Qwen3-VL 接口 | S10 |
| License | **未声明**（DataNorth 明确指出模型卡无 license 字段、仓库无 license 文件） | S10 |
| 支持推理栈 | Transformers / vLLM / SGLang / LLaMA-Factory / ms-swift / veRL | S1 |
| 技术报告 | 引用 "PhysBrain 1.5 Technical Report"（DeepCybo Team, 2026），但链接 404，arXiv 无此文 | S1 citation、S10 |

> **注意**：S1 页面顶部写 "8B Parameters · Qwen3-VL backbone"，未写 8.9B；8.9B/17.8GB/4.3GB/262144/16640 这组精确数字**仅来自 S10 对模型卡的转述**，本环境未直接读到模型卡原文，标 `[CITATION FROM SECONDARY]`。

---

## 3. 模型逻辑：Physical Loop 五步闭环

### 3.1 官方表述

S1 将模型逻辑概括为"镜像智能体-环境交互的闭环物理循环"：

> "observations guide reasoning and action, actions alter the environment, and the updated state feeds back as the next observation."

S1 的架构图将闭环画为 **4 个阶段**：Observe → Reason & Ground → Act → World Changes。

S9（官方新闻稿）将其扩展为**显式五步**：

1. **观察世界**（Observe）——感知 2D 场景并构建 3D 空间表示
2. **理解空间与任务**（Reason & Ground）——规划策略、定位目标、落地可供性
3. **判断动作后果**——预测状态转移
4. **执行**（Act）——生成目标导向的末端执行器轨迹
5. **反馈修正**——新观察回到下一轮

> 五步版本出自 S9 新闻稿原文；S1 官方页面图示为 4 步（将"判断后果"并入"World Changes"）。两者不矛盾，新闻稿版本更细。

### 3.2 三大能力如何在同一基座共同学习

S1 原文：

> "Language responses, structured spatial outputs, end-effector trajectories, and future world states are all formulated as discrete tokens and jointly learned under a unified next-token prediction objective — **no task-specific heads**."

S9 补充：

> "将语言回答、空间坐标、动作轨迹、未来画面与深度预测，全部纳入同一套训练框架，不需要为每种能力单独搭建模块或设计专门的输出层。"

机制要点（全部来自 S1 架构图与 S9）：

- **输入侧**：图像/视频观测 → Vision encoder；任务指令 + 具身形态/频率 → Shared token embedding；可选近期 ActionPiece token 序列作为运动上下文。
- **主干侧**：所有 token 汇入同一个 Shared LM output head（S1 架构图原文如此，S10 强调"文件仍是普通 Qwen3-VL 模型"）。
- **输出侧**：从同一主干分出三类 token 流——
  - Language tokens → 具身理解（Embodied Understanding）
  - Action tokens → ActionPiece → 末端执行器轨迹
  - Visual tokens → Image Reconstruction（S1）/ Vision Decoder（S9 转述架构图）→ RGB / Depth / Robot Mask
- **训练目标**：masked next-token prediction（S1 架构图底部原文 "Joint training with masked next-token prediction"）。

> **事实分层**：S1 架构图显示输出侧有"Shared LM output head"之后再分三类 token，但 S1 正文又说"no task-specific heads"。两者需调和为：**主干无任务专用头，但输出端有三组 token id 区间（语言/动作/视觉），由词表扩展实现**，而非三个独立线性层。这一调和来自 S10"DeepCybo turned robot arm movements and future camera views into new words in the model's vocabulary"。

---

## 4. 技术架构

### 4.1 整体数据流（依据 S1 架构图，逐框转录）

```
[Context Encoding] 观察场景、指定任务
  Image/video observations ──→ Vision encoder ─────────────┐
  Task instruction ──\                                      │
  Embodiment/frequency ──→ Shared token embedding ──────────┤
  Recent action ActionPiece tokens (optional motion context)┘
                                                              ↓
[Unified Modeling] 一个主干、一个词表、一个输出头
                    PhysBrain 1.5 (Shared LM) ──→ Shared LM output head
                                                              ↓
[Physical Capabilities] 理解、动作、预测未来状态
  Language tokens ──→ Embodied Understanding
  Action tokens    ──→ ActionPiece ──→ End-effector trajectory
  Visual tokens    ──→ Image Reconstruction ──→ RGB / Depth / Robot Mask
```

横断标注：**Language ∪ Action ∪ Visual**（S1 原文），联合训练目标为 masked next-token prediction。

### 4.2 VLM 接口

- 主干为预训练 Qwen3-VL，词表扩展 16,640 项（S10）。
- 标准推理接口：可直接用 Transformers / vLLM / SGLang 加载，无自定义推理代码（S1、S10）。
- FlashAttention：S5 README 明确说明"PhysBrain 1.5 在 FA4 上训练，FA4 评测结果正常，FA2 有可接受范围内的小幅波动"。

### 4.3 Human-as-Humanoid 管线（与 PhysBrain 1.5 的关系）

**重要澄清**：Human-as-Humanoid **不是 PhysBrain 1.5 模型本身的模块**，而是深度机智此前（2026-06-30）发表的独立论文（arXiv:2606.32009），其中**使用 PhysBrain 作为 VLM backbone**。S9 新闻稿将其作为 PhysBrain 1.5 动作数据来源的管线来介绍。两者关系：

- Human-as-Humanoid 论文（S7）第 5.2 节原文："The egocentric image and instruction are encoded by **PhysBrain**, which we use as an existing manipulation-aware VLM backbone."
- 论文中的动作生成主干是**独立的 conditional flow-matching DiT**，**不经过 VLM**："Proprioception and diffusion noise enter the action model directly. They are not routed through the VLM."
- S9 表述："PhysBrain 1.5 模型通过 Human-as-Humanoid 管线，从人类视频中学习手腕的运动方式"——这是新闻稿将数据管线与模型绑定的表述；严格按 S7，PhysBrain 提供视觉-语言 token，动作 DiT 独立生成 60-DoF chunk。

> **[UNVERIFIED 区分]** S1 Project Page 上的 "ActionPiece tokens encode end-effector trajectories in a unified action codebook" 与 S7 论文中的 "flow-matching DiT 预测 60-DoF chunk" 是**两套不同的动作表示**：前者是 PhysBrain 1.5 自回归词表中的 ActionPiece token；后者是 Human-as-Humanoid/PhysDex 的扩散动作专家。S9 新闻稿将二者混在"动作生成"一节叙述，但未说明它们是同一组件还是两套方案。本报告按"S1 官方页面为准"记录 ActionPiece，按"S7 论文原文"记录 flow-matching DiT，不强行合并。

---

## 5. 核心算法

### 5.1 动作生成：ActionPiece token

- S1 原文："ActionPiece tokens encode end-effector trajectories in a unified action codebook shared across control configurations and arm setups — one generalist checkpoint drives varied robotic platforms."
- 输入：任务指令 + 当前观测 + 可选近期动作历史。
- 输出：下一动作 chunk，编码为紧凑的 ActionPiece token 序列。
- 跨形态：S9 称"同一份模型可以适配不同形态、不同控制频率的机器人，无需为每台机器单独训练"——这是新闻稿主张，S1 仅称"shared across control configurations and arm setups"。
- **精度代价**（S10 批评性指出）："chopping a smooth arm movement into a fixed set of steps loses precision that a purpose built control module would keep."

### 5.2 未来状态预测：三模态

- S1 原文："Predicts the world one step ahead as spatially aligned RGB images, depth maps, and robot masks — a multimodal world model inside the same vocabulary."
- S9 新闻稿明确为"**预测 1 秒后**的物理状态，并同时输出三种模态：RGB 图像、深度图和机器人掩码"。
- S10 转述："a piece of the image it expects to see one second from now."
- 三模态在词表中作为 visual tokens 输出，经 Image Reconstruction/Vision Decoder 解码为空间对齐的 RGB / depth / mask（S1 架构图）。
- S1 展示的定性样例覆盖多种机器人形态（双臂厨房锅、碗叠放、香蕉放置等），每行输出 Input RGB / Predicted RGB / Predicted Depth / Predicted Mask 四列。

### 5.3 空间坐标与轨迹输出

- S1 定性样例图展示五类结构化输出格式：
  - 物体 ground：`[462, 458]`（点坐标）
  - 区域 ground：`[410, 650]`
  - 功能 ground：`[118, 156]`
  - 自由空间轨迹：`[152,105] [220,170] [280,240] [340,320]`（2D 点序列）
  - 接触丰富操作轨迹：`[375,525] [342,555] [323,601] [328,636] [314,666]`
- S5 代码中的坐标协议（`docs/final_point_metrics_protocol.md`）：输出为 `[x,y]`，范围 **0–1000**，解码时 `x_pixel = min(width-1, int(x/1000*width))`，越界点不截断到边缘但计入预测数 P。这与 S1 样例中的像素坐标格式一致。

---

## 6. 数据体系

### 6.1 Ego360 全景人类数据体系

S9 原文：

> "深度机智构建的 **Ego360 人类全景真实数据体系**。它通过全景视频记录任务执行时的周围环境，并同步保留全身姿态、手部运动与任务级语音。"

S9 引用 WAIC 2026（2026-07，上海）陈凯博士演讲图：

> "自研业界首创全景数采方案 DeepCybo Ego360：One Device For All and Record All In One"
> - 人体信息重建：基于全景相机的全身姿态估计替代现有 Ego-Exo 视频方案
> - 3D 环境及物理交互重建：基于全景相机的深度和点云提供更精准的物理实时交互信息
> - 胸前全景采集：低成本、无死角的时空数据

S1 官方页面对数据的表述更克制：

> "Pre-training draws its embodied supervision entirely from human interaction videos — **egocentric, synchronized ego–exocentric, and panoramic recordings** structured into task-centered episodes. Supervised fine-tuning then combines human demonstrations, real-robot trajectories, and simulated experience."

### 6.2 情境数采（In Context Data Collection）范式

S9 时间线：

- 2025 年 12 月：提出情境数采范式
- 2026 年 7 月：迭代为全景数采范式（Ego360）
- 2026 年 9 月 10 日：国家数据局具身智能座谈会，深度机智受邀展示人类全景真实数据与情境数采范式
- 同步推进：**模型驱动的全自动标注管线**

### 6.3 PhysBrain 1.0 的数据引擎（系列背景，S8）

PhysBrain 1.0 技术报告（arXiv:2605.15298）abstract：

> "Our data engine extracts scene elements, spatial dynamics, action execution, and depth-aware relations, then turns them into question-answer supervision for training PhysBrain VLMs."

即 1.0 的路线是：人类自我中心视频 → 数据引擎提取场景元素/空间动态/动作执行/深度关系 → QA 监督训练 VLM → 经 capability-preserving and language-sensitive adaptation 传到 VLA。1.0 在 ERQA、PhysBench、SimplerEnv-WidowX、LIBERO、RoboCasa 上报告 SOTA。S9 称 1.0 在仿真中平均成功率 80.2%（中文媒体报道）。

> **[分层说明]** Ego360、情境数采、全自动标注管线的具体数据规模、标注 schema、训练数据配比等**均未公开**（技术报告未发布）。以上仅为官方新闻稿一方陈述，无独立可核验细节。

---

## 7. 开源代码结构（EvalKit）

### 7.1 仓库概况

S5（`git clone --depth 1`，94 文件）：

- **定位**：面向 VLM 的空间与具身智能基准评测工具包；基于并扩展开源 [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit)。
- **Python**：3.10+，推荐 3.11。
- **不包含**：模型权重、数据集缓存、评测结果、运行日志、API key。
- **默认解码**：greedy，`temperature=0.0, top_p=1.0, top_k=-1, seed=3407`；MMSI-Bench 是唯一例外（显式启用温度采样）。

### 7.2 目录结构

```
benchmark/            28+ 个数据集适配器与基准实现（blink.py, cosmos.py, ...）
core/                 推理后端、媒体处理、共享指标、日志
  api_engine.py / hf_engine.py / vllm_engine.py / inference.py
  final_point_metrics.py      ← 点定位协议计算
  framework_integration.py    ← Qwen3-VL JSON 解析、坐标转换、mask 命中
  point_utils.py / lazy_media.py / logger.py
eval_*.py             单基准 CLI 入口（约 36 个）
scripts/
  benchmark_registry.py       ← 28 基准 canonical 计划
  eval_qwen3vl.py / .sh       ← 分片常驻模型 runner
  summarize_benchmark_scores.py ← 汇总与主指标选择
  export_benchmark_result_bundle.py
docs/final_point_metrics_protocol.md  ← 点定位指标协议
tests/                生成默认、hub 数据集、结果 bundle、汇总的单测
```

### 7.3 28 基准计划（`scripts/benchmark_registry.py` 原文）

代码注释明确："Return the current 28-benchmark plan (**non-judge, excluding Ego3D**)."

| 序号 | 基准名 | 数据集 | split | 主指标（来自 summarize 脚本） |
|---|---|---|---|---|
| 1 | ERQA | FlageVal/ERQA | test | overall_accuracy |
| 2 | RoboSpatial | chanhee-luke/RoboSpatial-Home | context+compatibility+configuration | non_strict_overall_score |
| 3 | EgoPlan2 | IffYuan/ego-plan | train | overall_score |
| 4 | SAT | FlagEval/SAT | default/test | overall_accuracy |
| 5 | Where2Place | FlagEval/Where2Place | test | non_strict_micro_f1 |
| 6 | RefSpatial-Bench | BAAI/RefSpatial-Bench | location+placement+unseen | non_strict_micro_f1 |
| 7 | Part-Affordance-2K | IffYuan/Part-Affordance-2K | train | non_strict_micro_f1 |
| 8 | ShareRobot-Trajectory | IffYuan/sharerobot_trajectory | train | normalized_rmse_score |
| 9 | VABench-Visual-Trace | IffYuan/vabench-v | train | normalized_rmse_score |
| 10 | Q-Spatial-Bench | andrewliao11/Q-Spatial-Bench | QSpatial_plus | success_rate |
| 11 | VABench-Point | IffYuan/VABench-P | test | non_strict_micro_f1 |
| 12 | Pixmo-Points | IffYuan/pixmo-points-eval | train | non_strict_micro_f1 |
| 13 | RoboAfford | Zray26/roboafford-eval | test | non_strict_micro_f1 |
| 14 | PIOBench | IffYuan/PIO-Bench | train | non_strict_micro_f1 |
| 15 | RoboRefit | VLyb/RoboRefit-corrected | test | non_strict_micro_f1 |
| 16 | BLINK | BLINK-Benchmark/BLINK | Counting+Relative_Depth+Spatial_Relation/val | overall_accuracy |
| 17 | CV-Bench | nyu-visionx/CV-Bench | default/test | overall_accuracy |
| 18 | VSI-Bench | IffYuan/vsi-bench | train | overall_score |
| 19 | EmbSpatial | FlagEval/EmbSpatial-Bench | test | overall_accuracy |
| 20 | PointBench | IffYuan/PointBench | train | non_strict_micro_f1 |
| 21 | COSMOS | IffYuan/COSMOS | train | overall_score |
| 22 | RoboVQA | VLyb/RoboVQA-16frames | local-16frames/train_explicit_style | overall_bleu |
| 23 | VLABench | VLyb/VLABench | local | overall_score |
| 24 | ERQA-PLUS | hugendas/erqa-plus | train | overall_accuracy |
| 25 | 3DSRBench | VLyb/3DSRBench | test | overall_accuracy |
| 26 | ViewSpatial | lidingm/ViewSpatial-Bench | test | overall_accuracy |
| 27 | MindCube | VLyb/MindCube-TinyBench | tinybench | overall_accuracy |
| 28 | MMSI-Bench | RunsenXu/MMSI-Bench | test | overall_accuracy |

> 另有 `eval_ego3dbench.py`、`eval_openeqa.py`、`eval_pio_s3_verified.py`、`eval_robofac.py` 等入口存在，但**不在 28 基准默认计划内**（registry 注释明确 excluding Ego3D；OpenEQA 与 PIO-S3 在 summarize 脚本的 BENCHMARK_ORDER 中出现但不在 build_specs 返回列表里）。

### 7.4 点定位评分协议（`docs/final_point_metrics_protocol.md`）

- 协议名：`physbrain_point_metrics_final_20260817`。
- 适用 10 个基准：Pixmo-Points, PointBench, Part-Affordance-2K, RoboRefit, RoboSpatial, VABench-Point, PIOBench, Where2Place, RefSpatial-Bench, RoboAfford（代码中 `POINT_BENCHMARKS` frozenset）。
- 每样本四计数：P（预测点数）、G（GT 目标数）、M（被覆盖的 GT 数，用于 recall）、H（落在 GT 区域内的预测点数，用于非严格 precision）。
- **严格指标**：TP = min(M,P,G)；FP = P-TP；FN = G-TP；micro 聚合；strict_micro_f1 仅作诊断。
- **主指标（非严格）**：precision = ΣH*/ΣP；recall = strict_recall；f1 = 调和平均；**point 基准摘要用 `non_strict_micro_f1` 为主指标**。
- 特殊规则：
  - PointBench 计数任务有 exact-count gate（预测数≠期望数则 TP=0）；全 966 样本 micro 聚合；3 个已知 mask 异常 ID（54, 445, 775）用最近邻 resize。
  - PixMo：预测点与 GT 代表点先做匈牙利匹配（欧氏距离），再查 mask 命中。
  - RoboSpatial：122 个 context 样本用点指标 + 228 个二分类样本用 yes/no 准确率，按 `(122*context_f1 + binary_correct)/350` 混合。
  - RoboRefit：用全部 2000 个 corrected mask，bbox 不计分。
- 输出格式：只接受标准 Qwen3-VL 最终 JSON（`point_2d`、显式 `[]`/`No object`、或 Markdown JSON fence），不扫 reasoning 文本。坐标 0–1000。

### 7.5 综合均分计算

S1 表格脚注原文：

> "Overall Average is the **unweighted mean** across the 28 benchmarks at the reported precision."

即：**28 个基准主指标的简单算术平均**（每个基准先归一到 0–100，再等权平均），不做加权。

S5 `summarize_benchmark_scores.py` 逻辑：每个基准从 `meta_result.json` 取 `metrics`，按 `PRIMARY_METRIC_BY_BENCHMARK` 选主指标，0–1 范围自动乘 100 显示。代码本身**不计算 overall mean**（只逐基准输出 markdown/tsv），overall 由 S1 官方汇总。

---

## 8. 评测维度与 28 基准

### 8.1 能力维度：五维（评测）vs 六维（展示）

S1 官方页面将 28 基准归入 **C1–C5** 五个评测 category（C6 仅作能力展示无基准）：

| 维度 | 名称 | 包含基准数 | 代表基准 |
|---|---|---|---|
| C1 | Foundational Visual-Spatial Perception（基础视觉空间感知） | 2 | BLINK, CV-Bench |
| C2 | Spatial and Multi-view Understanding（空间与多视角理解） | 9 | 3DSRBench, EmbSpatial-Bench, MindCube, MMSI-Bench, Q-Spatial-Bench, RoboSpatial-Home, SAT, VSI-Bench, ViewSpatial-Bench |
| C3 | Embodied Cognition, Reasoning, and Planning（具身认知、推理与规划） | 6 | COSMOS, EgoPlan-Bench2, ERQA, ERQA-PLUS, RoboVQA, VLABench |
| C4 | Spatial Grounding, Pointing, and Affordance（空间定位、指向与可供性） | 9 | Part-Affordance, PIOBench, PixMo-Points, PointBench, RefSpatial-Bench, RoboAfford, RoboRefit, VABench-Point, Where2Place |
| C5 | Visual Trace and Trajectory Reasoning（视觉轨迹与轨迹推理） | 2 | ShareRobot-Traj., VABench-V.-Trace |
| **C6** | **Action & Future Prediction（动作与未来预测）** | **0**（无基准，仅定性样例） | ActionPiece 轨迹、RGB/Depth/Mask 预测 |

合计：2+9+6+9+2 = **28** ✓

S9 新闻稿明确称"按**五大能力维度**组织——视觉空间感知、空间与多视角理解、具身认知与规划、空间指向与可供性、视觉轨迹推理"，与 S1 的 C1–C5 一致。**用户消息中"五维度"的说法与官方一致；C6 是能力展示而非评测维度。**

### 8.2 综合得分（S1 表格原文，8B 列）

| 模型 | 规模 | Overall |
|---|---|---|
| Gemini 3.6 Flash | 闭源 | 73.0 |
| GPT 6 Astra | 闭源 | 73.3 |
| Claude Opus 5 | 闭源 | 67.9 |
| Hy-Embodied-VLM-1.0 | 30A3B | 66.0 |
| Embodied-R1.5 | 8B | 64.9 |
| RynnBrain 1.1 | 9B | 63.1 |
| RoboBrain 2.5 | 8B | 58.2 |
| MiMo-Embodied | 7B | 57.4 |
| Cosmos 3 Nano | 8B+8B | 62.3 |
| ACE-Brain-0.5 | 8B | 59.0 |
| Qwen3-VL-Instruct | 8B | 59.5 |
| **PhysBrain 1.5-2B** | 2B | **66.6**（参考，不参与排名） |
| **PhysBrain 1.5-8B** | 8B | **72.5** |

S1 称 8B 在 28 项中 **14 项开源第一、10 项开源第二**。

### 8.3 独立批评性核对（S10）

DataNorth 指出三点重要限制：

1. **全部数字由 DeepCybo 自跑**："The model card says the team re-ran all comparison models itself so the metrics would match... no number here has been checked by anyone else."
2. **闭源模型被 handicap**："GPT 6 Astra was run at a low thinking setting and Gemini 3.6 Flash at minimal, while two of the open competitors were run with thinking switched on."
3. **28 基准全测理解**："All 28 benchmarks test one thing: whether the model understands what it is looking at... There are no robot success rates at all, in simulation or on hardware."

---

## 9. 用户提供数字的交叉核对表

| 用户消息中的数字/说法 | 核对结果 | 来源 | 状态 |
|---|---|---|---|
| 72.5 综合分 | S1 表格、S9 新闻稿、S10 报道三方一致 | S1/S9/S10 | ✅ 已核对 |
| 28 基准 | S1 表格 28 行；S5 registry 28 项；S10 "own suite of 28 tests" | S1/S5/S10 | ✅ 已核对 |
| 14 项开源第一 | S1 "ranks first on 14 benchmarks"；S9 "14 项开源第一" | S1/S9 | ✅ 已核对 |
| 2B 66.6 | S1 表格末列 66.6；S9 "PhysBrain 1.5-2B 取得 66.6 分" | S1/S9 | ✅ 已核对 |
| Ego360 | S9 新闻稿明确"Ego360 人类全景真实数据体系""DeepCybo Ego360"；S1 仅说 "panoramic recordings" 未点名 Ego360 | S9 | ✅ 已核对（新闻稿一方陈述） |
| Human-as-Humanoid | S9 明确；S7 arXiv:2606.32009 全文 | S7/S9 | ✅ 已核对（学术来源） |
| Physical Loop 五步 | S9 明确"五步"；S1 图示为 4 步 | S9/S1 | ✅ 已核对（新闻稿版本更细） |
| 未来 1 秒 RGB+深度+mask 三模态 | S9"预测 1 秒后"；S1"one step ahead"；S10"one second from now" | S9/S1/S10 | ✅ 已核对（"1 秒"为新闻稿与 S10 表述，S1 原文为 "one step ahead"） |
| Prime U 60 自由度 | S7 论文：PrimeU, 60-DoF upper-body；2×7-DoF arm + 2×20-DoF hand + 3-DoF neck + 3-DoF waist = 60 | S7 | ✅ 已核对（学术来源） |
| 4.8–7.2× 吞吐 | S7 abstract："4.8–7.2x raw demonstration-throughput gain over humanoid teleoperation"；S9 同 | S7/S9 | ✅ 已核对（学术来源） |
| 五维度 | S9 明确五维；S1 C1–C5 五维评测 + C6 展示 | S9/S1 | ✅ 已核对 |
| 8B / Qwen3-VL backbone | S1 原文 | S1 | ✅ 已核对 |
| 16,640 词表扩展 | 仅 S10 转述模型卡 | S10 | ⚠️ [CITATION FROM SECONDARY] |
| 262,144 上下文 | 仅 S10 转述模型卡 | S10 | ⚠️ [CITATION FROM SECONDARY] |
| 8.9B / 17.8GB / 2.2B / 4.3GB | 仅 S10 转述模型卡 | S10 | ⚠️ [CITATION FROM SECONDARY] |

---

## 10. PhysBrain 1.5 技术报告未公开的影响

S6（技术报告原文）**未获取**，原因：

1. S1 citation 块给出的 BibTeX 指向 `url = {https://github.com/DeepCybo-PhysAI/PhysBrain-1.5}`，该 GitHub URL 在本环境被 robots.txt 禁止自动访问。
2. S10 独立核实："That link returns a 404, and no paper for PhysBrain 1.5 exists on arXiv."
3. arXiv 搜索 "PhysBrain 1.5" 无结果；arXiv 上的 PhysBrain 论文为 1.0（2605.15298）与 Human-as-Humanoid（2606.32009），均非 1.5 技术报告。

**因此以下内容在公开渠道中无法核验，本报告不臆测：**

- 训练数据规模（Ego360 视频小时数、SFT 轨迹数）
- ActionPiece codebook 大小与聚类方法
- 词表扩展 16,640 中动作/视觉 token 的切分比例
- 未来状态预测的视觉 token 化方案（VQ? regression token?）
- 训练阶段配比（预训练/SFT 各自 token 数）
- 动作生成在真机/仿真的定量成功率
- 与 PhysBrain 1.0 的详细消融对比

---

## 11. PhysBrain 机制 → UDOS 可落地轻量化能力映射表

> **UDOS 背景**：CPU-only、约 52191 参数、合成参数化动力学小模型。不可能复现 2B/8B VLM、Ego360 全景视频预训练或真机控制。下表每条均为 **analogy（受启发的轻量化类比）**，**不是 reproduction（复现）**。验证方式限定在 CPU 小模型 + 合成数据上可执行的最小实验。

| # | PhysBrain 1.5 机制（来源） | 原始规模/形态 | UDOS 轻量化 analogy | 类比性质 | CPU 合成数据验证方法 |
|---|---|---|---|---|---|
| M1 | **Physical Loop 五步显式闭环 runner**：Observe→Reason→Act→World Changes→反馈（S1/S9） | 2B/8B VLM 自回归推理 | 在 UDOS 主循环中实现显式五状态 runner：`observe(s) → reason(obs,goal) → act(reason) → step_world(act) → observe(new)`，每步为纯函数；闭环长度 N 步可配置 | analogy（结构类比，非参数复现） | 用合成参数化动力学（如弹簧-阻尼、刚体碰撞）跑 100 条 episode，记录每步状态转移一致性；对比"无闭环"（单步开环）的轨迹发散度 |
| M2 | **统一自回归主干 + 多任务头（无任务专用头）**：Language/Action/Visual token 同词表联合 next-token 训练（S1） | 16,640 扩展词表、262K 上下文 | UDOS 用一个共享小 RNN/MLP 主干，输出三个低维 head：①语言意图 token（分类）②动作向量（连续回归）③未来状态向量（连续回归）；三任务共享主干参数，联合 loss | analogy（多任务共享主干思想） | 合成数据上训练后，逐步 dropout 某任务 head，观察其他任务性能下降幅度（证明共享表示）；对比三任务独立小模型的总参数量 |
| M3 | **未来状态三模态预测**：RGB+Depth+Robot Mask 同词表预测（S1/S9） | 高分辨率图像 token | UDOS 用低维向量代理三模态：①环境代理向量（代 RGB）②深度/距离向量（代 Depth）③机器人占据 mask 二值向量（代 Mask）；同一预测头输出三元组 | analogy（三模态解耦思想，非图像复现） | 合成动力学数据上，预测 t+1 的三元组，分别计算 MSE（环境/深度）与 IoU（mask）；消融实验：单向量预测 vs 三元组解耦预测的下游规划成功率 |
| M4 | **Human-as-Humanoid 形态无关动作重定向**：staged IK 将人类动作重定向到 60-DoF PrimeU（S7） | 60-DoF 人形、staged IK（手→臂→颈腰→guard） | UDOS 实现一个**形态无关动作适配器**：将"末端执行器目标向量"通过一个极简 IK/映射层转换到"合成双足/单臂参数化动作空间"；只学映射矩阵，不学感知 | analogy（形态对齐思想） | 合成两个差异显著的形态（单臂 3-DoF vs 双肢 6-DoF），用同一末端轨迹数据集训练重定向层；验证：重定向后关节轨迹在新形态上可执行（关节限位满足率） |
| M5 | **ActionPiece 统一动作 codebook**：跨控制配置/臂设置共享 codebook（S1） | VQ codebook、自回归 token | UDOS 用一个小 k-means/向量量化层将连续动作 chunk 离散化为 K 个"动作原型"（如 K=16/32），推理时自回归选择原型 | analogy（动作离散化思想） | 合成轨迹上，对比连续 MLP 动作输出 vs VQ 原型选择：①预测 MSE ②推理延迟（CPU 上）③动作平滑度 |
| M6 | **可供性 affordance 打分**：Part-Affordance/RoboAfford 基准（S1/S5） | VLM 点定位 non_strict_micro_f1 | UDOS 在合成场景中对"物体-可操作部位对"输出 affordance 分数（0–1），作为动作选择的先验 | analogy（可供性打分接口） | 合成数据生成 (物体, 可操作部位, 标签) 三元组，训练二分类打分器；验证：top-k affordance 命中真实可操作部位的比例 |
| M7 | **五维评测套件**：C1–C5 等权综合 28 基准（S1/S5） | 28 个真实数据集、GPU 评测 | UDOS 设计一个**微型五维评测套件**：每个维度 5–10 个合成任务，等权平均；维度对齐 C1–C5（空间感知/多视角/规划/定位/轨迹） | analogy（评测框架与等权聚合方法） | 在合成数据上跑 25–50 个任务子集，输出五维雷达图 + 综合均分；消融：改变某维度数据质量，观察综合分敏感度 |
| M8 | **点定位协议**：0–1000 归一化坐标、non_strict_micro_f1、匈牙利匹配（S5） | VLM JSON 输出、图像 mask | UDOS 合成"目标点定位"任务，输出归一化坐标 + 预测点数；用相同的 P/G/M/H 四计数与 non_strict_micro_f1 计算 | analogy（评分协议复用） | 合成 1000 个定位样本，复现 S5 的 P/G/M/H 统计与匈牙利匹配，验证 UDOS 定位精度指标 |
| M9 | **masked next-token prediction 联合训练**（S1 架构图） | Transformer causal attention | UDOS 在合成序列上用 masked prediction 作为预训练目标（输入部分观测，预测 mask 位置），再下游微调 | analogy（预训练目标） | 合成长序列数据，对比"直接监督训练"vs"先 masked 预训练再微调"的样本效率曲线 |
| M10 | **近期动作上下文作为输入**（S1：Recent action ActionPiece tokens, optional motion context） | VLM token 序列 | UDOS 将最近 H 步动作向量作为循环状态输入，参与当前步决策 | analogy（时序上下文） | 合成带惯性的动力学任务，对比有/无历史动作输入的预测误差 |

### 映射表使用说明

- **M1/M2/M3 是核心**：Physical Loop runner + 共享主干 + 三模态代理目标，构成 UDOS 可落地的最小闭环骨架。
- **M4/M5 是动作侧**：形态无关重定向与动作离散化，对应 PhysBrain 的跨形态卖点，但 UDOS 上不碰真机。
- **M6/M7/M8 是评测侧**：affordance 打分、五维套件、点定位协议，复用 PhysBrain 公开评测方法论。
- **M9/M10 是训练侧**：masked 预训练与动作上下文，是低成本可试的训练技巧。
- **不可映射项（明确排除）**：① Ego360 全景视频预训练（UDOS 无视频编码器）② 2B/8B VLM 词表扩展（UDOS 5 万参数）③ 真机 60-DoF 控制（UDOS 无执行器）④ FA4/262K 上下文推理（CPU 小模型）⑤ GPT-6 Astra/Gemini 对比基准（UDOS 无法复现跨模型比较）。

---

## 12. 局限、风险与未决项

### 12.1 证据局限

1. **1.5 技术报告未公开**（S6 [FULL TEXT NEEDED]）：所有训练细节、数据规模、消融、真机结果均无官方完整出处。
2. **HuggingFace 模型卡未直接读取**（S2/S3/S4 [FETCH FAILED]）：8.9B/17.8GB/262144/16640 等精确数字依赖 S10 转述。
3. **S1 官方页面与 S9 新闻稿均为一方陈述**：72.5 分、14 项第一等均由 DeepCybo 自跑，S10 独立指出无第三方复核。
4. **动作/预测能力无定量结果**：S1 仅有定性样例图；S10 明确指出"no robot success rates at all, in simulation or on hardware"。
5. **闭源对比存在 handicap**：S10 指出 GPT 6 Astra 用 low thinking、Gemini 3.6 Flash 用 minimal thinking 设置。

### 12.2 未决项

- `[CITATION NEEDED]`：ActionPiece codebook 大小、聚类算法、视觉 token 化方案——待技术报告公开。
- `[CITATION NEEDED]`：Ego360 数据规模、标注 schema、全自动标注管线细节——待技术报告公开。
- `[FULL TEXT NEEDED]`：PhysBrain 1.5 技术报告原文（arXiv/GitHub 404）。
- `[FETCH FAILED]`：HuggingFace 模型卡原文、HF Demo 空间。
- `[UNVERIFIED]`：S9 新闻稿中"上线 3 天超 3k 下载量""受邀国家数据局座谈会"等运营性说法，未独立核验。

---

## 13. 实际读到的资料清单与覆盖度

| 资料 | 实际读取范围 | 覆盖度 |
|---|---|---|
| S1 Project Page | 全文 7775 字符（两段读完）：架构图、28 基准表、定性样例、citation | **100%**（静态可抓部分） |
| S5 EvalKit 仓库 | README 全文、`scripts/benchmark_registry.py` 全文、`docs/final_point_metrics_protocol.md` 全文、`scripts/summarize_benchmark_scores.py` 全文、文件清单 | **高**（核心评测逻辑已读；`core/final_point_metrics.py`、各 `benchmark/*.py` 实现细节未逐行读） |
| S7 Human-as-Humanoid 论文 | abstract 全文 + HTML 正文关键 snippet（管线、PrimeU 参数表、staged IK 公式、DS-HKC、吞吐数据） | **高**（方法主线已读；消融表格与全部图表未逐字读） |
| S8 PhysBrain 1.0 技术报告 | abstract 全文 | **低**（仅 abstract；HTML 正文未读） |
| S9 ZAKER/量子位新闻稿 | 全文 16037 字符（三段读完） | **100%** |
| S10 DataNorth 报道 | 全文 2541 字符 | **100%** |
| S2/S3/S4 HuggingFace | 未获取 | **0%** [FETCH FAILED] |
| S6 技术报告（1.5） | 未获取 | **0%** [FULL TEXT NEEDED] |
| S11 HF Demo | 未访问 | **0%** |

### 未获取项清单

1. **PhysBrain 1.5 技术报告原文**（arXiv 无、GitHub 404）——[FULL TEXT NEEDED]
2. **HuggingFace 模型卡 8B/2B 原文**（web.fetch 持续失败）——[FETCH FAILED]
3. **HuggingFace 集合页原文**（JS 渲染）——[FETCH FAILED]
4. **HF Demo 空间**（JS 渲染）——未访问
5. **PhysBrain 1.0 论文 HTML 正文**（仅读 abstract）——[PARTIAL]
6. **EvalKit 各 benchmark/*.py 逐行实现**（已读 registry 与评分协议，未逐基准读适配器）——[PARTIAL]

---

*本报告基于 2026-09-13 可公开访问的网络资料撰写。所有数字与表述均标注来源；未公开内容显式标注 [FULL TEXT NEEDED] / [FETCH FAILED] / [UNVERIFIED]，未编造任何论文标题、作者、DOI、公式、模块名或分数。*
