# 世界模型外部佐证研究（轻量版）

> 用途：为 UDOS（CPU-only、约 5.2 万参数合成动力学小模型）的模块划分提供外部学术/工业界对照。
> 性质：**调研为辅、工程为主**。本文所有与 UDOS 的对应关系均为"轻量化机制类比"，**不是复现**，也不意味着 UDOS 在性能上对标下列任何大模型。
> 调研时间：2026-09-14。证据分级：【官方】=机构/论文原文；【媒体】=二手报道；【推断】=作者基于多来源的归纳。核不到的具体数字一律标 `[UNVERIFIED]`。

---

## 0. 证据分级与阅读约定

- 每条关键事实后附来源 URL。
- 【官方】：NVIDIA Newsroom / DeepMind Blog / Meta Newsroom / arXiv 原文 / 项目主页。
- 【媒体】：行业博客、课程讲义、二手解读，仅用于补充语境，不单独作为指标依据。
- 【推断】：作者对多个来源的归纳，不代表任何机构立场。
- `[UNVERIFIED]`：检索未直接命中原文，或仅有二手来源。

---

## 1. 世界模型四阶段框架

### 1.1 我们采用的四阶段（用户给定框架）

| 阶段 | 核心能力 | 代表工作 | 数据模态 | 关键指标（公开口径） |
|---|---|---|---|---|
| ① 语言世界模型 | 下一 token 预测、文本内推理，不具身 | GPT 系列、LLaMA 系列 | 文本 | 与本报告主题弱相关；LeCun 2022 称之为 "word model"，批评其不能做规划【官方：https://aegean.ai/book/world-models/jepa】 |
| ② 视觉世界模型 | 像素/视频未来帧生成，无动作条件或弱动作 | Sora、Genie（视频版）、视频扩散模型 | 视频/图像 | 生成质量（FVD/主观评分），不保证物理一致性【媒体：https://dl4ds.github.io/sp2026/static_files/lectures/25_world_models.pdf】 |
| ③ 空间 / 物理世界模型 | 在 3D 表示或潜空间中预测状态演化、可条件于动作 | V-JEPA 2-AC、OccWorld、Cosmos、Genie 2、Dreamer | 视频 + 动作 +（可选）3D 占据/深度 | 下游机器人成功率、预测一致性、规划时延（见 §3） |
| ④ 全域世界模型 | 多智能体、长时记忆、可交互仿真环境、可训练策略 | Genie 2/3 可玩环境、Omniverse/Mega 数字孪生、Dreamer 4 | 多智能体仿真 + 传感器流 | 实时性（fps/延迟）、可扩展性（机器人车队规模） |

### 1.2 外部对四阶段/分层的旁证

- **LeCun 路线（JEPA）**：在潜空间而非像素空间预测未来表示，H-JEPA 主张"低层预测秒级细粒度动态、高层预测分钟级粗粒度子目标"的层次堆叠【官方：https://aegean.ai/book/world-models/jepa；LeCun 2022 position paper】。这与我们的②→③→④递进方向一致。
- **Survey 分阶段（arXiv 2510.20668）**：Stage I 掩码式预训练 → Stage II 统一多模态 → Stage III 交互式生成 → Stage IV 记忆与一致性【官方：https://arxiv.org/pdf/2510.20668】。与本报告四阶段在"交互闭环"和"长时一致"上重合。
- **朱军团队五级路线（L1 生成 → L2 交互 → L3 在世界中行动 → L4 自主智能体 → L5 世界组织者）**【媒体：https://36kr.com/p/3977602492300290】。
- **能力分级（L1 Predictor → L4 Perfect WM）**：L2 潜空间世界模型、L3 生成式世界模型（Sora/Genie 2）、L4 尚不存在【媒体：https://airobotseidos.com/ai-world-models-cognitive-bridge/】。

> 【推断】上述多种分级在"语言/像素 → 空间/动作条件 → 交互/多体"这一主轴上高度收敛，UDOS 的 `future_multimodal` + `counterfactual` + `hierarchical` 组合可视为这一主轴在 5 万参数量级上的**极简投影**。

### 1.3 分层时间尺度（另一条正交轴）

外部综述还把世界模型按"时间跨度 × 空间粒度"切成三层【媒体：https://blog.gpufree.cn/...】：

| 高层 WM | 中层 WM | 底层 WM |
|---|---|---|
| 分钟级 / 场景级 / 任务规划 | 秒级 / 对象级 / 交互预测 | 毫秒级 / 关节级 / 接触动力学 |

【推断】这条轴与 §4 的脑-小-脊髓三层**几乎一一对应**：高层 WM ≈ 大脑，中层 WM ≈ 小脑，底层 WM ≈ 脊髓。UDOS 没有毫秒级关节环，但 `hierarchical`（分钟/任务级）→ `hybrid`/`policy`（秒级）→ `guard`（毫秒/事件级）的频率分层在结构上对齐。

---

## 2. 空间基础模型（Spatial Foundation Model, SFM）代表工作

| 工作 | 机构 | 时间 | 核心方法 | 模态 | 是否开源 | 关键指标 |
|---|---|---|---|---|---|---|
| SpatialVLM | Google DeepMind（Boyuan Chen 等，首作时在 MIT 实习） | 2024-01（arXiv 2401.12168），CVPR 2024 | 在 LLaVA 系 VLM 上，用自动生成的**米制 3D 空间 VQA** 数据微调，注入 chain-of-thought 空间推理 | RGB + 语言 | 权重/数据公开（项目页） | 声称首个 Internet 尺度米制 3D 空间推理数据集；具体样本量 [UNVERIFIED]【官方：https://spatial-vlm.github.io/；https://arxiv.org/pdf/2401.12168.pdf】 |
| V-JEPA 2（空间理解部分） | Meta FAIR | 2025-06（arXiv 2506.09985） | 1B ViT video encoder，100 万+ 小时互联网视频上掩码-去噪自监督，在**表示空间**而非像素空间做预测 | 视频 | 开源【官方：https://arxiv.org/html/2506.09985v1；https://about.fb.com/news/2025/06/our-new-model-helps-ai-think-before-it-acts/】 | 视频理解/预测 SOTA；零样本迁移到 Franka |
| OccWorld | 清华等 | 2023-11（arXiv 2311.16038） | 以 **3D 语义占据栅格**为场景表示，自回归 Transformer 预测未来占据并规划自车轨迹 | 多目图像 → 体素占据 | 开源（论文/代码） | 在自动驾驶占据预测上为早期代表作【官方：https://arxiv.org/pdf/2311.16038.pdf】 |
| 后续占据 WM 谱系 | 多机构 | 2024–2026 | OccLlama/occLLM（引入 LLM 推理）、OccSora（4D 生成）、OccTENS、OccSim（公里级长时） | 体素/占据 | 多数开源 | 趋势：从 2D 像素走向 3D 体素表示以保物理一致【官方：https://arxiv.org/html/2505.05512v1；https://arxiv.org/html/2603.28887v1】 |
| 场景图 / OctoMap | 弗莱堡大学等 | 经典（2010 起） | 概率占据、八叉树多分辨率建图 | 激光/深度 | 开源 | 机器人导航经典，与神经网络 WM 互补【官方：http://ais.informatik.uni-freiburg.de/staff/stachnis/pdf/wurm10icraws.pdf】 |

> 【推断】SFM 的共识：**显式 3D 表示（占据/点云/NeRF/3DGS）或米制 VQA** 比纯 2D 像素生成更能保证几何一致性；UDOS 没有视觉编码器，其"空间"概念只能在符号/状态向量层做类比。

---

## 3. 物理世界模型（Physical World Model, PWM）代表工作

| 工作 | 机构 | 发布时间 | 核心方法 | 模态 | 是否开源 | 关键指标 |
|---|---|---|---|---|---|---|
| **V-JEPA 2 / V-JEPA 2-AC** | Meta FAIR | 2025-06 | 冻结 1B encoder，在 62 小时 Droid 无标注机器人视频上训练 300M 参数 block-causal Transformer，动作条件下自回归预测下一帧表示；通过潜空间 MPC 选动作 | 视频 + 动作 | 开源【官方：https://arxiv.org/html/2506.09985v1】 | 零样本部署于两家实验室的 Franka 臂；简单 reach 任务成功率 100%，复杂任务下降；每步规划约 16 秒【媒体：https://aiwiki.ai/wiki/v_jepa_2】 |
| **NVIDIA Cosmos** | NVIDIA | 2024-01 CES 首发；2025-03 大版本（Cosmos Reason）；2026-05-31 Cosmos 3 | WFM 平台：视频 tokenizer + 扩散生成 + 推理；Cosmos 3 为 Mixture-of-Transformers（视觉-语言推理塔 + 扩散生成塔） | 文本/图像/视频/音频/动作 | 开放（Nano 16B / Super 64B）【官方：https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai；https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3】 | 官方称"排行榜领先的开放物理 AI 基础模型"；具体 benchmark 数值 [UNVERIFIED] |
| **Genie / Genie 2 / Genie 3** | Google DeepMind | Genie 2024-02（11B，arXiv 2402.15391）；Genie 2 2024-12-04；Genie 3 2025 | 仅从视频自监督学习 latent action，文本/图像/草图可条件生成可交互 3D 环境；Genie 2 单张提示图可键鼠游玩 | 视频 + 动作 + 图像提示 | **未开放权重**（研究演示 / Project Genie 预览）【官方：https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/；https://arxiv.org/pdf/2402.15391.pdf】 | 媒体称 Genie 2 达 720p/24fps、约 40ms 交互延迟【媒体：https://www.seeles.ai/resources/blogs/infinite-worlds-ai-real-time-generation.html】 |
| **Dreamer 系列** | Danijar Hafner 等（Google/多伦多大学） | PlaNet 2019；Dreamer 2020（ICLR）；V2 2021；V3 2023（arXiv 2301.04104）；Dreamer 4 2025–2026 | RSSM（Recurrent State-Space Model）：编码器 + 循环隐状态 + 动作条件预测；在潜空间"做梦"并用解析梯度反传训练 policy/critic | 像素 + 动作 + 奖励 | 开源【官方：https://danijar.com/project/dreamer/；https://arxiv.org/pdf/2301.04104.pdf；https://danijar.com/project/dreamer4/】 | V3 在 150+ 任务统一；V4 首次纯离线数据在 Minecraft 取得钻石【官方】 |
| Ha & Schmidhuber World Models | 2018 | VAE + MDN-RNN + Controller，首次系统提出"在学出的世界模型里训练策略" | 像素 + 动作 | 开源 | 概念奠基【媒体：https://blog.gpufree.cn/...】 |

> 【推断】PWM 分两条路线：(a) **像素/视频生成派**（Cosmos、Genie、Sora）——直观但物理一致性弱、贵；(b) **潜空间动力学派**（JEPA、Dreamer、RSSM）——不重建像素，只在压缩表示中预测，天然适配规划与 MPC。UDOS 走的是 (b) 的极简版。

---

## 4. 机器人神经系统分层：大脑 / 小脑 / 脊髓

### 4.1 三层对比表

| 层级 | 时间尺度 | 功能 | 代表方法 | 延迟预算（典型值） |
|---|---|---|---|---|
| **大脑（皮层/前额叶）** | 秒~分钟级 | 任务规划、子目标分解、语义理解、长程记忆；"做什么/为什么" | LLM/VLA、符号规划、MPC（慢规划层）、树搜索 | 100 ms ~ 数秒【媒体：https://asopi.tech/en/blog/20260708_3；https://landx.limxdynamics.com/blogs/aIV7nSdzEZ0IiZ54fhiZFZtR】 |
| **小脑** | 毫秒~百毫秒级 | 边缘轨迹平滑、误差校正、姿态稳定、运动技能学习；"怎么顺畅做" | Diffusion Policy / Flow Matching、轨迹优化、WBC（Whole-Body Control）、MPC（快环）、RL 技能 | 几十~200 Hz（约 5–20 ms）【媒体：https://www.neuromorphiccore.ai/when-robots-learn-to-think-with-their-spine/】 |
| **脊髓（反射弧）** | 毫秒级 | 本地反射：先制动/保护再上报；CPG 节律、关节级伺服、急停 | 反射控制器、PID、安全监控器、SNN 脉冲执行层、硬件急停 | < 1 ms ~ 数 ms，不等待大脑 |

### 4.2 学术旁证

- **三级双下行通路框架**（arXiv 2408.03525）：显式以大脑皮层 / 小脑 / 脊髓为 backbone，在六足机器人上验证【官方：https://arxiv.org/pdf/2408.03525】。
- **NeuroVLA**（arXiv 2601.14628）：皮层规划 + 自适应小脑运动稳定 + 脉冲脊髓快速执行，称首个 neuromorphic VLA 实物部署【官方：https://arxiv.org/html/2601.14628】。
- **CBMC-V3**（arXiv 2511.04109）：全 SNN 实现皮层/小脑/丘脑/脑干/脊髓五模块【官方：https://arxiv.org/pdf/2511.04109】。
- **Subsumption Architecture**（Rodney Brooks，1986/1991）：行为层逐级叠加，低层反射可"吞并"高层规划；"Intelligence Without Representation"【媒体：CSDN 综述；Brooks 经典论文】。这是脊髓/反射优先思想的工程源头。
- **Whole-Body Control / ZMP**：同时优化全身关节，替代早期独立关节控制；人形机器人平衡主流方法【媒体：https://damodev.csdn.net/...】。

> 【推断】三层共识：**高层慢而粗、低层快而硬；低层必须能在高层失联时独立保命**。这与 UDOS `guard.py` + `physical_loop.py` 的设计哲学同构。

---

## 5. 多体协同与数字孪生

- **NVIDIA Omniverse + Mega Blueprint**：以 OpenUSD 构建物理数字孪生，World Simulator 协调多台机器人与传感器流；面向"部署前先在数字孪生里测车队"。Isaac Sim 基于 Omniverse，配合 Isaac ROS / fleet orchestration【官方：https://www.nvidia.com/en-gb/use-cases/robotics-simulation/；https://build.nvidia.com/nvidia/mega-multi-robot-fleets-for-industrial-automation/blueprintcard】。
- **Omniverse vs Cosmos 分工**【官方原文】：Omniverse 负责"建/仿真 3D 世界与工作流"，Cosmos 负责"世界基础模型 + 数据处理/训练/评测框架"；二者可配合使用——Omniverse 出 simulation-ready 场景，喂给 Cosmos 生成数据。
- **Groot Dreams**（GTC 2025）：基于 Cosmos 的合成数据生成蓝图，单图 + 自然语言生成"梦"（合成世界状态），可大规模被动生成【官方：https://www.nvidia.com/en-us/on-demand/session/gtcdc25-dc51184】。
- **多智能体调度**：业界主流是"数字孪生世界仿真器 + fleet orchestrator"分离，而非单一大模型一次预测所有机器人；UDOS 无车队规模，仅在符号层面对照。

---

## 6. 代表工作横向对比表

| 模型 | 机构 | 时间 | 核心方法 | 模态 | 开源 | 关键指标 |
|---|---|---|---|---|---|---|
| V-JEPA 2-AC | Meta | 2025-06 | 1B 视频 encoder + 300M 动作条件潜空间预测器，block-causal | 视频+动作 | 是 | 62h Droid 数据；零样本 Franka；~16s/步规划【官方 arxiv 2506.09985】 |
| Cosmos 3 | NVIDIA | 2026-05 | Mixture-of-Transformers（推理塔+扩散塔） | 全模态 | 是（16B/64B） | 官方称 SOTA 开放；benchmark 数值 [UNVERIFIED]【官方 newsroom 2026-05-31】 |
| Genie 2 | DeepMind | 2024-12 | 动作条件生成式交互环境，latent action | 视频+动作 | 否 | 720p/24fps、~40ms（媒体口径）【官方 blog；媒体 seele.ai】 |
| SpatialVLM | Google DeepMind | 2024-01 | LLaVA + 米制 3D VQA 微调 | RGB+语言 | 是 | Internet-scale 3D 空间数据集；样本量 [UNVERIFIED]【官方 spatial-vlm.github.io】 |
| Dreamer V3/V4 | Hafner 等 | 2023/2025 | RSSM 潜空间想象 + 解析梯度 | 像素+动作 | 是 | V3 跨域统一；V4 Minecraft 钻石纯离线【官方 danijar.com】 |
| OccWorld | 清华等 | 2023-11 | 3D 语义占据自回归 Transformer | 多目→体素 | 是 | 占据 WM 早期代表作【官方 arxiv 2311.16038】 |
| MPC（慢/快环） | 控制论经典 | — | 在模型上滚动优化有限步轨迹 | 状态+动作 | 库众多 | 时间尺度决定其在大脑/小脑哪一层 |
| 反射控制 / Subsumption | Brooks 1986 起 | — | 感知-动作行为层叠加，低层吞并高层 | 传感器→执行器 | 概念开源 | 脊髓层思想源头 |

---

## 7. → UDOS 模块映射表（轻量化机制类比）

> UDOS 为 CPU-only、约 52k 参数合成动力学小模型。下表"类比"仅指**机制结构同构**，参数规模、模态、性能均不在一个量级。

| 外部概念 | UDOS 模块 | 轻量化机制类比 |
|---|---|---|
| V-JEPA 2 / JEPA：潜空间未来表示预测 | `future_multimodal.py` | 在低维隐状态上预测下一状态表示，不"重建"原始观测 |
| Dreamer RSSM：动作条件循环隐状态 | `dynamics.py` + `ctm_engine.py` | 循环隐状态 + 动作条件转移；多步想象用于评估而非大规模训练 |
| 反事实 / counterfactual rollout | `counterfactual.py` | 给定假设动作，前推若干步看状态差，供策略打分 |
| 潜在空间前向模型做 MPC | `policy.py` + `hierarchical.py` | 小窗口滚动评估候选动作序列（对应 V-JEPA 2-AC 的潜空间 MPC，但窗口短、搜索浅） |
| 空间基础模型 / 占据表示 | `identification.py` + `pce_format.py` | 无视觉；在结构化状态向量上做"谁在哪、可不可达"的符号化占据/affordance 标注 |
| Affordance（可供性） | `affordance.py` | 从当前状态直接读出"可执行动作集合"，对应 SFM/VLA 中的 affordance 概念 |
| 大脑（慢规划） | `hierarchical.py` + `reasoning.py` + `longhorizon.py` | 子目标分解、长时程规划，低频运行 |
| 小脑（轨迹平滑/协调） | `hybrid.py` + `policy.py` | 快慢混合控制：慢规划出子目标，快环跟踪并平滑 |
| 脊髓（本地反射弧，先保护再上报） | `guard.py` + `physical_loop.py` | 阈值/安全规则在主策略之外独立触发；越界先制动，事件事后上报 |
| Subsumption（低层吞并高层） | `guard.py` 优先级 | 安全反射优先级最高，可覆盖策略输出 |
| 数字孪生 / 合成数据 | `physical_loop.py` + `ego_data.py` + `adaptive.py` | 在内部合成闭环里产生经验，类比 Omniverse/Cosmos 的合成数据，但仅在 52k 模型内部 |
| 多智能体调度 | [UDOS 暂无对应模块] | 单智能体；`moe.py` / `multitask.py` 仅在任务级，非车队级 |
| ICM（内在好奇心） | `icm.py` / `icm_budget.py` / `icm_cross.py` / `icm_events.py` | 前向/逆模型误差作为探索信号，对应世界模型不确定性驱动探索 |
| 主动学习 / OOD | `active_learning.py` / `ood.py` | 用世界模型预测不确定性决定采样，对应合成数据优先策略 |

---

## 8. 关键发现（3–5 条）

1. **行业已收敛到"潜空间预测 > 像素重建"作为规划友好型世界模型范式**（JEPA/Dreamer 一脉 vs Sora/Genie 像素生成一脉）。UDOS 的 `future_multimodal` + `dynamics` 与前者同构，方向正确。
2. **大脑/小脑/脊髓三层 + 反射优先**已是机器人控制界的事实标准（arXiv 2408.03525、NeuroVLA、CBMC-V3 多篇 2024–2026 论文）。UDOS 的 `guard.py` 独立于 `policy.py` 触发，正是脊髓反射弧的轻量化类比，应在文档中明确这一分工。
3. **空间智能的显式化趋势**（OccWorld/占据栅格/SpatialVLM 米制 VQA）表明纯黑箱向量表示在几何一致性上受限；UDOS 虽无视觉，但 `identification` + `pce_format` 的结构化状态标注是这条路线在符号层的投影。
4. **Cosmos 3（2026-05）与 Genie 3/Project Genie（2026-01）标志着世界模型从"研究演示"进入"开放平台 + 可交互产品预览"阶段**，但两者均为十亿~百亿参数、GPU 集群产物；UDOS 的价值定位应是**机制原型/CPU 教学与闭环验证**，而非能力竞争。
5. **多体协同在业界靠"数字孪生 + fleet orchestrator"工程堆实现，不靠单一世界模型**。UDOS 目前无多智能体模块，若未来要扩展，建议在 `hierarchical.py` 之上加一层调度符号，而非引入第二个大模型。
6. **UDOS 的差异化定位**：外部大模型路线追求"更大数据 + 更多模态 + 更高分辨率"，UDOS 的 52k 参数 CPU 模型不可能、也不应该去拼这条路线。其真正可辩护的价值是：(a) 在 CPU 上闭环验证"潜空间预测 + 反事实 + 分层反射"的最小机制是否成立；(b) 作为 PCE-Format 协议下的**对照实现**，证明同一套状态/动作接口可以承载从 52k 到百亿参数的不同模型。

---

## 9. 实际读到资料清单与覆盖度自评

### 9.1 已读（官方/一手）
- SpatialVLM 项目主页 https://spatial-vlm.github.io/ ；arXiv 2401.12168 PDF 摘要。
- V-JEPA 2 arXiv 2506.09985 摘要 https://arxiv.org/html/2506.09985v1 ；Meta 新闻稿 https://about.fb.com/news/2025/06/our-new-model-helps-ai-think-before-it-acts/ 。
- NVIDIA Cosmos 3 Newsroom https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai ；developer blog https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3 ；2025-03 大版本中文博客 https://blogs.nvidia.cn/blog/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools/ 。
- Genie 2 DeepMind blog https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/ ；Genie arXiv 2402.15391 PDF 摘要。
- Dreamer 项目页 https://danijar.com/project/dreamer/ ；Dreamer V3 arXiv 2301.04104；Dreamer 4 https://danijar.com/project/dreamer4/ 。
- 分层控制 arXiv 2408.03525、2601.14628、2511.04109 摘要。
- OccWorld arXiv 2311.16038；Occupancy WM 综述 arXiv 2505.05512；OccSim arXiv 2603.28887。
- 世界模型综述 arXiv 2510.20668（Stage I–IV）。
- NVIDIA Omniverse/Mega 官方用例页与 blueprintcard。
- JEPA/LeCun 2022 位置论文转述 https://aegean.ai/book/world-models/jepa 。

### 9.2 已读（媒体/二手，仅作语境）
- gpufree 世界模型技术演进博客；seele.ai Genie 2 实时性；aiwiki V-JEPA 2 词条；36kr 朱军五级路线；asopi/neuromorphiccore 脑-小脑-脊髓对照表。

### 9.3 覆盖度自评
- 四阶段框架：**覆盖**（有多方分级旁证）。
- SFM：SpatialVLM / V-JEPA(空间部分) / OccWorld / 占据谱系 / OctoMap **覆盖**。
- PWM：V-JEPA 2 / Cosmos / Genie 2 / Dreamer 全系列 **覆盖**。
- 脑/小/脊髓：三层对比 + Subsumption + WBC **覆盖**。
- 多体/数字孪生：Omniverse/Mega/Groot Dreams **覆盖**；学术多智能体调度（如 MARL 经典工作）**未深入**。

### 9.4 未获取项 / 需后续补
- [UNVERIFIED] SpatialVLM 训练集精确样本量与 benchmark 分数（未读 PDF 全文）。
- [UNVERIFIED] Cosmos 3 Nano/Super 在具体 benchmark 上的数值（官方新闻稿未给具体数字）。
- [UNVERIFIED] Genie 2 参数量（官方未公开；仅媒体称"large-scale"）。
- [FULL TEXT NEEDED] 未逐篇精读 arXiv 2408.03525 / 2601.14628 / 2511.04109 全文，仅基于摘要与转载。
- [未获取] 经典 Subsumption 原文 Brooks 1986 / 1991 论文未直接 fetch。
- [未获取] 多智能体 RL/调度代表作（如 MAPPO、QMIX 一类）未检索，因 UDOS 无多体模块。
- [未获取] LeCun 2022 "A Path Towards Autonomous Machine Intelligence" 原文未直接 fetch，仅经二手转述。

---

*本报告为轻量外部佐证，不构成对任何模型的复现承诺；UDOS 的一切实现以仓库代码为准。*
