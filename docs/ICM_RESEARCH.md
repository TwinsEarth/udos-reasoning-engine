# ICM（In-Context Learning for Manipulation，物理AI上下文学习）四家巨头技术路线深度调研

> 报告日期：2026-09-14
> 调研对象：NVIDIA（MimicDroid + DreamDojo）、Skild AI（S1）、Generalist（GEN-1.5）、自变量机器人（WALL-WM）
> 事实纪律：**官方事实**（论文/官方博客原文）、**媒体陈述**（科技媒体转述）、**分析推断**（本报告判断）三类内容严格分色；所有关键数字均标注来源 URL；核不到的标 `[UNVERIFIED]`，抓不到的标 `[FETCH FAILED]`。

---

## 0. 执行摘要（TL;DR）

1. **ICM 范式正在从"权重"向"上下文"转移**：Skild S1、Generalist GEN-1.5、MimicDroid 三家在 2025-09 至 2026-08 的一年内，先后证明了"人做一遍、机器人看一遍视频当场执行、不改一行权重"在真机上可行。Skild 官方原话："Robotics thus far has been stuck in the BERT era."——这是把机器人学习从 BERT 时代（每任务微调）推向 ChatGPT 时代（上下文即任务说明）的产业级宣言。
2. **数据规模决定 ICL 涌现阈值**：Skild S1 在 1k→100k 小时的受控 scaling 实验中，未见任务成功率从语言提示的个位数爬升到视频提示的 66%（vs 9%）；Generalist GEN-1.5 在 50 万小时+物理交互预训练后，3–12 秒 demo 即触发 one-shot ICL。ICM 不是算法发明，而是数据规模越过临界点后的涌现。
3. **NVIDIA 走"两步走"，自变量走"一步到位"**：NVIDIA 把 ICL（MimicDroid，学术侧 UT Austin 主导）和世界模型（DreamDojo，44k 小时第一视角视频）拆成两条线；自变量 WALL-WM 则直接在预训练阶段就把 ICL 式事件对齐与世界模型目标耦合，用"事件"作为语言/视觉/动作的共同原子。
4. **工程卡点在实时性与上下文带宽**：arXiv:2602.18397 的实测显示，VLA 上下文从 1k 步扩到 10k 步，B100 上推理帧率从 11.7 Hz 跌到 1.2 Hz；Jetson Thor 这类边缘芯片只有在上下文压到 ~100 步时才能勉强到 8 Hz。20 Hz 实时闭环 + 长上下文 demo 是当前 ICM 落地的硬约束。
5. **对 UDOS 的启示**：UDOS 是 CPU-only、52191 参数的合成动力学小模型，**不可能也不应该复现** ICM 大模型；但 `incontext.py` / `temporal_memory.py` / `extended_context.py` / `pce_format.py` / `action_piece.py` / `retargeting.py` / `cache.py` / `batch.py` 这 8 个模块，恰好构成了 ICM 核心机制（上下文窗口、示例拼接、跨本体重定向、动作 token 化、物理 token 协议、推理缓存）的**轻量化机制类比**，可在合成数据上验证形状/接口/数值不变性，而非验证 ICL 效果本身。

---

## 1. ICM 范式定义

### 1.1 什么是 ICM

**官方事实**（Skild AI 官方博客，https://www.skild.ai/blogs/s1）：
> "We believe the main purpose of pre-training — if not the only one — is to enable, for robotics, the same shift we saw from BERT to GPT-3: *in-context learning*, or the ability to learn immediately from one or a few examples."

**官方事实**（Generalist 官方博客，https://generalistai.com/blog/gen-1.5）：
> "It can learn a new task in seconds, from a single example, without gradient updates or fine-tuning."

**定义（本报告归纳，分析推断）**：
ICM（In-Context Learning for Manipulation）= 在不更新模型权重（zero-gradient / no post-training）的前提下，把一段人类/机器人演示视频（或传感运动序列）作为"上下文 prompt"喂给预训练机器人基础模型，模型当场把演示映射到自身本体并闭环执行新任务。任务的"安装包"从权重（weights）转移到了上下文（context）。

### 1.2 三代范式对比表

| 维度 | 微调时代（BERT era） | ICL 时代（ChatGPT era） | ICM 时代（物理AI） |
|---|---|---|---|
| 任务安装方式 | 每任务采集数小时遥操作数据 + 微调专用策略 | prompt 里塞几个示例 | 一段 3–600 秒演示视频进上下文窗口 |
| 是否改权重 | 是（重度 SFT） | 否 | 否（zero-gradient） |
| 新任务部署时间 | 数天–数周数据采集 + 训练 | 秒级 | 分钟级（Skild：盆栽任务从演示到自主执行 11 分钟） |
| 任务说明载体 | 任务特定数据集 | 自然语言 | 演示视频/传感运动序列（physical prompt） |
| 典型数据规模 | 每任务 1–100 小时 | 数千 token prompt | 预训练 1k–500k 小时；in-context 仅 3–12 秒（GEN-1.5）至 10 分钟（S1） |
| 代表系统 | 传统 VLA 后训练、单任务策略 | GPT-3/4 ICL | Skild S1、Generalist GEN-1.5、MimicDroid |
| 失败模式 | 数据不够就崩 | 语言歧义、动作 grounding 缺失 | 长程脆化、对分布漂移敏感、实时性不足 |

来源：Skild 官方博客 Fig.1–Fig.4；Generalist 官方博客 Introduction 段。

---

## 2. 四家路线对比总表

| 维度 | NVIDIA MimicDroid | NVIDIA DreamDojo | Skild AI S1 | Generalist GEN-1.5 | 自变量 WALL-WM |
|---|---|---|---|---|---|
| **核心路线** | 用人类 play 视频作为唯一训练数据，轨迹对预测获得 ICL | 44k 小时第一视角视频学世界动力学，latent action 作代理标签 | 端到端为 ICL 预训练，视频 demo 进上下文窗口执行 | 大规模物理交互预训练，涌现 one-shot ICL | event-grounded VLA 预训练，事件级多模态对齐 + 世界模型 |
| **训练数据量** | 8 小时仿真 play（320k timesteps）+ 真机 [官方事实] | 44,000 小时（44,711h）第一视角人类视频，6015 任务 [官方事实] | 1k–100k 小时受控 scaling；发布时 100k 小时 [官方事实] | GEN-1 时 >50 万小时（2026-04）；GEN-1.5 连续训练 8 个月+ [官方事实/媒体陈述] | 未公开总小时数 [UNVERIFIED]；事件级 caption 数据生态 |
| **是否改权重** | 测试时 ICL 不改权重；与 test-time finetuning 对比 | 后训练阶段在目标机器人数据上微调 | **完全不改权重**（one set of weights） | ICL 模式不改权重；另有 1–10 梯度步快速适配模式 | event mode 不改权重；unified mode 保留 gradient-continuous VLA 路径 |
| **关键指标** | 真机 L1/L2/L3 成功率 0.53/0.23/0.08（vs Vid2Robot 0.28/0.08/0.00，近 2 倍）；比 test-time finetuning 高 26%/29% | 论文未给操作成功率；定位为世界模型基座 [FULL TEXT NEEDED] | 未见任务 per-step 成功率 **66%** vs 语言 VLA **9%**；已见任务 96% vs 89%；单视频 ≈ 380 demos | one-shot 平均 **59%**（±10%）；10 梯度步后 83%（±9%）；1 梯度步/1 分钟数据 66.5% | WALL-SS 桌面双臂平均任务进度 **69.1** vs π0.5 49.6 vs DreamZero 44.1（媒体转述论文） |
| **硬件依赖** | UT Austin + NVIDIA 联合；真机为人类oid [官方事实] | NVIDIA PREDICT/GROOT 基础设施；迁移到 Fourier GR-1 | 基于 NVIDIA AI 基础设施训练；部署硬件未公开 [UNVERIFIED] | 未公开具体硬件 | 未公开；WALL 系列已搭载真机入户 |
| **核心论文/链接** | arXiv:2509.09769（2025-09-11，ICRA 2026 Oral） | arXiv:2602.06949；GTC26 talk s81478 | https://skild.ai/blogs/s1（2026-08） | https://generalistai.com/blog/gen-1.5（2026-08-19） | arXiv:2606.01955（2026-06-01） |
| **发布时间** | 2025-09-11（arXiv v1） | 2026-02（arXiv/GTC26） | 2026-08（博客） | 2026-08-19 | 2026-06-01（arXiv）；2026-06-13 智源大会王昊介绍 |
| **本体归属** | UT Austin 主导（Yuke Zhu 兼 NVIDIA），Amazon Consumer Robotics 合作 | NVIDIA 官方研究 | Skild AI 官方产品 | Generalist AI 官方产品 | 自变量机器人（X Square Robot Team） |

> **重要更正（事实纪律）**：用户原始材料把 MimicDroid 归为"NVIDIA"。经 arXiv:2509.09769 原文核验，论文单位署名是 **UT Austin (1) + Amazon Consumer Robotics (2) + NVIDIA (3)**，第一作者 Rutav Shah、通讯作者 Yuke Zhu（UT Austin 教授，兼 NVIDIA affiliation）。这是**学术合作工作，不是 NVIDIA 主导的产品**，NVIDIA 仅是作者单位之一。NVIDIA 真正主导的是 DreamDojo（GTC26 官方演讲 + arXiv:2602.06949）。

---

## 3. 逐家技术路线详解

### 3.1 NVIDIA：MimicDroid（ICL 学术侧）+ DreamDojo（世界模型基座）

#### 3.1.1 MimicDroid

**官方事实**（arXiv:2509.09769 摘要，https://arxiv.org/abs/2509.09769，2025-09-11 提交）：
- 目标：让人形机器人从少量视频示例高效解决新操作任务。
- 核心创新：用 **human play videos**（人类自由交互的连续无标注视频）作为可扩展、多样化的训练数据源，替代劳动密集的遥操作数据。
- 方法：从 play 数据中提取行为相似的轨迹对，训练策略在一条轨迹条件下预测另一条轨迹的动作——通过这种"轨迹对"训练过程，模型在测试时获得对新物体/新环境的 ICL 能力。
- 跨本体桥接：先用 RGB 视频估计人类手腕姿态，利用运动学相似性 retarget 到人形；训练时用随机 patch mask 减少对人类特有线索的过拟合。
- 评测：开源仿真 benchmark，含递增泛化难度。

**官方事实**（arXiv HTML 正文，https://arxiv.org/html/2509.09769v1）：
> "MimicDroid achieves a success rate of **0.53 in L1, 0.23 in L2, 0.08 in L3**, nearly twofold higher than Vid2Robot, which obtains 0.28, 0.08, 0.00, respectively."

**官方事实**（项目主页 https://ut-austin-rpl.github.io/MimicDroid/）：
> "outperforming test-time finetuning by 26% on the abstract embodiment and 29% on the humanoid embodiment, while incurring only a 3% drop across [distributions]."

**官方事实**（项目主页 evaluation.html）：仿真 play 训练数据仅 **8 小时 / 320k timesteps**（spacemouse 在随机厨房环境自由交互）。

**分析推断**：
- MimicDroid 是"小样本 ICL + 人类视频"路线的学术里程碑，但数据量极小（8 小时仿真 play），与 Skild/Generalist 的十万/五十万小时工业级预训练不在一个数量级。
- 其核心贡献是证明"play 视频 → 轨迹对预测 → ICL 能力"这条数据构造路径可行，但未解决长程任务（分钟级以上）的 ICL。

#### 3.1.2 DreamDojo

**官方事实**（arXiv:2602.06949 摘要，https://arxiv.org/pdf/2602.06949v1）：
> "We introduce DREAMDoJo, a foundation world model that learns diverse interactions and dexterous controls from **44k hours of egocentric human videos**. Our data mixture represents the largest video dataset to date for world model pretraining."

**官方事实**（NVIDIA GTC26 演讲 s81478，https://www.nvidia.com/en-us/on-demand/session/gtc26-s81478/）：
> "DreamDojo is a research experiment we did using the PREDICT model... It was pre-trained on **44,000 hours of egocentric human video using latent actions as proxy labels**."

**官方事实**（DeepWiki 整理 https://deepwiki.com/NVIDIA/DreamDojo）：
- 三阶段训练：Stage 1 从 44k 小时人类第一视角视频学物理动力学，用 LAM 提取的 32 维 latent action（dims 352–384）作动作条件；Stage 2/3 在目标机器人数据上 post-training，迁移到 Fourier GR-1 等人形本体。

**媒体陈述**（aifuturefront，https://aifuturefront.com/nvidia-releases-dreamdojo-...）：具体数据量 44,711 小时，6,015 unique tasks。

**分析推断**：
- DreamDojo 本身**不是 ICL 模型**，而是一个世界模型基座（"action model backwards"——从执行反推画面）。它与 MimicDroid 共同构成 NVIDIA 物理 AI 的"两步走"：MimicDroid 提供 ICL 策略侧的学术范式，DreamDojo 提供视频预训练/世界模型侧的工业化基座。
- 两者在 NVIDIA GTC26 上被并列介绍为 GROOT Dreams 体系的一部分，但论文层面尚未看到"MimicDroid × DreamDojo"直接耦合的公开工作。

---

### 3.2 Skild AI S1

**官方事实**（https://skild.ai/blogs/s1，2026-08 发布，引用日期 2026-09-14 访问）：

**定位**：
- "Our flagship robotic foundation model **S1**, built from the ground up as an in-context learner. Show it a video of a task, short or long, seen or unseen, and it executes."
- "Unseen tasks / 10-minute horizons / One video prompt / No post-training."
- "One set of weights produced every example shown in this blog."

**训练配方**：
- 在 episodic data 上预训练，任务仅通过 in-context demonstration 指定。
- 演示可能来自不同场景、视角、本体；策略必须隐式学习演示者意图、功能对应、任务进度。
- 数据引擎四线并行扩展：遥操作（高硬件接近度、低可扩展性）、UMI（中）、第一视角视频（高多样/高扩展、低硬件接近）、仿真（中硬件接近、低多样、高扩展）。
- "For every dollar we spend on collecting data, we spend three on quality control."

**ICL scaling laws（Fig.4，受控对比：同数据/同架构/同算力，仅 prompt 不同）**：

| 预训练数据量 | ICL（视频 prompt） | 语言 prompt VLA |
|---|---|---|
| 1k 小时（已见任务） | 43% | 53% |
| 100k 小时（已见任务） | **96%** | 89% |
| 100k 小时（**未见任务**） | **66%** | **9%** |

- 未见任务上 ICL 比语言 VLA 高 ~7.3 倍；Skild 归因为：新技能 grounding（语言无法描述"翻煎饼"这种动作原语）+ 组合性（语言太粗，演示直接给出组合）。

**演示效率（Fig.7）**：
- 一条 in-context 视频 ≈ **380 条后训练遥操作 demo**（内插估计）。
- 后训练 2000 条 demo 可达 86%，最终反超 ICL 的 66%。
- 长程任务（>4 分钟）采集 380 条 demo 需 50–100 小时遥操作。

**鲁棒性（Fig.5/6）**：
- L1–L5 分布漂移：物体平移 15cm/30°、30cm/45°、换同类 affordance 物体、半动作换对侧手臂。
- L5 扰动下语言 VLA 退化程度是 ICL 的 3 倍。
- ICL 对"部署场景 vs 演示"的漂移也鲁棒，直到演示本身要求换执行计划。

**涌现行为**：
- 扰动鲁棒（执行中移动物体/换物体/改光照仍完成）；
- 错误恢复（滑板轮组装失败后重试）；
- 物理常识（演示用浇水壶但只有杯子，就用杯子；杯子已满就只补满）；
- 演示纠错（演示者过早打鸡蛋，S1 用受控动作重做）。

**时间线（Fig.8）**：2025-09 LocoFormer（运动 ICL）→ 2026-02 首个 in-domain manipulation ICL 结果 → 2026-05 首次翻煎饼 → 2026-08 S1 发布。

**媒体旁证**（NVIDIA 官方博客 https://blogs.nvidia.com/blog/skild-ai-s1-physical-ai/，2026-09-10）：
> "In Skild's tests on new, multistep tasks, its S1 robot succeeded about 66% of the time at each step, compared with 9% for a similar AI system — a more than sevenfold improvement... one short video example... roughly 380 hands-on training examples."

---

### 3.3 Generalist GEN-1.5

**官方事实**（https://generalistai.com/blog/gen-1.5，2026-08-19 发布，17 分钟阅读）：

**定位**：
- "GEN-1.5... can learn a new task in seconds, from a single example, **without gradient updates or fine-tuning**."
- 大多模态模型，处理 **30 秒记忆**的视频输入 + 其他传感/语言/本体感受输入，输出 **100 Hz** 动作轨迹。
- "the first model we know for which one-shot and few-shot learning of physical skills have emerged at scale."

**Physical Prompt（物理提示）**：
- 3–12 秒传感运动序列（人类手持夹爪或机器人自身 rollout）插入 30 秒上下文窗口；
- 上下文剩余部分滚动存放实时观测；
- 上下文拼接不连续时间跳跃——预训练从未见过这种时间断裂，但模型仍能工作。

**核心指标（Fig.2，10 个任务）**：

| 任务 | 10 梯度步/5 分钟数据 | one-shot ICL/12 秒 demo |
|---|---|---|
| Retrieve money from purse | 83.3% | 60.7% |
| Fold and crease paper | 69.3% | 50.0% |
| Twist lid off glass jar | 94.5% | 60.0% |
| Stack two small cups | 75.0% | 67.0% |
| Sweep trash with brush | 99.0% | 37.3% |
| Open book cover | 82.7% | 54.7% |
| Brush cube into bowl | 71.2% | 60.8% |
| Flip phone upside down | 81.0% | 78.0% |
| Unzip pencil pouch | 86.0% | 55.5% |
| Remove vacuum pad | 86.0% | 64.0% |
| **平均** | **83% (±9%)** | **59% (±10%)** |

**Few-step adaptation**：
- 1–10 梯度步，1–5 分钟数据（~10–50 demos）；
- 1 梯度步/1 分钟数据，held-out 任务成功率 66.5%；
- 10 步权重变化 < 0.15%——"fine-tuning slightly reconfigures knowledge already present rather than building new representations"。

**涌现能力**：
- **组合泛化**：两个独立 physical prompt 拼进上下文，模型自动衔接（中间重定位/重抓/纠错动作不在任一 prompt 中）；
- **零样本 sim2real**：仿真演示作 prompt，真机执行（预训练完全无仿真数据）；
- **人到机器人 ICL**：人用自己的手在机器人镜头前演示，机器人立刻用自己的手复现；
- **工具即兴**：演示用刷子扫块入碗，模型用香蕉当刷子、用簸箕把块铲进碗；
- **双手互用**：演示单手，模型双手旋转罐盖；
- **障碍物移除**：演示数据无纸盖碗，模型自动掀开再放回。

**数据规模**：
- GEN-0（2025）→ GEN-1（2026-04，"mastery" 99%+）→ GEN-1.5（连续预训练 8 个月+）。
- 媒体陈述（36 氪 https://36kr.com/p/3948968465268103）：2025-11 已积累 27 万小时，每周 +1 万小时；2026-04 GEN-1 数据 >50 万小时；GEN-1.5 未公布最新总量。**[注意：50 万小时是媒体转述 GEN-1 时点，非 GEN-1.5 官方数字]**

**为何 ICL 涌现（作者假设）**：
- 物理观测与动作分布可能呈现类似语言的 burstiness / Zipf 结构（Chan et al. 2022）；
- 物理工作含自然重复循环，模型学会检测并扩展这些模式；
- **未**显式设计 ICL 架构、**未**用 MAML 式内外循环、**未**加辅助目标——ICL 是从大规模物理预训练中**涌现**的。

---

### 3.4 自变量机器人 WALL-WM

**官方事实**（arXiv:2606.01955 摘要，https://arxiv.org/abs/2606.01955，2026-06-01 提交）：

标题：*WALL-WM: Carving World Action Modeling at the Event Joints*
作者列表（节选）：Shalfun Li, Victor Yao, Charles Yang, ... Yu Sun, ... Hao Wang, Qian Wang 等数十人（X Square Robot Team）。

核心摘要：
> "WALL-WM is a World Action Model that shifts video-action learning from **chunk-centric optimization** to **event-grounded Vision-Language-Action pretraining**, using semantically coherent action events as the atomic unit of learning."

> 现有 WAM 常见做法：从多模态/视频基座初始化，然后在固定长度 action chunk 上优化，条件是当前观测+指令。这种 chunk-centric 范式造成**粒度错配**：语言描述语义目标/事件，视觉通过连续场景演化，动作在控制级时间尺度——强迫三者进同一固定长度预测窗口，把 VLA 训练变成短视相关拟合。

> WALL-WM 方案：
> 1. 事件 ground 的 VLA 预训练；
> 2. 事件级 caption + 聚类均衡采样的数据生态；
> 3. 两种互补推理模式：
>    - **Event mode**：消费 next-event 描述，变长执行块；
>    - **Unified mode**：VLM + Staircase Decoding 条件化传统定长 chunk 推理，保留 gradient-continuous VLA 路径；
> 4. Muon optimizer 大规模预训练基础设施。
> "Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, achieving state-of-the-art performance in large-scale real-world generalization evaluation."

**官方事实**（智源大会世界模型论坛回顾，https://hub.baai.ac.cn/view/55626，2026-06-18）：
> 王昊介绍的 WALL-WM 以多视角输入和当前指令为条件，联合建模视觉和动作，核心是**变长预测**：复杂任务或动作耗时更长，简单任务耗时更短。事件模式为具身推理提供协议，使系统不仅知道要做什么，还知道每个事件大概持续多久，并以语言形式规划和反馈，把语言智能迁移到其他模态。

**媒体陈述**（凤凰网/爱范儿 https://tech.ifeng.com/c/8vw4TU9r5zl，2026-08-27）：
> 桌面双臂任务中，**WALL-SS 平均任务进度得分 69.1**；作为对比，π0.5 为 49.6，DreamZero 为 44.1，LingBot-VA 为 34.0。

> 注意：69.1 vs 49.6/44.1 是 **WALL-SS**（动作专家）在论文桌面双臂任务上的进度分，不是 WALL-WM 本体的 ICL 成功率。用户原始材料中"真机超 π0.5/DreamZero"应理解为 WALL 系列（含 WALL-WM/WALL-SS）整体在该基准上超过对照。**[UNVERIFIED 具体真机 ICL 成功率数字——论文摘要未给，媒体仅转述进度分]**

**媒体陈述**（澎湃 https://m.thepaper.cn/newsDetail_forward_33641837，2026-07-23）：
> "事件是连接语言、视觉和动作的天然尺度：事件基于语言表达，因此边界清晰；视觉也由事件分割，同一事件内的动作更容易预测。以事件为尺度做变长分割，能够实现对三个模态的天然统一。"

**分析推断**：
- WALL-WM 的"事件级"思路与 Generalist 的"physical prompt"在哲学上相通——都反对固定长度 chunk，主张用语义单元（事件 / 演示片段）作为学习原子。
- 差异：Generalist 是"预训练涌现 ICL"，WALL-WM 是"预训练阶段就把事件对齐显式写进数据/损失"；前者靠规模涌现，后者靠架构-数据协同设计。
- WALL-WM 论文摘要**未直接声称"ICL + 世界模型对齐"**，其 event mode 更接近"用下一个事件描述条件化变长执行"，与 Skild/Generalist 的"视频 demo 进上下文窗口"有形式差异。用户原始材料中"ICL+世界模型对齐"是对该路线的概括，论文原文表述为 event-grounded VLA + 世界动作建模。

---

## 4. ICM 与世界模型的关系

### 4.1 两条路线如何交汇（分析推断）

| 路线 | 代表 | 做法 | 优势 | 风险 |
|---|---|---|---|---|
| **两步走：ICL 策略 + 世界模型基座** | NVIDIA（MimicDroid 策略 + DreamDojo 世界模型） | 先用大规模人类视频预训练世界模型（DreamDojo），再在其上/旁训练 ICL 策略（MimicDroid） | 世界模型可独立用于 planning / 推演 / 安全预演；策略侧迭代快 | 策略与世界模型可能解耦，表征不对齐 |
| **一步到位：ICL 与世界模型在预训练中耦合** | 自变量 WALL-WM | 事件作为语言/视觉/动作/预测的共同原子；event mode 同时服务于 ICL 式条件化与世界状态预测 | 模态对齐天然；事件即规划单元 | 耦合架构调参复杂；公开数据少，可复现性待验证 |
| **纯 ICL 策略（无显式世界模型）** | Skild S1、Generalist GEN-1.5 | 直接大规模物理交互预训练，ICL 从 next-action prediction 中涌现 | 架构简单；scaling law 清晰；工业界可落地 | 无显式世界模型，长程规划/反事实推演能力弱（需靠 prompt 内演示补偿） |

### 4.2 关键观察

- **Skild 与 Generalist 都不做显式世界模型**：Skild 官方博客通篇未提 world model；Generalist 官方博客也未把世界模型作为核心。它们的 ICL 是从 next-action prediction 目标中涌现的，不是从 next-state prediction 目标中学到的。
- **NVIDIA 把世界模型单独成线**：DreamDojo 是"从动作反推画面"的生成式世界模型，MimicDroid 是"从演示条件化动作"的策略。GTC26 把两者放在 Groot Dreams 体系下，但论文层面尚未见耦合。
- **自变量的差异点**：WALL-WM 在预训练阶段就用事件把语言/视觉/动作/预测对齐，这是与前两家最本质的架构差异。事件既是动作 chunk 的粒度，也是世界模型预测的粒度，还是语言规划的粒度——一个原子，三个用途。

---

## 5. 工程卡点

### 5.1 20 Hz 实时性硬约束

**官方事实**（arXiv:2602.18397 "How Fast Can I Run My VLA?"，https://arxiv.org/html/2602.18397v1）：

| 上下文长度 | B100 帧率 | Jetson Thor / RTX 4090 帧率 |
|---|---|---|
| ~100 步 | — | ~8 Hz（勉强实时） |
| 1k 步 | 11.7 Hz | — |
| 10k 步 | **1.2 Hz**（不满足实时） | — |

**官方事实**（NVIDIA GROOT N1.7 on Jetson AGX Thor，https://www.modelscope.cn/learn/434629，2026-06-27）：
- Jetson AGX Thor 上端到端延迟 **49 ms**，达到 **20 Hz** 实时控制频率；
- 整个 per-frame 路径是两个静态 CUDA Graph，热路径无 Python/torch 计算。

**官方事实**（NVIDIA Cosmos 3 Edge，https://developer.nvidia.com/blog/post-train-nvidia-cosmos-3-edge-for-on-device-robot-control/，2026-08-19）：
- Jetson AGX Thor T5000 上，DROID action policy 每 chunk 生成约 1.53 秒（640×540，15 Hz）；
- 单 chunk 覆盖约 2.13 秒机器人运动；chunk 间流水线掩盖延迟。

**分析推断**：
- ICM 的上下文窗口远大于传统 VLA（GEN-1.5 是 30 秒记忆，Skild S1 是 10 分钟 demo）。按 20 Hz 采样，30 秒 = 600 步，10 分钟 = 12000 步。arXiv:2602.18397 的数据显示 10k 步时 B100 已降到 1.2 Hz——**长程 ICL demo 全量上云推理在当前硬件下不现实**，必须用 KV cache 增量 + chunk 流水线 + 边缘部署。

### 5.2 视频 token 带宽 / 延迟

**分析推断**：
- 10 分钟 demo 视频若按 30 fps 抽帧、每帧 1 个视觉 token，仅 prompt 部分就有 18000 token；加上实时观测和动作 token，上下文轻松到数万。
- 这就是为什么 Skild 强调"1 video prompt = 380 demos"但同时承认 2000 demos 后训练仍反超——**上下文窗口长度是 ICL 的硬天花板**。
- GEN-1.5 的 30 秒窗口 / 100 Hz 动作 ≈ 3000 action token，是当前公开数据中最克制的设计。

### 5.3 边缘算力

- 人形机器人主流边缘平台：**NVIDIA Jetson AGX Thor T5000**（GR00T N1.7、Cosmos 3 Edge 都已移植）；Apple/1X 等提到 Redwood [UNVERIFIED——用户原始材料中的 "Redwood" 未在本次检索中找到对应公开技术资料，可能指某芯片代号或项目代号]。
- 真机闭环需要：20 Hz 策略频率 + 视觉编码器 + VLM backbone + 动作专家。arXiv:2603.02271 实测 VLA 推理延迟比实时需求高 200–300 倍，自回归生成占 ~75% 延迟，主要瓶颈是内存带宽。

### 5.4 事件级模态对齐

- WALL-WM 的核心主张：固定频率预测让模型在无关紧要的瞬间反复"刷新"，却错过真正关键的事件。事件级对齐 = 语言边界 = 视觉切分 = 动作单元。
- 对应工程问题：如何自动检测事件边界？如何避免事件级 caption 噪声？如何让事件内动作可预测？这些在公开论文中仅给框架，工程细节 `[FULL TEXT NEEDED]`。

---

## 6. 三阶段 Scaling 框架（分析推断）

> 本节是本报告基于四家路线归纳的分析框架，非任何一家官方表述。

| 阶段 | Scaling 对象 | 核心问题 | 代表 | 数据/算力特征 | 对 UDOS 的类比 |
|---|---|---|---|---|---|
| **阶段 1：数据 Scaling（简智/堆数据）** | 物理交互数据小时数 | 更多遥操作/视频数据 → 更好的单任务策略 | 传统 VLA 后训练、Open X-Embodiment | 每任务 1–100 小时专用数据 | UDOS 合成参数化动力学小模型 = 无真实数据，靠参数化采样 |
| **阶段 2：思维链 Scaling（CTM/推演因果）** | 长上下文推演 / 世界模型预测步数 | 模型不仅预测下一动作，还推演多步后果 | DreamDojo 世界模型、WALL-WM event mode、智源 RoboBrain Orca | 44k 小时人类视频预训练世界模型；事件级变长预测 | UDOS `ctm_engine.py` 因果时间机制、`counterfactual.py` 反事实 |
| **阶段 3：上下文 Scaling（ICM/从演示即时理解）** | in-context 演示窗口长度 × 预训练数据规模 | 不训练权重，只靠 prompt 里的演示学会新任务 | Skild S1（10 分钟 demo）、GEN-1.5（30 秒窗口）、MimicDroid | 100k–500k 小时预训练 + 秒级演示 prompt | UDOS `incontext.py` InContextLearner、`extended_context.py` |

**关键判断**：
- 阶段 1 → 阶段 2 的跃迁：从"动作克隆"到"世界推演"；
- 阶段 2 → 阶段 3 的跃迁：从"推演正确"到"即时学会"。
- Skild 官方 scaling law 是阶段 3 的直接证据：1k 小时时 ICL 不如语言 VLA（43% vs 53%），100k 小时时 ICL 反超（66% vs 9%）——**上下文 Scaling 不是独立能力，是数据 Scaling 越过临界点后的涌现**。

---

## 7. UDOS 落地映射表

> **重要前置声明（事实纪律）**：UDOS 是 **CPU-only、52191 参数、合成参数化动力学小模型**（见 `udos/incontext.py` 模块 docstring），不预训练大模型、不处理真实视频/真机。以下映射**全部为"轻量化机制类比"**——即在合成数据上验证 ICM 概念的形状、接口、数值不变性，**不声称**复刻 ICL 效果。UDOS 实测中 `incontext.py` 的 0-shot MSE=0.0464 但 1-shot=0.6686 / 3-shot=1.1076 反而更差，这本身就是"小模型上 ICL 不涌现"的工程证据。

| UDOS 模块 | 现有职责（读码确认） | 对应 ICM 概念 | 类比强度 | 可在合成数据上验证的事 | 不可做/需标注 |
|---|---|---|---|---|---|
| `udos/incontext.py` `InContextLearner` | few-shot 示例窗口 + task_desc + 查询窗口沿时间维拼接 → `ExtendedContextWindow`；0/1/3-shot MSE 如实测量 | **ICM 核心：上下文窗口里塞演示** | ★★★ 接口类比 | 拼接形状正确；空示例守卫；task_desc 经 scene_params 注入；0/1/3-shot 可跑 | 0.0464→0.6686→1.1076 的退化是已知现象，不能包装成"ICL 效果"；与 Skild/Generalist 的涌现 ICL 无关 |
| `udos/temporal_memory.py` `TemporalMemory` | 环形缓冲（capacity=32）+ EMA 摘要（alpha=0.5），把旧帧前插到当前窗口前 | **上下文窗口的滚动历史 + 长程摘要** | ★★★ 机制类比 | 滑动窗口淘汰逻辑；EMA 摘要更新；与 predictor 集成不改权重 | 无演示视频，只有合成状态帧；无 10 分钟长程能力 |
| `udos/extended_context.py` `ExtendedContextWindow` | 正弦位置编码 + 历史截断/填充；W=6 时逐位等价旧路径 | **长上下文窗口的位置编码 + 截断策略** | ★★★ 基础设施类比 | PE 加/关对数值的影响；max_len 截断；pad_to 填充；W=6 锚点逐位一致 | 不实现 KV cache；无 attention mask 调优 |
| `udos/pce_format.py` `PhysicalToken`/`PhysicsSceneEncoder` | 位置(3)+速度(3)+受力(3)=9 维基础物理量 + 任意 attributes 字典；定长投影到 d_model；含 causal_parents | **物理 Token 协议（PCE-Format）**：把物理状态 token 化送进时序模型 | ★★★ 协议类比 | 变长属性 → 定长向量；causal_parents 字段；JSON 序列化/反序列化 | 不处理视觉 token；不做事件级 caption；与 WALL-WM 的事件对齐仅概念相似 |
| `udos/action_piece.py` ActionPiece | k-means++ 码本把连续动作量化为离散 token；encode/decode 往返；覆盖率统计 | **动作 token 化**（PhysBrain ActionPiece 思想） | ★★ 算法类比 | 码本收敛；往返误差有界；覆盖率统计；空/非法输入守卫 | 仅在合成动作向量上拟合；不碰真机动作/视频/VLM |
| `udos/retargeting.py` | 不同 DOF 维度动作向量代理不同本体；DOF 映射（端点对齐）+ 时间重采样 + 关节限幅 | **跨本体动作重定向**（Human-as-Humanoid 思想）；MimicDroid 的 human wrist → humanoid retarget | ★★ 算法类比 | 源维 S → 目标维 T 确定性近邻映射；时间重采样插值；clamp 到关节限位 | 不涉及 URDF/MJCF；不做运动学相似性 retarget；仅合成向量 |
| `udos/cache.py` `InferenceCache` | 输入张量哈希 + 模型参数 SHA-256 的 LRU 缓存；命中逐位一致；模型换件自动失效 | **长上下文推理的 KV cache / 前缀缓存**（ICM 工程卡点 5.2） | ★★ 工程类比 | 哈希 key 正确；LRU 淘汰；线程安全；参数变自动失效 | 不是 transformer KV cache；不做增量注意力 |
| `udos/batch.py` `BatchPredictor` | 变长序列按长度分组批量；大 batch 分片；可选缓存；与逐笔 predict_next 逐位一致 | **推理时批量/分片**（ICM 边缘部署的 batch 调度） | ★★ 工程类比 | 同长组拼 batch；跨长分别前向；max_shard 分片；空 batch 处理 | 不处理视觉编码器；不做 CUDA Graph 静态图 |

### 7.1 UDOS 现状与 ICM 差距（诚实清单）

1. **ICL 不涌现**：`incontext.py` 实测 1-shot/3-shot MSE 比 0-shot 更差，与 Skild/Generalist 的"涌现 ICL"现象相反。这是预期的——UDOS 没有大规模物理交互预训练，上下文拼接只是把无关示例硬塞进窗口。
2. **无视觉模态**：所有 token 都是 6 维合成物理状态，不处理 RGB/视频。MimicDroid/Skild/GEN-1.5/WALL-WM 的核心信号通道是视觉。
3. **无长上下文**：max_len 默认 64 帧，与 GEN-1.5 的 30 秒/100Hz（3000 token）、Skild 的 10 分钟差 2–3 个数量级。
4. **无世界模型预测**：UDOS 是 next-state 预测小模型，但不是生成式世界模型；不做 next-frame video generation。
5. **映射价值**：这些模块的真正用途是**为 UDOS 未来若接入更大模型/真实数据时，预埋接口形状**——PCE-Format 的 token 协议、ActionPiece 的动作 token 化、Retargeting 的跨本体映射、Cache/Batch 的推理基础设施，都是 ICM 系统中不可或缺的工程层，只是当前在 CPU 小模型上先做形状验证。

---

## 8. 资料覆盖度自评

### 8.1 逐家覆盖度

| 公司 | 覆盖度 | 已核验 | 未获取/待补 |
|---|---|---|---|
| **NVIDIA MimicDroid** | **已核验** | arXiv 摘要 + HTML 正文关键数字（0.53/0.23/0.08 vs 0.28/0.08/0.00；+26%/+29%；8 小时/320k 步训练数据）；项目主页；作者单位归属（UT Austin 主导）；ICRA 2026 Oral | 论文全文 PDF 未读（仅摘要+HTML 正文片段）；真机本体具体型号未确认；开源 benchmark 细节未深入 |
| **NVIDIA DreamDojo** | **部分核验** | arXiv 摘要（44k 小时）；GTC26 官方演讲 transcript（latent action 代理标签、基于 PREDICT）；DeepWiki 三阶段训练整理；MANUS 案例（in-lab 数据） | arXiv 全文 PDF 未读；具体成功率/下游操作指标未获取；与 MimicDroid 是否耦合未公开 |
| **Skild AI S1** | **已核验** | 官方博客全文（scaling law 图、66% vs 9%、380 demos、L1–L5 鲁棒性、时间线、参考文献）；NVIDIA 官方博客旁证；多家媒体交叉 | S1 具体模型架构/参数量/训练硬件未公开（官方说"future posts will go in depth"）；部署本体未公开 |
| **Generalist GEN-1.5** | **已核验** | 官方博客全文（10 任务逐一成功率、59%/83%/66.5%、30 秒窗口/100 Hz、涌现行为、1 梯度步/1 分钟 66.5%）；GEN-1 博客；36 氪媒体旁证 | 50 万小时是 GEN-1 时点（媒体），GEN-1.5 未公布最新总量；具体模型架构未公开 |
| **自变量 WALL-WM** | **部分核验** | arXiv 摘要（event-grounded VLA、event/unified 双模式、Staircase Decoding、Muon）；智源大会论坛回顾（王昊介绍、变长预测）；凤凰网/爱范儿媒体（WALL-SS 69.1 vs π0.5 49.6 vs DreamZero 44.1）；搜狐/澎湃/网易媒体解读 | 论文全文未读；WALL-WM 本体 ICL 成功率数字未在摘要给出；"ICL+世界模型对齐"是用户概括，论文原文未直接用此措辞；作者列表中文实名对应关系未核验；真机入户数据未核验 |

### 8.2 未获取项清单

1. `[FULL TEXT NEEDED]` MimicDroid 论文 PDF 全文（仅摘要 + HTML 正文片段）；
2. `[FULL TEXT NEEDED]` DreamDojo 论文 PDF 全文（44k 小时数据集组成、下游操作指标）；
3. `[FULL TEXT NEEDED]` WALL-WM 论文 PDF 全文（event mode 具体推理流程、真机 ICL 成功率、与 WALL-SS 的关系）；
4. `[UNVERIFIED]` 用户原始材料中 "Jetson Thor / Redwood" 中的 Redwood——未检索到对应公开技术资料；
5. `[UNVERIFIED]` 用户原始材料中 "MimicDroid 成功率提升近 2 倍"——已核验为 L1/L2/L3 三档 0.53/0.23/0.08 vs Vid2Robot 0.28/0.08/0.00，"近 2 倍"在 L1 档成立（0.53 vs 0.28 ≈ 1.9x），L3 档 Vid2Robot 为 0 不可比；
6. `[UNVERIFIED]` 用户原始材料中 "Skild 10 万小时预训练、纯 ICL 不动权重、未见任务成功率 66% vs 语言 VLA 9%"——已核验，全部与官方博客 Fig.4 一致；
7. `[UNVERIFIED]` 用户原始材料中 "Generalist 50 万小时真实物理交互"——50 万小时是 2026-04 GEN-1 时点媒体数字，GEN-1.5 官方博客未给最新总量；
8. `[UNVERIFIED]` 用户原始材料中 "GEN-1.5 one-shot 平均成功率 59%"——已核验，与官方博客 Fig.2 一致（59% ±10%）；
9. `[UNVERIFIED]` 用户原始材料中 ">10fps"——本次检索未找到四家官方明确声称 ">10fps" 的原文；GROOT N1.7 是 20Hz，Cosmos 3 Edge 是 15Hz 采样/640×540，GEN-1.5 是 100Hz 动作输出。此数字需用户澄清指向哪家；
10. `[FETCH FAILED]` 智源社区论文页 "Zero-WAM"（hub.baai.ac.cn/paper/f79d649e...）与 WALL-WM 关系未确认，可能是独立工作；
11. `[UNVERIFIED]` MimicDroid 与 DreamDojo 是否在 NVIDIA 内部已耦合——GTC26 把二者放同一演讲但未公开联合结果。

### 8.3 检索式记录

```
# MimicDroid
- arxiv.org/abs/2509.09769（直接访问，核验摘要）
- "MimicDroid NVIDIA UT Austin Yuke Zhu humanoid manipulation human play videos"
- "MimicDroid success rate real world nearly twofold teleoperation humanoid benchmark"

# DreamDojo
- "NVIDIA DreamDojo world model robot video pretraining 44000 hours"
- NVIDIA GTC26 talk s81478 / s82168 transcript
- arxiv.org/pdf/2602.06949v1
- DeepWiki NVIDIA/DreamDojo

# Skild S1
- "Skild AI S1 robot technical blog 100000 hours pretraining in-context learning"
- "Skild AI S1 blog unseen task success rate 66% language VLA 9% scaling benchmark"
- 直接抓取 skild.ai/blogs/s1（全文 snippet 模式）
- NVIDIA blogs.nvidia.com/blog/skild-ai-s1-physical-ai/

# Generalist GEN-1.5
- "Generalist GEN-1.5 robot one-shot in-context learning 500000 hours"
- 直接抓取 generalistai.com/blog/gen-1.5（分页两次读完全文）

# WALL-WM
- "自变量机器人 Zizai WALL-WM 世界模型 智源大会 2026 in-context learning"
- "WALL-WM 自变量 王潜 事件 世界模型 π0.5 DreamZero 真机"
- "WALL-WM 自变量 in-context learning 世界模型对齐 事件级 模态对齐 王昊"
- arxiv.org/abs/2606.01955（直接访问核验摘要）
- 智源社区 hub.baai.ac.cn/view/55626
- 凤凰网 tech.ifeng.com/c/8vw4TU9r5zl

# 工程卡点
- "VLA robot inference 20Hz real-time Jetson Thor edge latency video token bandwidth"
- arxiv.org/html/2602.18397v1 "How Fast Can I Run My VLA?"
```

---

## 9. 三类内容严格区分

### 9.1 官方事实（来自论文/官方博客原文）

- MimicDroid 作者单位、摘要、真机 L1/L2/L3 成功率、8 小时仿真 play 训练数据；
- DreamDojo 44k 小时第一视角视频、latent action 代理标签、基于 PREDICT；
- Skild S1 全部 scaling law 数字（66%/9%/96%/89%/380 demos/2000 demos 86%）、L1–L5 鲁棒性、时间线；
- Generalist GEN-1.5 全部 10 任务成功率、59%/83%/66.5%、30 秒窗口/100 Hz、1 梯度步/1 分钟数据；
- WALL-WM 摘要中 event-grounded VLA、event/unified 双模式、Staircase Decoding、Muon optimizer。

### 9.2 媒体陈述（科技媒体转述，非原始来源）

- Skild 估值 >140 亿美元（robohorizon.uk）；
- Generalist 50 万小时数据时点（36 氪，2026-04 GEN-1）；
- WALL-SS 69.1 vs π0.5 49.6 vs DreamZero 44.1 vs LingBot-VA 34.0（凤凰网/爱范儿转述论文）；
- 自变量 WALL-B/WALL-WM 发布节奏、融资、入户计划（澎湃/深圳新闻网/腾讯新闻）；
- GROOT N1.7 on Jetson Thor 49ms/20Hz（ModelScope 教程页，转述 NVIDIA 发布）；
- Cosmos 3 Edge on Jetson Thor 1.53s/chunk（NVIDIA developer blog，实际为 NVIDIA 官方）。

### 9.3 分析推断（本报告判断，非任何一方陈述）

- "ICM 范式正在从权重向上下文转移"的产业判断；
- 三阶段 Scaling 框架（数据→思维链→上下文）；
- NVIDIA 两步走 vs 自变量一步到位的路线对比；
- UDOS 8 个模块与 ICM 概念的映射强度评级；
- "上下文窗口长度是 ICL 硬天花板"的工程判断；
- "ICL 不涌现是因为没有大规模预训练"对 UDOS 现状的解释。

---

*报告结束。所有 [UNVERIFIED] / [FULL TEXT NEEDED] / [FETCH FAILED] 标记项见 §8.2。*
