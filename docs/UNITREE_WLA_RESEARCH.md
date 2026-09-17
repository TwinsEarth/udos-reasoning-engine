# 宇树 UnifoLM-WLA-1.0 机制调研 — UDOS v3.9.9 类比线依据

> 核对时间：2026-09-15。用途：为 UDOS v3.9.9（3.9.0…3.9.9，12 节点）提供**机制类比（analogy, not reproduction）**依据。
> 全部 CPU 合成数据、~52k 参数小模型，**不复刻 6B、不下载大权重**。
> 证据分级：【官方】= 一手项目页原文；【代码/权重-待核】= GitHub/HF 入口存在但本次未能逐件复核；【媒体】= 二手转述；【推断】= 本工程分析。

## 0. 对备料的勘误（落实）

1. **参数规模**：官方项目页原文为 **"6B parameters"**，其中具身推理骨干 **UnifoLM-ER-1-4B 基于 Qwen3-VL-4B**。【官方】备料中"6 亿"系误写，**以官方 6B / ER-1 4B 为准**。
2. **"全开源"存在交付物时间差**：项目页挂三个入口 Code/Models/Datasets（同指向 HF collection `unitreerobotics/unifolm-wla-10`）。本次复核：
   - `github.com/unitreerobotics/unifolm-wla` —— **被 robots.txt 拒绝自动抓取，代码/README 本次未能逐件核**，标【UNVERIFIED】。
   - HF collection `huggingface.co/collections/unitreerobotics/unifolm-wla-10` —— **本次抓取失败，权重/数据集是否可直接下、文件清单未能核**，标【UNVERIFIED】。
   - **License 媒体陈述冲突**：alpha-bionic（2026-09-11）称 checkpoint 为 **CC BY-NC-SA 4.0（非商用）**；humanoidnews.ru（2026-09-10）称 **Apache-2.0**。二者矛盾，**以官方 LICENSE 文件为准但本次未能核到，标【UNVERIFIED/媒体陈述冲突】**。
   - 结论：**不下载任何官方权重/代码/数据**；本线仅做机制类比，license 风险与本工程产物无关。
3. **前代 WMA-0 已真正开源**【代码-已核到二手结构】：DeepWiki 确认 `unitreerobotics/unifolm-world-model-action`，HF 有 `UnifoLM-WMA-0-Base` / `UnifoLM-WMA-0-Dual`，数据管线 prepare_training_data.py 把 LeRobot V2.1 转 MP4+HDF5+CSV。这是 CPU 类比最有抓手的**已开源前代**。
4. **16 项 benchmark 非统一独立榜单**：官方表注星号 `*` = 各模型技术报告/公开论文值；‡ = 用官方 API 自测；BLINK 只取 Relative Depth + Spatial Relation 两子任务（†）。ER-1 在其中 7 项领先开源、整体称"与头部专有相当"——**广口径自评，非同条件排行榜**【官方】。
5. **64 任务真机演示为覆盖声明**：项目页标注 Autonomous / 2x Speed，无每项次数/成功率/换本体泛化的独立证据【官方】。
6. 备料提及的 OminiA-0.3、UnifoLM-X2-1.0、研发投入/估值、王兴兴预测——**本次未在项目页一手证实**，标【待核/媒体陈述】，本线不采用。

## 1. 模型家族脉络【官方 + 二手结构】

- **UnifoLM-WMA-0**（World-Model-Action，已真正开源）：LatentVisualDiffusion 总编排 / AutoencoderKL / WMAModel 时空扩散 / ConditionalUnet1D；LeRobot V2.1 数据 → MP4(视角)+HDF5(状态/动作/归一化)+CSV(索引)。
- **UnifoLM-ER-1**（Embodied Reasoner，4B，Qwen3-VL-4B 底座）：具身推理。
- **UnifoLM-ER-Flow**：ER-1 + 未来动态区域预测 + 离散动作对齐 → 统一 VLA 表征。
- **UnifoLM-WLA-1.0**（6B）：ER-Flow 骨干（**Stop-Gradient 冻结**）+ **MMDiT 动作专家**，输出连续控制动作。

## 2. 三层架构 + 动作专家（核心，【官方】）

### 层 1 · 具身推理 ER-1
- 底座 Qwen3-VL-4B；**500 万+ ER 样本**，6 类监督：图像点预测、目标检测、多图推理、2D 轨迹预测、3D 目标检测、多图空间问答；与通用图文数据 co-train（保通用 VLM 能力同时注入空间理解）。
- 产出不仅是语言，还含结构化空间/轨迹锚点。
- 16 benchmark：ER-1-4B 在 RoboVQA 62.4 / RefSpatial-Bench 61.7 / EmbSpatial 88.9 等 7 项领先开源【官方表】。

### 层 2 · 未来动态区域预测（interaction-centric world model，关键差异化）
- t0/t1 相邻帧用**光流**提取"发生变化的动态区域 mask"；**VQ-VAE** 把动态 mask 编码为定长离散 token。
- 以"当前图 + 任务描述 或 动作"为条件，VLM 直接**预测未来动态区域的 mask token**（示例 token 序列 `0821370419421126`）。
- **主张：不生成整帧未来画面，只预测"动作会改变哪里"**（稀疏、面向交互主体）。
- → 与 UDOS 3.6 PWM **整态稠密 rollout** 正好构成"稀疏变化掩码 vs 稠密全态"可证据化 A/B。

### 层 3 · 离散动作学习（统一动作空间 + RVQ）
- 统一动作空间分**三个分量**【官方原文】：
  1. 末端执行器位姿 **EEF poses**；
  2. 末端执行器关节 **EEF joints**（夹爪/灵巧手）；
  3. 下肢关节 **lower-body joints**。
- **每个分量单独训练一个 RVQ（残差矢量量化）**，把连续动作轨迹离散为 token；三路在**共享时间步对齐**后同步送入 VLM，与视觉、语言在单 VLM 内联合对齐 → ER-Flow。
- 官方示例 token：`<EEF_START>18 42 07 31 56<EEF_END>` / `<HAND_START>04 29 51 16 38<HAND_END>` / `<LOWER_START>27 11 44 03 22<LOWER_END>`。

### WLA · MMDiT 动作专家（连续动作解码）
- ER-Flow 骨干对动作专家 **Stop Gradient（冻结）**；动作专家是 **MMDiT flow decoder（流匹配/扩散式去噪）**：输入 VLM hidden state + Embodiment State + 带噪动作 Noise Actions（配时间步嵌入 t）。
- 块内含 **GateMLP / Scale&Shift / NormGate / Self-Attention(q/k/v+RoPE+QK-Norm) / AdaLN 式调制**；输出**连续动作**（示例 -1.7 / 1.25 / 3.14 / 1.42）。
- 即"**离散 token 做跨模态对齐（ER-Flow，冻结）+ flow 解码器出连续动作（WLA 专家）**"双阶段。

## 3. 数据 / 本体 / 任务【官方】

- ≈**2500h 高质量真机**；含 **Unitree Open Datasets + BitRobot-HIW-500**（Humanoids in the Wild 500h）；**5M+ ER 样本**；多本体多场景。
- 平台 **Unitree G1**；末端：**平行夹爪 + 两类五指灵巧手**（跨末端复用同一策略）。
- **64 任务 = 10 全身移动操作**（取垃圾/放洗衣机/整理鞋/卫浴/厨房/铺床/取菜/放柜/上架/整理沙发）+ **54 桌面操作**（叠毛巾/叠布/叠裤/收纳盘/充电/收纸杯/拆风车/找绿块/插花/整理笔袋/乒乓球拍/工具墙/工具箱/打包手机/捡电池/踢足球…项目页有完整枚举）。
- 统一动作空间 → 主张**跨本体先验迁移**（不同本体/末端不单独训策略）。

## 4. → UDOS v3.9.9 机制映射表（CPU 合成数据可证据化类比）

| WLA-1.0 机制【官方】 | UDOS 3.9 轻量类比（opt-in，均需 A/B 证据） | 复用既有 |
|---|---|---|
| ER-1 统一具身推理头 | 编排既有 SFM(spatial)/affordance/unified-head，新增统一"具身推理聚合头"：空间关系+目标点+2D/3D 轨迹代理，**不重造** | spatial / affordance / unified_head |
| 光流→动态区域 mask | 合成相邻状态**差分**提取"真实变化分量"作监督（光流的状态域等价物），预测**稀疏 change mask** 而非全态 | dynamics 窗口 |
| VQ-VAE 编码动态区域 | 对 change mask 做小型离散码本量化，报码本利用率/重构误差/坍塌 | 新 changemask_vq |
| 稀疏变化 vs 整帧世界模型 | change-mask 稀疏预测 vs 3.6 PWM 稠密 rollout 的精度/成本 A/B | world_model (LatentWorldModel) |
| 统一动作空间三分 | 动作按 末端位姿/末端关节/下肢 三分组（合成 DOF 分组） | action/retargeting/neural_control |
| 三路 RVQ 动作分词 | 每分组一个轻量 VQ/RVQ 码本把连续动作离散为 action token；报利用率/重构误差/坍塌 | 新 action_rvq |
| ER-Flow 单 VLM 对齐 | 离散动作 token 与状态/任务 id 在同一表征内对齐（轻量对齐损失+一致性测试） | 新 embodied_align |
| ER-Flow stop-gradient + MMDiT flow 专家 | 主预测器冻结（延续外挂零梯度传统），外挂轻量 **flow-matching 少步动作解码器**，与直接回归动作 A/B；**明确非复现 6B MMDiT** | icm/hybrid 零梯度先例 |
| 跨本体/跨末端泛化 | 统一动作空间 + DOF 分组 + retarget：同一权重适配不同 DOF/末端代理，跨本体迁移 A/B | retargeting (ActionRetargeter/MorphologyLibrary) |
| 16 benchmark 多任务评测 | 合成任务族小型"多任务统一头"评测矩阵 | multitask / eval_suite |

## 5. 来源覆盖度与未获取项

| 来源 | 状态 | 结论 |
|---|---|---|
| 官方项目页 unigen-x.github.io/... | ✅ 已实访复核全文 | 架构图/benchmark 表/64 任务枚举/数据参数口径均一手确认 |
| github.com/unitreerobotics/unifolm-wla | ⚠️ robots 拒绝自动抓取 | 代码/README/license **本次未逐件核 [UNVERIFIED]** |
| HF collection unifolm-wla-10（权重+数据） | ⚠️ 抓取失败 | 文件清单/可下性/license **本次未核 [UNVERIFIED]**；媒体称 CC BY-NC-SA 4.0 与 Apache-2.0 冲突 |
| 前代 WMA-0 代码结构 | ✅ 二手 DeepWiki 已核 | 组件/数据管线可作 CPU 类比抓手 |
| 媒体（humanoidsdaily/alpha-bionic 等） | ✅ 二手 | 仅作旁证，不进本工程数值 |
| 完整训练集构成/推理延迟/真机成功率 | ❌ 未获取 | 标 [待核]，本线不引用 |

**本工程不下载任何官方权重/代码/数据**；全部机制在 CPU 合成数据上自洽复现，数字落 `benchmarks/results/*.json` 可复算。
