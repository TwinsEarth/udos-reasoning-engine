# UDOS 开源生态目录：模型库 / 数据库 / 动作库横向对比

> 调研日期：2026-09-16 ｜ 纯研究交付，未改代码、未升版本 ｜ 面向 UDOS 物理 AI 推演引擎（CPU-only、无 GPU、零重依赖机制原型）
> 机器可读总表：[opensource_catalog.json](./opensource_catalog.json)（79 条，统一 15 字段）

---

## 0. UDOS 选型方法论

UDOS 是 **CPU-only、无 GPU、零重依赖、不下载大权重/视频** 的物理 AI 推演引擎机制原型。因此对开源资源的选型遵循以下优先级：

1. **开放许可优先**：MIT / Apache-2.0 / BSD > CC-BY > CC-BY-NC（非商用）> 未公开。商用分发场景下非商用 license 的数据/权重只能作架构参考，不能打包。
2. **轻量可跑优先**：纯文本/BVH/NPZ/HDF5/Parquet 格式 > 需 GPU 推理的大权重 > 仅论文无代码。CPU 能解析格式/跑最小 demo 的资源可直接对接；大权重只作架构借鉴。
3. **格式可解析优先**：数据 schema 清晰、元数据独立于大文件的资源（如 LeRobot Parquet+MP4、RoboMimic HDF5、LAFAN1 BVH）可转为 UDOS PCE-Format；需专用仿真器（Isaac Sim/Omniverse）的仅作参考。
4. **架构可借鉴优先**：与 UDOS 现有模块同构的设计（如 DreamerV3 RSSM ↔ world_model latent imagine、ACT CVAE+chunk ↔ wla change-mask、Sakana CTM ↔ ctm_engine）即使权重不可用，代码和论文也有高参考价值。
5. **大权重只作参考不下载**：7B+ 级 VLA/世界模型（OpenVLA、π0、Cosmos、PhysBrain 8B）在 CPU 上无法推理，仅研究其 token 化、动作分词、架构设计。

---

## 1. 模型库对比总表（26 条）

### 1.1 VLA / 策略模型

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **LeRobot** (含 SmolVLA/SO-100) | HF | 码✅权✅数✅(481集) | Apache-2.0 | SmolVLA ~450M | RGB+本体+语言;动作chunk | SO-100/101,Koch | 高:CPU可解析数据/跑管线 | **高** — wla chunking、gpm 离线范式 |
| **OpenVLA** | Stanford/TRI | 码✅权✅数复用OXE | 码MIT;权重受Llama-2约束 | 7B;970k轨迹 | RGB+语言;256-bin动作 | 22本体 | 低:仅架构 | 中 — 动作分箱↔wla VQ;LoRA↔self_train |
| **Octo/Small/Base** | Berkeley等 | 码✅(MIT)权✅数复用OXE | MIT | 27M/93M;~800k轨迹/9本体 | RGB+语言/goal+本体;扩散头 | 9本体 | 中:27M可CPU慢跑 | **高** — embodiment adapter+扩散头↔wla flow/gpm |
| **openpi (π0/π0.5)** | Physical Intelligence | 码✅权✅数❌(10k h内部) | 码Apache类;权重[UNVERIFIED] | π0~3B/π0.5~3.3B;10k h | RGB+语言;flow/FAST连续动作 | 多本体/移动臂 | 低:仅架构 | **高** — flow动作专家/FAST分词↔wla flow/RVQ |
| **RT-X / Open X-Embodiment** | Google DeepMind+33实验室 | 码✅数✅(1M+/22本体)权部分 | 子数据集混合 | RT-1-X 35M/RT-2-X 55B | RGB+本体+语言 | 单/双/四足 | 中:CPU可解析tfrecord | 中 — 数据混合加权;RLDS时序↔ctm |
| **RDT-1B** | 清华TSAIL | 码✅权✅数✅ | 码MIT;权重license冲突[UNVERIFIED] | 1.2B DiT;46集1M+;ALOHA 6k | RGB×3+语言;64步动作;双臂14维 | ALOHA双臂 | 低:仅架构 | 中 — DiT扩散头↔wla flow |
| **CogACT** | 清华/微软亚研 | 码✅权✅ | MIT | ~7B级[UNVERIFIED] | RGB+语言;扩散头 | 操作臂 | 低 | 中 — 认知-动作协同↔latent_reasoner+wla |
| **SmolVLA** | HF LeRobot | 码✅权✅数✅ | Apache-2.0 | ~450M | RGB+语言+本体;扩散头 | SO-100等 | 中:管线可读 | **高** — 小模型+扩散头=UDOS轻量路线最佳对照 |
| **TinyVLA** | 社区 | 码✅权✅ | 随上游[UNVERIFIED] | 冻结主干+线性投影+扩散解码 | RGB+语言;扩散解码 | 操作臂 | 中:范式可读 | **高** — 冻结表征+小动作头=CPU-only gpm范式 |
| **MobileVLA** | 社区 | 码✅权✅ | 权重LLaMA-3条款 | LLaMA3-8B+LoRA | RGB+语言+本体 | 移动机器人 | 低 | 中 — LoRA/R1↔self_evolution |
| **GR-1/GR-2** (≠NVIDIA GR00T) | 清华/字节Seed | GR1/2码权未完整开放;GR00T码权✅ | GR00T码Apache/权重按版本 | GR1:800k Ego4D;GR2:38M视频;GR00T~3B | RGB+语言+本体;联合预测未来帧+动作 | 臂/人形 | 低:GR1/2仅论文 | **高** — 动作+未来帧联合预测↔world_model latent imagine |

### 1.2 动作生成 / 经典策略

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **Diffusion Policy** | Stanford/Columbia | 码✅(MIT)权✅示例 | MIT | 小模型;chunk 16-32 | RGB+本体;扩散生成动作块 | 通用臂/ALOHA | **高:CPU可跑PushT小demo** | **高** — chunking+去噪=wla flow最小可读实现 |
| **ACT (ALOHA)** | Stanford | 码✅(MIT)权✅ | MIT | 小CVAE;chunk~50 | RGB×2+本体;潜变量z | ALOHA双臂 | **高:CPU可跑小demo** | **高** — CVAE多峰动作+temporal ensembling↔wla change-mask |
| **VIMA** | Stanford(李飞飞) | 码✅(MIT)权✅数✅(VIMA-Bench) | MIT | 2M/20M/200M | RGB+多模态prompt | 仿真臂 | 中:2/20M可CPU | 中 — in-context多模态组装↔icm/reasoning_router |

### 1.3 世界模型

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **V-JEPA / V-JEPA 2** | Meta FAIR | 码✅(主MIT)权✅数❌ | **码MIT;权重CC BY-NC 4.0(非商用)** | ViT-L/H/g十亿级;1M+h视频 | RGB视频;隐空间预测;action-conditioned | 本体无关 | 低:仅架构 | **高** — 隐空间预测未来不重建像素=world_model范式;注意非商用 |
| **NVIDIA Cosmos** | NVIDIA | 码✅权✅数✅(Cosmos3) | Cosmos1:NVIDIA Open;Cosmos3:OpenMDW-1.1 | Diffusion 7B级;全模态 | 视频扩散世界生成+动作条件 | AV/机器人 | 低:仅架构/tokenizer | 中 — 连续视频tokenizer↔world_model/wm_conservation |
| **Dreamer / DreamerV3** | Hafner等 | 码✅(MIT)权❌(算法) | MIT | RSSM;12M→400M;单V100 | RGB/低维+动作;categorical潜状态 | 通用RL域 | **高:CPU可跑小域demo** | **高** — RSSM潜世界想象=world_model+neural_control+closed_loop最易读参照 |
| **Genie** | Google DeepMind | 码❌权❌数❌ | 无(未开源) | 11B;~30k h;160x90@1fps | 单图/文本→可交互2D世界;latent action | 虚拟世界 | 无 | 低:latent action思路↔wla动作隐变量 |

### 1.4 特殊关注（PhysBrain / UnifoLM / ICM线 / 未开源）

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **PhysBrain 1.5** | DeepCybo深度机智 | 码✅(EvalKit)权✅(2B/8B)数部分 | **[UNVERIFIED,基于Qwen3-VL-8B需继承license]** | 2B/8B;未来状态1.2M样本 | RGB+深度(MoGe-2)+机器人mask+语言+末端轨迹;自回归统一 | Franka等 | 低:仅架构/EvalKit | **高** — VLM统一编码动作+未来RGB/depth/mask=world_model+wla+spatial三合一 |
| **UnifoLM-WLA-1.0** | Unitree宇树 | 码✅权✅(HF)数部分(Z1/G1) | **[UNVERIFIED,2026-09-10宣布全开]** | **官方口径6B**(ER-1-4B主干+MMDiT flow;~2500h真机;5M+reasoning;64任务) | RGB+语言+本体;MMDiT flow | G1/Z1 | 低:仅架构 | **高** — WLA命名+flow动作专家直接对标UDOS wla/world_model |
| **MimicDroid / DreamDojo** | 多机构 | 论文✅;码权[UNVERIFIED] | [UNVERIFIED] | MimicDroid:play视频ICL;DreamDojo:44k h第一人称视频,连续latent action(VAE) | RGB人类视频;连续latent action | 人形/臂 | 无权重;架构可读 | **高** — 连续latent action自监督↔wla RVQ/icm;AC-WM↔world_model |
| **Skild S1** | Skild AI | 码❌权❌API❌论文❌ | 无(未开源) | 未披露;1段视频学10min任务(厂商自报66%) | RGB演示视频+本体;ICL不更新权重 | ABB/UR/MiR | 无 | 低:ICL范式↔icm |
| **Generalist GEN-1.5** | Generalist AI | 码❌权❌API❌ | 无(未公开) | 未披露;30s记忆;100Hz;3-12s demo | 视频+传感+语言+本体 | 自有机队 | 无 | 低:30s记忆窗↔icm/wm_events |
| **WALL-WM/OSS/SS** | 自变量X-Square | 码✅(wall-x)权部分数部分 | **[UNVERIFIED,以wall-x LICENSE为准]** | WALL-OSS-0.5;WALL-WM事件级;WALL-SS可推演60s | RGB+语言+动作;**事件级时序** | 自研本体/灵巧手 | 低:事件化代码可读 | **高** — 事件级(非固定窗)↔wm_events/ctm_engine;WALL-SS虚拟预演↔closed_loop |
| **CTM / ctm-imagenet** | Sakana AI | 码✅权✅ | Apache-2.0 | 连续时间ODE式;ImageNet checkpoint | 图像(非机器人);连续同步推理 | 无 | **高:ImageNet小ckpt可CPU跑** | **高** — 连续时间推理=UDOS ctm_engine(52191参数)命名与机制最直接外部参照 |
| **Doc-to-LoRA** | Sakana AI | 码✅checkpoint✅ | Apache-2.0 | Perceiver超网(8层cross-attn)→一次前向生成LoRA | 文本(非机器人);上下文→小适配器 | 无 | 中:小示例可CPU | 中 — 上下文→小适配器↔icm外挂记忆/self_train技能内化 |

---

## 2. 数据库 / 仿真任务库对比总表（26 条）

### 2.1 真实机器人数据集

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **Open X-Embodiment/RT-X** | Google DeepMind+21-34机构 | 码✅权✅数✅(1M+轨迹RLDS) | 码Apache2.0;数据多CC-BY4.0 | 1M+轨迹,22本体,60子数据集,527技能 | RGB多视角,部分深度,本体,语言,动作 | 22种 | 元数据/schema CPU可解析 | **高** — wla跨本体迁移+PCE-Format转换模板+ctm |
| **DROID** | Stanford/Google/18实验室 | 码✅权✅数✅(76K轨迹) | 数据CC-BY4.0;码Apache2.0 | 76K轨迹,350h,564场景,86任务,~1.7TB RLDS | 3xZed2 RGB-D+手腕RGB+语言+7DoF | Franka Panda | HDF5/元数据CPU可解析 | **高** — ctm时序动力学+wla动作分词+build_parametric_dataset |
| **BridgeData V2** | UC Berkeley RAIL | 码✅数✅(60K轨迹) | CC-BY 4.0 | 60,096轨迹,24环境,13技能,~400GB | RGB-D,本体,夹爪,语言 | WidowX 250 6-DoF | 元数据CPU可解析 | 中 — ctm(5Hz低频建模)+wla(低成本臂动作空间) |
| **RH20T** | 清华等 | 码✅数✅(110K序列,需申请) | 分段:CC-BY-SA4.0(可商用)/CC-BY-NC4.0(不可商用) | 110K+序列,140+任务,7种臂,~40TB | 8-10路RGBD,力/力矩(100Hz),音频,触觉(200Hz),语言 | 7种臂 | 元数据CPU可解析;体量极大 | 中 — wla(力触觉扩展)+spatial(多相机标定) |
| **RoboMIND 2.0** | 北京人形机器人创新中心 | 码✅权✅数✅(ModelScope/HF) | Apache 2.0 | V1:107K/4本体;V2:310K+/6本体/739任务/~15.58TB;含触觉+失败标签 | RGB-D,本体,动作,语言,触觉(V2),力/力矩,失败标签 | Franka/UR5e/双臂/天工人形等6种 | HDF5元数据CPU可解析 | **高** — wla(跨本体迁移核心)+ctm(失败恢复)+digital_twin |
| **AgiBot World** | 智元机器人 | 码✅权✅数✅(HF) | 开源许可 | 1,001,552轨迹,2,976h,217任务,87技能,106场景,~43.8TB;100+机器人 | RGB多相机,深度,本体,动作,语言,触觉,灵巧手 | AGIBOT G1/G2双臂人形 | 元数据CPU可解析;Alpha子集~8.5TB | **高** — wla(人形双臂)+ctm(长时序接触)+world_model(自带30K+WM训练集) |
| **Unitree + HIW-500** | 宇树+BitRobot+HF | 码✅权✅数✅(HIW-500 HF) | 开源(各数据集页) | HIW-500:500+h,23,743轨迹,11任务,12家庭,~10TB/~2.15TB LeRobot;Unitree:G1灵巧手/G1夹爪/Z1双臂/UnifoLM-WBT | 全身RGB,关节状态,全身动作,子任务标注(148K+) | Unitree G1人形 | **高** — LeRobot格式Parquet CPU解析 | **高** — wla(全身动作分词-同源)+ctm(长时序)+multi_agent(双臂) |
| **LeRobot社区数据集** | HuggingFace | 码✅(Apache2.0)权部分✅数✅(100+数据集) | 各数据集独立(多Apache/MIT/CC-BY) | 100+数据集,数千~数万示范,多平台 | RGB(MP4),本体(Parquet),动作(Parquet),部分语言 | 多种(SO-100/ALOHA/Franka等) | **很高** — Parquet纯CPU解析;MP4轻量 | **高** — PCE-Format(LeRobot→PCE模板)+wla(跨本体标准化) |
| **Ego4D** | Meta+国际联盟 | 码✅数✅(需license审批~48h) | 研究许可(非商用,不可再分发) | 3,670h视频,923参与者,74地点,9国 | RGB(头戴),音频,语音描述 | 人类(第一视角) | 标注JSON CPU可解析;视频抽帧CPU可行 | 中 — ctm(人类时序先验)+world_model(第一视角物理先验) |
| **Ego-Exo4D** | Meta+联盟 | 码✅数✅(需license~2天) | 研究+商用许可(不可再分发) | 1,286h视频,5,035 takes,740参与者,~38.4TB | 同步ego/exo RGB,音频,3D姿态,米制坐标系 | 人类(双视角) | 标注CPU可解析;视频抽帧CPU可行 | 中 — ctm(多视角对齐)+spatial(米制标定)+world_model |
| **EPIC-KITCHENS** | U Bristol | 码✅数✅ | 非商业研究许可 | EPIC-100:100h,45厨房,20M帧,89.9K动作 | RGB(GoPro),音频,自叙述,边界框 | 人类(厨房第一视角) | 标注CPU可解析;视频抽帧CPU可行 | 低-中 — ctm(厨房时序)+latent_reasoner(物体交互推理) |
| **Something-Something v2** | 20BN/Qualcomm | 数✅(需注册) | 研究用途免费(非商用) | 220,847短视频,174动作类 | RGB视频(2-6s) | 人手/物体交互 | **很高** — 短视频MP4 CPU轻松解码 | 低 — ctm(直觉物理)+world_model(物理常识) |
| **ALOHA / Mobile-ALOHA** | Stanford | 码✅权✅数✅(GDrive/LeRobot) | 开源(代码) | 静态~825ep(RT-X);Mobile 50示范/任务;ALOHA Unleashed 26K ep | RGB多相机,关节状态(14DoF双臂),底座速度 | 双臂ALOHA | **高** — HDF5元数据CPU可解析;LeRobot Parquet友好 | 中 — wla(双臂动作分词)+neural_control(低层控制器)+ctm(双臂协调) |

### 2.2 仿真任务库

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **LIBERO** | UC Berkeley/Sony AI | 码✅权可选✅数✅(HDF5~40GB) | MIT | 130任务(5套件),HDF5示范~40GB | RGB-D,本体,语言任务描述 | Franka Panda(MuJoCo) | 中 — MuJoCo CPU可跑(慢);HDF5元数据CPU可解析 | **高** — curriculum(终身学习基准)+closed_loop(仿真评估)+selfplan |
| **CALVIN** | U Freiburg | 码✅(MIT)权✅数✅(~166GB) | MIT | 34任务,1000语言指令,4环境ABCD | RGB-D多相机,本体,语言 | Franka Panda+平行夹爪 | 中 — PyBullet/MuJoCo CPU可跑(慢) | **高** — curriculum(长时序)+wm_scheduler(子任务调度)+latent_reasoner |
| **ManiSkill3** | UCSC(Su Lab) | 码✅(MIT)权✅数✅(2000+物体) | 码MIT;数据CC-BY4.0 | 20+任务族,2000+物体,4M+示范帧(v2) | RGB,RGBD,state-based | 人形/移动操作/单臂 | 低 — 并行仿真需GPU;CPU单环境极慢 | 中 — digital_twin(合成数据架构)+spatial(SAPIEN物理) |
| **RoboCasa / 365** | UT Austin/NVIDIA | 码✅(MIT)权✅数✅(厨房资产) | 码MIT;数据CC-BY4.0 | 365任务,2500厨房布局,3200+物体,600+h示范 | RGB-D,本体,语言 | PandaOmron 12DoF移动臂 | 中 — MuJoCo CPU可跑;大规模慢 | 中 — digital_twin(家庭仿真)+closed_loop(评估)+world_model |
| **RoboSuite** | Stanford/UT Austin/ASU | 码✅(MIT) | MIT | 9-13核心任务,6+机器人模型 | RGB,本体,低维物理状态 | Franka/Sawyer/UR5e/Kinova/Baxter/IIWA | **高** — MuJoCo原生CPU运行;轻量 | **高** — spatial/collision(MuJoCo物理参考)+neural_control+digital_twin |
| **RLBench** | U Edinburgh | 码✅(MIT) | MIT | 100+任务,关键帧示范 | RGB-D(4视角128x128),本体,语言 | 7-DoF Franka Panda | 中 — CoppeliaSim CPU可跑;PyRep/C++编译复杂 | 中 — closed_loop(关键帧分解)+curriculum(100+任务)+wm_events |
| **BEHAVIOR-1K** | Stanford | 码✅权✅数✅(10K示范HF) | 开源(见repo) | 1000家庭活动,50场景,10K遥操作示范(50任务) | RGB,本体,语言(BDDL一阶逻辑) | R1Pro人形(双臂+底座,23DoF) | 低 — Omniverse需NVIDIA GPU;示范元数据CPU可解析 | 中 — world_model(BDDL一阶逻辑形式化)+curriculum(1000任务)+digital_twin |
| **Isaac Lab** | NVIDIA | 码✅(BSD-3)权✅ | BSD-3(框架);Apache2.0(mimic扩展) | 多任务环境,GPU并行 | RGB,state,本体 | 多种(人形/四足/臂) | **很低** — 必须NVIDIA GPU+Isaac Sim | 低 — digital_twin(GPU并行架构参考)+multi_agent(并行环境) |
| **RoboMimic/MimicGen** | Stanford/NVIDIA | 码✅权✅数✅(HDF5;MimicGen 50K+) | RoboMimic:MIT;MimicGen码:NVIDIA SCL;数据:CC-BY4.0 | RoboMimic:~1000示范/任务x3质量档;MimicGen:50K+合成示范 | RGB(腕部+第三方),本体,物体位姿,动作 | Franka/Sawyer/UR5e/双臂Panda | 中 — HDF5元数据CPU可解析;MuJoCo CPU可跑 | 中 — build_parametric_dataset(合成pipeline)+ctm(多质量档)+closed_loop |

### 2.3 3D / 空间资产

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **Objaverse / XL** | Allen Institute for AI | 码✅(Python API)数✅(HF) | 集合ODC-By1.0;单物体各CC(80%+ CC-BY/CC0) | 800K(v1)/10.2M+(XL) 3D物体 | 3D网格(GLB/OBJ),渲染图,点云,caption | N/A(物体资产) | **高** — CPU可下载单个物体;网格解析CPU可行 | **高** — spatial/scene_graph(物体资产库)+occupancy(物体几何)+digital_twin |
| **ScanNet** | Stanford | 码✅(MIT)数✅(需协议) | 数据:非商业研究;码:MIT | 1,513场景,2.5M视图 | RGB-D视频,3D位姿,表面重建,语义分割 | N/A(室内场景) | 中 — 网格解析CPU可行;语义标注CPU处理 | 中 — spatial/occupancy(室内几何)+scene_graph(语义场景图)+digital_twin |
| **Matterport3D** | Matterport | 数✅(需注册) | 非商业学术研究 | 90建筑(原版);HM3D扩展1000空间;~200K RGB-D | RGB-D,3D网格,位姿,语义,平面图 | N/A(建筑级场景) | 中 — 网格加载CPU可行;建筑级注意内存 | 中 — spatial/scene_graph(建筑级空间)+occupancy(平面几何)+digital_twin |

### 2.4 特殊关注

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|
| **PhysBrain Ego360** | DeepCybo/中关村AI研究院 | 码✅(PhysBrain项目)权✅(Apache2.0 HF) **数据✗(Ego360未公开下载)** | 模型Apache2.0;Ego360未公开 | 未公开(论文中作训练源引用) | 全景视频,全身姿态,手部运动,任务级语音 | 未知 — 数据集未确认公开 | 低 — ctm(全景先验)+world_model(全景物理先验) |

---

## 3. 动作库 / 重定向 / 操作先验对比总表（27 条）

### 3.1 动作生成 / 模仿

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **MimicGen** | NVIDIA/UT Austin | 码✅权✗数✅(50K+) | 码NSCL·数CC-BY 4.0 | 50K demos from <200 human,18任务 | low-dim,RGB,子任务分段,无语言 | Franka/Sawyer/UR5e/双臂Panda | 分段/SE(3)变换可CPU读;生成需Isaac(GPU) | 中 — ctm/wla:子任务分段+桥插值类比wla三分组分段 |
| **DexMimicGen** | NVIDIA/UT Austin/UCSD | 码✅权✗数✅(20K+) | 承NVlabs(NSCL),[UNVERIFIED] | 20K+ from 60 human,9 bimanual tasks | low-dim,RGB,per-arm分段 | 双臂+灵巧手 | 同步/排序约束可CPU读 | 中 — wla:per-arm独立+协调同步=双臂change-mask |
| **RoboMimic** | ARISE(Stanford/UT) | 码✅权✗数✅(9套) | **MIT** | 6 suite×~1000 demo×3质量(PH/MH/MG) | RGB+本体状态+HDF5 | Franka Panda/Sawyer | **HDF5纯CPU可解析** | **高** — ctm/wla/closed_loop:动作chunk schema+三档质量 |
| **AnyTeleop** | UCSD/NVIDIA | 码✅(含dex-retargeting)权✗数✗ | dex-retargeting **MIT**;主仓[UNVERIFIED] | 多臂多手视觉遥操作系统 | RGB手眼/第三人称,手部关键点 | Franka/UR5e+Allegro/Shadow | 手关键点需GPU;IK优化可CPU | 中 — wla末端位姿/spatial:vision-only重定向几何对齐 |

### 3.2 重定向 / 人体到机器人

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **UCH(异构手统一控制)** | [UNVERIFIED]未命中同名仓库 | — | — | — | — | 异构灵巧手(推测) | 近邻=UniDexTok(22-DoF语义接口+共享codebook) | 中(待澄清) — wla RVQ:跨手共享codebook思想 |
| **Human2Robot/H2O** | CMU LeCAR | 码✅权✗数✅(retargeted) | [UNVERIFIED](LeCAR多为MIT) | SMPL→H1两阶段 | SMPL参数→关节角,无RGB | Unitree H1 | 阶段一纯几何优化**可CPU** | **高** — wla跨本体/spatial:形状对齐几何baseline+物理清理 |
| **OmniControl** | NTU S-Lab | 码✅权✅数✅(HumanML3D) | [UNVERIFIED](多MIT) | HumanML3D/KIT-ML,DiT | text+任意关节时空约束→SMPL动作 | SMPL人体(非机器人) | DiT推理需GPU;CPU仅架构 | **高** — wla/latent_reasoner:任意关节inpainting时空引导 |
| **HumanPlus** | Stanford | 码✅权✅数✅(40h) | [UNVERIFIED] | 40h mocap+40demo;33DoF 180cm | RGB(单目)→全身关节 | 自研33DoF/类H1 | shadowing需GPU;两阶段思想可CPU读 | **高** — neural_control:大mocap训跟踪器+少量BC训技能=大脑-小脑分层 |
| **ExBody** | CMU/Stanford | 码✅权✅数✅(CMU MoCap) | [UNVERIFIED] | ~780 motions;H1 | MoCap→H1 keypoint+关节 | Unitree H1(4DoF臂) | RL需GPU;上下体解耦思想可读 | 中 — neural_control:上体表达/下肢稳定解耦=SpinalReflex保底 |
| **ExBody-2** | CMU/Stanford | 码:出现clone命令暗示已放,但第三方称2026/4仍无公开→**矛盾** | [UNVERIFIED] | 两本体,设计因子ablation | MoCap→全身 | 两humanoid平台 | GPU训练 | 低 — neural_control:上下体解耦权重ablation参考 |
| **OmniH2O** | CMU LeCAR | 码✅(官网Code链接)权✅数✅ | [UNVERIFIED](LeCAR多MIT) | 通用kinematic接口;H1+G1 | kinematic pose统一接口+RGB-D | H1,G1 | RL需GPU;接口设计可读 | **高** — wla/multi_agent:kinematic pose作跨本体统一动作空间 |
| **PHC** | CMU(Kris Kitani) | 码✅权✅数(AMASS) | [UNVERIFIED] | 单策略模仿几乎全AMASS,可从跌倒恢复 | AMASS SMPL→torque | 28DoF仿真人形 | RL需GPU;perpetual思想可读 | 中 — world_model/closed_loop:不reset连续时序=ctm连续预测容错 |
| **Unitree LAFAN1_Retargeting** | Unitree(注:非PrimeU) | 码✅权✗数✅HF公开 | 数据'other'(承LAFAN1 **CC-BY-NC-ND**,非商用) | LAFAN1全量→H1/H1-2/G1 | BVH→关节角+root SE(3),无视觉 | Unitree H1/H1-2/G1 | **BVH纯文本+Pinocchio IK均可CPU** | **高** — wla/spatial:同源一对多跨本体对照;关节角直喂ctm |

### 3.3 抓取 / 操作

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **GraspNet-1B** | SJTU MVIG | 码✅权✅数✅ | **CC BY-NC-SA 4.0**(非商用) | 190场景,97,280图,88物,**1.1B grasps** | RGB-D,点云,6-DoF grasp,6D位姿 | 平行夹爪(非灵巧手) | 标注解析可CPU;网络需GPU | 中 — spatial/collision/SpinalReflex:1.1B力封闭位形先验查表 |
| **DexGraspNet 1.0/2.0** | PKU EPIC(王鹤组) | 码✅权✅数✅ | [UNVERIFIED](学术开源) | v2:8270场景,**427M grasp labels**(LEAP手) | 3D mesh+灵巧手关节角+力封闭标签 | Allegro/LEAP Hand | 力封闭优化可CPU;生成网络需GPU | 中 — spatial/wla末端关节组:427M手指协同配置先验 |
| **GraspXL** | ETH/Tübingen | 码✅权✅数✅(500k+) | [UNVERIFIED] | 500k+物体,跨MANO/Allegro | mesh+**时序**抓取动作序列 | MANO人手+Allegro | 序列解析可CPU | 中 — wla末端关节组flow:接近→接触→闭合时序先验 |
| **UniDexGrasp** | PKU EPIC | 码✅权✅数✅ | [UNVERIFIED] | 两阶段(proposal+goal-conditioned) | 点云→抓取轨迹 | Allegro Hand(PyBullet) | 策略推理需GPU | 低 — CortexPlanner:proposal+ranking对应多effort筛选 |

### 3.4 人体动作基础

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **AMASS(SMPL/SMPL-X)** | MPI IS | 码✅/body模型/SMPL CC-BY4.0,SMPL-X非商业/数✅注册 | **非商业科研专属(禁商用)** | ~42h,346人,11451动作,40+数据集,~43GB | SMPL/SMPL-X body参数,24/55关节,无RGB | SMPL人体(非机器人) | **NPZ+FK纯CPU可跑** | **高** — wla/ctm:几乎所有人形重定向的源动作先验(注意非商用) |
| **GRAB** | MPI | 码✅(GrabNet)权✅数✅注册 | 非商业科研(MPI系) | 10人×51物,全身抓握 | SMPL+MANO+物体mesh+**接触图** | MANO手+SMPL | 接触图计算可CPU | 中 — spatial/collision:手-物接触点真值作occupancy先验 |
| **DexYCB** | Berkeley/Stanford | 码✅权✗数✅ | [UNVERIFIED](YCB本体CC-BY4.0) | 582K帧,1000序列,10人,20物,8视图 | 多视图RGB-D,3D手pose,6D物pose | MANO人手 | 骨架标注可CPU | 低 — spatial_query:8视角多视角一致性ground truth |
| **HOI4D** | 清华等 | 码✅权✗数✅申请 | [UNVERIFIED](研究用) | 2.4M帧,4000序列,9人,800实例,16类,610房间 | 第一人称RGB-D,分割,6D位姿 | 人(ego) | 标注解析可CPU | 低 — scene_graph:类别级HOI边类型先验统计 |
| **ARCTIC** | MPI/NVIDIA | 码✅toolkit权✗数✅ | [UNVERIFIED](MPI非商业倾向) | 339序列,10人,11铰接物,2.1M图(8静态+1第一人称) | RGB(9视图)+SMPL-X/MANO+铰接物+接触 | 人双手(双臂) | mesh/关节标注可CPU | 中 — wla双臂:铰接物双手操作=双臂+物体主动关节联合建模 |
| **LAFAN1** | Ubisoft La Forge | 码✅MIT权N/A数✅ | 数据**CC-BY-NC-ND**(非商用禁演绎);码MIT | 496,672帧,15内容 | **BVH纯文本**关节角,无视觉 | 游戏人形(常重定向H1/G1) | **BVH最轻量CPU入口** | **高** — ctm/wla:纯文本关节角序列喂52191参数时序预测 |

### 3.5 IK / FK / 重定向工具链

| 名称 | 机构 | 开源三要素 | License | 规模 | 模态 | 适配本体 | CPU 可行性 | UDOS 对接·优先级 |
|---|---|---|---|---|---|---|---|---|
| **pytorch_kinematics** | UMich ARM Lab | 码✅/N/A/N/A | **MIT** | FK/Jacobian/damped IK,URDF/SDF/MJCF | URDF→关节角→末端SE(3)+Jacobian | 任意URDF机器人 | **纯PyTorch,CPU可跑小demo** | **高** — spatial/spatial_query:wla输出位姿→IK反解校验可达性/限位/自碰撞 |
| **yourdfpy** | C.Eppner(urdfpy继任) | 码✅/N/A/N/A | **MIT** | URDF解析/校验/可视化 | URDF XML→kinematic tree+碰撞mesh | 任意URDF | **纯Python零GPU** | **高** — spatial/scene_graph:UDOS零重依赖下解析异构URDF的候选纯Python层 |
| **Genesis** | Genesis-Embodied-AI社区 | 码✅/N/A/N/A | **Apache 2.0**(可商用) | Python+Taichi,CPU/Vulkan/Metal/NV后端,28k+★ | 多刚体/形变/流体统一物理 | URDF/MJCF任意 | 支持CPU后端,小场景可CPU;大RL需GPU | 中 — world_model/wm_conservation:轻量可移植物理架构参考 |
| **MuJoCo MPC(MJPC)** | Google DeepMind | 码✅/N/A/N/A | **Apache 2.0** | iLQG/GD/Predictive Sampling三shooting planner | MJCF→MPC→torque,contact-implicit | MuJoCo任意模型 | **MuJoCo步进CPU~100Hz已验证** | **高** — neural_control SpinalReflex:contact-implicit无学习反射fallback |

---

## 4. 对接 UDOS 优先级矩阵

### 4.1 高优先级（可直接对接 / CPU 可跑 / 许可友好）

| UDOS 模块 | 高优先级资源 | 切入点 | 理由 |
|---|---|---|---|
| **ctm_engine** | Sakana CTM/ctm-imagenet、LAFAN1(BVH)、Unitree LAFAN1_Retargeting、RoboMimic(HDF5) | 连续时间推理机制对照;纯文本关节角序列直接喂52191参数时序预测 | 同命名、CPU可跑、格式最轻 |
| **world_model** | DreamerV3(RSSM)、V-JEPA2(隐空间预测)、WALL-WM(事件级)、PhysBrain 1.5(未来状态统一token) | latent imagine范式对照;事件级时序↔wm_events;未来RGB/depth/mask统一token化 | DreamerV3 MIT+CPU可跑小demo;V-JEPA2注意非商用 |
| **wla** | ACT(CVAE+chunk)、Diffusion Policy(扩散去噪)、Octo(embodiment adapter+扩散头)、SmolVLA/TinyVLA(小动作头)、OmniH2O(kinematic pose统一接口)、Unitree/HIW-500(同源全身动作) | change-mask/VQ/RVQ动作分词对照;flow decoder对照;跨本体迁移对照 | ACT/Diffusion Policy MIT+CPU可跑;Octo 27M可CPU慢跑 |
| **neural_control** | HumanPlus(两阶段大脑-小脑)、ExBody(上下体解耦)、MJPC(contact-implicit反射)、MuJoCo MPC | 大脑-小脑-脊髓分层对照;SpinalReflex无学习反射fallback | HumanPlus两阶段=分层现成范式;MJPC Apache2.0+CPU~100Hz |
| **spatial/scene_graph** | Objaverse/XL(3D物体资产)、yourdfpy(URDF解析)、pytorch_kinematics(IK校验) | 物体资产库;机器人描述归一化;位姿可达性/限位/自碰撞验证 | Objaverse CC许可+CPU可下载;yourdfpy/pytorch_kinematics MIT+纯CPU |
| **digital_twin** | RoboSuite(MuJoCo CPU原生)、LIBERO/CALVIN(MuJoCo CPU可跑)、Objaverse(3D资产) | 仿真后端;合成环境;物体资产 | RoboSuite MIT+MuJoCo CPU原生运行 |
| **curriculum/selfplan** | LIBERO(130任务终身学习)、CALVIN(34任务长时序)、RLBench(100+任务) | 任务生成/分解基准;子任务调度 | LIBERO/CALVIN MIT+MuJoCo CPU可跑 |
| **PCE-Format** | LeRobot社区数据集(Parquet+MP4)、RoboMimic(HDF5)、Open X-Embodiment(RLDS) | 数据格式转换模板;元数据schema映射 | LeRobot Parquet纯CPU解析;格式最标准 |

### 4.2 中优先级（架构参考 / 需GPU / 部分可下）

| UDOS 模块 | 中优先级资源 | 说明 |
|---|---|---|
| gpm | OpenVLA(动作分箱)、RDT-1B(DiT扩散头)、CogACT(认知-动作协同) | 大权重仅架构参考 |
| icm | MimicDroid/DreamDojo(连续latent action)、Doc-to-LoRA(上下文→小适配器) | 论文新、代码未核实 |
| collab | — | 多智能体协作框架为UDOS自建,外部多为A2A/MCP协议参考 |
| latent_reasoner | OmniControl(时空inpainting)、Coconut类latent CoT | 隐式思考机制参考 |
| multi_agent | OmniH2O(kinematic pose统一)、AgiBot World(双臂) | 跨本体统一动作空间参考 |

### 4.3 低优先级（未开源 / 需GPU / 非商用 / 与物理AI无关）

| 资源 | 原因 |
|---|---|
| Skild S1、Generalist GEN-1.5、Genie | 完全未开源,仅概念锚点 |
| Isaac Lab、BEHAVIOR-1K(Omniverse)、ManiSkill3(GPU并行) | 必须NVIDIA GPU,CPU-only无法运行 |
| AMASS、GRAB、GraspNet-1B、V-JEPA2权重 | 非商用license,商用分发需规避 |
| EPIC-KITCHENS、Something-Something、DexYCB、HOI4D | 与机器人动作空间距离较远,主要作常识先验 |

---

## 5. 许可证与商用风险

| License 类型 | 代表资源 | 商用风险 | UDOS 策略 |
|---|---|---|---|
| **MIT / Apache-2.0 / BSD** | LeRobot、Octo、Diffusion Policy、ACT、DreamerV3、RoboSuite、LIBERO、CALVIN、pytorch_kinematics、yourdfpy、MJPC、Genesis、Sakana CTM、RoboMimic | **无风险**,可商用/修改/再分发 | 优先对接,可打包 |
| **CC-BY 4.0** | DROID数据、BridgeData V2、MimicGen数据、Objaverse集合、ManiSkill数据 | 低风险,需署名 | 可对接,保留署名 |
| **CC-BY-NC / CC-BY-NC-SA / CC-BY-NC-ND** | V-JEPA2权重、GraspNet-1B、AMASS、GRAB、LAFAN1数据、RH20T部分、Ego4D | **不可商用**,只能研究/架构参考 | 不打包,仅作参考 |
| **混合/继承约束** | OpenVLA(权重受Llama-2约束)、PhysBrain(基于Qwen3-VL需继承)、UnifoLM(license待核) | 需逐一核实上游license | 暂标[UNVERIFIED],不打包 |
| **未公开/需审批** | PhysBrain Ego360、Ego4D/Ego-Exo4D(需license审批)、ScanNet/Matterport3D(需协议) | 有条件可下,非即下即得 | 按需申请,不预设可用 |
| **完全未开源** | Skild S1、Generalist GEN-1.5、Genie | 无 | 仅概念锚点 |

---

## 6. "宣称开源 vs 实际可下"核实清单

| 项目 | 宣称状态 | 实际可下性 | 核实结论 |
|---|---|---|---|
| **PhysBrain 1.5** | 2026-09-08发布,宣称开源 | ✅ 权重在HF collection(2B/8B),EvalKit在GitHub;**license未在论文点明,标[UNVERIFIED]** | 权重可下,license待核 |
| **PhysBrain Ego360** | 论文中作训练源引用 | ❌ **未找到公开下载链接**。模型权重开源(Apache 2.0) ≠ 数据集开源 | **宣称使用但未确认公开数据下载** |
| **UnifoLM-WLA-1.0** | 2026-09-10宣布全开 | ✅ HF有WMA-0-Base等;官方口径**6B**(非6亿);**license [UNVERIFIED]** | 权重可下,license待核 |
| **Skild S1** | 宣称机器人基础模型 | ❌ **无权重/无API/无论文**,仅博客+合作部署 | **核实为未开源** |
| **Generalist GEN-1.5** | 宣称physical prompting | ❌ **无权重/无API/无代码**,仅研究发布,需直接合作 | **核实为未开源** |
| **自变量 WALL-WM** | 宣称事件级世界模型 | ✅ 已开源到github.com/X-Square-Robot/wall-x(论文+代码);权重/license需核仓库 | 代码可下,权重待核 |
| **Genie(DeepMind)** | 宣称生成交互环境 | ❌ 论文附录**明确不发布checkpoint/数据**,仅社区复现GenieRedux | **官方明确不发布** |
| **Unitree Open Datasets** | unitree.com/opensource列出G1灵巧手/G1夹爪/Z1双臂/UnifoLM-WBT | ✅ 页面确认开源,unitree_IL_lerobot框架在GitHub | 实际可下(需到各自仓库) |
| **BitRobot-HIW-500** | 宣称最大真实家庭人形遥操作数据 | ✅ HF上有LeRobot v3.0版本(~2.15TB),原始~10TB;23,743轨迹,500+小时 | 实际可下 |
| **AgiBot World** | 宣称100万轨迹开源 | ✅ HF agibot-world/AgiBotWorld-Alpha可下载,Beta~43.8TB/Alpha~8.5TB | 实际可下 |
| **RoboMIND 2.0** | 宣称开源 | ✅ ModelScope X-Humanoid/RoboMIND2.0可下载,Apache 2.0 | 实际可下 |
| **Unitree Human-as-Humanoid/Prime U** | 用户列为"Unitree(开放状态)" | ⚠️ **归属纠正**:arXiv:2606.32009的Human-as-Humanoid对应**深度机智60-DoF PrimeU,不是Unitree**;Unitree官方真实开源的是LAFAN1_Retargeting_Dataset(HF公开) | 归属已纠正 |
| **ExBody-2** | 宣称开源 | ⚠️ 出现clone命令暗示已放,但第三方称2026/4仍无公开;GitHub被robots拦截 | **状态矛盾,标[UNVERIFIED]** |
| **UCH(异构手统一控制)** | 用户列为待核实 | ❌ 未检索到以此缩写命名的明确开源仓库/论文,最接近的是UniDexTok | **[UNVERIFIED],需用户澄清** |

---

## 7. [UNVERIFIED] / [FETCH FAILED] 清单

### 模型库
- **RDT-1B 权重 license**:来源冲突(MIT vs CC-BY-NC),HF模型卡fetch error
- **PhysBrain 1.5 license**:论文未点明权重license(基于Qwen3-VL,可能继承其条款)
- **UnifoLM-WLA-1.0 license/各仓库权重可下性**:以官方GitHub/HF LICENSE为准
- **CogACT 精确参数量**:标~7B级,论文未给精确数
- **MimicDroid/DreamDojo 代码与权重**:仅有论文,仓库链接未一手核实
- **openpi 权重商用条款**:README细节未抓全
- **WALL-WM license**:以wall-x LICENSE为准

### 数据库
- **PhysBrain Ego360 数据集本身**:论文引用存在但未找到独立下载页面
- **Unitree各数据集具体大小和license**:未逐一打开各GitHub仓库核实
- **AgiBot World 具体license**:确认HF可下载但未核实完整license文本
- **RoboMIND 2.0 触觉数据具体规模**:官方新闻称1.2万+条,未在ModelScope核实精确数字
- **HIW-500 具体license**:确认HF公开但未核实完整条款

### 动作库
- **UCH(异构手统一控制)**:未命中同名仓库,需用户澄清
- **ExBody-2 GitHub仓库**:被robots.txt拦截,无法核实公开状态与LICENSE
- 下列项目**具体LICENSE文件**未直接读到(均为学术开源,推测MIT/BSD):DexMimicGen、AnyTeleop主仓、H2O、OmniControl、HumanPlus、ExBody、OmniH2O、PHC、DexGraspNet、GraspXL、UniDexGrasp、DexYCB、HOI4D、ARCTIC
- **HOI4D 参与人数**:项目页称9人,arXiv称4人,存在文献不一致

---

## 8. 全部来源 URL（按类别）

### 模型库
- LeRobot/SmolVLA: github.com/huggingface/lerobot · arxiv.org/html/2602.22818v1 · huggingface.co/lerobot
- OpenVLA: openvla.github.io · github.com/openvla/openvla
- Octo: octo-models.github.io · arxiv.org/pdf/2405.12213v2
- openpi: pi.website/blog/openpi · github.com/Physical-Intelligence/openpi
- RT-X/OXE: robotics-transformer-x.github.io · arxiv.org/pdf/2310.08864v4
- RDT-1B: rdt-robotics.github.io · github.com/thu-ml/RoboticsDiffusionTransformer
- CogACT: arxiv.org/html/2411.19650v1
- TinyVLA: tiny-vla.github.io · arxiv.org/html/2409.12514v2
- MobileVLA: arxiv.org/html/2511.17889
- GR-1/GR-2/GR00T: gr1-manipulation.github.io · gr2-manipulation.github.io · github.com/NVIDIA/Isaac-GR00T
- Diffusion Policy/ACT: github.com/real-stanford/diffusion_policy · github.com/tonyzhaozh/act
- VIMA: vimalabs.github.io · github.com/vimalabs/VIMA
- V-JEPA2: github.com/facebookresearch/vjepa2 · arxiv.org/html/2506.09985v1
- Cosmos: github.com/nvidia/cosmos · arxiv.org/pdf/2606.02800v4
- DreamerV3: github.com/danijar/dreamerv3 · arxiv.org/abs/2301.04104
- Genie: deepmind.google/research/publications/genie-generative-interactive-environments
- PhysBrain 1.5: arxiv.org/pdf/2609.14973 · huggingface.co/collections/DeepCybo/physbrain-15 · github.com/DeepCybo-PhysAI/PhysBrainEvalKit
- UnifoLM: unigen-x.github.io/unifolm-wla.github.io · github.com/unitreerobotics/unifolm-world-model-action
- MimicDroid/DreamDojo: arxiv.org/pdf/2509.09769v1 · arxiv.org/html/2602.06949v1
- Skild S1: skild.ai/blogs · datanorth.ai/news/skild-ai-launches-s1
- Generalist GEN-1.5: aiwiki.ai/wiki/generalist_gen_1_5
- WALL-WM: github.com/X-Square-Robot/wall-x · x2robot.com/pages/wm
- CTM: github.com/SakanaAI/continuous-thought-machines · huggingface.co/SakanaAI/ctm-imagenet · arxiv.org/abs/2505.05522
- Doc-to-LoRA: github.com/SakanaAI/doc-to-lora

### 数据库
- OXE/RT-X: robotics-transformer-x.github.io · arxiv.org/html/2310.08864v5
- DROID: droid-dataset.github.io · jiajunwu.com/papers/droid_rss.pdf
- BridgeData V2: rail-berkeley.github.io/bridgedata · arxiv.org/pdf/2308.12952v1
- RH20T: rh20t.github.io · arxiv.org/pdf/2307.00595v2
- RoboMIND 2.0: modelscope.cn/datasets/X-Humanoid/RoboMIND2.0 · arxiv.org/html/2512.24653v3
- AgiBot World: agibot-world.com · hf.co/datasets/agibot-world/AgiBotWorld-Alpha
- Unitree/HIW-500: unitree.com/opensource · bitrobot-foundation.github.io/humanoids-in-the-wild-500-hours
- LeRobot: github.com/huggingface/lerobot · pypi.org/project/lerobot
- Ego4D/Ego-Exo4D: ego4d-data.org · ego-exo4d-data.org · arxiv.org/pdf/2110.07058 · arxiv.org/html/2311.18259v3
- EPIC-KITCHENS: epic-kitchens.github.io · arxiv.org/pdf/1804.02748v1
- Something-Something: developer.qualcomm.com/software/ai-datasets/something-something
- ALOHA: mobile-aloha.github.io · arxiv.org/pdf/2401.02117 · aloha-2.github.io
- LIBERO: github.com/Lifelong-Robot-Learning/LIBERO · libero-project.github.io/datasets
- CALVIN: calvin.cs.uni-freiburg.de · github.com/mees/calvin
- ManiSkill3: maniskill.readthedocs.io · arxiv.org/html/2410.00425v1
- RoboCasa: robocasa.ai · github.com/robocasa/robocasa
- RoboSuite: robosuite.ai
- RLBench: github.com/stepjam/RLBench · arxiv.org/pdf/1909.12271v1
- BEHAVIOR-1K: behavior.stanford.edu · github.com/StanfordVL/BEHAVIOR-1K
- Isaac Lab: github.com/isaac-sim/IsaacLab · developer.nvidia.com/isaac/lab
- Objaverse: objaverse.allenai.org · CVPR2023
- ScanNet: scan-net.org · arxiv.org/pdf/1702.04405v1
- Matterport3D: matterport.com/partners/meta · arxiv.org/pdf/1709.06158
- PhysBrain: zgc-embodyai.github.io/PhysBrain · deepcybo-physai.github.io/PhysBrain-1.5 · arxiv.org/pdf/2512.16793 · arxiv.org/pdf/2609.14973
- RoboMimic/MimicGen: robomimic.github.io · mimicgen.github.io

### 动作库
- MimicGen/DexMimicGen: mimicgen.github.io · github.com/NVlabs/mimicgen · dexmimicgen.github.io · github.com/NVlabs/dexmimicgen
- RoboMimic: robomimic.github.io · github.com/ARISE-Initiative/robomimic
- AnyTeleop: yzqin.github.io/anyteleop · github.com/dexsuite/dex-retargeting
- H2O: human2humanoid.com · lecar-lab.github.io/publications.html
- OmniControl: neu-vi.github.io/omnicontrol · arxiv.org/abs/2310.08580
- HumanPlus: humanoid-ai.github.io · arxiv.org/abs/2406.10454 · github.com/MarkFzp/HumanPlus
- ExBody/ExBody-2: github.com/chengxuxin/expressive-humanoid · exbody2.github.io · arxiv.org/abs/2412.13196
- OmniH2O: omni.human2humanoid.com · arxiv.org/abs/2406.08858
- PHC: github.com/ZhengyiLuo/PHC · zhengyiluo.com/PHC-Site · arxiv.org/abs/2305.06456
- Unitree LAFAN1: huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset · unitree.com/opensource
- GraspNet-1B: graspnet.net · arxiv.org/abs/1912.13470
- DexGraspNet: pku-epic.github.io/DexGraspNet · github.com/PKU-EPIC/DexGraspNet
- GraspXL: arxiv.org/abs/2403.19649
- UniDexGrasp: pku-epic.github.io/UniDexGrasp · github.com/PKU-EPIC/UniDexGrasp
- AMASS: amass.is.tue.mpg.de · amass.is.tue.mpg.de/license.html · arxiv.org/abs/1904.03278
- GRAB: grab.is.tue.mpg.de · arxiv.org/abs/2008.11200
- DexYCB: dex-ycb.github.io · arxiv.org/abs/2104.04631
- HOI4D: hoi4d.github.io · arxiv.org/abs/2203.01577
- ARCTIC: arxiv.org/abs/2204.13662
- LAFAN1: github.com/ubisoft/ubisoft-laforge-animation-dataset
- pytorch_kinematics: github.com/UM-ARM-Lab/pytorch_kinematics · pypi.org/project/pytorch-kinematics
- yourdfpy: github.com/clemense/yourdfpy · yourdfpy.readthedocs.io
- Genesis: github.com/Genesis-Embodied-AI/Genesis · genesis-embodied-ai.github.io
- MJPC: github.com/google-deepmind/mujoco_mpc · arxiv.org/abs/2212.00541

---

## 9. 关键推荐（Top 8）

对 CPU-only、零重依赖的 UDOS，最值得优先对接/深读的 8 个资源：

1. **Sakana CTM / ctm-imagenet**（Apache-2.0，CPU可跑）— 与 UDOS ctm_engine 同命名同机制，最直接外部参照
2. **DreamerV3**（MIT，CPU可跑小demo）— RSSM 潜世界想象 = world_model latent imagine + neural_control + closed_loop 最干净可读实现
3. **ACT / Diffusion Policy**（MIT，CPU可跑小demo）— wla 动作解码器的两个经典祖先（CVAE+chunk vs 扩散去噪）
4. **LeRobot 社区数据集**（Apache-2.0，Parquet+MP4纯CPU解析）— PCE-Format 第一个转换模板，数据格式最标准
5. **RoboMimic**（MIT，HDF5纯CPU可解析）— 动作chunk schema + 三档质量划分，离线动作数据最佳现成模板
6. **LAFAN1 / Unitree LAFAN1_Retargeting**（BVH纯文本，CPU最轻量入口）— 纯文本关节角序列直接喂 ctm_engine，同源跨本体对照
7. **pytorch_kinematics + yourdfpy**（MIT，纯Python/CPU）— wla 输出动作后的可达性/限位/自碰撞验证 + URDF 解析归一化层
8. **Octo / SmolVLA / TinyVLA**（MIT/Apache-2.0，27M~450M可CPU慢跑）— "冻结表征+小动作头"= CPU-only gpm 第二引擎范式最佳对照

---

*本目录为纯研究交付，未修改 UDOS 代码、未升版本。所有外部数字均来自公开来源，核不到的已标 [UNVERIFIED]/[FETCH FAILED]。*
