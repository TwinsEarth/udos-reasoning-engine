# UDOS 开源资源集成报告 — v4.5.4 → v4.5.6

> 环境：CPU-only / 无 GPU / 零重依赖。本报告严格遵守**诚信四层 L0-L3**：
> 大权重/GPU 资源只做接口契约 + 能力声明 + 懒加载降级，**不下载、不假装运行、不编造推理输出/成功率**。
> "接口已适配" ≠ "模型已运行"。registry 无可学参数，零训练，不进主 state_dict。

## 1. 四层诚信定义

| 层 | 含义 | 本项目落地方式 |
|---|---|---|
| **L0 注册表(元数据)** | catalog 全量资源进统一 registry，纯声明、零运行成本 | 79 条全量内嵌 `udos/connectors/_catalog_data.py`（构建期由 `docs/opensource_catalog.json` 编译，运行时不读文件） |
| **L1 格式/数据适配** | 真实实现公开数据格式 → UDOS(PCE-Format/tensor) 的解析转换，最小内联 fixture 单测往返 | 每条形如 `normalize_trajectory` 通用归一化；专门格式 6 类（动作分块/VLA 256-bin/episode schema/SMPL→rot6d/BVH/URDF） |
| **L2 轻量可运行** | CPU 可跑、pip 可装、许可宽松的小库真实 import+smoke；惰性 import，缺包→absent 不崩核心 | yourdfpy 真实安装 + 内联 URDF FK smoke；pytorch_kinematics 惰性探测 |
| **L3 大权重/GPU** | 只做接口契约 + requires_gpu/weights + 懒加载 + 缺失 503/absent 优雅降级，文档化外部启动命令 | 20 个大模型契约就绪，invoke→503，不下载不假装运行 |

## 2. L0-L3 状态矩阵（v4.5.6 实测，full profile 79 条）

### 2.1 按级别 / 状态汇总

| 维度 | 取值 | 数量 |
|---|---|---|
| 级别 | L1（纯 python 格式适配） | 57 |
| 级别 | L2（CPU 可装小库，惰性） | 2 |
| 级别 | L3（大权重/GPU 契约） | 20 |
| 级别 | L0（仅元数据，无转换） | 0（全部至少 L1 归一化） |
| 状态 | available（本环境真实可用/转换通过） | 58 |
| 状态 | absent（需外部包/权重，未装未下载，契约就绪） | 21 |
| 状态 | degraded | 0 |
| 状态 | env_blocked | 0（本环境 pip 外网可用，yourdfpy 实装） |
| kind | model / dataset / action | 26 / 26 / 27 |

### 2.2 真实 smoke 通过清单（L2，有证据）

| id | 包 | 状态 | 证据 |
|---|---|---|---|
| `yourdfpy` | yourdfpy（已 pip 安装） | **available** | 对内联单关节 URDF 跑通 `update_cfg` + `get_transform('tip')`，FK 返回 4×4 位姿矩阵 |

### 2.3 [ENV 降级] / 未真跑清单

| id | 包/资源 | 状态 | 说明 |
|---|---|---|---|
| `pytorch_kinematics` | pytorch_kinematics | absent | 惰性 import 未安装（避免引入 pytorch3d 等重依赖）；契约就绪，`pip install pytorch_kinematics` 后即 L2 |
| 20 个 L3 大模型 | 见 §2.4 | absent | 需 GPU + 外部权重；本环境不下载 |

> 本环境 pip 索引可达（`yourdfpy` 实装成功），故**无 env_blocked**。若在无外网沙箱，上述 absent 会自动降级为 env_blocked 并标注 `[ENV BLOCKED: no package index]`——探测逻辑已实现，当前未触发。

### 2.4 L3 契约就绪但需 GPU/权重（20 个，invoke→503）

OpenVLA、Octo(small/base)、openpi(π0/π0-FAST/π0.5)、RDT-1B、CogAct、SmolVLA、TinyVLA、MobileVLA、GR-1/GR-2/GR00T、V-JEPA/V-JEPA2、NVIDIA Cosmos、Genie、PhysBrain 1.5、UnifoLM-WLA-1.0、WALL-WM/OSS/SS、CTM/ctm-imagenet、Doc-to-LoRA(D2L)、Skild S1、Generalist GEN-1.5、MimicDroid/DreamDojo。

> **返工调整**：DreamerV3 原列 L3，但其 RSSM 状态结构是纯表示/schema（不含可下载权重前向），已下沉为 L1 `WorldModelRSSMConnector`（仅 stoch/deter 状态 schema 归一化）。故 L3=20、L1=57。

> 这些**没有**任何一条在 CPU 上跑过推理。HTTP `POST /resources/{id}/invoke` 对它们统一返回 **503**，body 带 `resource_status=absent` + `requires{gpu,weights}` + `install_hint`。

## 3. 连接器架构卡片索引

| 模块 | connector | 真实对接的 UDOS 侧 |
|---|---|---|
| `udos/resource_registry.py` | `ResourceConnector` 协议 / `ResourceRegistry` | 统一四态能力探测（带线程缓存、永不抛） |
| `udos/connectors/base.py` | `GenericConnector` | `normalize_trajectory` → `udos/pce-action/v1` |
| `udos/connectors/specialized.py` `ActionChunkingConnector` | ACT / Diffusion Policy / RDT | 动作分块 [H,A] → canonical action 轨迹（对接 wla RVQ） |
| `…VLADiscreteConnector` | OpenVLA | 256-bin token ↔ 连续动作（量化往返误差 ≤ 0.5/255） |
| `…EpisodeSchemaConnector` | LeRobot/Open-X/RoboMimic/BridgeData/LIBERO/CALVIN | step 点分列名 → canonical obs/action/timestamps |
| `…SMPLPoseConnector` | AMASS/GRAB/HumanPlus/OmniH2O | axis-angle(72) → rotation-6d（对接 retargeting） |
| `…BVHClipConnector` | LAFAN1 | MOTION 数值行 → 关节轨迹 `udos/bvh-clip/v1` |
| `…URDFFKConnector` | yourdfpy / pytorch_kinematics | 关节角归一化 + yourdfpy 真 FK（L2） |
| `…WorldModelRSSMConnector` | DreamerV3 | RSSM stoch(类别概率归一化)+deter 状态 schema（L1，无权重） |

## 3.1 第三方代码库架构卡片（docs/third_party/）

按 SCOPE→ROUTE→EFFECT→BREAK→SHIP 只读方法，为真实接入的高优先级库逐张成卡：

| 卡片 | 证据等级 | UDOS connector |
|---|---|---|
| [yourdfpy](third_party/yourdfpy.ARCHITECTURE.md) | **confirmed**（读 site-packages 真实源码+真跑 FK） | URDFFKConnector |
| [pytorch_kinematics](third_party/pytorch_kinematics.ARCHITECTURE.md) | inferred（未装，标 absent） | URDFFKConnector |
| [ACT](third_party/ACT.ARCHITECTURE.md) | inferred | ActionChunkingConnector |
| [DiffusionPolicy](third_party/DiffusionPolicy.ARCHITECTURE.md) | inferred | ActionChunkingConnector |
| [LeRobot](third_party/LeRobot.ARCHITECTURE.md) | inferred | EpisodeSchemaConnector |
| [RoboMimic](third_party/RoboMimic.ARCHITECTURE.md) | inferred | EpisodeSchemaConnector |
| [LAFAN1](third_party/LAFAN1.ARCHITECTURE.md) | inferred | BVHClipConnector |
| [AMASS](third_party/AMASS.ARCHITECTURE.md) | inferred | SMPLPoseConnector |
| [DreamerV3](third_party/DreamerV3.ARCHITECTURE.md) | inferred | WorldModelRSSMConnector |

## 4. 性能 A/B（benchmarks/results/resource_registry_v456.json，CPU 真实计时）

| 指标 | 数值 |
|---|---|
| 装配 79 条 | 0.24 ms/op |
| probe 未缓存 → 缓存 | 0.0024 → 0.001 ms/op（缓存命中快 ~2.4×） |
| 列表 performance / full(autoprobe) | 0.082 / 0.075 ms/op |
| L1 转换热路径(action chunk) | 0.0045 ms/op |
| profile 切换 | 0.0019 ms/op |

**结论**：registry 为纯元数据/纯 python 转换，热路径在亚毫秒级；无性能劣化。未做无证据的"提速宣称"——本线为新增能力，A/B 记录基线而非宣称加速。

## 5. profile 语义（诚实定义）

- `performance`：**机械推导** = 高优先级 ∩ CPU可行(L1/L2，剔除 L3 重权重/GPU)。实测 **25 条**（L1=23 + L2=2，yourdfpy available / pytorch_kinematics absent）；不再手写窄白名单。
  - 高优先级解析：`udos_fit` dict 取 `priority` 字段、str 取"高——"前缀首字，字段实测高优先级=**36**（dict 21 + str 15；用户口径曾记为 27，以字段机械推导为准并据实记录）。performance 再剔除其中 11 个 L3 重权重模型，得 25。
- `full`：全量 79 条可发现、可探测、可优雅降级；批量探测带缓存，<30s 不卡启动。
- 契约（tests/test_resource_contract_v455.py 钉死）：performance ⊆ full、performance 无 L3、full=79、performance=25 且全部 priority=high。
- "全面兼容" = **接口统一 + 可发现 + 能力自检 + 缺失优雅降级**，**不等于全部在 CPU 跑通**。

## 6. HTTP 端点

- `GET /resources[?profile&kind&status&license&priority&level]` —— 纯 JSON 列注册表/能力/分级
- `POST /resources/{id}/probe` —— 真实能力探测（带缓存）
- `POST /resources/{id}/invoke` —— 统一调用；L3/缺失→503、未知 id→404、缺 action→400
- `POST /resources/profile` —— 运行时切换视图 profile（不改主权重）
- 环境变量 `UDOS_RESOURCE_PROFILE=performance|full`
