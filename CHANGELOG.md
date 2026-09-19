# Changelog

本文件记录 UDOS 推演引擎的显著变更，遵循 Keep a Changelog 与语义化版本。
定量结论以对应 `docs/VERIFICATION_v*.md` 与 `benchmarks/results/*.json` 为准。
v7 重写线（`udos7/`，纯引擎，不含 AGI/ASI 倒计时网站——网站属独立 v6.2 线）的结论以 `docs7/VERIFICATION.md` 与 `reports7/*.json` 为准。

## v7.5.0（百万级 Agent 矩阵集成收口；cpu-proto）

- 新增 `udos7/topology/matrix.py`：Stigmergy 黑板派单 → 分层 specialist 执行（熔断节点排除）→ 失败回滚 trace 并释放改派 → BFT-lite QA 委员会 2f+1 验收 → 内部市场按验收结算 → 哈希链 Trace + 治理审计的端到端闭环。
- 实测（24 工单，7 人 QA 含 2 拜占庭，3 个故障 specialist：crash/drop_context/byzantine 各一）：24/24 验收收口，3 个故障节点全部隔离，5 次返工改派，治理 ok、市场守恒、Trace 链完整；全领域故障+零重试预算时 kill-switch 触发并如实报告未收口，不冒充成功。
- 规模基准（b=8，显式模拟与闭式互验后外推）：百万级目标档实际 2,396,745 节点/7 层；星型中心扇入 20,971,520 条消息、串行 8,388,608 轮；分层每节点最大扇入恒 9、14 轮收敛。
- 新增 `scripts7/matrix_demo.py` 与 `reports7/matrix_scale_demo.json(.traces.jsonl)`；CLI `udos demo matrix`；`topology/__init__.py` 全量导出 68 个符号；新增 `docs7/TOPOLOGY_MATRIX_v7.5.0.md`。
- 修复版本号单一来源：此前 `udos7/__init__.py` 与 `pyproject.toml` 停在 `v7.4.2`（带 v 前缀导致 v7.4.3 起的版本替换未生效），现统一为 `7.5.0`，CLI 显示自动带 v 前缀。
- Tests 新增 `tests7/test_v750_matrix.py` 9 项；**全量回归 236 项全绿**（拓扑层 v7.4.1–v7.5.0 共新增 103 项）。

## v7.4.10（Stigmergy 环境媒介协作；cpu-proto）

- 新增 `udos7/topology/stigmergy.py`：共享黑板 + 原子认领（同一任务不会被两人认领，杜绝重复劳动）+ 完成标记 + 信息素蒸发/加权引导；Agent 间零直接通信。
- 实测（30 任务/5 Agent）：stigmergy 每任务 2 条痕迹共 60 条消息，contract-net 协商每任务 8 条共 240 条；40 任务/4 Agent 负载差 ≤1；高信息素任务被优先选中。
- Tests 新增 `tests7/test_v7410_stigmergy.py` 8 项。

## v7.4.9（Agent 内部市场与贡献结算；cpu-proto）

- 新增 `udos7/topology/market.py`：任务投标按质量/成本性价比授标（质量门槛、负载兜底）；贡献台账仅对验收通过的唯一完成者付费，重复劳动不付费，拜占庭/被拒结果不付费并可罚没保证金。
- 守恒断言：总支出 ≤ 总预算，余额和 = 已付 − 罚没，未付预算留在池内；实测 3 任务混合（通过/拒付罚没/通过）守恒成立。
- Tests 新增 `tests7/test_v749_market.py` 10 项。

## v7.4.8（多层熔断、隔离与回滚；cpu-proto）

- 新增 `udos7/topology/circuit_breaker.py`：Agent 级滑动窗口错误率断路器（closed/open/half_open + 冷却探测）、子矩阵级隔离比例熔断、全局 kill-switch；故障时在途工单登记并改派健康节点。
- TraceLedger 快照/回滚：派发前记录长度与末哈希，故障后截断半截状态并重派，回滚后哈希链仍可验。
- Tests 新增 `tests7/test_v748_circuit_breaker.py` 11 项。

## v7.4.7（分布式停止共识 BFT-lite；cpu-proto）

- 新增 `udos7/topology/consensus.py`：委员会对 stop/continue 投票，n≥3f+1、2f+1 法定人数；拜占庭谎报无法对抗诚实多数，同轮重复投票（equivocation）检出并整轮作废，超 f 缺席触发 view_change 而非提前终止/无限等待。
- 安全性断言：诚实票总数 n−f 下冲突决定不可能各凑齐法定人数；非委员会投票者被拒；签名占位确定性可验。
- Tests 新增 `tests7/test_v747_consensus.py` 10 项。

## v7.4.6（分层混合拓扑：b 叉聚合树与百万级扇出基准；cpu-proto）

- 新增 `udos7/topology/hierarchy.py`：顶层 Orchestrator / 中层 Handoff / 底层 Swarm 的 b 叉聚合树；小规模逐边显式模拟并与闭式计数互验，大规模（≤100 万 Agent）用已验证公式外推（explicit/analytical 分级标注）。
- 实测（branch=8，100 万 Agent）：星型中心扇入 >100 万消息、串行轮次=单元数；分层每节点最大扇入恒为 b+1=9、顶层只见 8 条领域汇总、并行轮次 2log_b N=14。
- Tests 新增 `tests7/test_v746_hierarchy.py` 9 项（含显式/公式一致性、对数轮次、百万外推）。

## v7.4.5（拓扑决策树：控制需求驱动选型；cpu-proto）

- 新增 `udos7/topology/decision.py`：流程明确→Orchestrator；需专家接力→Handoff；能力可封装→Agent-as-Tool；开放探索→Swarm；目标不清或高风险自治→escalate（高风险 Swarm 不放任）。
- 自治度等级随分支单调；规模 ≥1000 且需接力/探索时建议分层混合（v7.4.6 落地）。
- Tests 新增 `tests7/test_v745_decision.py` 12 项（含全分支参数化与高风险护栏）。

## v7.4.4（Owner/Trace/Stop Condition 治理：三类失败机械审计；cpu-proto）

- 新增 `udos7/topology/governance.py`：哈希链追加式 TraceLedger（篡改/断链可检出）、StopCondition（完成谓词+最大跳数，区分正常停止/无限循环）、audit_run 机械检测状态丢失、重复劳动、无人收口、提前终止。
- 故障注入实测：drop_context 在链式交接留下 todo_left 并被判定 state_loss；重复 claim 计 duplicate_work；byzantine specialist 被 QA 拒绝且工单 no_closer。
- Tests 新增 `tests7/test_v744_governance.py` 9 项。

## v7.4.3（Transfer Bundle：交接即责任转移；cpu-proto）

- 新增 `udos7/topology/transfer.py`：Goal/Context/Done/Todo/Trace/Owner 六字段交接包；validate 机械校验缺失字段、空 todo、done/todo 重叠；handoff 仅在包完整时转移 Owner，否则返回类型化缺口、责任不转移。
- advance 推进阶段并留痕，replay 仅凭交接包重建工作状态；故障注入验证 drop_context（缺 result）在交接点被拦截。
- Tests 新增 `tests7/test_v743_transfer.py` 9 项。

## v7.4.2（Agent-as-Tool：专家 Agent 工具化封装；cpu-proto）

- 新增 `udos7/topology/agent_tool.py`：ToolInput/ToolEnvelope 固定契约（report/confidence/error_type/trace_ref），主 Agent 只收信封、不收专家内部 trace，上下文增量有界。
- 类型化错误：bad_input / capability_gap / internal_error / timeout / low_confidence；专家内部异常不外泄；置信度门槛可配；专家实现可独立迭代而契约不变。
- Tests 新增 `tests7/test_v742_agent_tool.py` 8 项。

## v7.4.1（多智能体拓扑内核：Orchestrator/Handoff/Swarm；cpu-proto）

- 新增 `udos7/topology/`：统一工单契约（triage→specialist→qa 三阶段、内置真值）、可注入故障（drop_context/byzantine/duplicate/crash）的 AgentSpec、TraceEvent 父事件链与 RunMetrics（成功数/消息数/跳数/重复劳动/Owner 登记）。
- 三种拓扑跑**同一套处理器**：星型中心派发（2 消息/阶段）、链式 Transfer 接力（1 移交/阶段）、网状 contract-net 广播-投标-授标（负载均衡）。
- 实测（12 工单干净舰队）三拓扑均 12/12 成功；网状消息量 > 星型 > 链式；QA 双 Agent 在网状下被均衡使用。
- Tests 新增 `tests7/test_v741_topologies.py` 8 项。

## v7.4.0（第一人称经验数据飞轮：Coverage-aware 采集 + State Coverage + Yield；cpu-proto）

> 对标 egocentric data boom（Maxinsights/Dyna-2/GENE-26.5 公开报道，数字均 unverified）：把“人类经验→training-ready 数据”的工业管线在引擎内做成可跑、可对照的 CPU 原型。

- **Added `udos7/egodata/`**：
  - `episodes.py` 在 v7.3.7 具身环境上合成第一人称片段（手部轨迹代理、接触、子任务边界、4 任务×12 参数变体），按 25% 注入静止/镜头漂移/重复摆拍三类缺陷；
  - `coverage.py` 细粒度状态单元 `(任务,x,y,速度,接触)`，区分 Task Coverage 与 State Coverage；
  - `processing.py` 盲检 QC（轨迹长度、相机系抖动二阶差分、量化签名去重）、经验密度、Yield 良率台账；
  - `collection.py` 被动偏态采集 vs Coverage-aware 子模贪心采集（边际新状态最大化，1-1/e 近似）+ 同池消融。
- **实测（160 候选/48 预算=144 原始分钟）**：被动 Yield 0.667、覆盖 oracle 0.695、密度 7.80；Coverage-aware Yield **0.854**、覆盖 **0.951**、密度 **11.38**；消融（偏态池+贪心）0.889——收益约一半来自主动选择、一半来自采集分布多样化。
- **Added** `scripts7/egodata_flywheel_demo.py`（CLI `udos demo flywheel`）、报告、`docs7/EGODATA_FLYWHEEL_v7.4.0.md`。
- **Tests** 新增 `tests7/test_v740_flywheel.py` 10 项（确定性、三类缺陷盲检、Task/State 覆盖区分、密度排序、主动>被动、良率台账、基准确定性与证据分级、缺口引导）。
- **限制/闸门**：无真实第一视角视频/手部追踪/SLAM/MaxVLM；缺陷合成、状态单元手工离散。真实采集网络、视频理解大模型 key、跨本体真机验证列为闸门；外部 150 万/200 万/98%/1000 万小时等数字仅 unverified 参照。

## v7.3.9（高斯泼溅 3DGS-lite + Real-to-Sim / Sim-to-Real 闭环；cpu-proto）

> 在 v7.3.8 显式几何之上补 Atlas 类世界模型的另外三块：可微高斯泼溅输出、真实到仿真的重建管线、仿真到真实的失配与鲁棒性。外部 3DGS/Atlas 指标不与本仓数字互证。

- **Added `udos7/spatial/splat.py`**：轴对齐三维高斯 + 针孔一阶 footprint + 前到后 alpha 合成；TSDF 占据点初始化，Adam 拟合颜色/深度（150 高斯、40 步、36×28）。实测留出视角轮廓 IoU 0.729、覆盖率 1.00、深度 MAE 0.253（cpu-proto，非论文级 3DGS）。
- **Added `udos7/spatial/transfer.py`**：
  - 带噪观测（深度噪声/丢点、位姿抖动）→ TSDF → 6-连通聚类抽障碍球（z=0 平面截圆 + 合成真值标定的体素膨胀校正）；
  - Real-to-Sim：平面相交障碍簇数 2/真值 2，主障碍重建 r≈0.70（真值 0.69），重建仿真中 v7.3.7 hybrid 控制 8 回合 100%；
  - Sim-to-Real：规划/执行环境分离，有偏估计 + 扰动“真实代理”世界，朴素规划成功率 50%/均碰撞 0.50，半径膨胀裕量 0.22 的鲁棒规划 100%/0 碰撞。
- **Added** `scripts7/splat_transfer_demo.py`（CLI `udos demo splat`）、报告、`docs7/SPLAT_TRANSFER_v7.3.9.md`。
- **Tests** 新增 `tests7/test_v739_splat_transfer.py` 8 项（单高斯投影、拟合降损、新视角几何、噪声确定性与损失、平面障碍恢复、重建控制、裕量降碰撞、基准确定性与证据分级）。
- **已知限制**：无旋转协方差/EWA、静态场景；Real-to-Sim 全局规划仅覆盖单主障碍过门；半径校准来自合成真值。闸门：GPU 完整 3DGS、真实相机标定、MuJoCo(-MJX)/真机迁移。

## v7.3.8（新视角预测与空间上下文 New View Prediction / Spatial Context；cpu-proto）

> 对标 Atlas 类世界模型“给定带位姿的多视角观察，预测任意新视角”的基础任务；外部主张（AI-complete、Real/Sim-to-Sim 成本）仅作 unverified 参照。

- **Added `udos7/spatial/`**：
  - `camera.py` 针孔相机（look-at 位姿、投影/反投影闭合、逐像素世界视线）；
  - `scene.py` 球体基元合成场景 + 解析光线求交，渲染真值深度/颜色/掩膜；
  - `fusion.py` `VoxelContext` 多视角深度 TSDF 体素融合（显式空间上下文）+ 任意新视角深度预测 + IoU/覆盖率/精确率/深度 MAE；
  - `benchmark.py` 8 训练视角 / 6 错开留出视角，单视角 vs 多视角对比。
- **实测（CPU 可复现）**：多视角融合在留出视角上轮廓 IoU 0.890、覆盖率 0.996、深度 MAE 0.092，显著优于单视角 0.715/0.746/0.319；机制上验证新视角预测依赖相机几何与三维一致性。
- **Added** `scripts7/spatial_novelview_demo.py`（CLI `udos demo spatial`）、报告、`docs7/SPATIAL_NOVELVIEW_v7.3.8.md`。
- **Tests** 新增 `tests7/test_v738_novelview.py` 7 项（投影反投影闭合、渲染可见、融合占据、训练位姿一致性、多视角优于单视角、指标完全匹配、确定性）。
- **闸门**：真实多目采集/标定、GPU 上 NeRF/3DGS 训练、动态场景；学习型高斯泼溅见 v7.3.9（CPU 原型）。

## v7.3.7（具身混合控制：语义层 × 动作先验 TopK 候选 + 闭环接管；cpu-proto）

> 对标公开仿真报告中「通用大模型语义判断/修正 + VLA（π0.5）物理动作先验」的混合架构，在引擎 6 维状态契约上做可复跑 CPU 原型；外部报告数字（62.6 分/48%/14.4% 接管/Token 量）仅作 UNVERIFIED 参照，不与本仓数字互证。

- **Added `udos7/embodied/`**：
  - `env.py` 三维点质量闭环环境：有序目标、球形障碍接触（可终止失败）、中途扰动；任务套件 reach_free / ordered_sort / contact_gate / disturb_recover。
  - `hybrid.py` `MotionPrior`（K=32 候选动作段，PD 机动+横向避让）+ `SemanticCritic`（确定性规则语义裁决：顺序/进展/碰撞/努力度；hybrid 含 1 条短接管程序），三模式 direct / motion / hybrid；输出成功率、分数、碰撞、接管率、Token 代理量；接可观测 Tracer（motion.propose / semantic.critic / intervention 事件）。
  - `benchmark.py` 三模式×四任务配对对比，证据等级 cpu-proto，外部口径单列 external_reference(unverified)。
- **实测（CPU，seed 1000–1009，每任务 10 回合，可复现）**：hybrid 成功率 100%/均分 100/零碰撞/接管率 5%；direct 75%/77（contact_gate 0%）；motion 53%/69.6（ordered_sort 0%、disturb 10%）。语义层强在顺序与恢复、动作先验强在接触，互补性成立。Token 代理 hybrid 约为 direct 一半（方向同外部报告；proxy 非真实计费）。
- **Added** `scripts7/embodied_hybrid_demo.py`（CLI `udos demo embodied`）、`reports7/embodied_hybrid_demo.{json,traces.jsonl}`、`docs7/EMBODIED_HYBRID_v7.3.7.md`。
- **Tests** 新增 `tests7/test_v737_hybrid.py` 9 条（状态契约/有序目标/碰撞终止/K 候选/裁决选无碰撞/接管率边界/同 seed 确定性/套件互补性/Token 方向/Tracing）。
- **闸门**：LLM 语义裁判需 API key（token proxy→真实 usage）；MuJoCo(-MJX) 接触与学习型 π/世界模型需 GPU/HPC，未在本版冒充。

## v7.3.6（CLI 易用性：双击命令行菜单 + 根目录便捷入口；工具版）

> 解决“CLI 已实现但藏在 bin/ 子目录、未加 PATH，小白不知道怎么用”的问题。CLI 能力本身在 v7.3.4 引入、v7.3.5 接入共享环境，本版只加入口与引导，引擎行为不变。

- **Added 双击命令行菜单（无需记命令）**：仓库根 `UDOS-命令行菜单.bat`（Windows）与 `UDOS-命令行菜单.command`（macOS/Linux），数字选择：环境信息 / 跑 tests7 / 启动服务 / 健康检查 / 可观测六层演示 / Agent 军团演示 / 完整帮助 / 手动修复共享环境。
- **Added 根目录便捷入口**：Windows `udos.bat <命令>`、macOS/Linux `./udos.sh <命令>`，自动转发到 `bin/udos(.bat)`；说明只有 `pip install -e .` 后才能在任意目录直接敲 `udos`。
- **Docs**：`docs7/CLI_v7.3.md` 顶部新增“最快用法（不用记命令）”，明确双击菜单、相对路径命令与全局命令的区别。
- **Tests**：新增 `tests7/test_v736_menu.py` 4 条（两平台菜单存在且接好命令、POSIX 菜单 `bash -n` 语法通过、根便捷入口转发与真实 `version` 调用）；**tests7 全回归 99 项通过**。
- 说明：Windows `.bat` 运行时在 Linux 沙箱不可执行，仅做静态契约校验；POSIX 菜单与转发脚本已实际执行验证。

## v7.3.5（共享虚拟环境：torch 只装一次，多版本复用；工具版）

> 解决“每个版本解压后都要重复安装 torch”的问题。统一改为用户主目录下的**共享虚拟环境**，
> 并锁定 torch 同一版本；仅当依赖清单哈希变化时才增量安装。引擎/CLI 行为不变。

- **Added `bin/udos-env.py` 跨平台环境引导（纯标准库）**：
  - 环境位置解析：`UDOS_VENV` 环境变量 > 主目录共享环境（Windows `%USERPROFILE%\.udos\venv`，macOS/Linux `~/.udos/venv`）> 旧的仓库内 `.venv-udos/.venv-win`（已存在则沿用，不强迫迁移）。
  - 幂等 provision：以 `BOOTSTRAP_VERSION + 依赖清单` 的 sha256 作为标记（`<venv>/udos_provision.json`），并实际 `import torch,numpy,huggingface_hub,pytest` 自检；标记匹配且导入成功则**零安装直接复用**，否则才增量安装。
  - **锁定 `torch==2.14.0`**：Linux/Windows 走 CPU 专用索引 download.pytorch.org/whl/cpu（无 CUDA、体积小），macOS 走 PyPI 官方 wheel（arm64 含 MPS）；轻依赖官方源失败回退清华镜像。
  - 子命令 `ensure/path/info`；`ensure` 末行打印 venv 的 python 路径供包装脚本捕获。
- **Changed 包装器与一键脚本统一走共享环境**：`bin/udos`、`bin/udos.bat` 通过引导脚本取解释器；Windows 一键启动不再在每个版本目录建 `.venv-win`；mac `deploy/mac/common.sh` 的 VENV 默认改为 `~/.udos/venv`（尊重 `UDOS_VENV`，沿用旧 `.venv-udos`）。
- **Added** `deploy/requirements-cpu.txt` 作为可读/可手动安装的依赖清单（与引导脚本同源同版本）。
- **Docs**：`docs7/CLI_v7.3.md` 增补“一次安装、多版本共享”说明与自定义环境/离线建议。
- **Tests**：新增 `tests7/test_v735_shared_env.py` 9 条（环境解析优先级、平台路径、标记哈希、旧环境沿用、标记+导入双条件、info JSON），同步更新 CLI 包装器契约断言；**tests7 全回归 95 项通过**。
- **实测（Linux CPU）**：首次 ensure 建共享环境并写标记、第二次 ensure 静默零安装、`UDOS_VENV=... ./bin/udos version` 确认走共享环境；Windows `.bat` 运行时在沙箱不可执行，仅静态契约校验（与已实测引导逻辑同构）。

## v7.3.4（跨平台命令行工具 udos；功能版）

> 性质：在 v7.3.3 基础上新增统一命令行入口，macOS/Linux/Windows 用法一致。预测与可观测内核行为不变。

- **Added `udos7/cli.py` 跨平台 CLI（纯标准库）**：子命令 `version / info / serve / test / train / verify / demo / health / metrics / predict`；统一退出码（0 成功、1 运行/连接失败、2 参数或依赖缺失），便于 shell、批处理与 CI 串联。
- **Added 原生包装脚本**：`bin/udos`（macOS/Linux，自动选用 `.venv-udos`/`.venv`，回退 python3）与 `bin/udos.bat`（Windows，自动选用 `.venv-win`，回退 PATH 上的 python）；`pyproject.toml` 注册 console_scripts，`pip install -e .` 后可直接用 `udos`。
- **Added** `serve --auto-port`：首选端口被占用时自动绑定空闲端口，避免直接崩溃；`info` 输出平台/Python/torch/CUDA·MPS/跨平台资源/默认检查点是否存在（JSON）；`health/metrics/predict` 用 urllib 直连运行中引擎。
- **Added** `docs7/CLI_v7.3.md` 命令手册（含 macOS/Windows 示例与预测请求样例），Windows 部署 README 增补 CLI 一节。
- **Tests**：新增 `tests7/test_v733_cli.py` 14 条（版本/帮助/info、健康检查连不上返回 1、serve 派发与自动换端口、test 派发、demo 列举/非法名/obs-pro 真实跑通、predict 非法 JSON 返回 2、两平台包装脚本就位且可执行、console_scripts 注册）；**tests7 全回归 86 项通过**；真实启动服务后 `udos health/metrics` 端到端 rc=0。
- 证据：CLI 行为 verified（CPU 实测）；Windows `.bat` 在 Linux 沙箱无法执行，仅做静态契约校验，逻辑与已实测的 macOS/Linux 包装器同构（标 cpu-static for the bat runtime）。

## v7.3.3（AI 可观测性加深：六层功能补齐 + 跨平台；功能版）

> 性质：在 v7.3.0 可观测内核上按“3× 颗粒度”补齐生产栈缺口，新增 `udos7/observability/intelligence.py`
> 与跨平台资源采集、Windows 一键部署。预测/协同内核行为不变。零第三方依赖，新增能力 CPU 实测 verified；
> OTel Collector gRPC 直推、eBPF、GPU、LLM-as-judge 仍为部署/资源闸门（unverified-on-infra）。

- **Added 会话关联（Session Correlation）**：`session_index()` 以根 Span 的 `session_id` 把多条 trace 聚成会话，输出 traces/span 数/错误 trace/总延迟。
- **Added 实时护栏（Guardrails）**：`Guardrails.inspect()` 在输出到达用户前做 PII 泄露/毒性/格式违规/未接地(幻觉代理)/工具误用五类确定性拦截，拦截写入指标 `gen_ai_guardrail_blocks_total{type}` 与 Span 事件（拦截本身可观测）。
- **Added 成本归因与 Token 效率**：`CostAccounting` 按 model/user/feature 汇总调用数、输入/输出 Token、USD（价格表可覆盖，示例价）、截断浪费计数；`attach_tracer()` 可从离线 trace 的 `gen_ai.usage.*` 回填成本。
- **Added 智能分析层（确定性）**：`AnomalyDetector` 在线 Welford 均值/方差 + EWMA，按“纳入当前点之前”的统计量算 z 分（修复离群点自我稀释），spike/drop 告警；`root_cause()` 在 trace 内定位首个错误阶段、否则定位耗时热点及占比。
- **Added 拓扑图 + 火焰图数据**：`trace_topology()` 输出阶段节点/调用数/错误/父子边（供图数据库/拓扑图）；`flame_profile()` 计算每 Span 自时间 self_ms（总时长−直接子时长）与深度（供火焰图）。
- **Added OTLP/JSON 导出**：`to_otlp_json()/export_otlp_json()` 产出 OTLP resourceSpans 结构（traceId/spanId/parentSpanId/纳秒时间/status/attributes/events，PII 经 redact），可被 Collector/Tempo 以文件/管道摄取；真实 gRPC 推送标 unverified。
- **Changed 跨平台资源采集（macOS + Windows）**：`resource_usage()` 新增 platform 与网络收发字节；RSS 在 Linux 读 /proc、macOS/Linux 回退 `resource.ru_maxrss`、Windows 用 ctypes(psapi)；`network_io_counters()` Linux 读 /proc/net/dev，macOS/Windows 用可选 psutil，取不到诚实留 null。
- **Added Windows 一键部署**：`deploy/windows/UDOS-Windows一键启动.bat`（建 venv→装 CPU 依赖→tests7 回归→起 8777 服务，PyPI 失败自动切清华源）+ 小白 README（SmartScreen、PATH、端口、故障自查）；macOS 沿用 deploy/mac。
- **Tests**：新增 `tests7/test_v733_observability_pro.py` 11 条（会话分组、护栏四类拦截与边界、成本数学与 trace 回填、异常 spike、根因错误/热点、拓扑与火焰自时间、OTLP 结构与导出、跨平台资源），全绿；TDD 过程抓到并修复两个真 bug（session_index 误 `return rec`、异常检测把离群点计入统计导致漏报）。
- **Demo**：`scripts7/observability_pro_demo.py` 端到端跑通六层并产 `reports7/observability_pro_demo.{json,otlp.json}`。
- **诚实边界**：本版不运行容器/eBPF/GPU，不接 LLM 评分与商业内容审核；价格表为示例价；这些路径留空或门禁，不以 CPU 数字冒充生产能力。

## v7.3.0（AI 可观测性体系；功能版）

> 性质：在 v7.2.2 协同/军团之上新增**引擎内零依赖可观测层 `udos7/observability/`**，预测内核与协同层行为不变（默认 checkpoint 仍 v7.0.3）。对齐 OpenTelemetry GenAI 语义约定，落地“采集→管道→存储→分析→告警→治理”分层；引擎内可跑部分为 verified，Prometheus/Grafana/OTel Collector/eBPF/GPU/LLM-as-judge 给真实配置但标 unverified-on-infra（需 Docker/GPU/key）。

- **Added `observability/semconv.py`**：版本化语义契约 `gen-ai-semconv-v1-udos1`，标准 `gen_ai.*`（system/request.model/operation.name/usage.input|output_tokens/prompt/completion）与 UDOS 扩展 `udos.*` 前缀分离；定义 RAG/Agent 七段流水线。
- **Added `observability/tracing.py`**：Trace→嵌套 Span 树（contextvars 维护父栈）、属性/事件/Token 用量/状态、延迟；结构化 JSONL 日志（强制 trace_id/span_id/parent_id/step/status/latency_ms/token_usage）；`redact()` 邮箱/手机/身份证 PII 脱敏；`tail_sample_keep()` 尾采样（错误/超延迟/命中PII 100% 保留，成功 trace 按 trace_id 稳定哈希默认采 10%）。
- **Added `observability/metrics.py`**：Counter/Gauge/Histogram（p50/p95/p99）；AI 专用指标 TTFT、ITL、Token in/out、工具调用量与延迟、错误按类型、队列深度；`resource_usage()`（CPU 核数、RSS 可测，GPU 诚实留 null、evidence=cpu-proto）；`render_prometheus()` 文本 0.0.4 + `start_prometheus_exporter()` stdlib `/metrics` 与 `/health`（HTTP 抓取实测 verified）；七项 SLI 清单与多窗口**燃尽率告警** `burn_rate/alert_burn_rate`（page 14.4/14.4、ticket 6/1）。
- **Added `observability/evaluation.py`**：`OnlineEvaluator` 按采样率对生产流量跑确定性 judge——格式合规（JSON 可解析）、接地性（claim 命中上下文，cpu-proto）、工具调用正确性、毒性阻断词；滚动窗口**质量回归告警**；`LLMJudge` 无供应商 key 即 `GateError`（AL4），不提供伪评分。
- **Added `observability/instrument.py`**：`run_pipeline()` 对 意图分类→查询重写→检索→重排序→上下文压缩→LLM生成→事实校验 建嵌套 Span；`AgentRegistry`（名字/用途/状态/进程）；`observe_goal()` 非侵入包裹协调器 run 并记录决策数/失败数；`UDOS_OTEL_AUTO=1` + `enable_autoinstrument()` **零代码自动埋点**协调器 run（可 disable 还原）。
- **Added `deploy/observability/`**：Prometheus+Grafana docker-compose、抓取配置、TTFT/ITL/错误率 SLO 告警规则、预置 Grafana 看板与数据源、中文部署 README（开源自建路径 + OTel/Langfuse 双轨说明）。
- **Added `scripts7/observability_demo.py`**：20 条模拟请求端到端跑通追踪→指标→尾采样→JSONL→质量评估→Prom 文本，产 `reports7/observability_demo.{json,prom,jsonl}`。
- **Tests**：新增 `tests7/test_v73_observability.py` 12 条（脱敏、嵌套树/JSONL、尾采样三保留一丢弃、百分位、Prom 文本、HTTP /metrics 实测、资源快照、燃尽率 page/ticket、四类 judge、质量回归、LLM judge 门禁、流水线 Span 顺序、Agent 注册表、自动埋点开关还原），全绿。
- **诚实边界**：本版不运行容器/eBPF/GPU，不接 LLM 评分与商业审核；这些路径留空或门禁，绝不用 CPU 原型数字冒充生产可观测能力。复现与分层映射见 `docs7/OBSERVABILITY_v7.3.md`。

## v7.2.2（多 Agent 协同内核 + Agent 军团与 Scaling Law；功能版）

> 性质：在 v7.0.3 统一预测内核之上新增**协同层 `udos7/agents/`**，预测权重/checkpoint 不变（默认仍 `worldmodel_v7.0.3.pt`）。本版一次落地两个里程碑：**v7.1.1 多 Agent 协同**（专家/分类/分工/讨论/碰撞/提名/投票/选优 + 共享记忆 + 协调器）与 **v7.2.2 Agent 军团**（岗位/部门/组织/领域层级、完全异步 fan-out、Agent 版 Scaling Law 实测、AL0–AL5 门禁）。所有定量结论为 CPU 固定 seed 实测，证据分级 verified / cpu-proto / unverified，原始数据 `reports7/agent_scaling.json`。
>
> **版本线说明（legacy 冻结）**：旧 `udos/` 包（v5.5.5）按 v7 既定契约为**冻结对照版**，本轮不另造 v5.6.x/v5.7.x；用户要求的“多 Agent / 军团”等价能力落在主线 `udos7` 的 v7.1.1 / v7.2.2。AGI/ASI 倒计时网站属独立 **v6.2 产品线**，v7 仍是纯引擎、不含网站。

### v7.1.1 —— 多 Agent 协同内核（本版包含）
- **Added `agents/types.py`**：Task（goal/kind/payload/deps/branch/status，`is_ready` DAG 判定）、WorkProduct（作者/客观分/指标/置信/证据级/资源声明/分支/父事件）、Vote、Claim、Collision、Decision、只追加 Event（全局血缘）。
- **Added `agents/memory.py::SharedMemory`**：线程安全的唯一事实源——命名空间 KV + 单调版本 + **CAS**（过期版本写入失败）、每 Agent 私有 scratch（隔离工作区）、只追加事件日志、资源声明表（**读可共享，仅两个写锁碰撞**）、决策台账、`snapshot/merge`（KV 取高版本、事件按 id 去重按 ts 排序）供云端共享上下文。
- **Added `agents/workers.py` 专家 Agent**（重活为同步纯函数，由协调器线程池异步调度）：
  - `PredictionExpert` 三种**互为独立信息源**：`blind`（学习模型盲预测）、`explicit`（显式场景参数）、`analytic`（纯运动学解析积分基线）；
  - `ActionCompareExpert`（外部候选动作 dv 脉冲分别 rollout，按目标距离+风险排序的 MPC 式比较，不自动发明动作）；
  - `ValidatorExpert`（有真值用 MSE 客观打分、代码任务跑 harness）、`CriticExpert`（末位置共识离散度，超阈值标 needs_human）、`ClassifierExpert`、`DecomposerExpert`（确定性模板拆解；自由形式拆解需 AL4 LLM）、`CodePatchExpert`（**封闭候选策略集**、隔离分支草稿、不改主线）。
- **Added `agents/protocols.py`**：验证→讨论→提名→投票→碰撞裁决→选优→合并。最终分 = 0.7 客观验证分 + 0.3 加权投票（无投票完全由验证分决定，避免“声音大”左右结果）；代码任务**仅获胜分支**按 PR 语义合并（`merge_winning_patch`）；预测另给 `weighted_ensemble`。
- **Added `agents/coordinator.py::CoordinatorAgent`**：不写核心代码的项目经理/架构师。pull 共享上下文→分类→拆解 DAG→`asyncio` 信号量 + 线程池**完全异步**派发隔离分支→依赖调度（无就绪标 blocked、失败落 task_failed 不吞）→验证/讨论/投票选优→胜者合并→push 上下文。
- **Tests**：`tests7/test_v71_agents.py` 14 条（CAS 冲突、事件血缘、读写/写写碰撞、分类拆解、客观选优必中最高分、集成形状、动作排序、补丁仅胜者合并不污染主线、异步 DAG 依赖顺序与全部完成、跨协调器 LocalTransport 真实同步、AL4/AL5 门禁），全绿。

### v7.2.2 —— Agent 军团 + Scaling Law + AL 分级（本版）
- **Added `agents/legion.py`**：
  - `build_org(headcount, span)` 组织树（agent/team/department/domain/federation/planet/…）。**编制精确、对象有界**：每层节点数由 `org_level_sizes` 公式给出（可到亿级、O(层数)、实测 1 亿 headcount 构建 0.0012s、仅 2052 个预览节点），预算耗尽子树以 headcount 整数虚拟表示（materialized=False）。**亿级在线 LLM 子 Agent 不实例化、不冒充算力**。
  - `hierarchical_select` 分层投票（team 内选优→department 从胜者中再选）。
  - **Scaling Law 实测**（`scripts7/agent_scaling_bench.py`→`reports7/agent_scaling.json`）：
    - **CPU 密集型吞吐** `throughput_curve`：固定 48 个独立预测任务变并发，实测 1→2 Agent 1.59×，4 Agent 1.03×、8 Agent 0.70×——受物理核/GIL/torch 线程约束，**超订反降**，如实记录不外推；
    - **远程 I/O 型** `io_bound_curve`：用 asyncio.sleep 模拟远程 LLM/工具等待（云端数千子 Agent 的真实形态），并发 1/2/4/8 实测加速 1×/2.0×/4.0×/7.99×（simulation 时延，机制与协调器一致），近线性直到并发上限；
    - **集成质量** `quality_curve`：按**独立校准集** MSE 做 softmax 加权（不偷看 test）。实测加权集成 held-out MSE 0.0092，而**朴素等权平均差专家把结果拖到 0.213（约 23 倍差）**；k 超过独立信息源数后追加最强专家近相关副本，质量在 0.0092 附近**饱和**（拟合 mse(k)=a+b/k，gain_b≈−4e-5）。结论：**增益来自多样性 + 验证加权，而非堆人头**；输出含 best_single/naive_equal 对照。
- **Added `agents/automation.py`**：Epoch AI 口径 **AL0–AL5** 分级 + `CapabilityGate`（缺 LLM key/云上下文/RSI 即对 AL4/AL5 抛 `GateError`）与 `baseline_assessment` 诚实分级表（预测集成 AL2、窄域协调器 AL3 verified；自由云端 LLM AL4、RSI AL5 unverified）。
- **Added `agents/cloud.py`**：`LocalTransport`（进程内多协调器真实发布/订阅/快照合并，verified）与 `CloudTransport`（跨机云共享上下文适配器，未配置 endpoint/token 即 AL4 门禁，真实后端属部署闸门）。
- **Tests**：`tests7/test_v72_legion.py` 5 条（编制精确+对象有界+小规模全展开、分层投票选全局最优、CPU 吞吐实测、质量校准加权与饱和、I/O 近线性），全绿。
- **演示/基准脚本**：`scripts7/legion_demo.py`、`scripts7/agent_scaling_bench.py`。
- **诚实闸门（CPU 沙箱不可达，需用户资源）**：真实多供应商 LLM key（自由拆解/写码/交叉打分）、云端项目上下文与数千在线子 Agent、GPU/HPC（vLLM KV-offload、NEURON/CoreNEURON DHS、MuJoCo-MJX、0.5B/5B 端到端）。本版这些路径一律门禁拦截并标 unverified，绝不用 CPU 数字冒充。

## v7.0.3（解析运动学积分 + 学习门控混合预测头；补丁版）

> 性质：在 v7.0.2 统一内核上的**根因修复补丁**，单一 delta——把已能精确反演的加速度从“通用上下文”升级为“解析积分路径”，由一个只依赖确定性运动学特征的独立门控决定何时信任。所有数字为 CPU 固定 seed 实测（verified），同合同 A/B 与踩坑过程见 docs7/VERIFICATION.md。

- **根因（位置误差随视界二次增长，非“速度预测不准”）**：对 v7.0.2 checkpoint 分步探针发现 accel 误差几乎全是**位置误差**且随视界爆炸——oracle accel 分步 MSE `[0.0088, 0.042, 0.114, 0.245]`、位置 MSE 0.187 vs 速度 0.018。GRU 用通用残差 `next=last+Δ` 难以对恒定加速度做精确**二次积分**。确定性天花板探针：用观测 `a_lin` 做解析积分（`v+=a·dt，p+=v·dt+½a·dt²`），uniform/accel MSE **精确为 0**，spring 1.98、collision 0.29（恒定加速度模型在后两类失配）。
- **Added**：`WorldModelCore.analytic_kinematic_step`（固定 a 的恒定加速度解析积分）；混合预测头 `nxt = learned + g·(analytic−learned)`。`a` 与门控量在 rollout 中由**初始观测窗算一次并固定**（恒定加速度模型里 a 不随时间变；避免预测帧滑窗导致门在 spring 中途误开）。
- **Added**：运动学特征新增第 11 维 **`ca_conf`（恒定加速度一致性置信）**=速度对时间线性拟合 R² × (1−ω_valid)(1−jump_valid)。实测（seed2026）uniform/accel 中位与 p05 均=1.0、spring/collision=0（单用 R² 不够：spring .924、collision .860，必须乘 valid 标志）。
- **门控设计（三个被实测否决的错误方案，见 VERIFICATION）**：`g = ca_conf·tanh(MLP(kin)/2)`。① 零初始化 `clamp(linear,0,1)` 因 raw=0 恰处边界、**边界梯度为 0 冻死**（门全程 0、两臂逐位相同）→ 改 0 点可导的 tanh；② 门读共享 GRU 隐状态 h，门在 accel 饱和后改变共享表征梯度，**拖累 spring（盲路径 0.044→0.58）** → 改为只吃确定性 kin 的独立小头（11→32→6，末层零初始化），与共享路径解耦；③ rollout 每步重测 ca 会在 spring 预测帧上翻转误开 → a/ca 固定自初始窗。门末层零初始化 ⇒ 未训练 `tanh(0)=0` 即 **g≡0，严格恒等**（M1 契约 atol 1e-6 保持）。
- **Changed**：`KIN_DIM 10→11`；`persistence.load_worldmodel` 对 v7.0.2 旧权重做**零列填充迁移**（`scene.kin_encoder` [32,10]→[32,11]，新 ca_conf 列补 0）+ 门控小头缺失零初始化，加载后行为与 v7.0.2 逐位一致；任何 unexpected 键或非白名单 missing 键直接 RuntimeError。
- **实测（held-out seed=2026，verified；选中 hidden=256，967,796 参数，门控小头仅 582 个）**：
  - 盲路径 rollout4 MSE：overall **0.02482→0.01037（−58%）**、accel **0.06926→0.01410（−80%）**、uniform 0.00733→0.00573、spring 0.01933→0.01862（不退化）、collision 0.00334→0.00304（不退化）。
  - oracle overall 0.02090→0.00663、accel 0.06175→0.01090；**盲路径 accel（.0141）已逼近 oracle（.0109）**——解析积分用的是观测 a_lin 而非显式 accel_a。
  - 门控均值（test）：uniform 0.63、accel 0.66、spring **0.00**、collision 0.15（“窗内无跳变但视界内可能碰撞”保持谨慎，这是保留可学习门而非硬 g=ca_conf 的原因）。
  - conformal 覆盖仍达标（oracle 0.784/0.898/0.958，blind 0.776/0.900/0.956，±0.05）；参数扇形包络 0.859；延迟 predict_next≈2.36ms / rollout4≈4.83ms（CPU 2 线程 batch1）。
  - 选档（val 准则 3%）：64=95,348 参数/blind .01527、128=271,476/.01172、256=967,796/.01037，选 256。
- **门禁**：verify_v7 在 15 项基础上新增 6 条 v7.0.3 门禁（accel≤.035、overall 不退化、spring/collision 不退化超 15%、accel 门≥.3、spring 门≤.02），共 **21 项 all_pass**。
- **Tests**：新增 `tests7/test_v703_kin_gate.py` 7 条（未训练恒等、解析积分数学正确、ca_conf 分离、饱和路由、关通道无门、v7.0.2 权重零列填充迁移、小训练门启用且 accel 误差下降的突变敏感性）；`tests7` 共 **33 全绿**；legacy `tests/` 全量回归见 VERIFICATION。
- **诚实边界**：解析积分只在**恒定加速度模型成立**时可信，ca_conf 是合成四类数据上的确定性选择量；真实非平稳/接触动力学下需重新标定门控与 ca_conf，不得把 uniform/accel 的 0 天花板外推。GPU/0.5B/5B、vLLM KV-offload、NEURON DHS、MuJoCo-MJX 仍为 M5 闸门。

## v7.0.2（确定性可观测运动学通道 + 两个真实 bug 修复；补丁版）

> 性质：在 v7.0.1 统一内核上的**根因修复补丁**，不新增第二套链路。核心是把“状态里本就可直接反演的量”从盲估计隐藏标量改为确定性计算 + 零初始化学习注入。所有数字为 CPU 固定 seed 实测（verified），同合同 A/B 见 docs7/VERIFICATION.md。

- **根因（表征口径错误，非“加速度不可观测”）**：v7.0.1 让估计器输出“沿未知随机三维方向 d 的带符号标量 accel_a”，d 与 a 符号不可分离（d 翻转、(v0,a) 反号给出同一轨迹），故该标量 identifiability_skill=−0.53 是伪负结果。状态本身含三维速度，**恒定加速度=速度对时间（秒）的最小二乘斜率，三维向量可精确反演**。
- **Added**：`udos7/kinematics.py`（KIN_DIM=10）——窗心速度、三维加速度向量 `a_lin`（速度对秒的 LS 斜率，分母用 `Σ(t−t̄)²`）、弹簧角频率 ω（位置 PCA 主轴投影成标量 q，对内点做**带截距** `q̈=b1·q+b0`，b1=−ω²；门限 `w2>0.45² & r_spring<0.25 & r_spring<r_const`）、碰撞 x 轴速度跳变（`max|Δvx|>3·median+0.35`）；无效特征置 0 且 valid=0；坏形状/NaN/inf 拒绝。
- **Changed**：`SceneChannel` 新增末层**零初始化** `kin_encoder: Linear(10→scene_dim)`，与显式参数/估计器/场景桥在同一 ctx 求和；接入瞬间未训练仍严格恒等，训练后才承载可观测运动学。`WorldModelCore(..., use_kinematics=True)`；`persistence` 存取该标志，**加载 v7.0.1 旧 checkpoint 默认关闭运动学通道（state_dict 不错位，向后兼容）**。
- **Fixed（真实 bug）**：①服务校准器缓存键由 `use_explicit`（bool）改为 `(use_explicit, horizon)`，修复不同视界串用同一组 `q_per_step` 带宽；②`ConformalCalibrator.fit` 在请求 horizon 短于校准视界时用 `Y[:, :horizon]` 对齐（原 pred(H) vs Y(4) 维度 RuntimeError），并显式拒绝 horizon>校准视界。
- **Added**：`udos7/metrics.py::kinematic_recovery`（向量恢复技能/误报/召回/检出，分母按该类型样本数）；verify_v7 增 5 条运动学门禁 + 3 条相对 v7.0.1 改进门禁（共 15 项）。
- **实测（held-out seed=2026，verified）**：
  - 确定性恢复：accel 三维向量 skill=**1.000**（MAE≈0）、uniform 加速度误报范数 0、弹簧 ω 召回 **1.00**/有效 MAE 0.023/非弹簧误报 0、碰撞跨帧窗检出 0.74/非碰撞误报 0。
  - 盲路径 rollout4 MSE 同合同 A/B：overall **0.0459→0.0248（−46%）**、accel **0.1053→0.0693（−34%）**、spring **0.0637→0.0193（−70%）**、uniform 0.0077→0.0073、collision 0.0040→0.0033；oracle overall 0.0274→0.0209。
  - conformal 覆盖仍达标（oracle 0.805/0.917/0.971，blind 0.793/0.909/0.965，±0.05）；扇形包络 0.826；延迟 predict_next≈1.8ms / rollout4≈4.2ms；HTTP 冒烟 smoke_ok=true、半宽随 α 严格递增。
  - 选中档 hidden=256，**967,182 参数**（运动学编码器仅 352 个）；三档盲 test MSE 0.0251/0.0214/0.0248，选档按 val 准则（非 test）。
- **Tests**：新增 `tests7/test_v702_kinematics.py` 10 条；`tests7` 共 **26 全绿**；legacy `tests/` 全量回归 1656 通过/2 跳过/0 失败。
- **诚实边界**：确定性通道只承载状态可直接反演的量；碰撞时刻、被撞体参数、未来输入仍为隐含量。CPU 0.97M 档数字不外推 GPU/大规模；0.5B/5B、vLLM KV-offload、NEURON DHS、MuJoCo-MJX 仍为需 GPU/HPC 的 M5 闸门。

## v7.0.1（统一世界模型内核重写；新基线）

> 性质：相对冻结 legacy v5.5.5 的**重写**，删除“未训练演示 CTM + TinyBaseModel + LoRA 注入后从不 forward”的死路，合并为单一推理图。新建干净包 `udos7/ tests7/ scripts7/ checkpoints7/ reports7/ docs7/`，与 `udos/`（v5.5.5 冻结）并存。

- 单一推理图：观测逐帧编码 → 统一场景通道（显式参数＋估计器＋零初始化场景桥在同一 ctx 求和）→ 唯一 2 层 GRU → 残差解码 `next=last+Δ`（Δ 头零初始化，未训练即恒等）。
- 真三维四类动力学（uniform/accel/spring/collision），轨迹级 train/val/test/calib 四分（种子 42/1337/2026/314），窗口只在轨迹内切，杜绝泄漏。
- 不确定性：split conformal 按 α 真分水平（根治 legacy 三档名义覆盖全 0.8988 的 α 失效）＋参数蒙特卡洛扇形⊕残差 conformal 包络；entropy-certainty 与校准概率严格分离。
- 选中 hidden=256，966,830 参数（held-out 收敛选档，非预设）；综合门禁 7 项 all_pass；HTTP `/api/v7/*`（标准库 ThreadingHTTPServer）；MuJoCo CPU 最小虚拟小鼠因果代理标 cpu-proxy。

## v5.5.5（双引擎增量线收尾：全量回归 / 服务实启动 / 发布一致性）

> 性质：v5.4.5→v5.5.5"双引擎名副其实"增量线的**收尾与发布版**，无新算法；做全量回归、真实服务启动、版本/打包/容器/看板一致性核对与证据汇总。冻结锚点现场复核：主预测员 **52191 可学习参数 + 4 buffers**（更正早期笔记中"48 buffers"的误记）、场景头 **6788 参数**，均未改动。

- **全量回归（CPU）**：`pytest tests/` 全量 **1656 通过 / 2 跳过 / 0 失败 / 0 错误**（2 跳过为环境相关既有 skip）；本机性能相关 `benchmarks/results/*.json` 的覆写已按惯例还原，不污染跨机基线。
- **服务真实启动冒烟**（`scripts/v555_service_smoke.py` → `reports/v555_service_smoke.json`）：以 `python -m udos.server --checkpoint checkpoints/predictor_v4.3.9.pt` 真实拉起 HTTP 服务，`GET /health`=ok 且 `predictor_trained=true`、`GET /checkpoints`=200（34 件）、`POST /predict`=200。同一匀速窗口（vx=1.3）在线复现场景门效应：**盲预测下一位置倒退**（对期望位置误差 1.50、速度误差 1.50），带正确 `scene_params=[1.3,0,0,0]` 的条件化预测方向正确（位置误差 0.37、速度误差 0.38），`gate_corrects=true`。
- **证据汇总**：`scripts/v555_regression_summary.py` → `reports/v555_regression_summary.json`，汇总 5.4.9 三路 A/B、5.5.0 学习头、5.5.1 全局 conformal、5.5.2 类型条件化、5.5.3 门观测、5.5.4 LoRA 前向闭环的关键数字与冻结锚点，并显式列出遗留 v7 项（conformal 多水平失效、加速类非高斯结构误差、GPU 栈需云预算）。
- **打包/容器一致性（静态）**：`python -m udos.server --help` 干净导入；Dockerfile 的 COPY 路径（requirements.txt/udos/demos/checkpoints/predictor_v4.3.9.pt）全部存在，镜像 LABEL 与 compose image tag 已随 bump 同步到 5.5.5。**沙箱无 docker 守护进程，镜像未实际构建**，该项标记为未验证（需在有 Docker 的环境执行 `docker build` / `docker compose up` 后闭环）。
- **版本一致性**：`scripts/bump_version.py 5.5.5` 已同步运行时 `udos/__init__.py`、pyproject、Dockerfile LABEL、compose image、web 看板数据/标题与全部测试 `__version__` 断言。
- **能力边界（不变）**：全部结论为 CPU 合成数据（仅 x 轴非平凡）下的诚实档；主预测员/场景头权重全程冻结；LoRA 前向闭环位于演示基座通道，不接入物理预测员；GPU/多 LLM/双云/真实传感器为需资源授权的后续闸门。

## v5.5.4（消除 LoRA 死路：被注入基座真正参与场景前向编码）

> 性质：修复"GPM 生成 LoRA 并注入演示基座、却从不调用基座前向"的死路。改动只作用于 GPM/LoRA/演示基座（TinyBaseModel）这条演示通道；**冻结主预测员 PhysicsPredictor（52191）与场景头（6788）权重、物理 rollout 完全不变**。

- **根因（双重死路）**：①`internalize_scene` 注入 LoRA 后从不调用 `base_model.forward()`；②`GPMConfig` 默认 `init_scaler_b_zero=True`（LoRA 训练零扰动约定）使 B=0，即使调用前向注入差也为 0。
- **Added**：`UDOSReasoningEngine._scene_features` / `base_scene_encoding`（让被注入基座对场景特征真正跑一次前向，返回池化编码与补丁状态）/ `lora_path_self_test`（一次性：基线→注入→**注入差**→复位→**复位误差**，自测后不留补丁）。
- **Changed**：`internalize_scene` 注入前后各跑一次基座前向，记录注入前基线与 `||Δ||` 注入差，消息附带；`reset_scene` 复位后重跑前向并与注入前基线比对，消息附带**复位误差**。
- **Changed**：引擎**自建默认 GPM**（调用方未显式传 `gpm_config`）改为 live 演示档 `init_scaler_b_zero=False`，使 LoRA 前向通道默认可观察；调用方显式传入的 config（含训练零扰动档）原样尊重。`GPMConfig` 自身默认与 test_gpm 的零扰动约定不变。
- **Added**：`ReasoningResult.lora_injection_active` / `lora_patched_modules` 并进入 `summary()`；`reason()` 反映当前 LoRA 前向通道状态。
- **Added**：`scripts/lora_forward_selftest.py` → `reports/v554_lora_forward.json`。
- **实测（演示基座，四类场景）**：live 档注入差 匀速 0.60 / 加速 0.53 / 弹簧 0.38 / 碰撞 0.57（非零，LoRA 确实改变场景编码），**复位误差恒 0.0**，每场景 6 个线性层补丁、3840 LoRA 参数；训练档（scaler_B=0）补丁照样打过（前向闭环存在）但注入差 0.0、复位误差 0.0，符合 LoRA 零扰动约定。
- **诚实边界**：该闭环位于**演示基座**通道，证明"内化→注入→前向→无损复位"机制真实成立；不把未训练的演示基座输出接入冻结物理预测员（那样只会污染已验证的轨迹预测，属造假）。要让 LoRA 承载真实场景增益，需训练 GPM 超网络（另立任务，非本版范围）。
- **Tests**：`tests/test_v554_lora_forward_path.py` 7 项（live 注入差非零+复位逐位归零、自测确定性、编码随注入改变、internalize/reset 消息与精确复原、零 scaler 档闭环运行但零注入差、reason 通道状态、默认引擎 live）；连同 reasoning/gpm/contracts/conditioning/persistence/553/546/547 共 60 项回归全绿。

## v5.5.3（双引擎可观测性：来源 / 置信 / 场景门贡献 / 盲-感知轨迹差）

> 性质：只读可观测能力，不改任何模型权重与既有默认契约；主 predictor 恒 52191、场景头恒 6788。`reason(observe=False)` 默认行为与计算量逐位不变。

- **Added**：`udos/dual_engine_observe.py`
  - `observe_conditioning`：同一窗口、同一冻结预测员并排跑**盲 rollout**（无场景参数/记忆）与**感知 rollout**（本次场景条件），输出 `EngineObservation`。
  - 来源 `source`（metadata/attributes/learned_head/classical_router/blind）；v5.5.2 路由类型与一个 [0,1] 的**判别果断度**启发量（碰撞=跳变超阈比例、弹簧=简谐/匀加速模型相对裕度、线性类=lin_r2；是分类裕度，**非概率**）。
  - 场景门边际贡献：盲-感知轨迹的逐步位置/速度向量范数 `gate_position/velocity_delta`，及均值 `gate_contribution`、末步 `gate_final_delta`；可选挂接 v5.5.1/5.5.2 扇形给出逐步位置半宽。
  - 支持 `precomputed_conditioned` 复用 reason 已算 rollout，观测仅补一次盲 rollout。
- **Added**：`reason(..., observe=True)` 新增 opt-in，结果新增 `gate_observation`（JSON 友好 dict），并进入 `summary()`；默认 `observe=False` 时该字段为 None。
- **Added**：`scripts/dual_engine_observe.py` → `reports/v553_gate_observation.json`。
- **实测（held-out seed=2026，学习头条件，rollout4 MSE / 门贡献）**：场景门对四类均为命脉——匀速 盲 0.892→条件 0.017（门末步差均值 2.33）、加速 1.323→0.021（2.20）、弹簧 0.692→0.066（1.84）、碰撞 0.590→0.027（2.14）；路由准确率 匀速/弹簧 1.00、加速 .75、碰撞 .71（误判窗为近匀速窗，已在 5.5.2 验证下游无害）。干净合成数据上果断度饱和为 1.0（真实含噪数据才会拉开，如实记录）。
- **诚实边界**：可观测性只度量"场景通道改变了多少预测"，不保证预测正确（正确性仍由 5.5.0/5.5.2 的 A/B 与覆盖率支撑）；路由置信是启发式裕度而非校准概率；观测固定 dt=0.5 用于类型判别。
- **Tests**：`tests/test_v553_dual_engine_observe.py` 10 项（盲零贡献、显式弹簧门贡献大且更准、差非负与形状、果断度单位区间、复用预计算逐位一致、扇形半宽几何、输入校验、dict JSON 友好、reason 默认关闭与 opt-in 字段）；连同 552/551/550/reasoning/547 共 46 项回归全绿。

## v5.5.2（四类运动识别 + 估计路由 + 类型条件不确定性）

> 性质：新增能力，不改主模型/场景头权重；主 predictor 恒 52191、场景头恒 6788 参数。确定性运动路由器把经典估计的弹簧恢复率从个位数拉到满值，并用类型条件 conformal 修正 v5.5.1 的分类型欠/过覆盖。

- **Added**：`udos/dynamics_router.py`
  - `classify_dynamics`：仅凭窗口判别 uniform/accel/spring/collision。碰撞用帧间速度跳变稳健统计（`max|dv| > 3·median|dv| + 0.35`）；弹簧判别是**模型选择**——简谐模型 `acc=-ω²x` 相对残差 `rr` 须同时过门限且**严格小于**匀加速模型（常二阶差分）相对残差 `rc`，解决短窗内二次位置轨迹"看起来像简谐"导致的 accel→spring 误判。
  - `routed_scene_params`：按类型输出槽位干净的 4 维参数（匀速/碰撞 `[c,0,0,0]`、加速 `[c,a,0,0]`、弹簧 `[0,0,ω,0]`）；碰撞对方速度物理不可见，不编造。
  - `KindSpecificParamErrorModel`：按真实类型拟合逐槽位 bias/std，推理按预测类型 `.bind(labels)` 采样。
  - `fit_kind_conformal_inflation` / `apply_kind_inflation`：独立校准集上**逐类型** split-conformal 膨胀。
- **Added**：`scripts/kind_fan_coverage.py` → `reports/v552_kind_fan.json`。
- **实测（held-out seed=2026，确定性路由，rollout4 recovery）**：匀速 1.000、加速 0.991、**弹簧 1.001（v5.4.9 经典仅 0.03、v5.5.0 学习头 1.09）**、碰撞 0.917。混淆矩阵：匀速 100%、弹簧 100%（对匀速/加速零误报）、加速 85.4%（余 14.6% 为近匀速窗，归 uniform 无害）、碰撞 72.9%（不含跳变窗局部即匀速，归 uniform 无害）。
- **类型条件 conformal（名义 80%，M=48）**：分类型覆盖由 v5.5.1 的 匀速.84/加速.61/弹簧.85/碰撞.95 收紧为 **匀速.816/加速.780/弹簧.813/碰撞.793（pooled .801）**。逐类膨胀因子 匀速 1.00/加速 **8.25**/弹簧 1.50/碰撞 0.84——加速类需很宽带，暴露高斯参数误差不足以覆盖其 rollout 结构误差（诚实记录，留后续模型改进）。
- **诚实边界**：路由器只识别生成器四类一维运动，非通用时序分类器；类型条件 conformal 保证各类型边际（样本×步×维）覆盖；分类误判窗（近匀速）下游已验证无害。
- **Tests**：`tests/test_v552_dynamics_router.py` 11 项（四类合成窗分类与槽位、弹簧无 v0/a 污染、碰撞跳变、输入校验、批量确定性、held 弹簧召回≥.98 且匀速零误报、类型误差模型拟合/绑定/去偏、膨胀几何与校验）；连同 551/550/549/548/547/reasoning 共 54 项回归全绿。

## v5.5.1（参数不确定性：蒙特卡洛轨迹扇形 + 覆盖率校准）

> 性质：新增能力，不改主模型/场景头权重；主 predictor 恒 52191、场景头恒 6788 参数。把 v5.5.0 的点估计升级为带不确定性的 p10/p50/p90 轨迹扇形，并在独立 held-out 上验证覆盖率。

- **Added**：`udos/scene_fan.py`
  - `ParamErrorModel.fit`：在校准集上拟合 `P_hat−P_true` 的逐槽位 bias/std（全局、类型无关、对角高斯，std 有下限）。
  - `monte_carlo_rollout`：对点估计去偏后采样 M 组参数，各跑一次冻结主预测员 rollout，取分位数得 `TrajectoryFan(low/median/high)`。
  - `fit_conformal_inflation` + `calibrated_band`：在**独立校准集**（seed=314）上以中位为中心、按扇形半宽等比求 split-conformal 膨胀因子，使 pooled 经验覆盖达到名义 80%（p10..p90）。
  - `coverage_fraction` / `per_step_coverage`：pooled 与分步覆盖率。
- **Added**：`scripts/scene_fan_coverage.py` → `reports/v551_fan_coverage.json`。
- **实测（held-out seed=2026，M=64，名义 80%）**：原始 MC pooled 覆盖 **0.730**（欠覆盖）；conformal 膨胀因子 **1.264** 后 pooled **0.8134**（达标），分步覆盖 0.804/0.824/0.814/0.804（跨步均匀）。分类型：匀速 0.840、弹簧 0.852、碰撞 0.951（过覆盖）、**加速 0.610（欠覆盖）**。
- **诚实边界**：全局对角高斯不区分运动类型，pooled 达标但单类型欠/过覆盖；该缺口由 v5.5.2 运动识别 + 类型相关误差模型收紧。本扇形只传播"场景参数估计不确定性"，不含模型结构/观测噪声；conformal 保证的边际单位是（样本×步×维）pooled 覆盖。
- **Tests**：`tests/test_v551_scene_fan.py` 7 项（误差模型形状/正性、采样去偏中心与条件 std、扇形顺序与同 seed 复现、零方差退化为点 rollout、非法分位拒绝、覆盖率助手、独立 held-out pooled 覆盖落入 0.77–0.88 且不显著低于原始 MC）；含 550/549/548/547/reasoning 共 43 项回归全绿。

## v5.5.0（双引擎名副其实 · 大版：冻结主模型，端到端训练学习型场景头）

> 性质：**新增能力 + 主链接线**。主预测员 PhysicsPredictor 全程冻结，52191 参数锚点 `predictor_v4.3.9.pt`（eval_mse 0.045556）**一个权重都不改**；只训练一个 6788 参数的独立 `SceneEstimationHead`（独立 state_dict / 独立权重文件，可挂载、摘除、回滚）。这是"GPM 场景记录员"第一次经训练后真正调制训练过的主预测员，并根治 v5.4.9 暴露的弹簧短板。

- **Added**：`udos/scene_head.py`
  - `SceneEstimationHead`：观测窗口 `[B,W,6]` → 4 维隐藏参数的两层 MLP（64 隐层，末层零初始化，起步近似场景盲）。
  - `differentiable_rollout`：与 `PhysicsPredictor.rollout` 算子序列完全一致、但保留 autograd 的自回归滚动；冻结参数作为常数，梯度只回到头。**训练目标即真实推理 rollout MSE**，不是另一条旁路。
  - `train_scene_head` / `save_scene_head` / `load_scene_head`：冻结主模型、AdamW 端到端训练，独立权重存取。
- **Changed**：`UDOSReasoningEngine.attach_scene_head()` 与 `reason()` 条件优先级——显式真值参数（metadata/attributes）> 学习头估计（`scene_params_source="learned_head"`）> 场景盲旧路径。**未挂载头时逐位回退旧行为**（严格向后兼容）。
- **产物**：`checkpoints/scene_head_v5.5.0.pt`（约 30KB，6788 参数）；`scripts/train_scene_head.py` 可复跑；`reports/v550_head_ab.json` 四方 A/B。
- **实测四方 A/B（held-out seed=2026，主模型冻结，训练 seed=42）**：

  | 指标(rollout4 MSE) | blind | explicit | classical(v5.4.8) | **learned(v5.5.0)** | learned 恢复率 |
  |---|---|---|---|---|---|
  | 整体 | 1.3595 | 0.0784 | 0.2418 | **0.0448** | **102.6%** |
  | 匀速 | 0.7143 | 0.0709 | 0.0709 | **0.0149** | 108.7% |
  | 匀加速 | 3.3355 | 0.0599 | 0.0890 | **0.0464** | 100.4% |
  | 弹簧 | 0.7405 | 0.1404 | 0.7223（恢复 3%） | **0.0859** | **109.1%** |
  | 碰撞 | 0.6475 | 0.0423 | 0.0848 | **0.0319** | 101.7% |
- **诚实边界**：学习头是"**下游任务最优的场景条件器**"，不是精确物理参数估计器——弹簧 ω 的名义估计有偏（MAE≈0.57），但轨迹 MSE 最优；恢复率 >100% 表示"为冻结模型定制的有效参数"比"喂名义真值参数"更适配该模型的既有偏差，并非"比真值更懂物理"。可解释的精确反演仍以 v5.4.8 经典估计器为准（互补）。碰撞对方速度 v2 在主体窗口物理不可见，学习头同样无法凭空恢复，其增益来自可观测槽。
- **Tests**：`tests/test_v550_scene_head.py` 8 项（零初始化、形状拒绝、可微 rollout 与推理逐位一致、梯度确实回到参数、短训练损失下降、**已发布产物弹簧恢复率≥0.80 且整体不劣于显式 1.1 倍**、reason 挂载/未挂载接线与回退、挂载/摘除）；含 545–549/reasoning/service 共 61 项回归全绿。

## v5.4.9（双引擎名副其实 5/…：盲/显式/估计三方 A/B）

> 性质：新增可测评估模块 + 复现脚本 + 真实报告，**不改主模型权重/checkpoint、不改 reason 主链**；主 predictor 恒 52191 参数、eval_mse 恒 0.045556。

- **Added**：`udos/scene_estimation_ab.py` 的 `three_way_scene_ab(predictor, dataset)`，在同一 held-out 上比较三种场景条件——`blind`（不喂场景）、`explicit`（喂真值参数 P，上界）、`estimated`（喂 v5.4.8 估计器反演、不可观测槽位置 0），报告单步/H 步 rollout 的整体与分类型 MSE、`est_over_blind` 与增益恢复率 `recovery=(blind−estimated)/(blind−explicit)`，并附估计器逐槽位可观测率。
- **Added**：`scripts/ab_scene_estimation.py` 落 `reports/v549_scene_ab.json`（默认 held-out seed=2026，训练 seed=42，可复跑）。
- **实测结论（seed=2026, n=2560）**：整体 recovery 单步 **91.8%**、4 步 **87.2%**（est/blind MSE 0.149/0.178，即仅凭观测窗口、无 GPM 也消掉约 82–85% 的场景盲误差）；分类型 4 步 recovery：匀速 **100%**、匀加速 **99.1%**、碰撞 **93.0%**（v2 不可观测是剩余缺口）、弹簧 **3.0%**（短窗 ω 召回 0.77 + 线性 v0 槽污染）。弹簧短板明确交给 v5.5.0 学习型估计头与 v5.5.2 类别路由解决，不粉饰。
- **Tests**：`tests/test_v549_scene_ab.py` 7 项（结构完整、显式严格优于盲、匀速 recovery≥0.95、整体 recovery≥0.70、弹簧不造成净伤害、v2 可观测率 0/ω 召回带、确定性）；含 545–548/reasoning 共 46 项回归全绿。

## v5.4.8（双引擎名副其实 4/…：场景隐藏参数估计器）

> 性质：新增独立、确定性、无需训练的经典运动学估计器，**不改主模型权重/checkpoint、不接线 reason 主链**（盲/显式/估计三方 A/B 在 v5.4.9，学习型估计头在 v5.5.0）；主 predictor 恒 52191 参数、eval_mse 恒 0.045556。

- **Added**：`udos/scene_estimator.py` 的 `estimate_scene_params(window, dt)` 与 `SceneEstimate`，仅凭观测窗口 `[B,W,6]` 反演 4 维隐藏参数并给逐槽位物理可观测性掩码：
  - `v0`：速度对时间最小二乘的**窗口局部**截距（匀速=全局 v0；匀加速=全局 v0+a·s·dt），速度本就在 raw 中，恒可观测；
  - `accel_a`：速度最小二乘斜率，仅当速度线性优度 `lin_r2≥0.95`（匀速 a=0/匀加速）标记可观测，弹簧/碰撞非常数加速置掩码 False；
  - `spring_omega`：由简谐关系 `acc=−ω²x`（位置二阶差分）反演，仅当 `速度非线性(lin_r2<0.95) 且 ω²≥0.09 且 简谐拟合相对残差 rr≤0.10` 才标记可观测，否则 NaN；
  - `other_v2`：碰撞对方速度主体窗口物理不可见，**恒 NaN/不可观测，绝不编造**。
- **实测工作点（生成器 W=6/dt=0.5，seed 42/7/2026）**：弹簧 ω 召回约 0.78（短窗弧度不足是真实可辨识性天花板，精度优先），匀速/匀加速/碰撞的 ω 误报率为 0；ω 估计误差 ≤0.04（0.8/1.2/1.5 三点），匀加速斜率与窗口局部截距精确。
- **Added**：`SceneEstimate.values_filled(fill=0)` 供只有 4 维入口、尚无掩码通道的主预测员消费，仅替换不可观测槽位、可观测槽位逐位不变。
- **Tests**：`tests/test_v548_scene_estimator.py` 11 项（四类槽位契约、v2 全数据集恒不可观测、三 seed 精度/召回工作点守卫、批量与单条一致、NaN 填充、短窗/NaN/dt/坏形状拒绝）；先 RED 后 GREEN；含 training/v21/545–547 共 52 项回归全绿。

## v5.4.7（双引擎名副其实 3/5：GPM 记忆桥）

> 性质：新增能力（独立外挂模块，零初始化，向后兼容），**不改主模型权重/checkpoint**；主 predictor 恒 52191 参数、eval_mse 恒 0.045556。本版把 GPM 的场景嵌入（latent 维）接通到训练过的主预测员 CTM（scene_dim=32），消除"GPM 丰富场景记忆只喂未训练演示 CTM"的第二处断点；桥未训练前输出严格为 0，真实增益在 v5.5.0 联合训练后产生。

- **Added**：`udos/gpm_memory_bridge.py` 的 `GPMSceneBridge`（`Linear(latent→scene_dim)`，权重/偏置零初始化），由推理引擎按 predictor 实际 `ctm.cfg.scene_dim` 懒构建，独立 `nn.Module`、不进 `PhysicsPredictor.state_dict`，可独立训练/存权重/回滚。
- **Changed**：`PhysicsPredictor` 的 `forward/predict_next/rollout` 新增可选 `scene_bias`，`_resolve_context` 做加性融合 `ctx = scene_encoder(params) + bias`；**零偏置短路**保证零桥在场景盲时 ctx 仍为 None（避免零张量经 scene_proj 偏置失真），有参数时浮点 +0 恒等；`scene_bias` 与 `scene_params` 同样拒绝 NaN/inf。
- **Changed**：`reason()` 计算桥偏置并传入主预测员；`ReasoningResult`/`summary()`/HTTP `/reason` 透出 `gpm_bridge_active`、`gpm_bridge_norm`（零桥 norm=0，训练后 >0 即 GPM 记忆真正调制主预测）。`attach_predictor` 更换模型时重置桥按新维度懒重建；构造开关 `use_gpm_bridge`（默认开，零初始化无副作用）。
- **反死路证据**：与 LoRA"注入演示基座却从不 forward"不同，测试证明桥权重置非零后主预测输出真实改变（`test_nonzero_bridge_changes_prediction`）；零桥在带参/场景盲/关桥三路径与 v5.4.6 在 6 位 JSON 舍入容差内逐位一致。
- **Tests**：`tests/test_v547_gpm_memory_bridge.py` 7 项契约（零初始化输出 0、带参/盲/关桥逐位兼容、非零桥真实改变预测、独立于 52191 锚点且可独立存取、`_resolve_context` 加性/短路规则、维度懒建）；先 RED 后 GREEN；training/v21_multistep/service 等回归 58 项全绿。

## v5.4.6（双引擎名副其实 2/5：主预测链接线）

> 性质：新增能力（行为变更，向后兼容），**不改模型、算法、checkpoint**；主 predictor 恒 52191 参数、eval_mse 恒 0.045556。本版把 v5.4.5 的场景参数通道真正接进训练过场景门的主预测员，消除"GPM 记录场景却不影响物理轨迹"的断点。

- **Changed**：`UDOSReasoningEngine.reason()` 在挂载 `PhysicsPredictor` 时，经 `extract_scene_params()` 从 PCE 场景提取 4 维隐藏参数并传入 `predictor.rollout(raw, H, scene_params=...)`；无场景参数时维持旧版场景盲 `rollout(raw, H)`，逐位兼容。GPM 内部演示 CTM 支路保持原样。
- **Added**：`ReasoningResult` 新增 `predictor_conditioned`（主预测员是否真正消费场景参数，区别于仅指内部演示 CTM 的 `scene_conditioned`）、`scene_params`、`scene_params_source`；`summary()` 与 HTTP `POST /reason` 顶层透出。
- **实测增益（真实 HTTP 端到端，非单元构造）**：启动加载 `predictor_v4.3.9.pt` 的服务，对同一 spring 窗口（ω=1.2，4 步 rollout）POST `/reason`，场景盲位置 MSE 3.673 → 带正确隐藏角频率 0.535，约 **6.87 倍**提升；`predictor_conditioned` 由 false 变 true、`scene_params_source=metadata`。独立测试集（seed=7）平均增益为单步约 13.7 倍、4 步约 16.8 倍（见 v5.4.5 条目背景）。
- **Tests**：`tests/test_v546_predictor_scene_link.py` 5 项契约（主链 future 与带场景直接 rollout 数值一致、带场景显著更准、无场景逐位兼容、结果字段透出、attributes 与 metadata 通道等价）；先取得 5 项 RED 再实现到 GREEN；`test_service`/`test_conditioning`/`test_v21_multistep` 等回归全绿。

## v5.4.5（双引擎名副其实 1/5：场景参数通道）

> 性质：新增能力（数据通道），**不改模型、算法、checkpoint**；主 predictor 恒 52191 参数、eval_mse 恒 0.045556。背景：读码定位到双引擎断点——`reason()` 调训练过场景门的 `PhysicsPredictor.rollout()` 时未传任何场景参数，独立测试集实测同窗口单步场景盲 MSE 0.551 vs 带场景 0.040（约 13.7 倍）、4 步推演 1.406 vs 0.084（约 16.8 倍），GPM 场景记忆从未进入主预测员。本版先建规范承载通道，主链接线在 v5.4.6。

- **Added**：`udos/scene_bridge.py` 的 `extract_scene_params()`/`SceneParams`，从 `PhysicsScene` 的 `metadata["scene_params"]`（dict 按槽名，或长度 4 的 list 按固定槽位顺序）或 `PhysicalToken.attributes` 提取 4 维隐藏物理参数 `(v0, accel_a, spring_omega, other_v2)`；优先级 metadata > attributes > 默认 0.0；无任何场景信息返回 `None`（场景盲旧路径逐位兼容）；NaN/inf、错误长度、不可解析值显式 `ValueError`（与 `predict_next` 有限值纪律一致）。
- **Tests**：`tests/test_v545_scene_bridge.py` 13 项契约（dict/list 槽位顺序、attributes 回退、metadata 优先级、缺失返回 None、NaN/inf/错长拒绝、未知键忽略、确定性、PCE 序列化往返、float32/全有限）。
- **Tooling**：`scripts/bump_version.py` 增强为支持单参数（自动读当前版本），补齐 web 看板落点，收尾校验 `__version__` 与 tests 无残留旧断言；保护 finesim/CHANGELOG/benchmarks 等历史标注不被替换。

## v5.4.4（开源修订：打包硬伤 + 版本/看板一致性 + 能力边界精确化）

> 性质：打包/文档/默认配置补丁。**不改模型、算法、checkpoint**；主 predictor 恒 52191 参数、eval_mse 恒 0.045556，全量 1570 passed + 2 skipped（共收集 1572）、0 failed。起因是外部对 v5.4.3 的只读代码评审，逐条经源码复核属实后修复（评审中"README 首页仍为 4.5.3"一条经核不成立）。

- **build fix (P0)**：`Dockerfile` 曾 `COPY third_party/ctm`，而开源仓库并无 `third_party/` 目录，导致 `docker build` 必然失败。已删除该 COPY；`.dockerignore` 改为整体忽略 `third_party/`；镜像注释改为"不内置上游源码，CTM/D2L 适配器在显式启用时经 huggingface_hub 运行时拉取"。
- **build fix**：容器与 `docker-compose.yml` 默认 checkpoint 由过时的 `predictor_v3.3.3.pt` 更新为最新正式件 `predictor_v4.3.9.pt`；镜像 OCI LABEL 与 compose image tag 同步到 5.4.4。
- **consistency**：版本号单一来源统一到 5.4.4（`udos/__init__.py` 的 `__version__`/docstring、`pyproject.toml`、Dockerfile LABEL、compose、Web 控制台 title/角标/总览）；修正 `udos/__init__.py` docstring 中"上游开源代码库已随工程克隆到 third_party/"的过时表述。
- **web console**：`scripts/build_console_data.py` 的总览版本改取 `udos.__version__`、测试数改为实时 `pytest --collect-only`（兼容 pytest 9 的每文件计数输出），并补齐脚本对内联 HTML `const DATA` 与 title/角标的回写，消除 `dashboard_data.json` 与 HTML 脱节；本次仅外科更新总览两字段，完整保留 registry/perf 的 v4.5.6 历史快照；看板明确标注"注册表/性能为静态快照，仅 /health、/resources、/reason/latent 实时"；QA 阈值更新为只增≥1572。
- **docs**：README"能力边界"新增"双引擎耦合范围"（GPM 生成的 LoRA 仅注入演示用 `TinyBaseModel` 且 `reason()` 不调用其前向；GPM `scene_embedding` 只进内部轻量 `CTMPhysicsEngine` 支路；对外物理轨迹来自独立挂载的 `PhysicsPredictor.rollout`，不消费场景嵌入），并明确自然语言 `query` 仅回显、多模态为低维代理头、"自进化"是冻结主模型的配置搜索、资源注册表列出≠权重已运行。
- 验证与未执行项见 `docs/VERIFICATION_v5.4.4.md`；发布说明见 `RELEASE_NOTES_v5.4.4.md`。

## v5.4.3（首个开源版本）
- 双引擎（CTM 连续思维机 + GPM 场景内化）认知架构内核以 Apache-2.0 开源；新增精细生物物理数值核 `udos/finesim/`（被动电缆 / Hodgkin–Huxley / NMDA 时序抑制可证伪对照 / Payeur 四类树突处理 / 突触位置鲁棒性 / Hines 串行 vs DHS 层级并行对拍）。详见 [RELEASE_NOTES_v5.4.3.md](RELEASE_NOTES_v5.4.3.md) 与 `docs/VERIFICATION_v5.4.3.md`。

## v5.0.1（安全补丁）
- **security fix (P0)**：移除 `udos/debug.py` 硬编码调试口令；`DebugPanel` 改为 `enabled` 显式参数 + `UDOS_DEBUG` 环境变量（默认关）。清除 README/demos/docs 全部复述。新增安全守卫测试钉死无硬编码口令/秘密字面量。


## [3.9.0] — 宇树 UnifoLM-WLA 机制类比线首训（统一 ER 头 + 动作三分组）

> 性质：analogy, not reproduction。全部新能力为 CPU 合成数据**外挂零梯度、opt-in**，不进主 state_dict；主 predictor 恒 52191、eval_mse 恒 0.045556。内化冻结 v3.8.6 主权重（不重训）。

### Added
- `udos/embodied.py`：`EmbodiedReasoningHead`（编排 spatial/affordance/unified-head，不重造）+ `ActionTriGroup`（EEF pose/EEF joints/lower-body 三分组骨架）。
- `udos/wla.py`：`ChangeMask`/`ChangeMaskVQ`/`RVQActionTokenizer`/`ActionStateTaskAlign`/`FlowMatchingDecoder`。
- 正式件 `checkpoints/predictor_v3.9.0.pt`（第 26 代）；`benchmarks/results/training_v3.9.0.json`。

## [3.9.0.dev1] — 相邻状态差分→稀疏 change-mask（对标光流动态区域）
- `ChangeMask`：末两帧差分→|Δ|>阈值二值掩码，未变化维恒等复制（"只盯动作会改变哪里"）；报 changed_ratio。

## [3.9.0.dev2] — 变化区小型 VQ 码本
- `ChangeMaskVQ`：纯 torch k-means 码本，报利用率/recon_mse/坍塌（max_code_share>0.9 判坍塌）。

## [3.9.0.dev3] — 稀疏 change-mask vs 3.6 PWM 稠密 rollout A/B
- `scripts/wla_sparse_vs_dense_ab.py` → `wla_sparse_vs_dense_ab.json`：稀疏逐步 MSE 低于稠密（稠密末步潜在漂移累积），成本约 1/10，采纳。

## [3.9.0.dev4] — 每分组 RVQ 动作分词
- `RVQActionTokenizer`：L 级残差量化，每级报利用率/最大码占比/整体 recon_mse。

## [3.9.0.dev5] — 动作 token-状态-任务对齐
- `ActionStateTaskAlign`：确定性投影到共享空间，alignment_loss + same/cross-task 一致性测试。

## [3.9.0.dev6] — 冻结骨干外挂少步 flow-matching 解码器（对标 MMDiT）
- `FlowMatchingDecoder`：n_steps Euler 去噪；`scripts/wla_flow_vs_regress_ab.py`。**被直接回归反证（0.264 vs 0.169）→ opt-in 留候选账本**（`docs/WLA_CANDIDATE_LEDGER.md`）。

## [3.9.1] — 统一动作空间跨本体/跨末端迁移 A/B
- `scripts/wla_cross_embodiment_ab.py`：复用 retargeting，零样本迁移严格落目标限位（within_limits=true，违例率诚实报告）。

## [3.9.2] — 多任务统一头评测矩阵 + HTTP 端点
- `scripts/wla_multitask_matrix.py`：任务族×ER 代理矩阵；新增 `POST /wla/er`（正常 200/非法 400/未知 404）；`tests/test_v392_service.py`。

## [3.9.3] — 全 checkpoint 兼容 + 性能基准 + 加固
- `scripts/feature_latency_v39.py` → `feature_latency_v3.9.2.json`；`tests/test_v393_hardening.py`：全部 checkpoint 可加载、主参恒 52191、输出有限、外挂边界守卫。

## [3.9.4] — 边界精修 + 文档对齐
- ROADMAP/ARCHITECTURE/DEPLOYMENT/README 增补 v3.9 线；候选账本与研究报告定稿；版本断言同步。

## [3.9.9] — 终点正式训练 + 全量回归 + 线末收口
- 终件 `checkpoints/predictor_v3.9.9.pt`（第 27 代）；全量 pytest 全绿报总数/覆盖率；27 代 checkpoint 全兼容。

## [4.1.0] — 自规划自监督线起点（任务/课程自动生成器 + 可解性自验证器 + 正式训练）

> 性质：analogy, not reproduction。对标智谱"环境自造 + 自产验证器"。全部新能力为 CPU 合成数据**外挂零梯度、opt-in**，不进主 state_dict；主 predictor 恒 52191、eval_mse 恒 0.045556。内化冻结 v3.9.9 主权重（不重训）。

### Added
- `udos/curriculum.py`：`LessonSpec`（不可变课程规格）+ `CurriculumGenerator`（按 stage 自动生成参数化物理课程，复用 `build_parametric_dataset`）+ `SolvabilityVerifier`（用冻结主预测器自身前向做可解性自验证：有限性 + 物理界内 + rollout 不发散）。
- 正式件 `checkpoints/predictor_v4.1.0.pt`（第 28 代）；`benchmarks/results/training_v4.1.0.json`；`make ckpt410`。
- `tests/test_v410_curriculum.py`（13）。

### Evidence
- n_params=52191；eval_mse=0.045556（锚点逐位保持）；自生成课程自验证 solvable_ratio=1.0（24/24）；28 代 backcompat 清单；外挂只读主权重（md5 前后一致）。

## [4.1.0.dev1] — 课程难度递进（环境复杂度自动扩展）
- `CurriculumGenerator.auto_progression`：逐 stage 记录自验证可解率与难度代理（horizon/spread 随 stage 自动扩张）；`auto_expand_horizon`：从 start 起按 step 递增 horizon，用自验证器探测难度前沿直到可解率跌破 floor 或达 max_h。
- 测试：stage 单调扩张、难度前沿停止条件、非法参数 ValueError。

## [4.1.0.dev2] — 自生成课程 vs 固定课程 A/B
- `scripts/curriculum_ab_v410.py` → `benchmarks/results/curriculum_self_vs_fixed_v4.1.0.json`；`make curriculum-ab`。
- 实测：同探针预算下两者可解率均 1.0（delta=0.0），但自生成课程自适应扩张探明模型难度前沿 horizon=8（固定课程只探 horizon=2）；结论：可解覆盖持平、自生成额外暴露难度前沿 → opt-in 采纳。

## [4.1.0.dev3] — 自监督信号：PWM rollout 一致性伪标签
- `udos/selfsup.py`：`PWMConsistencyPseudoLabeler`——用外挂 `LatentWorldModel.imagine_rollout` 想象未来作伪标签，与主预测器真实 rollout 逐步 MSE 映射为逐步一致性置信 w=exp(-mse/median)；零梯度、主权重只读。
- `tests/test_v410_selfsup.py`：形状/置信∈[0,1]/懒拟合/no-grad/非法输入。

## [4.1.0.dev4] — 自监督信号：物理守恒 + 多视角一致伪标签
- `PhysicsMultiviewGate`：复用 `ConservationChecker`（动量/能量）+ `spatial.multiview_consistency_error`（两正交视角重投影），两道违反量映射为 [0,1] 门控分并与 PWM 置信相乘；匀速运动守恒门≈1。
- 测试：匀速严格守恒、门控乘积有界、非法维数/步数 ValueError、零梯度。

## [4.1.0.dev5] — 伪标签增益 A/B（有/无自监督损失门）
- `scripts/pseudolabel_gain_ab_v410.py` → `benchmarks/results/pseudolabel_gain_v4.1.0.json`；`make pseudolabel-ab`。
- 实测：PWM 一致性伪标签有区分度（均值 0.48）；但叠加守恒+多视角硬门对加速/振动轨迹过度降权（均值权重 0.03，144/144 被压，有效一致性反降 -0.466）。**结论：PWM 一致性单独采纳为自监督信号；守恒+多视角硬门被反证 → opt-in（默认关）留候选账本**（`docs/WLA_CANDIDATE_LEDGER.md`）。

## [4.1.0.dev6] — 自规划目标分解（goal → 子目标链）
- `udos/selfplan.py`：`GoalDecomposer`——给定起始窗口与目标态，线性插值候选中间子目标，复用 `policy.MPCActionSelector` 按 rollout 奖励（-MSE to subgoal）+ 风险/安全打分优选每步；输出子目标链与到 goal 的残差序列（单调下降）。纯前向、零梯度、主权重只读。
- `tests/test_v410_selfplan.py`（6）：链长/形状、残差递减、非法输入 ValueError、确定性、零梯度。

## [4.1.1] — "何时停止/自我纠正"判据（置信门控 + 收敛停止 + OOD 主动验证）
- `udos/selfplan.py`：`StopCorrectController`——对自规划子目标链做三判据：①置信门控（confidence_fn < 阈值 → self_correct 重规划）；②收敛停止（残差平台 → stop）；③OOD 主动验证（`ood.DistributionDriftDetector.is_ood` 命中 → verify 交回复核）。决策优先级：低置信 > OOD > 收敛 > 采纳。复用 calibration/ood/guard 语义，纯前向、零梯度。
- 版本全来源同步 4.1.0 → 4.1.1（`udos/__init__.py`、`pyproject.toml`、`Makefile`、`Dockerfile`、`docker-compose.yml`、测试断言）。
- `tests/test_v411_stopcorrect.py`（8）。

## [4.1.2] — HTTP 端点集成 + 加固 + 性能基准
- 新增 5 个端点（复用既有 `_parse_window`/`_coerce_*`/`ServiceNotReady` 错误语义）：`POST /curriculum/generate`、`POST /curriculum/solvable`、`POST /selfsup/pseudo-label`、`POST /selfplan/decompose`、`POST /selfplan/decide`；懒单例（课程/自验证器/伪标签器/分解器）。
- 错误语义延续：未训练/未挂载 → 409，非法/缺字段/null → 400，未知路由 → 404，进程不崩；`benchmarks/perf_baseline_v412.py` → `benchmarks/results/perf_baseline_v412.json`（`make perf412`）。
- `tests/test_v412_service.py`（11）：5 端点逐路径正常 200 + 非法 400 + 未知 404 + 未训练 409。
- 版本全来源同步 4.1.1 → 4.1.2。

## [4.1.9] — 最终正式训练 + 全量回归 + 线末收口
- 终件 `checkpoints/predictor_v4.1.9.pt`（第 29 代）；`benchmarks/results/training_v4.1.9.json`；`make ckpt419`。
- 内化冻结 v4.1.0 主权重（不重训）：n_params=52191、eval_mse=0.045556 锚点逐位保持；**全 29 代 checkpoint 逐件可加载且主参一致**。
- 全线零梯度终检：课程可解率 1.0、PWM 伪标签一致性均值 0.5、目标链子目标数 3、停止判据决策 stop。
- 2 次正式训练（4.1.0=第28代、4.1.9=第29代）完成；10 迭代全落地。

## [4.2.0] — 完全自训练线起点（世界模型自生成 (s,a,s') 三元组骨架 + 正式训练）

> 性质：analogy, not reproduction。对标 RSI/自蒸馏思想，CPU 合成数据**外挂零梯度、opt-in**。全部新能力不进主 state_dict；主 predictor 恒 52191、eval_mse 恒 0.045556。内化冻结 v4.1.0 主权重（不重训）。

### Added
- `udos/self_train.py`：`SelfGeneratedTriplets`（不可变三元组容器）+ `TransitionTripletGenerator`（世界模型自生成 (s,a,s') 骨架）。
  - 三元组：`s_t`=真实窗口末帧物理状态（教师侧锚定）、`a_t`=自博弈候选动作（确定性网格骨架）、`s_{t+1}`=世界模型自治想象下一状态（外挂 encode→transit→decode，零梯度）。
  - `triplet_quality`：action_spread / next_state_std / transition_norm / momentum_residual，全确定性可复算。
- 正式件 `checkpoints/predictor_v4.2.0.pt`（第 30 代）；`benchmarks/results/training_v4.2.0.json`；`make ckpt420`。
- `tests/test_v420_self_train.py`（11）：形状/确定性/零梯度(md5 不变)/非法 ValueError/非有限拒绝。

### Evidence
- n_params=52191；eval_mse=0.045556（锚点逐位保持）；30 代 backcompat 全可加载。
- 自生成三元组实测：n=64、action_spread=0.826、next_state_std=2.014、transition_norm=1.437、**momentum_residual=0.811（偏大，诚实暴露：骨架阶段 action 尚未真正闭环进转移，dev1 自博弈后改进）**。
- **防退化纪律**：v4.2.0 仅交付骨架与可复算度量，不宣称 RSI 必然提升；momentum_residual 偏大照实留账（action 独立通道，未参与世界模型转移）。

## [4.2.0.dev1] — 自博弈探索 + 规则校验（守恒/约束）
- `udos/self_train.py`：`SelfPlayExplorer`——策略与世界模型互搏。对每个窗口枚举动作网格，世界模型想象 next_state 并叠加冲量定理 Δv_x=a·dt/m，再复用 `ConservationChecker` 校验动量/能量守恒违反，逐样本选**守恒违反最小**的动作（argmax 物理自洽性，确定性 tie-break）。纯前向、零梯度、主权重只读。
- 与 v4.2.0 骨架区别：action 经冲量定理显式注入下一速度（不再独立标签），守恒校验由 ConservationChecker 严格量化（非骨架近似残差）。
- `tests/test_v420_dev1_selfplay.py`（7）：动作分布/确定性/零梯度(md5)/规则通过率/非法 ValueError。

## [4.2.0.dev2] — 执行验证 + 模型评审（ensemble/calibration 筛选）
- `udos/self_train.py`：
  - `ExecutionValidator`：纯前向检查自生成 next_state 可执行性（有限性 + 位置/速度界内 + 转移范数不发散），报 pass/fail/各维违例数。
  - `ModelReviewer`：ensemble 类比（n_perturb 次小扰动想象方差作不确定性）+ calibration 类比（方差中位数阈值筛除高不确定样本）；保留评审通过子集。纯前向、零梯度、主权重只读。
- `tests/test_v420_dev2_review.py`（8）：clean 通过/超界拒绝/零梯度/确定性/非法 ValueError。
- **防退化纪律**：评审筛选只报保留率与方差分位，不宣称筛选后必然提升下游。

## [4.2.0.dev3] — 人工抽检 hook + 数据质量账本
- `udos/self_train.py`：
  - `HumanAuditHook`：确定性抽样式交回抽检（sample_rate × 种子），record_decision 记录 approve/reject/pending；**不自动通过任何样本**（抽检是治理门）。
  - `DataQualityLedger`：逐批次记录三元组质量/执行验证/模型评审/抽检决策画像，`to_json` 落 `benchmarks/results/`；日志走 stderr 不污染 stdout。
- `tests/test_v420_dev3_audit.py`（9）：抽样确定性/决策计数/空输入拒绝/账本落 JSON/非法 batch_id。

## [4.2.0.dev4] — teacher→student 自训练递归（仅一代可验证闭环，最高优先级防退化）
- `udos/self_train.py`：`TeacherStudentLoop`——teacher=冻结主预测器（只读），student=外挂 `_StudentHead`（6→32→6 残差小 MLP，**不入主 state_dict**，主参恒 52191），在自生成 (s_t→s_{t+1}) 上拟合。
- **判定在真实 holdout**（非自生成数据）：`improved`/`degraded`/`collapsed`（ratio>10x）。未证优时照实报 degraded 并留候选账本，**不宣称 RSI 提升**。
- `tests/test_v420_dev4_teacherstudent.py`（7）：train loss 下降/teacher md5 不变/verdict 诚实/未 fit 报错/空 triplet 拒绝。
- **实测证据（防退化核心）**：student 在自生成数据上 train loss 0.403→0.055（自欺式下降），但在真实 holdout 上 **student_mse=0.249 vs teacher_mse=0.045556，ratio=5.47x → verdict=degraded**。这是历史"自蒸馏学生 mse=2.63 被 REJECT"的同型重演：**student 过拟合世界模型想象，未真改进**。结论：teacher→student 一代闭环**未证优**，opt-in 留候选账本（`docs/WLA_CANDIDATE_LEDGER.md` 范式）。

## [4.2.0.dev5] — 数据回流三档配比代理（预/中/后训练）
- `udos/self_train.py`：`DataRefluxMixer`——纯配比代理（不真训练主预测器）。pre=100:0、mid=70:30、post=30:70（原始:自生成）。后训练档若 student_verdict∈{degraded,collapsed} 自动标 caution（高比例回流有崩塌风险）。
- `tests/test_v420_dev5_reflux.py`（8）：三档配比/caution 触发/自生成用量夹取/非法 stage/非法配比。

## [4.2.0.dev6] — 退化检测与回滚（teacher→student 真单调改进 vs 漂移/崩塌量化）
- `udos/self_train.py`：
  - `DegradationDetector`：输入代际 student 在真实 holdout 上的 mse 序列，判 `improving`（严格单调降/斜率负）/`drifting`（波动上升）/`collapsed`（某代 > 前代×collapse_ratio）。
  - `RollbackManager`：代际 student 快照注册，`should_rollback` 在当代 mse 超历史最佳×patience_ratio 时建议回滚，`best` 返回历史最佳代。纯治理，不改主权重。
- `tests/test_v420_dev6_degrade.py`（9）：improving/drifting/collapsed 三判定/快照最佳/回滚触发/重复代拒绝。

## [4.2.1] — 收益递减判据 + 自训练 A/B（自产数据 vs 原始数据）
- `udos/self_train.py`：
  - `DiminishingReturnsCriterion`：连续 patience 代真实 holdout mse 改善 < min_delta → diminishing_returns=True，建议 stop_self_training（防 RSI 无界递归）。
  - `SelfTrainAB`：诚实 A/B——A 臂 student 在自生成三元组训练、B 臂在原始数据训练，同架构/种子，在同一真实 holdout 比 mse。主预测器全程只读，主参恒 52191。
- `scripts/self_train_ab_v421.py` → `benchmarks/results/self_train_ab_v4.2.1.json`；`make self-train-ab`。
- `tests/test_v421_ab.py`（6）：递减触发/持续下降/A/B 两臂/teacher md5 不变/诚实 verdict。
- **实测（诚实 A/B）**：A 臂（自产数据）holdout mse=0.257 vs B 臂（原始数据）mse=0.0145，delta=+0.242，**self_generated_wins=False**。结论：**自产数据训练 student 远不如原始数据（差 ~17x）**，teacher→student 一代闭环在本线未证优 → opt-in 留候选账本，不宣称 RSI 提升。

## [4.2.2] — HTTP 端点集成 + 加固 + 性能基准
- 新增 2 个端点（复用 `_require_predictor`/`_coerce_int`/`_parse_window` 错误语义）：
  - `POST /selftrain/triplets`：世界模型自生成 (s,a,s') 三元组 + quality 度量。
  - `POST /selftrain/self-play`：自博弈探索 + 守恒规则校验报告。
- 懒单例（`_ensure_selftrain`：拟合 wm + 构造 generator/explorer）；错误语义延续：未训练 409、非法 n 400、未知路由 404、不崩进程。
- `tests/test_v422_service.py`（7）：两端点正常 200 + 非法 400 + 未知 404，版本断言同步 4.2.1。

## [4.2.9] — 最终正式训练 + 全量回归 + 线末收口
- 终件 `checkpoints/predictor_v4.2.9.pt`（第 31 代）；`benchmarks/results/training_v4.2.9.json`；`make ckpt429`。
- 内化冻结 v4.1.0 主权重（不重训）：n_params=52191、eval_mse=0.045556 锚点逐位保持；**全 31 代 checkpoint 逐件可加载且主参一致**。
- 全线零梯度终检：自生成三元组/自博弈规则校验/执行验证/模型评审/抽检账本/teacher→student/回流配比/退化检测/回滚/收益递减/A/B 全链路跑通。
- 2 次正式训练（4.2.0=第30代、4.2.9=第31代）完成；10 迭代全落地。
- **线末诚实结论**：teacher→student verdict=collapsed（student_mse 0.465 vs teacher 0.0456，ratio 10.21x）；自产数据 A/B 未优于原始数据（0.257 vs 0.0154）；回流 caution=True、收益递减=True。**全线机制为 opt-in 外挂，未证 RSI 提升，留候选账本**。

## [4.3.0] — 完全自进化线起点：系统配置自优化搜索器 + 正式训练
> 性质：analogy, not reproduction。收口智谱"完全自训练"三维之**基础设施自我优化**（数据自产=4.2 / 环境自造=4.1 / 基础设施自优化=4.3）。全部 CPU 合成、外挂零梯度、默认 opt-in；主 predictor 恒 52191、eval_mse 恒 0.045556。内化冻结 v4.1.0 主权重。
- `udos/self_evolution.py`：`ConfigSpec`（batch 分片/缓存/集成权重/控制频率/码本规模旋钮）+ `ConfigEvaluator`（固定基准负载上量"输出保真 max_abs_diff"硬门 + 确定性成本代理）+ `ConfigSearcher`（网格枚举，仅在保真配置里取成本最优，默认永远在候选内）。
- **核心硬门**：候选配置必须在固定负载上使预测输出相对默认配置 ≤ atol（不允许靠降质换速度）；纯效率旋钮（shard/cache）输出逐位不变。
- 正式件 `checkpoints/predictor_v4.3.0.pt`（第 32 代）；`benchmarks/results/training_v4.3.0.json`；`make ckpt430`。
- 实测：36 候选全过保真门，default_cost=8.0 → best_cost=3.5（缓存开 + cortex_hz=1.0 + codebook=4），**降本 56.25%，输出保真**；n_params=52191、eval_mse=0.045556 锚点逐位保持；32 代 backcompat 全可加载。

## [4.3.0.dev1] — 自动 搜索→验证→选用 闭环（网格/随机）
- `SearchVerifySelectLoop`：显式三段闭环（search 网格/种子随机采样 → verify 保真硬门 → select 按成本择优落 archive）；维护每轮 history，确定性可复算。
- `tests/test_v430_self_evolution.py`：闭环确定性、非法 mode/rounds 守卫。

## [4.3.0.dev2] — 配置自优化收益 A/B（搜索后 vs 默认，落 JSON）
- `ConfigAB`：与 4.2 线 `SelfTrainAB` 同纪律的诚实 A/B；A 臂搜索后配置、B 臂默认配置，同 `ConfigEvaluator` 同负载；仅当 A 臂**保真且成本严格更低**才 `A_wins=True`，否则照实 `A_not_better` 留候选账本。
- `scripts/self_evolution_bench_v43.py` → `benchmarks/results/self_evolution_v43.json`（`make self-evolution-bench`）。

## [4.3.0.dev3] — 自进化 orchestrator：4.1 课程 → 4.2 数据 → 4.3 配置 串联
- `SelfEvolutionOrchestrator.run_generation`：一代 = 环境自造（CurriculumGenerator 出课程）→ 数据自产（TransitionTripletGenerator 自生成三元组+质量）→ 基础设施自优化（ConfigSearcher），三维各有产出，主权重只读。

## [4.3.0.dev4] — 多代自进化运行：性能/效率曲线（诚实记录真改进 vs 退化）
- `MultiGenerationRunner`：跑 G 代照实记录每代 best_cost 曲线；末尾判 `efficiency_improving` / `drifting` / `collapsed`，**不宣称飞轮必然上升**。实测 3 代成本稳定 3.5（确定性网格 best 逐代一致），honest_note 明确标注。

## [4.3.0.dev5] — 全局 自我停止 / 自我纠正 判据
- `GlobalStopCorrectCriterion`：组合收益递减（连续多代改进不足→`stop_self_evolution`）与保真击穿/成本突增（→`rollback_to_last_preserved` / `rollback_and_research`）。实测：成本曲线无进一步下降 → 推荐 `stop_self_evolution`。

## [4.3.0.dev6] — 长程任务闭环类比（拆解→模块当工具→合成环境交互→错误恢复→验证）
- `LongHorizonLoop`：horizon 拆成 n_sub 子目标区间，逐子步调用主预测器 rollout；NaN/越界时用上一帧兜底恢复（不崩进程）；末了验证轨迹全有限+步数守恒。对应公告"任务拆解/工具调用/环境交互/错误恢复/结果验证"。

## [4.3.1] — HTTP 端点集成 + 加固 + 性能基准
- 新增 3 个 POST 端点：`/self-evolution/search`、`/self-evolution/ab`、`/self-evolution/long-horizon`；错误语义延续（未训练 409 / 非法 400 / 未知路由 404 / 不崩进程）；日志仍 stderr，不污染 HTTP 体与 /metrics。
- `tests/test_v431_self_evolution_service.py`：逐路径 200/400/404 + `/health` 200 + `/metrics` 纯文本。

## [4.3.2] — 文档对齐 + 边界精修
- 调研 `docs/SELF_TRAINING_EVOLUTION_RESEARCH.md`（智谱完全自训练三维 RSI 定义 / HKEX 2026-09-13 公告事实分层：60%/15%/25% 用途、714 港元配售 + RMB201.4 亿零息可转债；"1GW 国产算力""收购中科加禾"在本可核附件查无实据，标 [UNVERIFIED]）。
- 边界精修：ConfigSpec 全旋钮入参守卫、Evaluator 工作负载形状/repeats 校验、LongHorizonLoop n_sub∈[1,horizon] 校验；全量回归。

## [4.3.9] — 最终正式训练 + 全量回归 + 全局收口打包（v4.3 线终点 ✅）
- 终件 `checkpoints/predictor_v4.3.9.pt`（第 33 代）；`benchmarks/results/training_v4.3.9.json`；`docs/VERIFICATION_v4.3.9.md`。
- 2 次正式训练（4.3.0=第32代、4.3.9=第33代）；**全 33 代 checkpoint 逐件可加载、主参恒 52191、eval_mse=0.045556**；外挂模块不入主 state_dict。
- 全局终点：打包 `udos-engine-v4.3.9.zip`，全新解压 /tmp 独立复跑通过。

## [4.4.1] — 多智能体协作&协同&协调线终点（四拓扑 + 决策树 + Bundle + 治理三件套）✅

> 性质：analogy, not reproduction。全部新能力为 CPU 合成数据**外挂零梯度、opt-in**（swarm 默认关），不进主 state_dict；主 predictor 恒 52191、eval_mse 恒 0.045556。**本线零正式训练**（协作框架为纯前向调度算法，无可学组件），checkpoint 代数维持 33 代不变、旧锚点逐位未动。调研见 `docs/MULTI_AGENT_COLLAB_RESEARCH.md`。

### Added
- `udos/transfer_bundle.py`：`TransferBundle` 五要素（Goal/Context/Done/Todo/Trace）+ 完整性校验，缺 Goal/Trace 关键要素拒绝交接。
- `udos/collab_governance.py`：治理三件套 `OwnerLedger`（唯一负责人/归属转移）、`TraceChain`（事件链可重建责任链、检测断裂）、`StopGuard`（最大跳数/循环检测）、`ClaimLock`（认领去重）。
- `udos/collab_topology.py`：四拓扑统一接口（star/chain/tool/mesh）+ `TopologySelector` 图1 决策树四问与"控制需求不足拒绝自治"分支，默认 star。
- `udos/collab_agents.py`：`CapabilityRegistry` 把 9 个 UDOS 能力（CTM/GPM/SFM/PWM/分层神经控制/curriculum/self_train/self_evolution/wla）注册为带 Tool Schema（name/input/output/confidence/error_type）的 Agent-as-Tool。
- `udos/collab_orchestrator.py`：星型 Orchestrator（拆分→分配→结果收集器去重/校验→统一收口）。
- `udos/collab_handoff.py`：链式 Triage→Specialist→Return + Bundle 交接 + 回退补救。
- `udos/collab_swarm.py`：网状 Swarm（**默认关 opt-in**）：能力发现/局部协商/再委派/冲突-未对齐检测 + 治理开销统计。
- HTTP：`POST /collab/select`、`/collab/run`、`/collab/handoff`、`GET /collab/trace/{id}`；错误语义 400/404/409/500 不崩进程、路径穿越 400。
- `scripts/collab_ab_v44.py` → `benchmarks/results/collab_ab_v44.json` + Makefile `collab-ab`；`docs/MULTI_AGENT_COLLAB_RESEARCH.md`。

### Changed
- 版本全来源同步 4.3.9 → 4.4.1（`udos/__init__.py`、`pyproject.toml`、`Makefile`、`Dockerfile`、`docker-compose.yml`、测试断言）。
- A/B 证据：star/chain/mesh 各 20 合成任务；star trace 完整率 1.0、mesh 消息量/治理开销最高、收敛跳数 3>2；三论断（star 最可控 / swarm 治理成本高 / 拓扑比数量重要）均 SUPPORTED，账本见 JSON。

## [4.5.3] — 隐式思考 / Latent Reasoning 线终点（CTM 潜空间分支探索 + 四档 effort + 自适应路由 + 多专家协作）✅

> 性质：analogy, not reproduction。全部新能力为 CPU 合成数据**外挂零梯度、opt-in**（none 档默认逐位等价），不进主 state_dict；主 predictor 恒 52191、eval_mse 恒 0.045556。**本线零正式训练**（隐式探索为对编码 token 的确定性扰动分支，无可学组件），checkpoint 代数维持 33 代不变、旧锚点逐位未动。调研见 `docs/LATENT_REASONING_RESEARCH.md`。

### Added
- `udos/latent_reasoner.py`：`LatentReasoner`——在 CTM 连续隐藏状态做 K 条潜路径并行探索（latent best-of-K，非 token 层 beam），按终态 certainty 聚合/选路，输出隐式摘要（路径数/收敛分/路径间分歧/置信）；四档 effort 骨架 none/low/high/max（K=1/2/4/8，sigma=0/0.02/0.05/0.08）；**none 档逐位委托 `engine.reason()` 与 v4.4.1 默认逐位等价**（prediction_vector 锚点）。
- `udos/reasoning_router.py`：`ReasoningRouter`——难度信号（CTM 收敛/集成分歧/OOD/任务复杂度/停机判据）合成难度分 → 推荐 effort 档位 + 是否显式化 + 可解释理由（规则零梯度）。
- `udos/latent_collab.py`：多专家在思考深度上协作（convergence/stability/parsimony 三专家 Agent-as-Tool 打分、Orchestrator 聚合选路、专家分歧大则升级 effort 与 star→chain/mesh），TraceChain 记录。
- HTTP：`POST /reason/latent`（effort=none|low|high|max，非法 effort→400、未挂载→409，返回 answer/effort/隐式 tick/路径数/是否显式化/延迟/可选显式链与隐式摘要）、`POST /reason/route`（难度评估+推荐档位+理由）；错误语义 400/404/409/500 不崩进程。
- `scripts/latent_pareto_v45.py` → `benchmarks/results/pareto_v45.json` + Makefile `latent-pareto`；`docs/LATENT_REASONING_RESEARCH.md`。

### Changed
- 版本全来源同步 4.4.1 → 4.5.3（`udos/__init__.py`、`pyproject.toml`、`Makefile`、`Dockerfile`、`docker-compose.yml`、测试断言）。

### 证据与账本（不预设隐式更好）
- 逐档实测：none 8 ticks/最低延迟、low 16、high 32/2 显式 token、max 64/11 显式 token。
- **REJECT**：隐式探索未改善物理 MSE（主预测器外挂不动，逐档 rollout MSE 恒 1.664042）；low/high/max 延迟≥none（K 次前向），隐式不省延迟。隐式档位价值在可观测性/可追溯，非精度。
- 自适应路由分桶：easy/mid→none(8 ticks)、hard→high(32 ticks)；平均分桶 16 ticks vs 全 max 64 ticks——把算力花在难题上。
- 主参恒 52191；33 代 checkpoint md5 逐位不变；none 逐位等价锚点全绿。

## [4.5.4] — 高性能级开源资源集成（L0 注册表 + L1 格式适配 + L2 真 smoke，CPU-only 诚实落地）

> 性质：analogy, not reproduction。`udos/resource_registry` 无可学参数，零训练、不进主 state_dict；主 predictor 恒 52191、eval_mse 恒 0.045556；33 代 checkpoint 与四个锚点 md5 逐位不变。**诚信四层 L0-L3**：本环境 CPU-only / 零重依赖，大权重/GPU 资源只做契约+能力声明+懒加载降级，不下载、不假装运行、不编造推理输出。

### Added
- `udos/resource_registry.py`：统一 `ResourceConnector` 协议（id/kind/license/level/requires{gpu,weights,pkg}/priority/profile_tags/status）+ 四态 `available|degraded|absent|env_blocked`；`probe()` 带线程安全缓存永不抛；`load()` 懒加载；`invoke()` 统一调用；`ResourceRegistry` 支持 profile（performance/full）与 kind/status/license/priority/level 过滤。
- `udos/connectors/`：models/datasets/actions 三类 connector；`GenericConnector` 通用轨迹归一化（PCE-Format v1）；`specialized.py` 真实格式适配器——动作分块(ACT/Diffusion Policy)、VLA 256-bin 离散动作(OpenVLA)、LeRobot/Open-X/RoboMimic episode schema、SMPL axis-angle→rot6d(AMASS/GRAB)、LAFAN1 BVH 片段、URDF/FK(yourdfpy)。
- `udos/connectors/_catalog_data.py`：由 `scripts/build_catalog_data.py` 从 `docs/opensource_catalog.json` 构建期编译（79 条），运行时不读 JSON。
- HTTP：`GET /resources[?profile&kind&status&license&priority&level]`（纯 JSON 列注册表/能力/分级）、`POST /resources/{id}/probe`（真探测）、`POST /resources/{id}/invoke`（L3/缺失→503、未知 id→404、缺 action→400）；延续 400/404/409/500 不崩进程、`/metrics` 纯文本、日志 stderr。
- `tests/test_resource_registry.py` + `tests/test_resources_service.py`（24 例）：L1 fixture 往返一致、L2 yourdfpy 真 smoke、L3 优雅降级、过滤/缓存/状态码矩阵。

### L0-L3 诚实落地（performance profile 默认）
- L2 真跑：yourdfpy 真实 pip 安装 + 对内联 URDF 跑通 `update_cfg/get_transform` FK（available，有 smoke 证据）。
- L2 契约：pytorch_kinematics 惰性 import，未装→absent + `pip install` 提示，不破坏核心 import。
- L3 契约（21 个大权重/GPU）：OpenVLA/π0/Octo/RDT/CogAct/SmolVLA/TinyVLA/MobileVLA/GR00T/V-JEPA/Cosmos/Genie/PhysBrain/UnifoLM/WALL/CTM-ImageNet/Doc-to-LoRA/Skild/Generalist/MimicDroid/DreamerV3 —— requires_gpu+requires_weights，invoke→503，不下载不假装运行。
- L1：其余条目纯 python 格式适配/归一化，inline fixture 单测往返。

### Changed
- 版本全来源同步 4.5.3 → 4.5.4（`udos/__init__.py`、`pyproject.toml`、`Makefile`、`Dockerfile`、`docker-compose.yml`、测试断言）。
- 新增 `UDOS_RESOURCE_PROFILE` 环境变量选择默认 profile（默认 performance）。

## [4.5.5] — 通用全能级（全量 79 条可发现 + profile 运行时切换 + 兼容性契约锚点）

> 性质：延续零训练、零主权重改动。"全面兼容"诚实定义＝**接口统一 + 可发现 + 能力自检 + 缺失优雅降级**，不等于全部在 CPU 跑通；L3 大权重/GPU 资源只做契约与能力声明。主参恒 52191、33 代 checkpoint 与锚点 md5 逐位不变。

### Added
- 全量 79 条资源进 registry（L0 全量声明 + 每条至少 L0/L1；L3 显式 requires_gpu/weights + 懒加载降级）。
- 统一 profile：`performance`（4.5.4 高性能子集）/ `full`（全量可发现、可探测、可优雅降级）。
- HTTP `POST /resources/profile` {profile}：运行时切换视图 profile（非法→400）；切换清空探测缓存但**不改主权重、不改默认数值**。
- `tests/test_resource_contract_v455.py`（6 例）：全量 schema 合法/license 非空/四态自洽、每 connector probe 永不抛、profile 切换前后引擎参数逐位相等、full 批量 79 条探测 <30s 不卡启动。

### 证据
- 全量 79 条枚举、schema 合法；L3 条目均声明 gpu/weights；absent 条目均带 detail 与 install_hint。
- profile 切换前后 `engine.parameters()` 逐位相等（锚点）。

### Changed
- 版本全来源同步 4.5.4 → 4.5.5。

## [4.5.6] — 资源线修 bug + 性能 + QA 收口（CPU-only 诚实交付）

> 性质：零训练、零主权重改动。registry 无可学参数；33 代 checkpoint 与四锚点 md5 逐位不变。

### Fixed（RED→GREEN，见 docs/VERIFICATION_v4.5.6.md）
- B-1：SMPL 零旋转 rot6d 只返回 4 维 → 恒为 6 维单位旋转。
- B-2：`GET /resources?profile=full` 的 summary 未随查询 profile 更新 → `summary(profile)` 透传。
- B-3：yourdfpy 误用不存在的 `forward_kinematics()` → 真实 API `update_cfg/get_transform`，FK smoke 通过。

### Added
- `scripts/bench_resource_v456.py` + Makefile `resource-bench` → `benchmarks/results/resource_registry_v456.json`。
- `docs/VERIFICATION_v4.5.6.md`（release decision = go）、`docs/RESOURCE_INTEGRATION_v4.5.md`（L0-L3 矩阵）。

### 证据
- **1489 passed / 0 failed**（基线 1459，新增 31，只增）；覆盖率 93%。
- L0-L3：L1=56 / L2=2(yourdfpy 真跑) / L3=21(契约 503)；状态 available=57 / absent=22；无 env_blocked（外网可用）。
- 性能：装配 0.24ms、probe 缓存 0.001ms、full 列表 0.075ms、L1 转换 0.0045ms。
- 四锚点 md5 逐位不变；33 代 checkpoint；主参恒 52191。

### Changed
- 版本全来源同步 4.5.5 → 4.5.6。

## [3.8.7] - Hardening patch（fix/hardening，不重训、不改主权重）

> 性质：v3.8.6 之上的加固补丁，完全对标 v3.3.4 加固专项五工作线；**不重训、不改主权重、不出新 checkpoint**。

### Fixed
- **FIX-301**：新端点（`/twin/scene`、`/twin/step`、`/wm/imagine`、`/wm/conservation`、`/icm/predict`）直接 `int(body.get(...))`/`float(body.get(...))`，当 JSON 值为 `null`（Python None）时 `int(None)` 抛 `TypeError`，不被 `except ValueError` 捕获，落入 `except Exception` → **500 而非 400**。修复：do_POST 异常处理扩展为 `except (ValueError, TypeError)`；新增 `_coerce_int`/`_coerce_float` 辅助方法，新端点统一使用；`neural_step` 增加 `candidate_actions` 每项须为 dict 的校验。

### Changed
- 版本全来源同步 3.8.6 → 3.8.7（`udos/__init__.py`、`pyproject.toml`、`Makefile`、`Dockerfile`、`docker-compose.yml`、126 个测试断言）。
- 关键懒初始化路径补 INFO 日志（WM/神经控制器/孪生场景）；日志仍全 stderr，不串 `/metrics` 与 HTTP 响应体。

### Added
- `tests/test_v387_hardening.py`（24）：FIX-301 RED/GREEN + null/坏字符串/默认值矩阵 + candidate_actions dict 校验 + caplog 断言 + capsys 红线。
- `benchmarks/perf_baseline_v387.py` + `benchmarks/results/perf_baseline_v387.json`：新路径性能基线。
- `docs/QA_HARDENING_v3.8.7.md`、`docs/VERIFICATION_v3.8.7.md`。

### Evidence
- 1184 passed / 0 failed / 0 skipped；覆盖率维持 93%；25 代 checkpoint md5 逐位不变；主参恒 52191；HTTP 矩阵全绿；性能无 ACCEPT 候选。

## [3.8.6] - 最终训练重建 + 全量验证 + 25 代兼容（3.8 线终点 ✅）

> 性质：3.8 线对外发布件；主 predictor 重训但**主参恒 52191、eval_mse=0.045556（与 v3.8.0/v3.7.0 逐位同口径）**。

### Added
- 正式件 `checkpoints/predictor_v3.8.6.pt`（seed=42/n=48/epochs=60/patience=12，同 front/hybrid_weight=0）；离线全特性 A/B `benchmarks/results/training_v3.8.6.json`（ICM+PWM+分层+多体+闭环 零外挂自检）。
- `scripts/build_v386_checkpoint.py` + Makefile `ckpt386`。
- `docs/VERIFICATION_v3.8.6.md`：测试/覆盖率、各线正反证据与被回退候选、25 代兼容、HTTP 矩阵、分层/多体延迟预算、已知限制。

### Evidence
- 主参恒 52191；全部新特性 `zero_grad_state_dict_md5_unchanged=true`；backcompat **25 件**（v2.1.0..v3.8.6）；HTTP /twin/step、/twin/scene 200/400/409/404 全矩阵验证。多体为合成参数化代理，数字孪生为合成场景。

## [3.8.5] - Patch + 文档对齐

> 性质：3.8 线 patch，**不重训、不改权重**。

### Added
- 边界测试 `tests/test_v385_edge.py`（8）：多体零智能体守卫、孪生极端配置（单体/大 bounds/50 障碍）、闭环空窗/非有限守卫、调度零/负预算与 max<base 守卫、协调器负 buffer 守卫。
- 文档对齐 3.8 线：`docs/ROADMAP.md`（补 v3.4–v3.8.x 路线）、`docs/ARCHITECTURE.md`（外挂层数据流）、`docs/DEPLOYMENT.md`（/twin 端点运维）、`README.md`（3.8 模块清单）。

### Evidence
- 边界全绿；主 52191 参数不变。

## [3.8.4] - 综合评测（跨 3.4-3.8 全部特性）

> 性质：3.8 线评测节点，**不重训、不改权重**。

### Added
- `scripts/comprehensive_eval_v38.py` → `benchmarks/results/comprehensive_eval_v3.8.0.json`：ICM(3.4)+PWM(3.6)+分层控制(3.7)+多体/WM调度/闭环/孪生(3.8) 在同一 predictor 上叠加调用，验证组合不冲突且主权重 md5 不变。
- 测试 `tests/test_v384_eval.py`（4）：综合 JSON 全 ok/零梯度、同模型多体+闭环+孪生叠加、backcompat 24 件。

### Evidence
- 7 个特性全部 ok；`zero_grad_state_dict_md5_unchanged=true`；主 52191 参数不变。

## [3.8.3] - 加固 + 24 代 checkpoint 兼容 + 性能基准

> 性质：3.8 线加固节点，**不重训、不改权重**。

### Added
- `scripts/feature_latency_v38.py` → `benchmarks/results/feature_latency_v3.8.0.json`：3.8 线新特性单次延迟（baseline predict_next / 多体消解 / WM 调度 / 闭环一步 / 孪生一步）。
- 测试 `tests/test_v383_service.py`（5）：24 代 checkpoint 全可加载且主参恒 52191、v3.8.0 件元数据、延迟 JSON 字段、/twin/* 端点复核。

### Measured
- 24 件（v2.1.0..v3.8.0）；baseline predict_next ≈3.05ms，闭环一步 ≈3.91ms，多体消解 ≈0.22ms（8 体），孪生一步 ≈0.61ms，WM 调度分配 ≈0.003ms。

## [3.8.2] - 系统集成 + HTTP 端点 /twin/step、/twin/scene

> 性质：3.8 线集成节点，**不重训、不改权重**。

### Added
- **HTTP `POST /twin/scene`**：创建/查询合成数字孪生场景（`n_agents`/`n_obstacles`/`seed`/`bounds`），返回快照+summary；无字段且已有场景即查询。
- **HTTP `POST /twin/step`**：数字孪生一步（多体冲突消解+积分），返回 step/冲突计数/快照；可选 `window` 在已挂 predictor 时附跑一步 `ClosedLoopOrchestrator`（大脑→小脑→脊髓→WM反馈→感知更新）。
- `UDOSService` 懒构造 `DigitalTwinScene` 与 `ClosedLoopOrchestrator`（opt-in 外挂，不挂默认钩子）。
- 测试 `tests/test_v382_integration.py`（10）：创建/查询/步进 200、未建场景 409、非法参数 400、未知路由 404、附跑闭环、/loop/step 默认不变。

### Evidence
- 错误语义沿用全工程：ServiceNotReady→409、ValueError→400、未知路由→404、异常→500；主 52191 参数不变。

## [3.8.1] - 多体协同 A/B + 冲突消解

> 性质：3.8 线 A/B 节点，**不重训、不改权重**；收益不稳处保持 opt-in。

### Added
- `scripts/multi_agent_ab_v38.py` → `benchmarks/results/multi_agent_ab_v3.8.0.json`：对穿场景下「有协调 vs 无协调」碰撞/到达/延迟对比，扫描智能体数 N∈{2,4,8}。
- 测试 `tests/test_v381_ab.py`（5）：A/B JSON schema、碰撞单调下降、opt-in、让行候选显式、诚实权衡说明。

### Measured
- 对穿 20 步：无协调碰撞事件 1/6/28（N=2/4/8），有协调 0；碰撞降 100%。**诚实权衡**：让行减速导致到达率下降至 0（安全换速度），故协调器保持 opt-in，不夸大收益。单次协调开销 0.16–0.95ms 随 N 增长。

## [3.8.0.dev3] - 合成数字孪生场景

> 性质：3.8 线内部节点，**不重训、不改权重**；孪生为合成参数化场景。

### Added
- **`udos/digital_twin.py`**：`DigitalTwinScene`——参数化合成多体+多物体+障碍场景生成器（可配 `n_agents`/`n_obstacles`/`seed`/`bounds`/形态半径/限速），同 seed 确定性逐位可复现；`step()` 走 `AgentCoordinator` 冲突消解+积分；`snapshot()`/`from_snapshot()` 场景快照与精确回放。零梯度、无可训参数。
- 测试 `tests/test_v38_twin.py`（9）：参数化生成、同 seed 确定性、异 seed 相异、快照回放逐位一致、step 推进、非法配置守卫、错类型守卫、零障碍退化。

### Evidence
- 合成数字孪生（analogy not reproduction）；主 52191 参数不变。

## [3.8.0.dev2] - 规划-执行-反馈全域闭环

> 性质：3.8 线内部节点，**不重训、不改权重**；一步闭环复用既有全部模块。

### Added
- **`udos/closed_loop.py`**：`ClosedLoopOrchestrator`——一步完整闭环：大脑规划→小脑执行→脊髓反射（`HierarchicalController`）→WM 想象反馈（`LatentWorldModel`，与本步 command 比较得 L1/L2 反馈误差，仅观测不改权重）→空间感知更新（command 滑窗推进观测，状态机式多步）。纯推理、零梯度、确定性、opt-in。
- 测试 `tests/test_v38_closed_loop.py`（8）：闭环串联各模块、反馈字段、感知窗滑窗、多步推进、反射 winner=spinal、确定性、零梯度 md5 不变、空/非法守卫、reset。

### Evidence
- 闭环 `zero_gradient=true`、`analogy_not_reproduction=true`；主 52191 参数不变；反射触发时 priority_winner=spinal。

## [3.8.0.dev1] - 世界模型调度器（多体 WM 想象预算分配）

> 性质：3.8 线内部节点，**不重训、不改权重**；多体 WM 想象预算分配为确定性规则。

### Added
- **`udos/wm_scheduler.py`**：`WMScheduler`——为多体分配世界模型想象预算。每体至少 1 步（horizon=1 即回退真实单步预测，逐位锚定 `predict_next`）；总预算为全部 horizon 之和上限；高优先级集中获额外想象步至 `max_horizon`，同级按 agent_id 字典序。复用 `LatentWorldModel.imagine`，零梯度、无可训参数。
- `scripts/wm_budget_ab_v38.py` → `benchmarks/results/wm_budget_ab_v3.8.0.json`：低预算（全回退真实）vs 高预算（优先级想象）A/B。
- 测试 `tests/test_v38_wm_scheduler.py`（11）：优先级分配、预算上限、不足预算守卫、回退标记、同级决胜、复用 WM、horizon=1 逐位锚点、缺 window 守卫。

### Evidence
- 实测 5 体：低预算 5/5 全回退真实；高预算 4 体想象、最高优先 3 体各 4 步、used=15≤cap；主 52191 参数不变。

## [3.8.0] - 多体协同核心 + 正式训练（全域调度线首训）

> 性质：3.8 线首训节点；主 predictor 重训但**主参恒 52191、评估口径不变**（eval_mse=0.045556，与 v3.7.0 逐位同口径）。

### Added
- **`udos/multi_agent.py`**：`MultiAgentScene`（N 智能体状态容器，每体独立 id/状态[6]=pos(3)+vel(3)/目标/优先级/形态半径/限速，有序可快照/回放）与 `AgentCoordinator`（**非学习**冲突消解：优先级让行 + 侵入深度成比例减速，同级按 agent_id 字典序决胜，被让行者/冲突对显式记录）。纯确定性几何算法，无可训参数。
- 正式件 `checkpoints/predictor_v3.8.0.pt`（seed=42/n=48/epochs=60/patience=12，同 front/hybrid_weight=0）；离线自检 `benchmarks/results/training_v3.8.0.json`。
- 测试 `tests/test_v38_multi_core.py`（15）：多体容器、冲突消解、优先级/同级决胜、侵入单调性、单体退化恒等、空/非法守卫、快照回放、积分+限速、确定性、零外挂。

### Evidence
- 主参恒 52191；多体自检 `zero_grad_state_dict_md5_unchanged=true`、`learned=false`、`analogy_not_reproduction=true`；backcompat 24 件（v2.1.0..v3.8.0）。多体为合成参数化代理，非真机多机器人。

## [3.7.3] - Patch 精修 + 文档对齐（3.7 线终点）

> 性质：3.7 线终点 patch，**不重训、不改权重**。

### Added
- 测试 `tests/test_v373_edge.py`（6）：零/负频率与负延迟预算守卫、反射冲突（碰撞+越界同步双事件）、PID 大误差数值稳定、大脑无候选（None 退化 vs [] 报错）、cortex_every=1 与 reset 清空反射日志。
- 修复 `HierarchicalController.reset()` 漏清 controller 级 `reflex_log`。
- 文档：`docs/VERIFICATION_v3.7.3.md` 对齐 3.7 线交付与实测指标。

### Evidence
- 终点全量 pytest 全绿；23 代 checkpoint（v2.1.0..v3.7.0）；分层实测 脊髓0.005ms≪小脑0.035ms≪大脑3.2ms。

## [3.7.2] - 加固 + 23 代 checkpoint 兼容 + 性能基准

> 性质：3.7 线加固节点，**不重训、不改权重**。

### Added
- **`scripts/neural_feature_latency_v37.py`** → `benchmarks/results/feature_latency_v3.7.0.json`：分层神经控制各组件单次延迟（baseline predict_next / neural full step / cortex plan / cerebellum track / spinal reflex）。
- 测试 `tests/test_v372_service.py`（6）：23 代 checkpoint 全可加载且主参恒 52191、v3.7.0 件元数据、两份 latency JSON 字段、/neural/step 200 复核。

### Measured
- 23 件（v2.1.0..v3.7.0）；单步 neural_full_step ≈ 3.44ms（大脑规划 ≈3.27ms 主导），小脑 ≈0.035ms，脊髓 ≈0.005ms。

## [3.7.1] - 集成 + HTTP 端点 /neural/step、/neural/reflex/log

> 性质：3.7 线集成节点，**不重训、不改权重**。

### Added
- **HTTP `POST /neural/step`**：一步分层神经控制（window/scene_params/candidate_actions）；返回 command[6]/priority_winner/safety_state/各层延迟/priority_matrix。未训练 409、非法输入 400、未知路由 404。
- **HTTP `POST /neural/reflex/log`**：查询反射事件日志（reset=true 清空），含 scheduler_summary。
- `UDOSService` 懒构造 `HierarchicalController`（只读外挂，opt-in，不挂默认钩子）。
- 测试 `tests/test_v371_integration.py`（8）：loop+neural 组合、/loop/step 默认不变、两端点 200、400/409/404、reflex log reset。

### Evidence
- 错误语义沿用全工程：ServiceNotReady→409、ValueError→400、未知路由→404、异常→500；主 52191 参数不变。

## [3.7.0.dev6] - 分层延迟预算量化 A/B

> 性质：3.7 线延迟节点，**不重训、不改权重**。

### Added
- **`scripts/neural_latency_v370.py`** → `benchmarks/results/neural_latency_v3.7.0.json`：三层单次延迟（cortex/cerebellum/spinal 的 mean/p50/max）+ 频率-延迟三维扫描（cortex_every=1/2/5/10）+ 预算合同 vs 实测 p50。
- 测试 `tests/test_v37_latency_ab.py`（5）：JSON 字段、四维扫描、分层延迟不等式、预算内标志。

### Measured
- 实测 p50：**脊髓 0.005ms ≪ 小脑 0.043ms ≪ 大脑 3.19ms**，三层均在合成预算（0.5/5/40ms）内；大脑每 1 步规划次数 > 每 10 步。

## [3.7.0.dev5] - 安全边界与反射优先级（脊髓>小脑>大脑）

> 性质：3.7 线安全节点，**不重训、不改权重**。

### Added
- **`CortexPlanner` 安全校验**：复用 `decision.safety_boundary` 校验最优目标是否落在区间下界之上，输出 `target_safe`/`boundary_reason`。
- **安全状态机**：`HierarchicalController.safety_state ∈ {nominal, caution, reflex}`（反射触发→reflex；大脑目标越界→caution；否则 nominal）；step 输出显式 `priority_matrix`（脊髓0<小脑1<大脑2）。
- 测试 `tests/test_v37_safety.py`（6）：优先级矩阵暴露、违例制动、状态机恢复、正常 nominal、target_safe 字段。

### Evidence
- 碰撞当步 `safety_state=reflex`/`winner=spinal`；下一良性步回到 `nominal`。

## [3.7.0.dev4] - 三回路多频率分层调度

> 性质：3.7 线调度节点，**不重训、不改权重**。

### Added
- **`HierarchicalController` 调度计数**：`run_counts`（cortex/cerebellum/spinal 实际调用次数）+ `scheduler_summary()`（频率比、ran_*、cortex_ratio）；大脑每 `cortex_every` 步规划一次、小脑/脊髓每步。
- 测试 `tests/test_v37_scheduler.py`（4）：多频率调度、频率比自动推算、scheduler_summary 字段、空守卫。

### Evidence
- cortex_every=4 跑 10 步：cortex=3（step0/4/8）、cerebellum=10、spinal=10；大脑不每步。

## [3.7.0.dev3] - 脊髓本地反射弧（碰撞→制动 / 越界→截断 / 超速→减速）

> 性质：3.7 线脊髓节点，**不重训、不改权重**。

### Added
- **`SpinalReflex` 升级**：碰撞→零速制动、越界→位置截断、超速→按比例减速；反射同步在当步 `act()` 完成（反射延迟 <1 步），"先制动再上报"，事件记入 `events`/`reflex_log`；无可训参数。
- `HierarchicalController` 新增 `collision_obstacles`/`collision_radius`/`speed_limit` 入参。
- 测试 `tests/test_v37_spinal.py`（8）：碰撞制动、越界截断、超速减速、反射优先级可证（final==spinal）、事件日志累积、良性不触发、单元直测。

### Evidence
- 碰撞当步 `priority_winner=spinal`，速度维归零，小脑命令被覆盖丢弃；两次同窗口 step 累积 2 条反射事件。

## [3.7.0.dev2] - 小脑边缘轨迹平滑/跟踪协调（PID 代理 + 前馈 + 低通）

> 性质：3.7 线小脑节点，**不重训、不改权重**。

### Added
- **`CerebellumTracker` 升级**：PID 代理（kp/ki/kd）+ 前馈补偿（kff 预判目标速度）+ 一阶低通平滑（alpha∈[0,1]）；增量跟踪 `command = state + smoothed`，内部积分/微分跨步累积，`reset()` 清空；无可训参数。
- 测试 `tests/test_v37_cerebellum.py`（7）：跟踪收敛、低通平滑有界、PID 参数与 alpha 越界守卫、前馈对目标跳变响应、空目标守卫、reset 后确定性。

### Evidence
- 低通相位滞后允许轻微超调，稳态误差 < 0.1；前馈项在目标跳变当步非零。

## [3.7.0.dev1] - 大脑慢规划（复用 policy MPC，多步候选评估）

> 性质：3.7 线大脑节点，**不重训、不改权重**。

### Added
- **`CortexPlanner` 升级**：封装 `MPCActionSelector` 做多步候选 rollout 评估；慢频率（cortex_every 调度）输出目标轨迹点，非规划步复用缓存目标供小脑消费；安全违例候选显式记录 `rejected_candidates`；未提供候选时诚实退化到 `predict_next`（与 v3.7.0 逐位一致）；空候选集显式 ValueError。
- 测试 `tests/test_v37_cortex.py`（6）：规划输出、慢频率缓存、与 policy 最优一致、空候选守卫、无候选退化。

### Evidence
- 同候选集大脑最优 best_index/best_score 与直接 `MPCActionSelector.select` 一致；cortex_every=3 时 step1/2 ran=False 复用目标。

## [3.7.0] - 三层控制架构核心（大脑/小脑/脊髓）+ 正式训练

> 性质：3.7 线首训节点，**同口径训练**（seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0），主 predictor 仍 52191 参数。analogy, not reproduction。

### Added
- **`udos/neural_control.py`**：`ControlLayer` 基类（频率/延迟预算/优先级）、`CortexPlanner`（大脑，只读 predictor 目标规划）、`CerebellumTracker`（小脑，比例跟踪）、`SpinalReflex`（脊髓，越界截断反射）、`HierarchicalController`（三层容器 + 统一 step + 多频率调度 + 优先级解析 脊髓>小脑>大脑）。
- **`scripts/build_v370_checkpoint.py` / `make ckpt370`** → 正式件 `checkpoints/predictor_v3.7.0.pt` 与 `benchmarks/results/training_v3.7.0.json`：分层控制零外挂自检（首步目标 == predict_next 逐位、step 前后主权重 md5 不变、三层合同就位）。
- 测试 `tests/test_v37_neural_core.py`（10）：三层结构、step 接口、频率比、空守卫、与 policy/loop 接口一致、确定性、零梯度。

### Evidence
- 三层优先级常量 脊髓(0)<小脑(1)<大脑(2)；cortex_every 由频率比自动推算（2Hz/20Hz=>10）；23 代 checkpoint（v2.1.0..v3.7.0）。

## [3.6.3] - Patch 精修 + 边界加固 + 文档对齐

> 性质：3.6 线终点 patch，**不重训、不改权重**。

### Added
- 测试 `tests/test_v363_edge.py`：零/负 horizon 守卫、未 fit 外挂恒等转移确定性、守恒单步守卫与大批检验、wm_events 非有限/非正半径守卫、文档对齐（VERSION_PLAN_3.6 / README / CHANGELOG 含 PWM）。

### Evidence
- 终点全量 pytest 全绿（1009+）；22 代 checkpoint 兼容；HTTP `/wm/imagine`、`/wm/conservation` 200/400/409 复核通过。3.6 线 PWM（潜在空间世界模型）交付闭环。

## [3.6.2] - 加固 + 22 代 checkpoint 兼容 + 性能基准

> 性质：3.6 线加固节点，**不重训、不改权重**。

### Added
- **`scripts/wm_feature_latency_v36.py`** → `benchmarks/results/feature_latency_v3.6.0.json`：WM 各组件单次延迟（encode_latent/transit/imagine_h4/imagine_uncertain/contact/conservation + 基线 rollout）。
- 测试 `tests/test_v362_service.py`（6）：22 代 checkpoint 全可加载且主参恒为 52191、v3.6.0 件元数据、latency JSON 字段、两端点复核。

### Measured
- backcompat **22 件**（v2.1.0..v3.6.0）；imagine_h4 ≈ 4.4ms vs 基线 rollout ≈ 19.7ms（外挂潜在想象约 4.5× 快，精度见 dev4/dev6 报告）。

## [3.6.1] - 集成 + HTTP 端点 /wm/imagine、/wm/conservation

> 性质：3.6 线集成节点，**不重训、不改权重**。

### Added
- **HTTP `POST /wm/imagine`**：潜在空间多步想象（window/horizon[1..16]/scene_params）；第 0 步锚定主单步，返回 `states`/`step_mse_vs_real`/`wm_params`。
- **HTTP `POST /wm/conservation`**：对真实或想象 rollout 做动量/能量守恒检验（source=real|imagine, mass, spring_k），返回 `momentum_violation`/`energy_violation`/`conserved`。
- **`PhysicalLoopRunner` opt-in 想象模块**：`use_world_model=True` 时 `future_state` 步走 `LatentWorldModel.imagine`（默认关 => 逐位走 `predictor.rollout`）。
- 测试 `tests/test_v361_integration.py`（11）：两端点 200、400（越界 horizon/非法 source/缺 window）、409（未训练）、404（未知路由）、loop 默认逐位走 rollout + opt-in 想象组合、第 0 步锚定。

### Evidence
- 错误语义沿用全工程：ServiceNotReady→409、ValueError→400、未知路由→404、异常→500；WM 外挂只读，主 52191 参数不变。

## [3.6.0.dev6] - 长 horizon 想象 vs 真实 rollout 对比实验

> 性质：3.6 线迭代节点，**不重训、不改权重**。

### Added
- **`scripts/wm_imagine_ab_v36.py`** → `benchmarks/results/wm_imagine_v3.6.0.json`：长 horizon（H=8/16/32）想象 rollout 误差累积 vs 真实 rollout、潜在空间压缩率（窗口展平 36 / 潜在 32 = 1.125）。
- 测试 `tests/test_v36_imagine_ab.py`（6）：JSON 字段、三档 horizon 齐全、第 0 步 MSE=0 锚点、误差随 horizon 单调非降、压缩率与 analogy 标注。

### Measured
- 第 0 步严格 0（锚点）；H=32 末步 MSE ≈ 35.5，误差随 horizon 累积（合成外挂转移预期现象）；不宣称想象可替代主 rollout。

## [3.6.0.dev5] - 世界模型不确定性 / 置信度 / 高不确定回退

> 性质：3.6 线迭代节点，**不重训、不改权重**。

### Added
- **`LatentWorldModel.imagine_uncertain()`**：n_samples 次种子受控带噪想象（潜在注入高斯噪声），以集成方差作不确定代理（复用 ensemble 思想，单模型+确定性扰动）。返回 `mean`/`std`/`uncertainty`/`confidence`∈[0,1]；`fallback_threshold` 可使高不确定步回退主 `predictor.rollout`，`used_fallback` 记录回退步。
- 测试 `tests/test_v36_uncertainty.py`（9）：形状、第 0 步锚定零方差且后续步方差>0、置信度区间、回退步逐位等于真实 rollout、无回退路径、同 seed 确定性、零梯度 md5、空/非法守卫。

### Evidence
- 纯前向、种子确定性；第 0 步锚定真实单步（无噪），噪声仅作用于潜在外挂转移。

## [3.6.0.dev4] - 世界模型与预测器一致性 A/B

> 性质：3.6 线迭代节点，**不重训、不改权重**。

### Added
- **`scripts/wm_consistency_ab_v36.py`** → `benchmarks/results/wm_consistency_v3.6.0.json`：A=主 `predictor.rollout` vs B=`LatentWorldModel.imagine_rollout`，对比逐步 MSE、动量/能量守恒违反量、单步延迟。
- 测试 `tests/test_v36_consistency_ab.py`（6）：JSON 落盘字段齐全、第 0 步 MSE=0 锚点且末步发散、主参 52191 不变、opt-in 裁决、被否决候选保留。

### Measured
- 逐步 MSE：第 0 步严格 0（锚点），末步 ≈ 2.54（外挂潜在转移随 horizon 累积误差）；延迟 imagine ≈ 5.56ms vs 真实 rollout ≈ 19.46ms（想象约 3.5× 快但精度差）。**裁决 opt-in**，默认不替换主 rollout。
- 被否决候选：①默认把 imagine 设为主 rollout（破坏逐位锚点/兼容铁律）；②外挂 WM 参数并入主 state_dict（破坏 52191 与 22 代兼容）。

## [3.6.0.dev3] - 物理守恒 (动量/能量) 一致性检验

> 性质：3.6 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/wm_conservation.py`**：`ConservationChecker`（在 rollout 轨迹 [B,H,6] 上检验动量代理 `m*v`、能量代理 `0.5*m*v²+势能` 的一致性，输出违反量）。支持可选弹簧势 `0.5*k*x²` 与重力势 `m*g*z`；返回逐样本 `momentum_per_step`/`momentum_violation`/`energy_per_step`/`energy_violation`/`conserved`。
- 测试 `tests/test_v36_conservation.py`（8）：匀速段严格守恒（违反≈0）、加速轨迹违反检出、弹簧 K+U 守恒（合成 spring 轨迹可验）、<2 步/非法质量/非有限守卫、`is_conserved` 助手。

### Evidence
- 纯解析、零梯度；合成低维代理上的不变量诊断，不宣称真实物理引擎严格守恒。

## [3.6.0.dev2] - 接触 / 碰撞事件预测 (几何代理)

> 性质：3.6 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/wm_events.py`**：`ContactPredictor`（从想象/真实 rollout 的物理状态轨迹 [B,H,6] 用碰撞几何代理预测接触事件，**非学习**）。逐样本输出 `distances`/`contact_flags`/`closing_speed`/`contact_prob`∈[0,1]、新进入接触的 `contact_steps` 事件序列、速度反转 `impact_steps` 代理；球-球判定与 `udos/collision.py` 同口径（中心距≤r1+r2）。
- 测试 `tests/test_v36_contact.py`（9）：接近即接触+事件步、远离零事件、速度反转 impact 代理、与 `CollisionDetector.ball_ball` 数值一致、空轨迹/非法半径/非有限守卫、batch 形状。

### Evidence
- 纯解析几何、零梯度；不接真实物理引擎（无冲量/摩擦/解算），analogy not reproduction。

## [3.6.0.dev1] - 多步"想象" rollout (潜在轨迹 + 解码物理状态)

> 性质：3.6 线迭代节点，**不重训、不改权重**。

### Added
- **`LatentWorldModel.imagine_rollout()`**：H 步潜在 rollout，同时返回 `states`[B,H,raw]（解码回物理状态）与 `latents`[B,H,latent]（每步潜在状态）；`compare_real=True` 时额外跑 `predictor.rollout` 并返回 `real` 与逐步 `step_mse`[H]。第 0 步锚定 `predict_next`（H=1 逐位等价）。
- 测试 `tests/test_v36_imagine.py`（8）：多步形状、H=1 锚点、与真实 rollout 逐步 MSE（第 0 步严格 0）、零梯度 md5、空/非法 horizon 守卫、`imagine` 与 `imagine_rollout.states` 逐位一致。

### Evidence
- 纯前向外挂；H>1 步为潜在转移解码（合成类比），与真实 rollout 的差异如实报告、不追求零误差。

## [3.6.0] - PWM 物理世界模型线首训 + 潜在空间前向世界模型核心 + 正式训练

> 性质：3.6 线首件；**唯一一次正式训练**；潜在空间世界模型起步（纯推理外挂、零梯度）；backcompat 扩至 22 代。

### Added
- **`udos/world_model.py`**：`LatentWorldModel`（从 `predictor.obs_encoder` 末帧激活提取潜在状态 `z_t`=[B,d_input]；外挂小 MLP 前向转移 `z_{t+1}=z_t+MLP([z_t,action])`，末层零初始化未拟合时恒等；外挂线性读出 `decode(z)->[pos3,vel3]`）；`imagine()` 多步潜在 rollout，第 0 步逐位锚定 `predict_next`（H=1 与 `predictor.rollout` 逐位等价）；`fit()` 离线在冻结 predictor 潜在轨迹上拟合外挂转移+读出（优化器只含世界模型参数）。
- **正式训练重建** `checkpoints/predictor_v3.6.0.pt` + `benchmarks/results/training_v3.6.0.json`（`scripts/build_v360_checkpoint.py`，Makefile `ckpt360`；与 v3.5.0 同口径 seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）。
- 测试 `tests/test_v36_wm_core.py`（13）：潜在状态提取、转移形状与默认恒等、decode 形状、imagine rollout、H=1 逐位锚点、零梯度 md5 锚点、fit 不动主权重、外挂参数量显式、空窗口/非法 horizon/缺 action 守卫。

### Measured
- 主 predictor 仍 **52191 参数**；外挂世界模型自身 **2310 参数**（transit 2112 + decode 198），不入主 state_dict；imagine/fit 前后主权重 md5 不变（零梯度可证）。
- H=1 imagine 与 `predictor.rollout(window,1)` 逐位等价；backcompat **22 件**（v2.1.0..v3.6.0）。

### Evidence
- analogy, not reproduction：潜在空间为合成低维代理（d_input=32），不宣称复现 V-JEPA/Cosmos/Genie/视频世界模型；新能力 opt-in，默认路径逐位等价。

## [3.5.0] - SFM 空间基础模型线首训 + 3D 几何核心 + 正式训练

> 性质：3.5 线首件；**唯一一次正式训练**；SFM 纯推理外挂起步；backcompat 扩至 21 代。

### Added
- **`udos/spatial.py`**：`SpatialObject`（pos3/vel3/radius 球体代理）、`SpatialScene`（多物体容器 + 位置/速度/半径矩阵 + 成对距离 + 整体坐标变换）、`SpatialTransform`（平移/旋转/缩放，内部 3×3 线性矩阵，解析逆与复合，可逆性可验）。
- **正式训练重建** `checkpoints/predictor_v3.5.0.pt` + `benchmarks/results/training_v3.5.0.json`（`scripts/build_v350_checkpoint.py`，Makefile `ckpt350`；同口径 seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）。
- **pce 互转**：`SpatialObject.from_physical_token / to_physical_token`，与 `PhysicalToken(position[3]/velocity[3])` 对齐。
- 测试 `tests/test_v35_spatial_core.py`（17）：物体几何、变换可逆性、轴角旋转、复合、空场景守卫、pce 互转、零外挂。

### Measured
- SFM 为纯 numpy 解析几何：`roundtrip_max_err ≈ 9e-16`（apply∘inverse 逐分量可逆）；主 predictor 仍 **52191 参数**，空间推理前后 state_dict md5 不变（零梯度可证）。
- 正式件 `eval_mse` 与 v3.4.5 同口径一致（见 `training_v3.5.0.json`）；backcompat **21 件**（v2.1.0..v3.5.0）。

### Evidence
- analogy, not reproduction：空间为合成低维代理，不复现 SpatialVLM/真实 3D/点云；新能力 opt-in，默认路径逐位等价。

## [3.5.0.dev1] - SceneGraph 场景图（空间关系）

> 性质：3.5 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/scene_graph.py`**：`SceneGraph`（物体节点 + 空间关系边），谓词 above/below/left/right/near/far/inside 全部由解析几何计算（非学习、零梯度）；支持 `relate/edges/query/neighbors` 遍历查询，可配置 up/right 轴与 near/far 阈值。
- 测试 `tests/test_v35_scene_graph.py`（9）：方位关系正确、near/far/inside、图遍历边数、空图守卫、版本断言。

### Evidence
- 纯几何外挂，不触碰主 predictor；与 SpatialScene 接口一致。

## [3.5.0.dev2] - OccupancyGrid 占据网格 + 有符号距离场

> 性质：3.5 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/occupancy.py`**：`OccupancyGrid`（规则 3D 体素网格，球体半径填充占据，分辨率可配 8/16/32）、`DistanceField`（占据为负/自由为正的有符号距离场，分块计算不爆内存）。
- 测试 `tests/test_v35_occupancy.py`（8）：球体填充、越界守卫、SDF 符号与单调性、空网格守卫、体素中心往返。

### Evidence
- 纯 numpy 确定性；SDF 自由侧随距离单调不降。

## [3.5.0.dev3] - 碰撞检测 + 最近邻

> 性质：3.5 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/collision.py`**：`CollisionDetector`（球-球代理碰撞 + 穿透深度、球-AABB 代理碰撞、场景接触枚举、跨帧"新进入接触"事件）、`NearestNeighbor`（给定点/物体暴力最近 k 邻，自动剔除自身）。
- 测试 `tests/test_v35_collision.py`（8）：球-球接触与穿透、球-AABB、场景接触对、连续帧事件、空场景/<2 物体守卫、最近邻排序。

### Evidence
- 解析几何，无冲量/摩擦解算；空场景显式 ValueError。

## [3.5.0.dev4] - 多视角坐标变换一致性

> 性质：3.5 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/spatial.py` 扩展**：`OrthographicView`（正交投影代理，非真实相机；eye/look_at/up 构造右-上-前正交基，project/unproject 精确可逆）、`multiview_consistency_error`（同一世界点经多视角投影-反投影的最大不一致误差）。
- 测试 `tests/test_v35_multiview.py`（6）：投影-反投影往返、跨视角一致性、基向量正交、视点重合/up 平行守卫、极端斜视可逆。

### Evidence
- 正交投影无透视歧义，往返误差 ~机器精度；非真实相机内参/畸变。

## [3.5.0.dev5] - 空间查询引擎

> 性质：3.5 线迭代节点，**不重训、不改权重**。

### Added
- **`udos/spatial_query.py`**：`SpatialQueryEngine`（射线-球解析相交、场景最近命中 raycast、两点视线遮挡 line-of-sight、球形范围检索 range_search、轴对齐盒检索 box_query）。
- 测试 `tests/test_v35_query.py`（10）：射线命中/未命中、最近命中、视线遮挡、范围升序、盒查询、空场景与零方向守卫。

### Evidence
- 解析几何，复用 spatial/collision；空场景显式 ValueError。

## [3.5.0.dev6] - 空间 A/B + 与 affordance 对比

> 性质：3.5 线迭代节点，**不重训、不改权重**。

### Added
- **`scripts/spatial_ab_v35.py`** + `benchmarks/results/spatial_ab_v3.5.0.json`：SFM `range_search`（精确欧氏距离）vs `AffordanceScorer`（softmax 可达性打分）的延迟与语义对比；占据分辨率 8/16/32 扫描；被否决候选保留。
- 测试 `tests/test_v35_ab.py`（5）：A/B JSON 落盘、分辨率扫描含 8/16/32、`decision=opt-in default off`、被否决候选字段齐全。

### Measured
- 合成 40 物体：range_search ≈ 0.076 ms、affordance ≈ 0.144 ms；两者语义 **IoU≈0.33**（精确距离 vs 归一化打分），不互相替代；SFM 作为独立 opt-in 能力，默认不接管 affordance 路径。

### Evidence
- 收益不稳 => 默认关；旧服务输出逐位等价。

## [3.5.1] - 集成 + HTTP /spatial/query、/spatial/collision

> 性质：3.5 线集成节点，**不重训、不改权重**。

### Added
- **HTTP 端点**：`POST /spatial/query`（op=range/box/raycast/los，零外挂不需训练）、`POST /spatial/collision`（场景接触检测）。请求体带 `objects` 临时构造场景；缺省走服务注册场景（未注册 -> 409）。
- **错误语义**：非法输入 -> 400；无可用场景 -> 409；未知路由 -> 404；异常 -> 500；日志只进 stderr 不污染响应。
- 测试 `tests/test_v351_integration.py`（10）：range/raycast 200、坏 op/空 objects 400、无场景 409、碰撞接触对、未知路由 404、默认 health 路径逐位一致。

### Evidence
- SFM 端点独立 opt-in，不依赖主 predictor；旧端点行为不变。

## [3.5.2] - 加固 + 21 checkpoint 兼容 + 性能基准

> 性质：3.5 线加固节点，**不重训、不改权重**。

### Added
- **backcompat 扩至 21 件**（v2.1.0..v3.5.0）全部可加载；`test_v352_service.py` 断言恰好 21 件。
- **`scripts/sfm_feature_latency_v35.py`** + `benchmarks/results/feature_latency_v3.5.0.json`：10 个 SFM 组件单次查询延迟（50 物体合成场景，CPU 2 线程）。
- 测试 `tests/test_v352_service.py`（6）：21 checkpoint 加载、v3.5.0 正式件元数据、latency JSON 字段、A/B 与 training JSON 在位。

### Measured
- 单次延迟中位数量级：scene_graph 0.005ms、occupancy/DF 0.01~0.015ms、raycast 0.5ms、碰撞全检测 ~3ms（50 物体 O(N²)）；均远低于主推理路径。

### Evidence
- 全量回归无退化；21 代 checkpoint 逐件有限。

## [3.5.3] - Patch 精修 + 边界测试 + 文档对齐（3.5 线终点）

> 性质：3.5 线终点补丁，**不重训、不改权重**。

### Added
- **边界测试** `tests/test_v353_edge.py`（10）：零半径物体构造期拒绝、相切共面碰撞（穿透=0）、极端大坐标数值稳定、射线切球退化、空查询守卫、0 角度旋转可逆、退化 up 拒绝、网格边界夹取、等高物体无 above/below。
- **文档对齐**：`ROADMAP.md`/`ARCHITECTURE.md`/`DEPLOYMENT.md`/`README.md` 增补 v3.5 SFM 线（模块清单、零外挂、HTTP 端点、21 代 backcompat）。

### Evidence
- 3.5 线共 10 节点（3.5.0..3.5.3）全部落地；正式件 `predictor_v3.5.0.pt`；HTTP 200/400/409/404 验证。

## [3.4.5] - 最终训练重建 + 全量验证 + 20 代兼容 + 收尾

> 性质：3.4 线终点发布件；**唯一一次最终正式训练**；backcompat 20 代。

### Added
- **最终正式训练重建** `checkpoints/predictor_v3.4.5.pt` + `benchmarks/results/training_v3.4.5.json`（`scripts/build_v345_checkpoint.py`，Makefile `ckpt345`；同口径 seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）。
- **backcompat 扩至 20 件**（v2.1.0..v3.4.5）全部可加载；save/load 逐位一致测试。
- **`docs/VERIFICATION_v3.4.5.md`**：测试/覆盖率、正反证据、被否决候选、checkpoint 兼容、HTTP 矩阵、已知限制。

### Measured
- 同配方重训无漂移：`eval_mse=0.045556`，与 v3.4.0/v3.3.3 逐位一致；n_params 仍 52191。
- ICM 离线 A/B：`icm_k0=0.045→k1=0.023→k3=0.015→k5=0.014`（单调改善）；naive `k0=0.093→k3=4.19`（退化，REJECT）；零梯度 md5 不变=true。

### Evidence
- 全量 pytest **846 passed / 0 failed / 0 skipped**；20 代 checkpoint 兼容；正式件指标齐全；CHANGELOG 3.4 线共 12 条；零梯度可证。

## [3.4.4] - 全特性集成 + 综合评测（不改权重）

> 性质：发布节点，**不重训、不改权重**。

### Added
- **全特性组合集成测试** `tests/test_v344_integration.py`：loop + ICM + 事件切分 + 预算 cap + 跨本体归一化 + PCE 包往返 + 碰撞事件切分 组合不冲突、默认路径逐位一致。
- **综合评测**落 `benchmarks/results/icm_comprehensive_v3.4.4.json`（ICM k0/k3 MSE 与改善量、backcompat 19 件）。

### Evidence
- 新增测试（3）：全特性组合可跑且有限、综合评测 JSON 落盘且 k3 不退化、版本断言。
- backcompat 仍 19 件。

## [3.4.3] - Patch 精修 + 边界测试 + 文档对齐（不改权重）

> 性质：发布节点，**不重训、不改权重**。

### Added
- **边界测试** `tests/test_v343_edge.py`（10）：空记忆退化为 0-shot、零预算 cap、未缓存残差 KeyError、匀速事件无边界退化、三流短序列守卫、无 memory 参数、PCE 空包/坏版本、episode 动作维度校验、版本断言。
- **文档对齐**：`ROADMAP.md` / `ARCHITECTURE.md` / `DEPLOYMENT.md` / `README.md` 增补 v3.4 ICM 线（架构图、运维端点、零梯度与 backcompat 说明）。

### Evidence
- 全量 pytest 全绿；文档链接/章节有效。

## [3.4.2] - 集成加固 + 19 checkpoint 兼容 + 性能基准（不改权重）

> 性质：发布节点，**不重训、不改权重**。

### Added / Measured
- **backcompat 扩至 19 件**（v2.1.0..v3.4.0）：全部可 `load_predictor`，预测输出有限、形状 [6]。
- **性能基准** `scripts/icm_feature_latency_v342.py`，落 `benchmarks/results/feature_latency_v3.4.0.json`（2 线程 CPU）：
  - 纯 0-shot predict_next **3.34 ms**；
  - ICM k=3 检索聚合 **3.72 ms**（仅 +0.4 ms 开销）；
  - 事件切分 0.08 ms / 三流对齐 0.10 ms / PCE 提示词解析 0.21 ms。
- ICM 路径端到端延迟可忽略，满足 20Hz 实时预算。

### Evidence
- 新增测试 `tests/test_v342_service.py`（3）：19 件 checkpoint 全加载、latency JSON 落盘可复算、版本断言。
- 全量回归无退化。

## [3.4.1] - ICM 与 PhysicalLoop 集成 + 服务端点（不改权重）

> 性质：发布节点（3.4 线首个对外端点），**不重训、不改权重**。

### Added
- **PhysicalLoopRunner opt-in ICM**：新增 `use_icm/icm_memory/icm_k` 参数；`_integrated_predict` 在 ICM 记忆库非空时走 `ICMAggregator` 检索聚合（零梯度），`use_icm=False` 默认逐位委托 `predictor.predict_next`。
- **HTTP 端点**（避开既有 `/icl/predict`）：
  - `POST /icm/predict`：演示条件化预测（未训练/记忆库空 → 409；非法输入 → 400）；
  - `POST /icm/demo/register`：注册演示到服务级记忆库（未训练 → 409；缺字段 → 400）。
- 错误语义：客户端 400、未挂载 409、未知路由 404、异常 500 不崩进程；日志走 `logging_config`（stderr），不污染响应体。

### Evidence
- 新增测试 `tests/test_v341_integration.py`（8）：loop 默认路径逐位一致、loop+ICM 组合可跑、`/icm/predict` 空库 409、注册后预测 200、注册缺字段 400、k<0 400、未知路由 404、版本断言。
- 真实起 HTTP 端口（`create_server(..., checkpoint=predictor_v3.4.0.pt)`）验证 200/400/409/404 矩阵。

## [3.4.0.dev6] - 三路线对照实验（数据/思维链/上下文 Scaling，不改权重）

> 性质：dev 节点，**不重训正式件**（仅对照用短训）；明确合成类比，非 LLM scaling law 复现。

### Added / Measured
- **`scripts/icm_three_route_ab.py`**，落 `benchmarks/results/icm_three_route_v3.4.0.json`，三路线样本-性能-延迟三维对照：
  - **数据 Scaling**：n_per_kind=16/32/48 → mse **0.462 / 0.182 / 0.162**（数据越多越好、边际递减）；
  - **思维链(tick) Scaling**：CTM iterations=4/8/16 → mse **0.173/0.182/0.158**，延迟 **0.45/0.87/2.23 ms**（tick 越多精度略升、延迟 ~5×）；
  - **上下文 Scaling (ICM)**：k=0/1/3/5/10 → mse **0.057/0.029/0.020/0.019/0.019**，延迟平稳 ~3.6 ms。
- **结论**：在该合成基准上，**上下文 Scaling（ICM）单位成本收益最高**——mse 从 0.057 降到 0.019 而延迟几乎不增。

### Evidence
- 新增测试 `tests/test_v34_three_route.py`（3）：JSON 落盘可复算、三维度齐全、上下文路线不爆炸、版本断言。

## [3.4.0.dev5] - k-shot scaling 曲线 + 权重逐位不变锚点（不改权重）

> 性质：dev 节点，**不重训、不改权重**。

### Added / Measured
- **ICM k-shot scaling 曲线**（同合成基准）：k=0/1/3/5/10-shot MSE = **0.0485 / 0.0319 / 0.0205 / 0.0185 / 0.0181**——单调改善、~5-shot 后饱和，绝不退化。
- **权重逐位不变锚点**：多轮 ICM 推理（k=1/3/5/10 各 5 次）前后 `state_dict` md5 逐位一致。
- **ICM vs naive 对照**：ICM k3=0.0206 vs naive k3=2.331（~100× 差距），正面证明检索+聚合优于朴素拼接。

### Evidence
- 新增测试 `tests/test_v34_shot_scaling.py`（5）：k-shot 曲线有限不退化、权重 md5 锚点、vs naive ICL 对照、空记忆守卫、版本断言。
- analogy, not reproduction。

## [3.4.0.dev4] - 上下文预算/检索压缩与延迟-收益 A/B（不改权重）

> 性质：dev 节点，**不重训、不改权重**；呼应 "8000 步 token 不能全上云"。

### Added
- **`udos/icm_budget.py`** `ContextBudgetManager`：存储预算（episode 数上限）、检索截断（`cap_k` 把请求 k 压到预算内）、原型压缩（把 k 个残差原型按权重分桶聚成 n_proto 个，零梯度均值）；opt-in 默认 `budget=None` 不截断（与 v3.4.0 逐位一致）。
- **延迟-收益 A/B** `scripts/icm_budget_ab_v340.py`，落 `benchmarks/results/icm_budget_ab_v3.4.0.json`。

### Evidence
- A/B 实测（同基准）：budget=4 mse=0.0204 / lat=3.87ms；budget=8 mse=0.0192 / lat=3.70ms；budget=16/32 因 k 上限=8 不再变化（**边际收益递减、延迟平稳**）。
- 新增测试 `tests/test_v34_budget.py`（8）：cap_k 边界、不限预算 opt-in、压缩 no-op、分桶压缩、预算收缩 k、A/B JSON 落盘可复算、坏预算守卫、版本断言。

## [3.4.0.dev3] - PCE 物理提示词数据包接口（HTTP-ready JSON，不改权重）

> 性质：dev 节点，**不重训、不改权重**；扩展 `udos/pce_format.py`。

### Added
- **`pce_format.DemonstrationPrompt`**：可 HTTP 传输的 JSON 提示词包——因果块（`input_window[W,6] → action[6] → result[6]`）+ 适配块（跨本体 `source_dof/target_dof/src_freq/dst_freq`）；`to_dict()/dumps()` 序列化为 `PCE-DemonstrationPrompt/v1` 格式。
- **`pce_format.PCEPromptParser`**：`from_dict()` 校验包版本并解析；`load_episodes()` 把提示词包转成 `DemonstrationEpisode` 列表供 ICM 注册；往返一致。
- 导出 `DemonstrationPrompt` / `PCEPromptParser` 到 `udos/__init__.py`。

### Evidence
- 新增测试 `tests/test_v34_pce_prompt.py`（8）：包结构、JSON 往返、未知包版本拒绝、非对象拒绝、空包零 episode、坏形状守卫、空 prompt_id 守卫、版本断言。
- 不调主模型、不改权重；analogy, not reproduction。

## [3.4.0.dev2] - 跨本体演示归一化后 ICL（复用 retargeting，不改权重）

> 性质：dev 节点，**不重训、不改权重**；复用 `udos/retargeting.py` 成熟件。

### Added
- **`udos/icm_cross.py`** `CrossEmbodimentICM`：源本体演示轨迹 `[T_src, src_dof]` →
  时间重采样（`ActionRetargeter.resample`）→ DOF 映射+关节限幅到 6 维（`ActionRetargeter.retarget`）→
  滑窗切 (window, result) 对 → 注册进 `DemonstrationMemory` 并缓存残差；
  预测期委托 `ICMAggregator` 做跨本体演示条件化预测。
- 导出 `CrossEmbodimentICM` 到 `udos/__init__.py`。

### Evidence
- 新增测试 `tests/test_v34_cross_embodiment.py`（7）：do4→do6 归一化注册、目标 dof 必须=6 守卫、源 dof 不匹配守卫、do=0 源本体不可重定向、时间重采样路径、k=0 走 0-shot、版本断言。
- 纯函数式零梯度；analogy, not reproduction。

## [3.4.0.dev1] - 事件级切分与三流对齐（WALL-WM analogy，不改权重）

> 性质：dev 节点，**不重训、不改权重**；纯函数式零参数外挂。

### Added
- **`udos/icm_events.py`**：
  - `EventSegmenter`：变点检测，变点信号=相邻帧速度差 L2 范数，自适应阈值 `median + k·MAD`（`min_gap` 抑制抖动）；匀速段无边界（合法退化），碰撞瞬时速度交换可稳定检出边界；
  - `ThreeStreamAligner`：沿事件边界把 **物理token状态流 / 动作流(Δs) / 结果流(sₜ₊₁)** 切成等长对齐片段；`align_episodes()` 直接产出可注册为 `DemonstrationEpisode` 的 (window, result) 对。
- 导出 `EventSegmenter` / `ThreeStreamAligner` 到 `udos/__init__.py`。

### Evidence
- 新增测试 `tests/test_v34_events.py`（9）：匀速无边界、碰撞检出边界、短序列/末维守卫、三流等长、对齐对可注册 ICM、无边界退化整条、版本断言。
- 零参数、零梯度、纯合成轨迹验证；analogy, not reproduction。

## [3.4.0] - ICM 上下文记忆核心：检索+聚合（零梯度）+ 首训正式件 + backcompat 19 件

> 日期：2026-09-14。性质：**大版本起点**（ICM In-Context Memory）。
> 正式件 `checkpoints/predictor_v3.4.0.pt`（52191 参数，与 v3.3.3 同配方，eval_mse 逐位一致 0.045556）。
> analogy, not reproduction；CPU-only 合成数据；ICM 推理路径**零梯度**、默认 opt-in。

### Added
- **`udos/icm.py`** ICM 上下文记忆三件套：
  - `DemonstrationEpisode`：一次 "输入轨迹→动作→结果" 的 PCE 因果块（`action = result − 窗口末帧`），注册期一次算好 L2 归一化检索嵌入；
  - `DemonstrationMemory`：按查询窗口与演示输入窗口的余弦相似度检索 top-k（纯确定性、零参数、不调主模型）；
  - `ICMAggregator`：零梯度原型聚合——**在输出残差空间**聚合，`p_icm = p0 + λ·Σ softmax(sᵢ)·rᵢ`，残差 `rᵢ = resultᵢ − predict_next(inᵢ)` 注册期缓存。
- **正式训练重建** `checkpoints/predictor_v3.4.0.pt` + `benchmarks/results/training_v3.4.0.json`（`scripts/build_v340_checkpoint.py`，Makefile `ckpt340`；seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）。

### Changed（先复现再超越）
- **复现并解释既有 few-shot 退化**：naive `InContextLearner` 朴素拼接原始窗口，离线 A/B 实测 `naive_k0=0.093 → k1=2.30 → k3=4.19`（反而更差）；根因=把 k·W 帧原始窗口拼进 52k 小模型注意力视野，**稀释**了默认 6 帧主信号。
- **ICM 检索+聚合路径 k-shot 不退化**（同基准）：`icm_k0=0.045 → k1=0.023 → k3=0.015 → k5=0.014`，单调改善、绝不爆炸；λ 收缩保证 λ=0 时 `p_icm≡p0` 逐位一致。
- **零梯度原则**：ICM 全程 `@torch.no_grad`，聚合器**可训参数=0**，不入主 state_dict；权重不变锚点测试 `state_dict` md5 在 ICM 推理前后逐位一致（`zero_grad_state_dict_md5_unchanged=true`）。
- 版本号升至 **3.4.0**（`udos/__init__.py` / `pyproject.toml` / Makefile / Dockerfile / docker-compose / tests 断言同步）。

### 被否决/保留候选（诚实账本）
- naive 朴素拼接 few-shot ICL：0.046→1.11（历代已记录），本节点在新基准再次复现为 0.093→4.19，**REJECT**，ICM 改用残差空间检索聚合。

### Evidence（真实运行）
- 新增测试 `tests/test_v34_icm_core.py`（13）：episode 结构、memory top-k 降序、零梯度 md5 锚点、k-shot 不退化、naive 退化复现、空记忆/λ=0/k=0 opt-in 逐位等价。
- 同配方重训无漂移：`eval_mse=0.045556`，与上一线 v3.3.3 `prev_line_v333_mse=0.045556` 完全一致；n_params 仍 52191。
- backcompat 扩至 **19 件**（v2.1.0..v3.4.0）；训练 103.1s / 2 线程 CPU。

## [3.3.4] - hardening patch：6 真缺陷修复 + 统一 logging + 2 项性能 ACCEPT（不改权重）

> 日期：2026-09-14。性质：**加固补丁**，不重训、不改权重、不删旧测试（只增）。
> 正式件 checkpoint 仍为 `checkpoints/predictor_v3.3.3.pt`，md5 `f993bcbdd476473c28dd4604bbbe11d6`（与 v3.3.3 一致）。

### Fixed（6 项，引用 DIAG-ID）
- **DIAG-001**（P2）：`GET /eval/5d` 未训练时由 500 修正为契约约定的 **409**——`do_GET` 补 `ServiceNotReady -> 409` 分支，与 `do_POST` 对齐。
- **DIAG-002**（P2）：`Dockerfile` / `docker-compose.yml` 启动加载的 checkpoint 由 v3.2.0 修正为 **v3.3.3**（发布元数据与实际权重代际对齐）。
- **DIAG-003**（P3）：`POST /icl/predict` 非法 `window` 类型由 500 修正为 **400**——`torch.as_tensor` 加 try/except 转 `ValueError`，与 `/predict` 等端点对齐。
- **DIAG-004**（P3）：残缺 PCE scene（`/internalize`、`/reason`）由 500 修正为 **400**——`_scene()` 入口校验 tokens 非空；合法 PCE 路径不受影响。
- **DIAG-008**（P3）：`/health` 读 `scene_memory.keys()` 加服务锁（预防性线程安全加固）。
- **DIAG-009**（P3）：checkpoint 目录锚定工程根（基于 `__file__` 定位），不再依赖进程 cwd。

### Changed
- **日志体系**：全量切换为标准库 `logging`，新增 `udos/logging_config.py`；**42 个 udo 模块**接入 logger；默认级别 WARNING、输出 `sys.stderr`、格式 ISO 时间；`UDOS_LOG_LEVEL` 环境变量覆盖；`/metrics` 保持纯 Prometheus 文本（无 JSON/日志串入）；`caplog` 可断言；数值逐位等价锚点测试守护。
- **性能优化（2 项 ACCEPT，均有同合同 A/B + 逐位等价证据）**：
  - **C1**：`physical_loop.run()` 末尾冗余前向复用 `understand.target_state`，loop 单步 p50 **52.7 → 46.9 ms（−11%）**。
  - **C2**：`BatchPredictor` 默认 `max_shard` 32 → **64**，bs=64 批量推理 p50 **14.58 → 8.64 ms（−41%）**。
- 版本号升至 **3.3.4**（代码版本单一来源同步：`udos/__init__.py` / `pyproject.toml` / Dockerfile LABEL / docker-compose 与 Makefile 镜像 tag；测试版本断言 71 处同步）。

### 性能被反证候选（诚实保留）
- C3a HTTP `/predict` 序列化（框架/list 转换绑定，无安全单 delta）、C3b GPM/CTM 前向冗余拷贝（B=1 无可省）——均 REJECT，未改代码。

### Evidence（真实运行）
- 全量 pytest：**738 → 766 passed（+28）** / 0 failed / 0 skipped；覆盖率维持 **93%**。
- 新增测试：`tests/test_v334_bugfix.py`（12）、`test_v334_logging.py`（7）、`test_v334_numerical_equivalence.py`（4）、`test_v334_perf_optim.py`（5）。
- 18 代 checkpoint 全部可加载；正式件权重 md5 不变。
- A/B 原始数据 `benchmarks/results/perf_ab_v334.json`，post 基线 `benchmarks/results/perf_baseline_v334_post.json`。
- 详见 `docs/QA_HARDENING_v3.3.4.md` 与 `docs/VERIFICATION_v3.3.4.md`。

## [3.3.3] - 3.3 线终点：最终训练重建 + 全部新特性离线 A/B + backcompat 18 件

### Added
- 最终正式训练重建 `checkpoints/predictor_v3.3.3.pt` + `benchmarks/results/training_v3.3.3.json`
  （`scripts/build_v333_checkpoint.py`，Makefile `ckpt333`）；含全部新特性离线评估 A/B：
  action_piece / ego_augment / moe / distill_v2 / prune_v2 / robustness。
- backcompat 由 17 件扩至 **18 件**（v2.1.0..v3.3.3），全量加载 predict。

### Changed
- 版本号升至 **3.3.3**（对外发布）；正式件仍 52191 参数、默认旧架构。

### Fixed
- 无（最终件与 v3.3.0 同口径确定性复现）。

### Evidence（真实运行，seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）
- `train_seconds=102.5`，`eval_mse=0.045556`（naive=0.163415），`ece=0.056546`，
  coverage=0.8888，`batch_inference_max_diff=1.97e-06`。
- 离线 A/B：action_piece roundtrip=0.008166；ego_augment 有限；moe 2244 参数；
  prune_v2（0.5 通道，不重训）mse=2.840944；student_v2 15027 参数 mse=2.630763；
  robustness_score=55.18。
- 全量 pytest：**738 passed / 0 failed**，覆盖率 **93%**（5731 语句 / 402 未覆盖）。
- backcompat 18 件全部可加载；`predictor_v3.3.0.pt`、`predictor_v3.3.3.pt` 均可 reload 一致。

### 被否决/降级候选（诚实保留）
- 通道剪枝 0.5 不重训（mse 2.84）与 d_model 减半学生（mse 2.63）均显著劣化全量（0.0456），
  保持 opt-in、推荐 full；MoE/蒸馏/剪枝/鲁棒性均为合成数据机制类比，非外部榜单。
- 不做 zip 打包与独立 /tmp 复跑（由后续 QA 代理完成）。

## [3.3.2] - 3.3 线边界精修 + 文档对齐

### Added
- `tests/test_v332_edge.py`（7 例）：MoE 专家数=1 退化路由、蒸馏温度=0 守卫、
  剪枝稀疏度=1.0/负值守卫、激活缓存空态与清空防护、鲁棒性极端噪声（sigma=10，分数仍在 [0,100]）、
  服务未训练态 loop_step 抛 ServiceNotReady、版本断言。

### Changed
- 版本号升至 `3.3.2`；ROADMAP / ARCHITECTURE / DEPLOYMENT / README 更新至 3.3 线。

### Fixed
- 无（均为边界守卫补齐）。

### Evidence（真实运行）
- 全量 pytest：**738 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 同前：3.3 新特性保持 opt-in，正式件口径不变。

## [3.3.1] - 全特性集成 + 五维/鲁棒性综合评测 + backcompat 17 件

### Added
- `tests/test_v331_integration.py`（5 例）：loop+retarget+affordance+multitask+multimodal+
  action_piece+ego_augment+moe+robustness 全特性组合不冲突、默认路径逐位一致、
  五维评测+鲁棒性评测综合报告落 `benchmarks/results/integration_report_v3.3.1.json`、
  backcompat v2.1.0..v3.3.0 共 **17 件**全量加载 predict。

### Changed
- 版本号升至 `3.3.1`；checkpoint 谱系扩至 17 件。

### Fixed
- 小样本下 ActionPiece 码本利用率可为 0，集成测试放宽为 `0<=utilization<=1`（合法区间）。

### Evidence（真实运行）
- 17 件 checkpoint 全部 `predict_next` 有限；综合报告五维分与鲁棒性分数均落 [0,100]。
- 全量 pytest：**731 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 全部 3.3 新特性仍为 opt-in 外挂；组合验证仅证明互不冲突，未改变正式件口径。

## [3.3.0.dev6] - 鲁棒性加固（噪声/OOD/外推/对抗扰动）

### Added
- `udos/robustness.py`：`RobustnessEvaluator(model, noise_sigmas, fgsm_eps,
  extrapolation_scale)` 只读综合评估：噪声 sigma 网格退化、OOD 命中率/误报率（集成现有
  `ood.DistributionDriftDetector`）、外推幅度 MSE 退化、FGSM 代理一步对抗扰动退化；
  加权输出 `robustness_score ∈ [0,100]`。
- `tests/test_v33_robustness.py`（6 例）：四项指标齐全、分数范围、OOD 检测器集成、
  噪声网格与 noise_augment 一致、FGSM 退化记录、同种子可复现、未训练模型不崩溃。

### Changed
- 版本号升至 `3.3.0.dev6`。鲁棒性评估为只读外挂，不训练不改权重。

### Fixed
- FGSM 代理接口修正为接收单步目标 `[B,R]`。

### Evidence（真实运行）
- 全量 pytest：**726 passed / 0 failed**（720 + 6 新）。
- 未挂 OOD 检测器时 hit_rate/false_alarm 显式为 None（不静默退化）。

### 被否决/降级候选（诚实保留）
- FGSM 为单步梯度符号代理，非完整对抗训练；鲁棒性分数为合成数据相对度量，非外部榜单。

## [3.3.0.dev5] - 内存优化（梯度检查点 + 激活 FP16 缓存）

### Added
- `udos/training.py`：`TrainConfig.gradient_checkpointing`（默认 False）开启后，训练时对每个
  CTM 前向用 `torch.utils.checkpoint(use_reentrant=False)` 重算换激活内存；
  `ActivationFp16Cache`（推理外挂）按输入哈希缓存 obs_encoder 中间嵌入（FP16 存储，内存减半，
  命中直接取回）。
- `tests/test_v33_memory.py`（5 例）：梯度检查点训练 loss 下降、默认关逐位一致、
  FP16 缓存命中、FP16 字节数 = 等价 FP32 的一半、缓存清空。

### Changed
- 版本号升至 `3.3.0.dev5`。两项优化默认全关，不改旧训练/推理路径。

### Fixed
- FP16 量化-上采样有 ~1e-3 数值误差（非逐位），测试容差放宽并照实记录；
  修复效率 JSON 版本断言改为 `startswith("3.3.0")`（dev 节点版本递增不锚定单一版本）。

### Evidence（真实运行）
- 梯度检查点开启训练 3 epoch loss 下降；FP16 缓存取回误差 max < 5e-3。
- 全量 pytest：**719 passed / 0 failed**（715 + 5 新 - 1 已修复版本断言）。

### 被否决/降级候选（诚实保留）
- CPU 上激活内存绝对值小，梯度检查点主要在大 batch/多步 rollout 才有可见收益；
  FP16 缓存仅在重复相同输入时省内存，二者均 opt-in。

## [3.3.0.dev4] - 效率 Pareto A/B（参数/延迟/精度三维）

### Added
- `scripts/efficiency_pareto_v33.py` + `benchmarks/results/efficiency_pareto_v3.3.0.json`：
  full / pruned_v2 / student_v2 / moe_addon 四形态的 `n_params - predict_ms - eval_mse` 三维对比。
- `tests/test_v33_efficiency.py`（4 例）：JSON 结构、四形态三维指标、推荐配置标注、
  剪枝退化被照实记录、opt-in 默认关。

### Changed
- 版本号升至 `3.3.0.dev4`。正式件默认不变。

### Fixed
- 无。

### Evidence（真实运行，CPU 2 线程，480 样本）
- full：52191 参数 / 6.70 ms / eval_mse=0.04750。
- pruned_v2（通道剪枝 0.5，不重训）：52191 参数 / 6.46 ms / eval_mse=2.93773（精度大幅退化）。
- student_v2（d_model 64→32，少量蒸馏）：15027 参数 / 4.55 ms / eval_mse=2.29728。
- moe_addon：2244 参数 / 0.19 ms（路由头外挂，状态-MSE 继承 full）。
- **推荐配置 = full**（剪枝/学生不重训精度损失 >5%，退回全量）；全量 pytest **715 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 0.5 通道剪枝不重训、d_model 减半学生均显著劣化 in-distribution 精度 => 二者保持 opt-in，
  不默认开启；延迟收益在小模型尺度不显著。

## [3.3.0.dev3] - 结构化剪枝 v2（通道/注意力头级，而非逐元素幅值）

### Added
- `udos/lite.py`：`StructuredPrunerV2(prune_ratio, head_dim=None)` —— 按 Linear 输出通道
  （权重一行）L2 范数排序，整行（含 bias）一起置零；`head_dim` 给定时按注意力头分组整头剪枝。
  提供 `prune/report/unprune/fine_tune`：剪枝后可 mask 冻结下少量微调恢复。
- `auto_student_config(cfg, scale)` 工具（d_model 减半=0.5 / 四分之一=0.25，带下限）。
- `tests/test_v33_prune.py`（6 例）：通道稀疏度、整行零通道、vs v1 逐元素剪枝对比、
  微调恢复、头分组、save/load 剪枝态、unprune 还原。

### Changed
- 版本号升至 `3.3.0.dev3`。剪枝保留形状（整行置零）以跨层安全；真正删通道重建形状列为被否决候选。

### Fixed
- 微调阶段对 requires_grad 叶子参数 in-place 重应用 mask 改用 `torch.no_grad`。

### Evidence（真实运行）
- 全量 pytest：**711 passed / 0 failed**（705 + 6 新）。
- 同 prune_ratio=0.4 下，v2 整通道零行数显著多于 v1（v1 逐元素很少整行为 0）。

### 被否决/降级候选（诚实保留）
- 真正删除输出通道并重建相邻层形状（cross-layer shape surgery）实现复杂、易错，本期保留为
  整行置零（保留形状）的结构化剪枝；不宣称已获得实测推理加速。

## [3.3.0.dev2] - 知识蒸馏 v2（温度软标签 + 中间特征匹配 + 学生自动架构）

### Added
- `udos/lite.py`：`DistillationTrainerV2(alpha, temperature, feature_weight, student_scale, lr)`
  —— 回归任务下温度 T 作用于 teacher 表示软化（软标签项 `T²·MSE(student, teacher/T)`，
  T=1 退化为 v1），加中间逐 tick 解码轨迹特征匹配损失；学生由 `auto_student_config`
  自动 d_model 减半/四分之一。
- `tests/test_v33_distill.py`（5 例）：蒸馏 loss 下降、温度可调（T=1 vs 8 权重不同）、
  学生自动架构与参数量 < teacher、vs v1 对比、save/load 学生。

### Changed
- 版本号升至 `3.3.0.dev2`。

### Fixed
- 无。

### Evidence（真实运行）
- 全量 pytest：**705 passed / 0 failed**（700 + 5 新）。
- 四分之一学生 `d_model=8`、半量学生 `d_model=16`，均小于 teacher `d_model=32`。

### 被否决/降级候选（诚实保留）
- 回归任务无 logits/softmax，"温度"为分类 KD 的类比实现（软化 teacher 表示），不宣称复现 KL 蒸馏。

## [3.3.0.dev1] - 轻量 MoE 任务路由（推理外挂，opt-in 默认关）

### Added
- `udos/moe.py`：`LightweightMoE(in_dim, out_dim, num_experts, top_k, seed)` ——
  N 个专家线性层（权重 batched 存为 `[E,in,out]`）+ 线性门控路由选 top-k 专家，
  softmax 加权混合专家输出。总参数量 = `E*(in*out+out)+in*E+E`，随专家数线性增长、完全可控。
- `tests/test_v33_moe.py`（6 例）：前向形状有限、门控确为 router logits 的 top-k、
  权重归一、参数量公式与可控、top_k=E 用全部专家、空输入/维度/非法 top_k 守卫、
  opt-in 默认关时 PhysicsPredictor 输出逐位不变。

### Changed
- 版本号升至 `3.3.0.dev1`。MoE 为独立推理外挂，不挂载进主模型，主预测路径零改动。

### Fixed
- 无。

### Evidence（真实运行）
- 全量 pytest：**700 passed / 0 failed**（694 + 6 新）。
- 示例 `LightweightMoE(16,8,num_experts=4,top_k=2)` 参数量 = 4*(16*8+8)+(16*4+4) = 644。

### 被否决/降级候选（诚实保留）
- MoE 仅验证路由机制可用；尚未接入正式训练口径，in-distribution 精度收益待 dev4 效率 Pareto A/B 判定。

## [3.3.0] - 3.3 线起点：CTM 架构精炼（残差+LayerNorm+可配置初始化）+ 正式训练

### Added
- `udos/ctm_engine.py` 增量三项 opt-in 架构能力（`CTMConfig`）：
  `residual=True`（循环单元残差 `new_act = f(prev) + prev`）、`act_norm=True`
  （每次更新后 `LayerNorm(d_model)`）、`init_mode ∈ {legacy,xavier,he}`（构造后重初始化全部
  `nn.Linear` 权重）。**默认全关**：`residual=False/act_norm=False/init_mode="legacy"`，
  守卫分支不执行，与旧架构逐位等价。
- 正式训练重建 `checkpoints/predictor_v3.3.0.pt` + `benchmarks/results/training_v3.3.0.json`
  （`scripts/build_v330_checkpoint.py`，Makefile `ckpt330`；默认配置=旧架构）。
- `tests/test_v33_arch.py`（7 例）：默认配置两次同种子 state_dict/前向逐位相等、
  显式关=缺省、残差/LayerNorm opt-in 前向、初始化可配置、新配置 save/load round-trip、轻量训练不崩溃。

### Changed
- 版本号升至 `3.3.0`；正式件主模型参数量仍为 **52191**（默认旧架构，未挂任何 opt-in）。
- `persistence.load_predictor` 经 `CTMConfig(**dict)` 重建，新字段带默认值，旧 16 件 checkpoint 逐位兼容。

### Fixed
- 无（opt-in 分支不触碰默认 RNG 消耗顺序与前向图）。

### Evidence（真实运行）
- 正式训练：seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0，
  `train_seconds=104.5`，`eval_mse=0.045556`（naive=0.163415，untrained=4.415056），
  `ece=0.056546`，coverage=0.8888，`batch_inference_max_diff=1.97e-06`。
- opt-in 离线快照：legacy 引擎 46136 参数，residual+act_norm+he 引擎 46264 参数（+128 = 2×64 LayerNorm），
  二者前向非逐位相同且全部有限。
- 全量 pytest：**694 passed / 0 failed**（687 旧 + 7 新）。

### 被否决/降级候选（诚实保留）
- 残差/LayerNorm/He 初始化均为 opt-in；in-distribution 精度提升未在正式件口径验证，
  正式件仍用旧架构默认，收益留待 dev 节点 A/B。

## [3.2.3] - 3.2 线终点：边界精修 + 文档对齐

### Added
- `tests/test_v323_edge.py`（10 例）：极端增强参数（179° 旋转 / 扰动 10 / 噪声 5）、
  时间缩放 clamp、非法构造参数、W=0 守卫、记忆超容量保留最近帧、reset 后 summary 报错、
  ICL 示例维度不匹配、未训练态 augment 可用但 icl 409、H=0 守卫、版本断言。

### Changed
- 版本号升至 `3.2.3`；README / ARCHITECTURE / DEPLOYMENT / ROADMAP 更新至 3.2 线。

### Fixed
- 边界精修：确认 TemporalMemory.falsy 语义（__len__ 空为 falsy）在所有集成点用 is None 判定。

### Evidence（真实运行）
- 全量 pytest 全绿（见下方收口）；checkpoints/predictor_v3.2.0.pt 可加载。
- 16 件 checkpoint 全量向后兼容。

### 被否决/降级候选（诚实保留）
- 3.2 线新能力均为推理外挂 opt-in；in-distribution 精度未因增强/ICL 提升，正式件口径不变。

## [3.2.2] - 集成加固 + backcompat 16 件 + 新服务端点 + 性能基准

### Added
- server 新增 `POST /augment/generate`（无状态 Ego360 增强，不需已训练模型）与
  `POST /icl/predict`（few-shot 上下文注入预测；未训练 409 / 非法 400）。
- `benchmarks/results/feature_latency_v3.2.0.json`：CPU(2线程) 性能基准。
- `tests/test_v322_service.py`（8 例）：backcompat v2.1.0..v3.2.0 共 **16 件**全量加载 predict、
  两个新端点正常/未训练/非法输入、延迟 JSON 结构。

### Changed
- 版本号升至 `3.2.2`；checkpoint 谱系扩至 16 件。

### Fixed
- 无（端点 opt-in，不改既有端点行为）。

### Evidence（真实运行, CPU 2 线程, n_iters=30）
- backcompat：16 件 checkpoint 全部可加载并 predict_next 输出有限。
- 延迟：predict=3.190ms，rollout(4)=12.281ms，augment=0.050ms，icl=3.259ms。

### 被否决/降级候选（诚实保留）
- augment/generate 为无状态工具端点；icl/predict 在未训练件上返回 409（不伪造预测）。

## [3.2.1] - 数据增强/记忆/ICL 与 Physical Loop 集成

### Added
- `PhysicalLoopRunner` 新增 opt-in 参数 `use_memory` / `memory` / `icl_examples`：
  - TemporalMemory 作为 loop 可选记忆（每次 run 末帧推入，下次 run 前插长上下文）；
  - ICL 作为 loop 上下文注入（icl_examples 拼接查询窗口）；
  - SyntheticEgoAugmenter 维持训练时 opt-in（正式件口径不变）。
- `tests/test_v32_integration.py`（5 例）：默认 loop 逐位等价 predict_next、
  memory 在 loop 内累积（run 1 次 buffered==1、run 2 次==2）、ICL 注入可运行、
  增强成对形状保持、B>1 守卫。

### Changed
- 版本号升至 `3.2.1`；**默认未挂外挂时 loop.prediction 与 predictor.predict_next 逐位一致**
  （`_integrated_predict` 默认委托旧路径）。

### Fixed
- 无（opt-in 集成，不改变默认行为）。

### Evidence（真实运行）
- 默认 loop 锚点 `torch.equal(loop.prediction, predictor.predict_next(X))` 严格为真；
  现有 v2.8 loop 测试 8 例全绿无回归。

### 被否决/降级候选（诚实保留）
- memory/ICL 集成路径仅支持 batch=1；B>1 显式报错不静默。

## [3.2.0.dev6] - 长时域 Rollout 与记忆协同

### Added
- `udos/longhorizon.py`：`LongHorizonRollout`（H>4 长时域递归预测；use_memory=True 时每步把
  预测状态推入 TemporalMemory 并前插历史帧累积上下文，缓解纯自回归误差累积；H=8/12/16）。
- `tests/test_v32_longhorizon.py`（6 例）：H=4/use_memory=False 逐位等价 predictor.rollout、
  H=8/12/16 形状与有限性、记忆累积 buffered==horizon、与 HierarchicalRollout 逐位一致、
  horizon=0 守卫、B>1 守卫。

### Changed
- 版本号内部升至 `3.2.0.dev6`；默认 use_memory=False 与旧 rollout 逐位一致。

### Fixed
- 修正 `self.memory or TemporalMemory(...)` 的 falsy 陷阱（TemporalMemory.__len__ 使空缓冲为
  falsy），改为 `is None` 判定，避免误新建记忆导致累积丢失。

### Evidence（真实运行, v3.1.0 正式件）
- H=4 锚点 `torch.equal(lh.rollout(w,4), predictor.rollout(w,4))` 严格为真。
- use_memory=True 跑 H=8 后记忆 buffered==8；与 hierarchical rollout 逐位一致。

### 被否决/降级候选（诚实保留）
- 记忆累积为状态帧前插代理；长时域误差是否下降需 H=16 多步 MSE 实测（接口已就绪，
  增益待 3.2.1 集成后如实复跑）。

## [3.2.0.dev5] - In-Context Learning（任务描述 + few-shot 示例）

### Added
- `udos/incontext.py`：`InContextLearner`（few-shot 示例窗口 + 任务描述向量 + 查询窗口沿
  时间维拼接为扩展上下文，经 ExtendedContextWindow 送入 predictor；任务描述前 4 维映射
  scene_params 通道；0/1/3-shot 可运行测量）。
- `tests/test_v32_icl.py`（7 例）：上下文拼接形状 [1,(k+1)W,6]、0-shot 退化为查询窗口、
  任务描述注入、0/1/3-shot MSE 有限、空示例/形状不一致/任务描述过短守卫。

### Changed
- 版本号内部升至 `3.2.0.dev5`；纯推理外挂、不改权重；0-shot 与旧路径一致。

### Fixed
- shot_mse 修正为逐样本取 scene_params（避免批维广播错位）。

### Evidence（真实运行, v3.1.0 正式件）
- 0-shot MSE=0.0464；1-shot=0.6686；3-shot=1.1076。
- **结论：在未训练 ICL 的冻结 ~52k 小模型上，拼接无关示例窗口引入额外 token 反而抬高误差，
  未观察到 in-context 收益**。

### 被否决/降级候选（诚实保留）
- InContextLearner 维持 opt-in；当前小模型无位置预训练、未在示例-查询对上训练，ICL 不成立。
  接口保留，供未来更大模型/多模态上下文复用。

## [3.2.0.dev4] - 数据增强 A/B（增强 vs 原始 + 类型消融）

### Added
- `scripts/ego_augment_ab_v32.py`：原始 vs 增强训练的 eval_mse / 噪声鲁棒 / OOD 鲁棒对比，
  叠加 view / perturb / noise / time_scale 单类型消融；落
  `benchmarks/results/ego_augment_ab_v3.2.0.json`。
- `tests/test_v32_augment_ab.py`（6 例）：A/B JSON 结构、四类消融齐全、opt-in 默认关、
  如实记录无提升、增强确定性可复现、版本断言。

### Changed
- 版本号内部升至 `3.2.0.dev4`；**增强保持 opt-in 默认关**，不改正式件训练口径。

### Fixed
- 无（离线测量 + 证据诚实记录）。

### Evidence（真实运行, quick n=12/epochs=12）
- raw eval_mse=0.4985 / noise_mse=0.5006 / ood_mse=17.8000；
  全量增强 eval_mse=1.8451（in-distribution 变差）/ ood_mse=3.3971（OOD 鲁棒大幅提升）。
- 单类型：仅 noise 增强 eval_mse=0.4101（略优于原始）；view/time_scale 显著抬高 in-distribution MSE。
- augment_mse_gain=-1.347（无 in-distribution 提升）=> **被否决为正式数据默认，保留 opt-in**。

### 被否决/降级候选（诚实保留）
- 全量几何增强破坏"仅沿 x 轴运动"的训练分布，in-distribution MSE 反升；仅 OOD/噪声鲁棒性受益。
  结论：增强维持 opt-in，不进正式件；纯噪声增强单独看略优，留待多种子复核。

## [3.2.0.dev3] - 时间记忆机制（滑动窗口 + 摘要）

### Added
- `udos/temporal_memory.py`：`TemporalMemory`（容量 C 的环形缓冲保存超出窗口的历史帧 +
  EMA 历史摘要向量；update/reset/summary/history；build_extended_window 把旧帧前插当前窗口）。
- `tests/test_v32_memory.py`（7 例）：缓冲增长与超容量淘汰最旧、EMA 摘要数值正确、
  长时域旧帧可查、reset 清空、与 ExtendedContextWindow/predictor 集成预测有限、
  k=0 逐位返回原窗口、空记忆/维度守卫。

### Changed
- 版本号内部升至 `3.2.0.dev3`；纯推理外挂、不改权重；未挂记忆时旧路径逐位不变。

### Fixed
- 无（新能力落地）。

### Evidence（真实运行）
- capacity=3 推入 5 帧后保留最近 3 帧（[2,3,4]）；EMA alpha=0.5 下 0->2 的摘要=1.0。
- 集成 ExtendedContextWindow 后扩展窗口 [10,6] 预测 [1,6] 有限。

### 被否决/降级候选（诚实保留）
- 记忆为状态帧缓存 + EMA 代理，未重训；长时域增益待 dev6 LongHorizon 验证。

## [3.2.0.dev2] - 上下文窗口扩展（更长历史）

### Added
- `udos/extended_context.py`：`ExtendedContextWindow`（在已训练 predictor 上支持
  window>6 长历史输入；正弦位置编码扩展 opt-in、历史截断为最近 max_len 帧、
  首帧重复填充；不修改主模型权重）。
- `tests/test_v32_context.py`（6 例）：PE 表形状、W=6 默认逐位等价锚点（torch.equal）、
  W=12/24 长窗口可推理且输出有限、截断保留最近帧、填充尾部对齐、空窗口/W=0 守卫。

### Changed
- 版本号内部升至 `3.2.0.dev2`；**默认 window=6 与旧路径逐位一致**（use_pe=False 且
  窗口恰为 6 时直接委托 predict_next）。

### Fixed
- 无（新能力落地）。

### Evidence（真实运行）
- W=6 锚点：`torch.equal(ecw.predict_next(X), predictor.predict_next(X))` 严格为真。
- W=12/24（use_pe=True）输出有限、量级与 W=6 同阶（未爆炸）。

### 被否决/降级候选（诚实保留）
- 位置编码为外挂加法、未重训，W>6 的精度增益不保证（仅验证可运行性与有限性）；默认仍 W=6。

## [3.2.0.dev1] - 多视角合成数据生成与视图一致性

### Added
- `udos/ego_data.py`：`MultiViewGenerator`（同一场景参数化状态序列，按已知 xy 平面
  刚体旋转 R_k 生成 K 个视角；视图间对应关系矩阵 C[k,j]=R_j R_k^T）。
- `tests/test_v32_multiview.py`（7 例）：多视图形状 [K,T,6]、第 0 视角逐位等于基准、
  视图变换可逆（R_k^T R_k=I）、跨视图一致性误差 ~0、自对应矩阵为 I、越界/空场景守卫。

### Changed
- 版本号内部升至 `3.2.0.dev1`；主模型与默认路径不变。

### Fixed
- 无（新能力落地）。

### Evidence（真实运行）
- 四视角 45° 步长下 invertibility_error 与 cross_view_consistency 均 <1e-5（解析已知变换）。
- 第 0 视角（angle=0）与基准序列逐位一致。

### 被否决/降级候选（诚实保留）
- 多视图为纯几何代理，不涉及真实多目相机标定；视图一致性仅在合成刚体变换上验证。

## [3.2.0] - 3.2 线起点：Ego360 启发合成多视角数据增强 + 正式训练

### Added
- `udos/ego_data.py`：`SyntheticEgoAugmenter`（受 Ego360 启发，合成参数化状态序列上
  做视角 xy 旋转/平移、轨迹扰动、高斯噪声注入、时间缩放；analogy, not reproduction）。
- `scripts/build_v320_checkpoint.py` + Makefile `ckpt320`：正式训练重建
  `checkpoints/predictor_v3.2.0.pt` + `benchmarks/results/training_v3.2.0.json`。
- `tests/test_v32_augment.py`（9 例）：输出形状、旋转正交可逆、参数=0 逐位不变、
  扰动可控、(X,Y) 成对标签对齐、时间缩放形状保持、空数据集/非法参数守卫。

### Changed
- 版本号对外升至 `3.2.0`；主模型 52191 参数与默认路径不变。正式件仍用**原始数据**训练
  （增强为训练时 opt-in，不进正式件口径，见设计约束 2）。

### Fixed
- 无（新能力落地）。

### Evidence（真实运行）
- 正式训练同口径 seed=48/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0；
  **52191 参数，eval_mse=0.045556，naive_mse=0.163415，untrained_mse=4.415056，
  best_epoch=57，train_seconds=106.6，coverage=0.8888，calibrated_ece=0.056546**。
- ego 离线快照：view_roundtrip_max_abs_err=0.0（刚体变换严格可逆），augment_mean_deviation=0.1791。
- reload 逐位一致（single_step_mse 差 <1e-9，batch vs 逐笔 max_diff <1e-5）。

### 被否决/降级候选（诚实保留）
- 数据增强仅 opt-in；正式件未采用增强训练（收益待 dev4 A/B 验证，见后续条目）。

## [3.1.3] - 3.1 线终点：边界精修 + 文档对齐

### Added
- `tests/test_v313_edge.py`（9 例）：码本大小=1 退化、空 token 序列、极端动作值（1e6）、
  codec 粗级=1、服务未训练态 409、in-context 超长示例、越界历史守卫、单 token 解码形状。
- `docs/ROADMAP.md` / `ARCHITECTURE.md` / `DEPLOYMENT.md` / `README.md` 对齐至 3.1 线
  （ActionPiece tokenizer/codec/predictor/decoder/curriculum/prompter + 两端点 + 15 件 backcompat）。

### Changed
- 版本号对外升至 `3.1.3`（3.1 线终点）；主模型 52191 参数与默认路径不变。

### Fixed
- 无（边界补漏 + 文档收尾）。

### Evidence（真实运行）
- 9 例边界全绿；15 件 checkpoint 全量可加载；两端点 200/400/409。
- 全量回归：**607 -> 616 passed / 0 failed**（详见最终报告总数与覆盖率）。

### 被否决/降级候选（诚实保留）
- 3.1 全部能力维持 opt-in 默认关；课程学习无显著增益、tokenized 未替主 CTM 均如实记录。

## [3.1.2] - 集成加固 + backcompat 15 件 + /action/tokenize·detokenize + 延迟基准

### Added
- `udos/server.py`：新增 `POST /action/tokenize`（连续动作 -> token）与
  `POST /action/detokenize`（token -> 连续动作）；懒构造合成动作 k-means tokenizer；
  未训练 409、非法输入 400。
- `scripts/benchmark_v31_features.py`：predict_next / action_tokenize /
  action_detokenize / token_next_token 延迟基准，落
  `benchmarks/results/feature_latency_v3.1.0.json`。
- `tests/test_v312_service.py`（7 例）：两端点 200/400/409、detokenize 往返、
  backcompat 15 件全加载、latency JSON 落盘可复算。
- backcompat 扩到 v2.1.0..v3.1.0 共 **15 件**（test_v30_integration 改为
  `test_backcompat_15_checkpoints_load`）。

### Changed
- 版本号对外升至 `3.1.2`；主模型 52191 参数与默认路径不变。

### Fixed
- 无（集成加固 + 新端点）。

### Evidence（真实运行）
- 延迟基准（N=50）：predict_next p50=3.17ms；action_tokenize p50=0.174ms；
  action_detokenize p50=0.016ms；token_next_token p50=0.014ms。
- 15 件 checkpoint 全部可加载并产出有限预测；两端点 200/400/409 符合契约。
- 全量回归：**600 -> 607 passed / 0 failed**（新增 7 例）。

### 被否决/降级候选（诚实保留）
- 两端点不挂载旧 /predict 默认路径；tokenizer 为合成数据懒构造，非真机动作码本。

## [3.1.1] - ActionPiece 与 Physical Loop 集成（opt-in，默认路径逐位不变）

### Added
- `udos/physical_loop.py`：`PhysicalLoopRunner` 新增 opt-in 参数
  `use_tokenized / tokenizer / token_predictor / action_examples`；开启后
  `predict_action` 步骤用 token_predictor 自回归预测下一个 token、经 tokenizer
  解码为连续动作扰动，跨 run 累积 `token_history`；in-context 示例经
  `action_examples` 注入。**默认 use_tokenized=False 时走原 MPC 路径，逐位不变**。
- `tests/test_v31_integration.py`（6 例）：默认路径逐位一致、tokenized 路径可跑且有限、
  缺 tokenizer 守卫、与 2.8-3.0 特性组合不污染主路径、token_history 累积。

### Changed
- 版本号对外升至 `3.1.1`；主模型 52191 参数与默认 `predict_next`/loop 路径不变。

### Fixed
- 无（纯 opt-in 集成）。

### Evidence（真实运行）
- 默认 loop.run 输出与 `predictor.predict_next` **torch.equal 逐位一致**；开启
  use_tokenized 后两次 run 累积 token_history>=3，预测扰动有限、不回改主权重。
- 全量回归：**594 -> 600 passed / 0 failed**（新增 6 例）。

### 被否决/降级候选（诚实保留）
- tokenized 路径仅 opt-in 暴露，不替换 MPC；in-context 注入仍为外挂。

## [3.1.0.dev6] - In-context action prompting（few-shot 示例轨迹）

### Added
- `udos/action_piece.py`：`InContextActionPrompter`（few-shot 示例轨迹 + 查询状态
  拼接上下文，在示例上即时建 n-gram，对查询后缀预测下一个 token；
  `build_context()` / `predict_next()` / `few_shot_accuracy(n_shots)`）。
  **analogy, not reproduction** —— 受 PhysBrain 长上下文 / in-context learning 启发，
  在合成 token 序列上做轻量类比，不预训练大模型。
- `tests/test_v31_prompting.py`（6 例）：拼接正确、示例驱动预测、1/3/5-shot 单调不下降、
  空示例/空查询/非法 shot 守卫、与 TokenizedActionPredictor 接口一致。

### Changed
- 版本号内部升至 `3.1.0.dev6`；主模型与默认路径不变（推理外挂）。

### Fixed
- 无（纯新增）。

### Evidence（真实运行）
- 合成周期 token 任务（vocab=8）：**1-shot=0.725, 3-shot=0.975, 5-shot=1.000**
  （单调上升，远超随机基线 0.125；更多示例 -> n-gram 计数更完整）。
- 全量回归：**588 -> 594 passed / 0 failed**（新增 6 例）。

### 被否决/降级候选（诚实保留）
- in-context prompting 仅作 loop 上下文注入 opt-in（见 3.1.1），不替换主 CTM；
  few-shot 精度提升来自合成任务强结构，不外推真机长上下文。

## [3.1.0.dev5] - 序列课程学习（easy->hard 纯数据调度）

### Added
- `udos/action_piece.py`：`SequenceCurriculum`（按轨迹复杂度/长度排序，
  `easy_to_hard()` / `random_order()` / `split_easy_hard(frac)`；
  `convergence_ab()` 在合成 next-token 任务上对比课程 vs 随机顺序）。
  **纯数据调度，不改模型权重、不改正式件训练口径**。
- `tests/test_v31_curriculum.py`（7 例）：排序正确、随机序保集合、easy/hard 切分、
  空集守卫、课程不劣于随机、纯调度不改输入张量。

### Changed
- 版本号内部升至 `3.1.0.dev5`；主模型与正式件训练口径不变。

### Fixed
- `_train_epoch` 内 `float(loss)` 改为 `float(loss.detach())`，消除 grad-tensor 转标量警告。

### Evidence（真实运行）
- 合成 next-token 任务（vocab=8, 12 epochs）：**课程 hard 精度=0.6258，随机=0.6258**；
  课程不劣于随机（curriculum_better=True）。
- 全量回归：**581 -> 588 passed / 0 failed**（新增 7 例）。

### 被否决/降级候选（诚实保留）
- **课程学习在本线性小任务上未观察到显著收敛速度/最终精度增益**（与随机持平），
  照实记录；课程排序工具保留为 opt-in 纯数据调度，不接入正式件训练。

## [3.1.0.dev4] - Tokenized vs 连续动作 A/B（码本大小 8/16/32/64 扫描）

### Added
- `scripts/action_piece_ab_v31.py`：在合成动作上对比 tokenized 量化 vs 连续均值基线，
  扫描 K∈{8,16,32,64} 的 quant_mse / 推理延迟 / 码本参数量，落
  `benchmarks/results/action_piece_ab_v3.1.0.json`，含 `verdict` 与 `opt_in_default_off`。
- `tests/test_v31_ab.py`（4 例）：A/B JSON 可复跑、四档扫描单调不增、opt-in 默认关时
  主路径逐位一致、被否决/降级候选在 verdict 保留。

### Changed
- 版本号内部升至 `3.1.0.dev4`；主 predictor.predict_next 默认路径不变（外挂 opt-in）。

### Fixed
- 无（纯 A/B 评测）。

### Evidence（真实运行）
- 合成 6D 动作（6 簇）：连续均值基线 MSE=6.14；tokenized 量化 MSE 随 K 递减
  **K=8:0.06325, K=16:0.05280, K=32:0.04095, K=64:0.02887**（相对基线低约 100×），
  encode+decode 延迟 ≈1.75ms，码本参数 48~384；best=K=64。
- 诚实判定：tokenized 量化在合成动作上确有 MSE 收益，但**仍维持 opt-in 默认关**
  （不替换主 CTM，不改变旧默认输出）。
- 全量回归：**577 -> 581 passed / 0 failed**（新增 4 例）。

### 被否决/降级候选（诚实保留）
- 未把 tokenized 设为默认动作路径；码本大小 K=64 仅为离线最优，生产默认仍需 dev6/
  集成节点权衡；A/B 仅在合成动作上成立，不外推真机。

## [3.1.0.dev3] - token→连续动作解码器与线性插值平滑 + retargeting 集成

### Added
- `udos/action_piece.py`：`TokenActionDecoder`（硬解码 = 逐 token 码本中心分段常数；
  `smooth_decode()` 相邻 token 间线性插值平滑；`continuity()` 步间跳变度量；
  可选关节限位 `joint_limits` 后处理 clamp；`within_limits()` 校验）。
- 与 `ActionRetargeter` 集成：解码出的源形态动作可直接喂给 retarget。
- `tests/test_v31_decoder.py`（8 例）：硬解码形状/有限、平滑降跳变、interp_steps=1
  等价硬解码、限位 clamp、retargeting 接口一致、未拟合/非法限位守卫。

### Changed
- 版本号内部升至 `3.1.0.dev3`；主模型与默认路径不变（推理外挂）。

### Fixed
- 无（纯新增）。

### Evidence（真实运行）
- 合成 6D 动作（K=8）：硬解码步间跳变=2.405，interp_steps=4 平滑后=0.6013
  （**4.0× 更连续**）；紧限位 [-0.5,0.5] 下输出严格落在限位内；retarget 到 4 自由度
  后输出形状 [5,4] 且在目标限位内。
- 全量回归：**569 -> 577 passed / 0 failed**（新增 8 例）。

### 被否决/降级候选（诚实保留）
- 平滑解码为离线演示；是否替代硬解码由 dev4 A/B 决定，不挂入主控制回路。

## [3.1.0.dev2] - 自回归 next-token 动作预测 (n-gram 推理外挂)

### Added
- `udos/action_piece.py`：`TokenizedActionPredictor`（n-gram 计数模型，给定历史 token
  序列自回归预测下一个 token；短历史向低阶回退；`generate()` 自回归生成；
  `heldout_accuracy()`；`decode_to_actions()` 委托 tokenizer 把 token 解码回连续动作）。
- `markov_token_sequences()`：合成准周期 token 序列（next≈(prev+step) mod vocab 以
  stickiness 概率），供 next-token 验证。
- `tests/test_v31_token_predict.py`（9 例）：精度优于随机、logits 形状/非负、
  generate 长度与范围、token→连续解码、低阶回退、空/未拟合/越界守卫。

### Changed
- 版本号内部升至 `3.1.0.dev2`；主 CTM 不变（token 预测为推理外挂，不替换主模型）。

### Fixed
- 无（纯新增）。

### Evidence（真实运行）
- 合成 Markov token 序列（vocab=8, order=2, stickiness=0.9）：
  **held-out next-token 精度=0.9177，随机基线=0.125**（7.3× 随机）。
- 全量回归：**560 -> 569 passed / 0 failed**（新增 9 例）。

### 被否决/降级候选（诚实保留）
- n-gram 仅验证序列建模可行性，不接入主 CTM；模型仍为纯计数无权重外挂。

## [3.1.0.dev1] - 多尺度粗+细两级码本与码本质量指标

### Added
- `udos/action_piece.py`：`ActionPieceCodec`（粗粒度码本 + 细粒度残差码本两级量化；
  encode 输出 (coarse_ids, fine_ids)，decode = 粗中心 + 残差中心；`quality()` 给出
  two_level_mse / coarse_mse / 粗&细困惑度 / 粗&细利用率 / gain_vs_coarse）。
- `token_perplexity()`：离散 token 分布困惑度 = exp(熵)（均匀=n_vocab，退化=1）。
- 三种初始化策略 kmeans++/random/uniform_grid 在 codec 两级分别可配。
- `tests/test_v31_codec.py`（8 例）：两级误差 <= 粗级、利用率/困惑度区间、
  三初始化对比、encode/decode 形状、save/load、未拟合/长度守卫。

### Changed
- 版本号内部升至 `3.1.0.dev1`；主模型 52191 参数与默认路径不变（推理外挂）。

### Fixed
- 无（纯新增）。

### Evidence（真实运行）
- 合成 6D 动作（4 簇高斯混合）：两级量化 two_level_mse < coarse_mse（残差细化有效），
  粗/细利用率 > 0，困惑度落在 (1, n_vocab]；三初始化策略均拟合成功且往返 MSE>=0。
- 全量回归：**552 -> 560 passed / 0 failed**（新增 8 例）。

### 被否决/降级候选（诚实保留）
- 两级码本仅为离线验证，不挂入主训练；最终码本大小与是否替代单级待 dev4 A/B。

## [3.1.0] - 3.1 线起点：ActionPiece 离散动作 token 化 + 正式训练重建

### Added
- `udos/action_piece.py`：`ActionPieceTokenizer`（k-means 码本把连续动作向量量化为
  离散 token；k-means++/random/uniform_grid 初始化；encode/decode、码本学习、
  token 覆盖率/利用率、往返误差、state_dict save/load；空/越界/未拟合守卫）。
  **analogy, not reproduction** —— 受 PhysBrain ActionPiece 启发，仅在合成动作向量上验证。
- `scripts/build_v310_checkpoint.py`：在 v3.0.3 同口径上重建正式件
  `checkpoints/predictor_v3.1.0.pt` + `benchmarks/results/training_v3.1.0.json`，
  离线快照新增 `features_v31_actionpiece`（用独立集相邻状态差作动作代理拟合 16 码本）。
- Makefile 新增 `ckpt310` target。
- `tests/test_v31_tokenizer.py`（10 例）：码本收敛、往返误差有界、覆盖率非平凡、
  确定性 seed、空/未拟合/越界/维度守卫、save/load 往返。

### Changed
- 版本号对外升至 `3.1.0`；主模型仍为 **52191 参数**（ActionPiece 为推理时外挂，
  不参与主训练，默认路径不变）。

### Fixed
- 无（新能力首节点）。

### Evidence（真实运行）
- 正式训练（seed=42 / n_per_kind=48 / epochs=60 / patience=12 / front / hybrid_weight=0）：
  **eval_mse=0.045556，final_loss=0.059488，best_epoch=57，train_seconds=110.2，
  coverage=0.8888，ECE=0.056546，n_params=52191**；reload 后逐位一致（batch max diff=1.97e-6），
  `udos_version` 烘焙为 3.1.0。
- ActionPiece 码本（合成动作，K=16）：**roundtrip_mse=0.008166，utilization=1.0，
  16 个 token 全部用到，inertia_final=0.0490**。
- 五维（UDOS 内部基准）：composite=87.86（与 v3.0.3 逐位一致，主模型未变）。
- 全量回归：**542 -> 552 passed / 0 failed**（新增 10 例）。

### 被否决/降级候选（诚实保留）
- ActionPiece 不替换主 CTM，仅作推理外挂；默认路径逐位不变；码本大小 K=16 为
  离线快照值，最终 A/B 收益待 3.1.0.dev4 节点验证（不稳则 opt-in）。

## [3.0.3] - 3.0 线终点：最终训练重建 + 全特性离线 A/B + backcompat 14 件

### Added
- `scripts/build_v303_checkpoint.py`：在 v3.0.0 口径上重建最终正式件
  `checkpoints/predictor_v3.0.3.pt` + `benchmarks/results/training_v3.0.3.json`，
  离线快照包含全部新特性：features_v27（policy/active/hier）、features_v28（loop）、
  features_v29（retarget）、features_v30_multimodal（rgb/depth/mask 形状）、
  features_v30_eval5d（五维+composite，UDOS 内部基准），并附
  `new_features_offline_ab` 段（multitask/retarget/affordance/future_multimodal/eval5d）。
- Makefile 新增 `ckpt303` target。
- backcompat 扩到 v2.1.0..v3.0.3 共 **14 件**（test_v30_integration 改
  `test_backcompat_14_checkpoints_load`）。

### Changed
- 版本号对外升至 `3.0.3`（3.0 线终点）；主模型仍为 **52191 参数**（所有 3.0 特性均为
  推理时外挂，不参与训练）。

### Fixed
- 无（收尾件，仅训练重建 + 回归 + 文档）。

### Evidence（真实运行）
- 正式训练（seed=42 / n_per_kind=48 / epochs=60 / patience=12 / front / hybrid_weight=0）：
  **eval_mse=0.045556，final_loss=0.059488，best_epoch=57，train_seconds=108.1，
  coverage=0.8888，ECE=0.056546，n_params=52191**；reload 后逐位一致（max diff<1e-9），
  `udos_version` 烘焙为 3.0.3。
- 五维（UDOS 内部基准，非 PhysBrain）：visual_spatial=96.61, multiview=49.34,
  embodied_planning=100, spatial_affordance=100, visual_trajectory=93.37, composite=87.86。
- 全量回归：**542 passed / 0 failed**；覆盖率 TOTAL 4457 stmts / 338 miss = **92%**。
- 两正式件 predictor_v3.0.0.pt 与 predictor_v3.0.3.pt 均存在且可加载。

### 被否决/降级候选（诚实保留）
- 默认开启多模态未来头被否决（无状态 MSE 增益、2.29× 参数开销）；一致性损失挂入主
  训练被否决（hybrid_weight=0 不变）；五维分数不随版本单调提升如实记录；所有 3.0 能力
  维持 opt-in 默认关。

## [3.0.2] - 边缘精修 + 文档对齐（3.0 线补丁件）

### Added
- `tests/test_v302_edge.py`（8 例）：multimodal H=0 守卫、mask_dim=0 守卫、全模态关返回空、
  eval_suite 零维度/缺分数守卫、五维权重和≠1 自动归一化（含权重和=0 守卫）、
  alignment_loss NaN 防护与错误形状守卫、服务端点未训练态 409。

### Changed
- `docs/ROADMAP.md` / `ARCHITECTURE.md` / `DEPLOYMENT.md` / `README.md` 对齐至 3.0 线
  （未来多模态预测 + UDOS 内部五维评测 + 两端点）；版本号对外升至 `3.0.2`。

### Fixed
- `FiveDimensionEvaluator.composite_score` 增加分数缺维度守卫（此前只查权重，空分数会
  KeyError）；零方差/常数输入的相关性 NaN 经 `nan_to_num` 退化为 0。

### Evidence（真实运行）
- 全量回归：**534 -> 542 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 3.0 全部新能力维持 opt-in（推理时外挂，不改旧 predict_next 默认路径）。

## [3.0.1] - 3.0 集成加固 / backcompat 13 件 / 服务端点 / 延迟基准

### Added
- `udos/server.py`：`POST /future/predict`（多模态未来预测，返回 rgb/depth/mask
  形状与值）与 `GET /eval/5d`（UDOS 内部五维评测 + composite，显式标注非 PhysBrain）；
  未训练 409、非法输入 400。
- `scripts/benchmark_v30_features.py`：predict_next / future_multimodal_head /
  RGB/Depth/Mask 独立头 / eval5d_dim1 延迟基准，落
  `benchmarks/results/feature_latency_v3.0.0.json`。
- `tests/test_v30_integration.py`（8 例）：全特性组合默认路径逐位一致、
  backcompat 13 件全加载、两端点 200/400/409、latency JSON 落盘。

### Changed
- backcompat 覆盖扩到 v2.1.0..v3.0.0 共 **13 件**；版本号对外升至 `3.0.1`。

### Fixed
- 无（纯集成 + 新端点）。

### Evidence（真实运行）
- 延迟基准（N=50）：predict_next p50=3.27ms；future_multimodal_head p50=0.18ms；
  rgb/depth/mask 独立头 p50≈0.01ms；eval5d_dim1 p50=30.5ms（含一次前向评测）。
- 全量回归：**526 -> 534 passed / 0 failed**（新增 8 例）。

### 被否决/降级候选（诚实保留）
- 两新端点默认不挂载旧 /predict 路径；/eval/5d 分数为 UDOS 内部基准。

## [3.0.0.dev6] - 综合均分 + 四代五维演化

### Added
- `FiveDimensionEvaluator.composite_score(scores, weights=None)`：五维加权平均，
  权重可配置、自动归一化（和不必为 1）；缺维度/权重和<=0 抛 ValueError。
- `scripts/eval5d_evolution.py`：对 v2.7.3/v2.8.0/v2.9.0/v3.0.0 四代正式件跑五维评测，
  落 `benchmarks/results/five_dim_evolution_v3.0.0.json`；Makefile 新增 `eval5d`。
- `tests/test_v30_eval_evolution.py`（5 例）：等权/自定义权重 composite 正确、
  非法权重守卫、四代 JSON 落盘且分维度在 [0,100]。

### Changed
- 版本号升至 `3.0.0.dev6`。

### Fixed
- 无。

### Evidence（真实运行）
- 四代五维 composite（seed=2025）：v2.7.3=87.86, v2.8.0=87.86, v2.9.0=87.86,
  v3.0.0=87.86（核心 predictor 同口径 52191 参数，代理指标稳定；外挂特性不改变主预测）。
- 全量回归：**521 -> 526 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 五维分数未随版本单调提升（外挂特性不改主预测器），如实记录，不伪造增长。

## [3.0.0.dev5] - 五维评测套件（UDOS 内部基准）

### Added
- `udos/eval_suite.py`：`FiveDimensionEvaluator`（**UDOS 内部基准, 非 PhysBrain
  榜单分数**，metadata 显式标注）。五维均 0-100：①视觉空间感知代理=状态重建精度；
  ②多视角空间理解代理=scene_params 辨识精度；③具身认知与规划代理=policy 动作优选；
  ④空间指向与可供性代理=affordance 打分准确率；⑤视觉轨迹推理代理=rollout 轨迹精度。
  纯前向、确定性、不修改 predictor。
- `tests/test_v30_eval_suite.py`（7 例）：五维分数 [0,100]、算术均分、空模型守卫、
  分数可复现、显式标注非 PhysBrain。

### Changed
- 版本号升至 `3.0.0.dev5`。

### Fixed
- dim3 policy 奖励参考切片到 MPC horizon=2（与 rollout 形状 [1,2,6] 对齐）。

### Evidence（真实运行）
- v3.0.0 五维（seed=2025）：visual_spatial=96.61, multiview_spatial=49.34,
  embodied_planning=100.0, spatial_affordance=100.0, visual_trajectory=93.37。
- 全量回归：**514 -> 521 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 五维分数为 UDOS 内部合成代理指标, 与 PhysBrain 外部数字严格分开, 不外推。

## [3.0.0.dev4] - 未来预测 A/B 证据

### Added
- `scripts/future_multimodal_ab_v30.py`：多模态未来头（三模态联合）vs 单模态未来头
  （仅状态）的 eval_mse / 延迟 / 参数对比，落
  `benchmarks/results/future_multimodal_ab_v3.0.0.json`；含一致性损失开 vs 关段。
- `tests/test_v30_future_ab.py`（4 例）：A/B JSON 落盘可复算、默认 opt-in 关旧路径
  逐位一致、一致性损失不改变推理输出、被否决候选保留。

### Changed
- 版本号升至 `3.0.0.dev4`。

### Fixed
- 无。

### Evidence（真实运行）
- A/B（N=50）：单模态 FutureStateHead **924 参数**，p50=0.017ms；多模态
  FutureMultimodalHead **2112 参数**（参数开销 **2.29x**），p50=0.044ms；多模态不输出
  状态（state_mse=n/a），一致性损失在代理上=1.62 且不改变推理输出。
- 全量回归：**510 -> 514 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- **否决**默认开启多模态未来头（无状态预测 MSE 增益，仅增参/增延迟）；
- **否决**将一致性损失挂入正式训练主损失（训练口径 hybrid_weight=0 不变）。

## [3.0.0.dev3] - 跨模态空间对齐一致性损失

### Added
- `udos/future_multimodal.py`：`CrossModalAlignmentLoss`（参数-free）约束三模态在
  同一时间步的空间一致性——RGB 统计与深度排序的 Pearson 相关、mask 与 |RGB 活跃|
  的相关，损失 = (1-corr)。作为 multitask 训练的 opt-in 辅助损失，**推理不计算**；
  含 NaN/零方差防护（退化相关=0）。**analogy, not reproduction**。
- `tests/test_v30_alignment.py`（6 例）：损失有限非负、已知投影关系下相关数据损失
  < 打乱数据、小投影几步优化损失确有下降、常数输入 NaN 防护、推理不返回损失。

### Changed
- 版本号升至 `3.0.0.dev3`；不进入正式件训练口径（3.0.0 主模型仍 52191 参数）。

### Fixed
- 零方差/常数列导致的相关性 NaN：`nan_to_num` 退化为 0，损失有界。

### Evidence（真实运行）
- 新增 6 例；全量回归：**504 -> 510 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 一致性损失为 opt-in 辅助项，未挂入正式训练（主训练口径 hybrid_weight=0 不变）。

## [3.0.0.dev2] - 对象 mask 代理头

### Added
- `udos/future_multimodal.py`：新增 `MaskProxyHead`（latent 线性投影 + 可选
  scene_params 阈值偏置，经 sigmoid 输出软 mask `[B,H,mask_dim=4]∈[0,1]`）；
  `FutureMultimodalHead` 改为委托该头，并支持直接调用时传入 scene_params。
  mask 与 RGB/深度共享同一 H 步对齐。**analogy, not reproduction**。
- `tests/test_v30_mask.py`（7 例）：mask 形状/值域 [0,1]、与 RGB/深度 H 对齐、
  空场景守卫、scene 阈值偏置单调生效、头单独开关、mask_dim=0 守卫。

### Changed
- `FutureMultimodalHead.forward(latent, scene_params=None)` 增加可选 scene_params
  （经 MultiTaskHead 注册时仍以 head(latent) 调用，行为不变）；mask 输出由原始投影
  改为 [0,1] 软 mask（形状逐位一致）；版本号升至 `3.0.0.dev2`。

### Fixed
- 无。

### Evidence（真实运行）
- 新增 7 例；全量回归：**497 -> 504 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- mask 为未训练随机投影 + 阈值偏置的软 mask，不对应真实实例分割。

## [3.0.0.dev1] - RGB 代理头 + 深度代理头细化

### Added
- `udos/future_multimodal.py`：新增 `RGBProxyHead`（latent->[B,H,rgb_dim=8]，8 通道
  代理 [mean_r,mean_g,mean_b,var,hist_bin0..3] 颜色统计）与 `DepthProxyHead`
  （latent->[B,H,depth_dim=4]，4 通道代理 [rel_dist,depth_grad,near_ratio,far_ratio]
  深度排序）；`FutureMultimodalHead` 改为内部委托这两个独立头。两头独立 MSE 损失、
  可联合推理。**analogy, not reproduction**。
- `tests/test_v30_rgb_depth.py`（7 例）：RGB/深度头形状有限、仿射叠加性（投影线性性
  /非退化）、独立损失与联合推理、不污染旧 predictor、非法维度守卫。

### Changed
- `FutureMultimodalHead` 的 RGB/深度投影重构为独立可复用头；对外输出形状与节点 21
  逐位一致（仍 [B,H,8]/[B,H,4]）；版本号升至 `3.0.0.dev1`。

### Fixed
- 无（纯重构 + 新增独立头，不改旧默认输出）。

### Evidence（真实运行）
- 新增 7 例；全量回归：**490 -> 497 passed / 0 failed**。

### 被否决/降级候选（诚实保留）
- 两头发起时为未训练随机线性投影，仅验证形状/线性性/隔离；不宣称有真实语义。

## [3.0.0] - 未来状态多模态代理目标 + 正式训练重建（3.0 线起点）

### Added
- `udos/future_multimodal.py`：`FutureMultimodalHead`（multitask 注册新头），从共享
  latent 输出未来 H 步三模态低维代理：RGB 代理 `[B,H,rgb_dim=8]`、深度代理
  `[B,H,depth_dim=4]`、对象 mask 代理 `[B,H,mask_dim=4]`；三模态共享 backbone，
  各模态 `use_rgb/use_depth/use_mask` 独立可开关。**analogy, not reproduction**：
  不碰真实 RGBD 图像，用状态向量的不同线性投影代理三种模态。
- 正式训练重建 `checkpoints/predictor_v3.0.0.pt` +
  `benchmarks/results/training_v3.0.0.json`（含 features_v30 离线快照）。
- `scripts/build_v300_checkpoint.py`（`--quick` 小规模复现）；Makefile 新增 `ckpt300`。
- `tests/test_v30_multimodal.py`（7 例）：三模态形状有限、共享 backbone 无梯度冲突、
  模态独立开关、默认关旧路径逐位一致、config 往返、非法 horizon 守卫。

### Changed
- 版本号对外升至 `3.0.0`；训练口径同 v2.9.0（seed=42/n_per_kind=48/epochs=60/
  patience=12/front/hybrid_weight=0），主模型参数量仍为 **52191**。

### Fixed
- 无（纯新增推理时外挂头，不改旧默认输出路径）。

### Evidence（真实运行）
- 正式训练：**52191 参数**，60 epochs（best_epoch=57），train_seconds=110.2，
  final_loss=0.059488，eval_mse=**0.045556**，coverage=0.8888，ECE=0.056546。
- 全量回归：**483 -> 490 passed / 0 failed**（新增 7 例）。
- checkpoints/predictor_v3.0.0.pt 可加载（udos_version=3.0.0，reload 一致）。

### 被否决/降级候选（诚实保留）
- FutureMultimodalHead 默认 **opt-in**（MultiTaskHead.enable 默认关），不改旧
  predict_next 路径；三模态为低维线性投影代理，不宣称复现 VLM/RGBD 预训练。

## [2.9.3] - 边缘精修 + 文档对齐（2.9 线终点补丁件）

### Added
- `tests/test_v293_edge.py`（5 例）：retargeting dof=0 占位形态不可重定向、affordance
  零物体 (N_parts=0) 守卫、spatial_relation 单物体 (n_obj<2) 守卫、频率比极端值
  （100x 过采样 / 0.01x 欠采样）端点一致且有限、服务端点未训练态 409。

### Changed
- `docs/ROADMAP.md` / `ARCHITECTURE.md` / `DEPLOYMENT.md` / `README.md` 对齐至 2.9 线
  （形态无关动作重定向 + 空间可供性 + 两端点）；版本号对外升至 `2.9.3`。

### Fixed
- 收紧失败路径：dof=0 / 零物体 / 单物体 / 极端频率比均显式或安全退化，不静默 inf/越界。

### Evidence（真实运行）
- 全量回归：**478 -> 483 passed / 0 failed**；覆盖率 **92%**（TOTAL 4196 stmts / 329 miss）。
- checkpoints/predictor_v2.9.0.pt 可加载（udos_version=2.9.0，52191 参数，eval_mse 0.0456）。

### 被否决/降级候选（诚实保留）
- 2.9 全部新能力维持 opt-in（推理时外挂，不改旧 predict_next 默认路径）。

## [2.9.2] - 集成加固: 2.9 特性组合 / 服务端点 / backcompat 12 件 / 延迟基准

### Added
- `udos/server.py`：`POST /retarget/convert`（动作重定向，支持预设名或显式形态 dict）
  与 `POST /affordance/score`（可供性打分）；均未训练 409、非法输入 400。
- `scripts/benchmark_v29_features.py`：predict_next / retarget_convert /
  affordance_score / spatial_relation_head 延迟基准，落
  `benchmarks/results/feature_latency_v2.9.0.json`。
- `tests/test_v29_integration.py`（8 例）：全特性组合默认路径逐位一致、
  backcompat 12 件全加载、两端点 200/400/409、latency JSON 落盘。

### Changed
- backcompat 覆盖扩到 v2.1.0..v2.9.0 共 **12 件**；版本号对外升至 `2.9.2`。

### Evidence（真实运行）
- 全量回归：**470 -> 478 passed / 0 failed**（新增 8 例）。
- 延迟基准（N=50）：predict_next p50=3.20ms；retarget_convert p50=0.02ms；
  affordance_score p50=0.12ms；spatial_relation_head p50=0.22ms（外挂投影，几乎零开销）。

### 被否决/降级候选（诚实保留）
- 两新服务端点默认关闭重定向/可供性，按需调用，不改 /predict 默认语义。

## [2.9.1] - 重定向 A/B + affordance A/B 证据汇总 + 校验加固

### Added
- `scripts/v29_feature_ab.py`：汇总两类 A/B 落 `benchmarks/results/v29_feature_ab.json`
  —— retargeting（重定向 vs 截断）+ affordance（有引导 vs 无引导的动作成功率代理）；
  显式记录被否决候选与 opt-in 默认。
- MorphologyConfig 校验加固：control_freq/关节限位 NaN/inf 守卫、kinematics 类型守卫。
- `tests/test_v29_ab.py`（4 例）：A/B JSON 落盘可复算、引导成功率>无引导、
  opt-in 默认关、被否决候选记录、NaN/inf 限位与非法频率守卫。

### Evidence（真实运行）
- 全量回归：**466 -> 470 passed / 0 failed**（新增 4 例）。
- **affordance A/B**（N=200, 4 部位）：引导成功率 **1.00** vs 无引导恒选 part_0 **0.28**。
- **retarget A/B**：eval_mse 1.68 vs 截断 3.98（2.37×）。

### 被否决/降级候选（诚实保留）
- retarget/affordance/spatial_relation 进旧 predict_next 默认路径 -> **维持 opt-in**；
- affordance 成功率为合成代理，不宣称真机抓取指标（conclusion=`synthetic_proxy_only`）。

## [2.9.0.dev6] - affordance + 动作联合推理 (AffordanceActionPlanner)

### Added
- `udos/affordance.py`：`AffordanceActionPlanner`——结合 AffordanceScorer + 可选
  ActionRetargeter，生成面向 best_part 的接近动作序列 [B,H,6]；**affordance 低 /
  不可达部位不生成动作**（轨迹全零、no_valid_action）；`plan_action_dict` 产出与
  policy/PhysicalLoopRunner 兼容的 best_action（state_perturbation[6]）。
- `tests/test_v29_affordance_action.py`（4 例）：联合推理动作有限/面向近部位、
  低 affordance 过滤为零动作、空物体守卫、opt-in 挂 predict_action_fn hook 后
  loop 最终预测仍与 predict_next 逐位一致。

### Evidence（真实运行）
- 全量回归：**462 -> 466 passed / 0 failed**（新增 4 例）。
- 全部超距 -> no_valid_action=true 且动作轨迹恒零；hook 未挂时旧 MPC 路径不变。

### 被否决/降级候选（诚实保留）
- planner 进 loop.predict_action 默认：维持 opt-in（经 predict_action_fn hook 注入，
  loop 最终预测与 planner 解耦，逐位等价）。

## [2.9.0.dev5] - 空间关系头 (物体间空间关系推理)

### Added
- `udos/multitask.py`：`SpatialRelationHead`——从共享 latent 线性投影出每物体 2D
  坐标，再由坐标差派生物体对空间关系矩阵 [B,N_obj,N_obj,5]，5 通道代理
  `[right,left,up,down,contact]`；天然满足对称性 `right(i,j)==left(j,i)`；
  n_obj<2 守卫；通过 `MultiTaskHead.register_head/remove_head` 独立开关。
- `tests/test_v29_spatial_relation.py`（5 例）：关系矩阵形状有限、对称性约束、
  可控坐标下左右/上下/接触正确、头关闭/移除旧路径逐位不变、n_obj=1 守卫。

### Evidence（真实运行）
- 全量回归：**457 -> 462 passed / 0 failed**（新增 5 例）。
- 对称性质 `right == left.T`、`up == down.T` 逐位成立；头移除后 predict_next `torch.equal` 不变。

### 被否决/降级候选（诚实保留）
- 关系头随 MultiTaskHead 默认 enable=False 维持 opt-in。

## [2.9.0.dev4] - 可供性 affordance 打分模块

### Added
- `udos/affordance.py`：`AffordanceScorer(state, object_proxy)`——给定机器人状态
  + 物体部位代理向量 [B,N_parts,6]（pos3+vel3），输出可操作部位归一化打分
  [B,N_parts] + best_part + 操作建议；打分基于**距离（closeness）/ 相对速度
  （approach）/ 可达性（reach_radius 过滤）**合成特征；无可达部位诚实返回全 0。
  **analogy, not reproduction**——用状态向量子集代理"物体部位"，不碰 RGBD/点云。
- `tests/test_v29_affordance.py`（5 例）：打分形状/归一化求和=1、可达性过滤、
  空物体守卫、无可达部位诚实全 0、与 loop.understand 集成。

### Evidence（真实运行）
- 全量回归：**452 -> 457 passed / 0 failed**（新增 5 例）。
- 超距部位得分恒 0、可达部位归一化和=1；affordance 为推理时外挂不改主路径。

### 被否决/降级候选（诚实保留）
- affordance 默认进旧路径：维持外挂（推理时按需构造 AffordanceScorer）。

## [2.9.0.dev3] - 零样本形态切换合成验证 + 重定向 A/B

### Added
- `ActionRetargeter.zero_shot_transfer`：源形态训练动作策略直接重定向到未见目标形态，
  输出 clamp 后动作有限且严格满足目标限位；诚实报告 clamp 前违例率
  (`violation_rate`) 与 `within_limits`。
- `scripts/retarget_ab_v29.py`：重定向 (端点对齐 map+clamp) vs 直接截断 (naive
  前 T 维+clamp) 对"理想端点映射参考"的重建 MSE，落
  `benchmarks/results/retarget_ab_v2.9.0.json`。
- `tests/test_v29_zeroshot.py`（4 例）：零样本切换后动作有限/限位违例率为 0、
  非法维度/空动作守卫、未见形态可重定向、A/B JSON 落盘可复算、opt-in 不改主路径。

### Evidence（真实运行）
- 全量回归：**448 -> 452 passed / 0 failed**（新增 4 例）。
- **A/B**（60->4 DOF, N=2048）：retarget eval_mse=1.68 vs truncate=3.98，
  **2.37× 重建误差下降**；两法 clamp 后均 100% 在限位内；pre-clamp 违例率 0.62。

### 被否决/降级候选（诚实保留）
- 上述收益为**合成随机投影代理**，非任务精度收益；结论标 `synthetic_proxy_only`，
  重定向维持 opt-in（推理时按需构造，不进旧 predict_next 默认路径）。

## [2.9.0.dev2] - 动作时间重采样 (跨控制频率对齐)

### Added
- `ActionRetargeter.resample`：把源频率动作轨迹 [...,T_src,S] 重采样到目标频率，
  目标帧数 `T_dst=round((T_src-1)*dst/src)+1`，**首/末帧与源首/末帧逐位一致**，
  支持频率比非整数；`kind="linear"`(分段线性，单调不超调) / `"cubic"`(Catmull-Rom)。
- `ActionRetargeter.action_energy`：梯形积分 ∫‖a‖²dt 代理能量，用于重采样前后
  守恒检查。
- `tests/test_v29_resample.py`（6 例）：2x/0.5x/1.5x 帧数正确、端点一致、
  线性保单调、能量近似守恒、三次样条有限、loop.predict_action 组合默认逐位一致。

### Evidence（真实运行）
- 全量回归：**442 -> 448 passed / 0 failed**（新增 6 例）。
- 能量守恒（平滑正弦 60->180Hz）：重采样前后能量比 0.90~1.10（残差为粗网格梯形离散误差）。

### 被否决/降级候选（诚实保留）
- 矩形 Riemann 能量口径：在非整数频率比下随采样率漂移，**改用梯形积分**口径。

## [2.9.0.dev1] - 形态配置体系与预设形态库 (MorphologyLibrary)

### Added
- `udos/retargeting.py`：`MorphologyLibrary`——预设 3 种合成形态
  `prime_u_60dof`(60 DOF/100Hz) / `arm_7dof`(7 DOF/500Hz) /
  `gripper_4dof`(4 DOF/1000Hz)，每种含 DOF/控制频率/关节限位/运动学占位；
  `get/names/register` 注册表、`are_compatible` 兼容性检查、`to_dict/from_dict`
  整体序列化往返。
- `tests/test_v29_morphology.py`（5 例）：预设参数正确、库序列化往返、兼容性检查、
  未知形态 KeyError / 非法注册 ValueError、与 ActionRetargeter 接口一致。

### Evidence（真实运行）
- 全量回归：**437 -> 442 passed / 0 failed**（新增 5 例）。
- 大跨度 60->4 DOF 重定向输出严格夹到 gripper 限位 [-1,1] 内、有限值。

### 被否决/降级候选（诚实保留）
- 预设形态库仅作合成占位，不承诺对真实 URDF/MJCF 本体的可移植性。

## [2.9.0] - Human-as-Humanoid 形态无关动作重定向适配器 + 正式训练重建

### Added
- `udos/retargeting.py`：`MorphologyConfig`（DOF 数 / 控制频率 / 关节限位 / 运动学参数，
  含构造校验与 `to_dict/from_dict` 序列化、`compatible_with` 兼容性粗检）+
  `ActionRetargeter`（源形态动作轨迹 -> 目标形态：端点对齐的确定性 DOF 映射 + 关节限幅
  clamp；空动作 / 末维不符 / 非有限值显式 ValueError）。**analogy, not reproduction**——
  用不同维动作向量代理不同机器人形态，不涉及 URDF/真机。
- `scripts/build_v290_checkpoint.py`：同 v2.8.0 口径（seed=42/n_per_kind=48/epochs=60/
  patience=12/front/hybrid_weight=0）正式训练重建 `checkpoints/predictor_v2.9.0.pt` +
  `benchmarks/results/training_v2.9.0.json`（含 features_v29 重定向离线快照）；
  Makefile 新增 `ckpt290` target。
- `tests/test_v29_retarget.py`（6 例）：形态配置校验、DOF 映射端点对齐/恒等维逐位一致、
  关节限幅 clamp、空动作/维度/非有限守卫、dof=0 不可重定向、序列化往返。

### Changed
- 版本号对外升至 `2.9.0`；retargeting 为推理时外挂，主模型参数仍为 52191。

### Evidence（真实运行）
- 正式件：`predictor_v2.9.0.pt` 可加载（udos_version=2.9.0，**52191 参数**）；
  eval_mse=**0.045556**，train_seconds=111.4，best_epoch=57，coverage=0.8888，
  ece=0.056546。
- 重定向离线快照：6->4 DOF 映射输出 [4,4]，严格落在目标限位 [-0.5,0.5] 内
  （abs_max=0.5，within_target_limits=true）。
- 全量回归：**431 -> 437 passed / 0 failed**（新增 6 例）。

### 被否决/降级候选（诚实保留）
- 重定向默认挂载：维持 opt-in（推理时按需构造 ActionRetargeter，不改主模型默认路径）。

## [2.8.3] - 边缘精修 + 文档对齐（2.8 线终点补丁件）

### Added
- `tests/test_v283_edge.py`（7 例）：loop 空 window（B=0/W=0）与非有限值显式 ValueError、
  multitask 头 latent 维度不匹配显式报错、future_state horizon=0 拒绝、feedback 极端偏差率
  （dev_threshold=-1）触发但默认不改权重、服务端点未训练态 409 语义。

### Changed
- `docs/ROADMAP.md` / `ARCHITECTURE.md` / `DEPLOYMENT.md` / `README.md` 对齐至 2.8 线
  （PhysicalLoop 五步编排 + 共享 backbone 多任务头 + 两端点）；版本号对外升至 `2.8.3`。

### Fixed
- 收紧失败路径：空/非有限窗口不静默 inf 传播；头维度不匹配不静默产生错形状。

### Evidence（真实运行）
- 全量回归：**424 -> 431 passed / 0 failed**（新增 7 例边缘测试）。
- checkpoints/predictor_v2.8.0.pt 可加载（udos_version=2.8.0，52191 参数）。

### 被否决/降级候选（诚实保留）
- 多任务三头默认开启维持 opt-in；hierarchical 单自回归头下与平铺逐位一致（无误差下降，opt-in 脚手架）。

## [2.8.2] - 集成加固: loop+multitask 组合 / 服务端点 / backcompat 11 件 / 延迟基准

### Added
- `udos/server.py`：`POST /loop/step`（五步闭环单步，未训练 409、非法 horizon/输入 400）
  与 `POST /multitask/predict`（三头联合推理，未训练 409）；懒挂 `self._loop`/`self._mth`。
- `scripts/benchmark_v28_features.py`：predict_next / physical_loop_step / multitask 三头
  延迟基准（p50/p95/mean），落 `benchmarks/results/feature_latency_v2.8.0.json`。
- `tests/test_v28_integration.py`（8 例）：loop+multitask 组合不冲突且默认路径逐位一致、
  backcompat 11 件全加载、两端点 200、400/409 语义、latency JSON 落盘可复算。

### Changed
- backcompat 覆盖扩到 v2.1.0..v2.8.0 共 **11 件**；版本号对外升至 `2.8.2`。

### Evidence（真实运行）
- 全量回归：**424 passed / 0 failed**（较 2.8.1 的 416 新增 8）。
- 延迟基准（N=50）：predict_next p50=3.20ms；physical_loop_step p50=33.19ms；
  multitask 三头 p50=0.18ms（外挂投影，几乎零开销）。

### 被否决/降级候选（诚实保留）
- 多任务三头联合默认开启 → 维持 opt-in（enable=False），服务端点按需懒建。

## [2.8.1] - 未来状态头 + 多任务 A/B 证据 + opt-in 门控

### Added
- `udos/multitask.py`：`FutureStateHead`（共享 latent -> 未来状态代理 [B,H,6] + 每步
  不确定性 [B,H,1]，softplus 非负，与 predictor.rollout 对齐）。
- `scripts/multitask_ab_v28.py`：共享 backbone 三头 vs 独立头 A/B，落
  `benchmarks/results/multitask_ab_v2.8.0.json`（延迟/encode 次数/参数量/未来头代理 mse）。
- `tests/test_v28_multitask_ab.py`（5 例）：未来头形状有限、A/B JSON 落盘可复算、
  opt-in 默认关逐位一致、enable 后三头联合推理、被否决候选保留。

### Changed
- 版本号对外升至 `2.8.1`；MultiTaskHead 默认 `enable=False`（opt-in）。

### Evidence（真实运行）
- 全量回归：416 passed / 0 failed（较 dev6 的 411 新增 5）。
- **A/B**：共享 encode 一次 0.255ms vs 独立 encode 三次 0.596ms => 2.34× 延迟优势；
  future_head 代理 mse=8.10（随机初始化线性头，不代表任务精度收益）。

### 被否决/降级候选（诚实保留）
- 三头联合默认开启：无训练头，精度不带来源收益，**降级为 opt-in**（enable 默认关）。

## [2.8.0.dev6] - 空间坐标头 + 动作轨迹头

### Added
- `udos/multitask.py`：`SpatialCoordHead`（共享 latent -> [B,N_pts,3] 空间坐标代理，
  3 通道代理 xyz）；`ActionTrajectoryHead`（共享 latent -> [B,H,action_dim]，action_dim
  默认对齐 RAW_DIM=6，与 policy 动作空间一致）；均为纯线性投影、推理时外挂、不参与训练。
- `tests/test_v28_heads.py`（6 例）：坐标头形状有限、动作头形状有限、共享 backbone
  推理模式无梯度冲突、头关闭时旧路径不变、共享 vs 独立头延迟对比。

### Evidence（真实运行）
- 全量回归：411 passed / 0 failed（较 dev5 的 405 新增 6）。
- 推理模式跑两头后主模型参数 grad 均为 None；共享 encode 一次跑两头 <= 独立两次 encode。

## [2.8.0.dev5] - 统一多任务头骨架（共享 backbone 接口）

### Added
- `udos/multitask.py`：`MultiTaskHead`——共享 backbone 接口
  `encode(window, scene_params) -> latent[B,latent_dim]`（复用 predictor obs/scene
  encoder，确定性池化+对齐，无新参数）；头注册机制 `register_head/get_head/list_heads/
  remove_head`；`enable` 开关（默认 False）；`config_dict/load_config` 配置往返。
- `tests/test_v28_multitask_base.py`（8 例）：头注册/查询/列表、空头守卫、encode 形状有限、
  latent_dim 可配、默认路径逐位不变、enable 路由、配置 save/load。

### Evidence（真实运行）
- 全量回归：405 passed / 0 failed（较 dev4 的 397 新增 8）。
- 默认 enable=False 时 forward 返回 {}，encode 后主模型 predict_next `torch.equal` 不变。

## [2.8.0.dev4] - feedback / correction 阶段（online + adaptive 集成）

### Added
- `PhysicalLoopRunner._step_feedback`：用 future_state.deviation 的 L2 范数 +
  `online.OnlineAdapter`（StreamingDriftDetector）联合判定是否需要修正；
  `dev_threshold`（默认极大，偏差单独不触发）、`enable_recalibration`（默认 False）、
  `recalibration_data` opt-in；触发时 correction 信号回写 `loop_state["correction"]`。
- 默认仅记录、不改权重/校准；opt-in 时触发 PAVA 再校准（主权重仍不改）。
- `tests/test_v28_feedback.py`（6 例）：无偏差不触发、偏差超阈记录、默认不改权重、
  opt-in PAVA 再校准、correction 回写 loop_state。

### Evidence（真实运行）
- 全量回归：397 passed / 0 failed（较 dev3 的 391 新增 6）。
- 默认触发时 predictor 前后 `torch.equal` 逐位一致；opt-in 下 drift 触发后 recalibrated=true。

## [2.8.0.dev3] - future_state 阶段（rollout 预测集成）

### Added
- `PhysicalLoopRunner._step_future_state`：对 predict_action 选中动作（经 `_apply_action`
  叠加 state_perturbation / scene_param 覆写）做 `predictor.rollout` 多步预测，输出
  trajectory[B,H,6]；挂载 conformal 半宽时给出 lower/upper 不确定性区间；
  deviation = 首步轨迹 vs understand.target_state。
- `_apply_action` 辅助：把选中动作覆写到窗口副本（只读，不改原窗口）。
- `tests/test_v28_future_state.py`（6 例）：轨迹形状有限、无扰动偏差逐位为 0、
  horizon=1 退化 [B,1,6]、与旧 rollout 逐位一致、动作扰动改变轨迹。

### Evidence（真实运行）
- 全量回归：391 passed / 0 failed（较 dev2 的 385 新增 6）。
- 无动作扰动时 future_state.trajectory 与 `predictor.rollout` `torch.equal`；
  无扰动偏差信号逐位为 0（首步 rollout 即 predict_next）。

## [2.8.0.dev2] - predict_action 阶段（MPC 集成进 loop）

### Added
- `PhysicalLoopRunner._step_predict_action`：懒构造 `MPCActionSelector`，调用方经
  `run(..., candidate_actions=[...])` 提供候选；未提供时默认单个无操作动作；显式空列表透传
  => `no_valid_action=True`。输出 best_action/best_score/best_index/ranked_actions/best_risk，
  best_action 追加进 `loop_state["action_history"]`（跨 run 累积）。
- `run(..., candidate_actions=None)` 新参。
- `tests/test_v28_predict_action.py`（7 例）：空候选 no_valid_action、已知最优选择正确、
  风险惩罚单调性、动作历史累积、与旧 policy 逐位一致、默认候选兜底。

### Fixed
- 空候选列表 [] 不再被误填为默认动作（区分"未提供 None"与"显式空 []"）。

### Evidence（真实运行）
- 全量回归：385 passed / 0 failed（较 dev1 的 378 新增 7）。
- 同一 selector 下 loop.predict_action 的 best_index/best_score 与直接
  `MPCActionSelector.select` 逐位一致。

## [2.8.0.dev1] - observe / understand 阶段实现

### Added
- `PhysicalLoopRunner._step_observe`：接收 window+scene_params，调用 `predictor.obs_encoder`
  做特征提取（输出 features + scene_context），纯只读前向。
- `PhysicalLoopRunner._step_understand`：输出结构化理解向量
  `understanding_vector` = concat([B,6] 目标状态, [B,1] 风险标量, [B,1] 不确定标量)
  固定 [B,8]；`risk_flags` 含 ood_flag/ood_score/uncertain/interval_width（挂载 OOD 检测器
  与 conformal 半宽时给出真实值，未挂载时风险通道诚实退化为 0/None，不伪造证据）。
- `tests/test_v28_observe_understand.py`（6 例）：observe 形状有限、understand 含目标/风险字段、
  空/NaN 守卫、只读无副作用、旧件风险通道诚实退化。

### Changed
- 版本号升至 `2.8.0.dev1`。

### Evidence（真实运行）
- 全量回归：378 passed / 0 failed（较 2.8.0 的 372 新增 6）。
- 2.8.0 正式件上 ood_flag 为 bool、interval_width>0、understanding_vector=[1,8] 有限。

## [2.8.0] - Physical Loop 五步闭环 runner 骨架 + 正式训练重建（2.8 线起点）

> analogy, not reproduction —— Physical Loop 为受 PhysBrain 1.5 启发的轻量化类比编排层，
> 非复现（UDOS 为 CPU-only ~52k 参数合成动力学小模型）。

### Added
- `udos/physical_loop.py`：`PhysicalLoopRunner`，显式五步编排
  observe→understand→predict_action→future_state→feedback，整合现有
  predictor/policy/online/adaptive（不重复造轮子）；**每步可插拔**（构造时可传
  observe_fn/understand_fn/... 覆盖内置默认）；loop_state 记录每步 name/耗时/输出形状，
  动作历史与 correction 信号跨 run 累积，纯元数据无副作用。
- `tests/test_v28_loop.py`（8 例）：空循环守卫、五步顺序、默认未挂载 hook 时
  与直接 `predict_next` 逐位一致（`torch.equal`）、loop_state 完整、可插拔 hook、
  不破坏 checkpoint save/load。
- `scripts/build_v280_checkpoint.py`（`make ckpt280`）：同口径正式训练，产出
  `checkpoints/predictor_v2.8.0.pt`（**52191 参数**）+ `benchmarks/results/training_v2.8.0.json`
  （含 features_v28：loop 形状/五步顺序/逐位一致标志）。

### Changed
- 版本号对外升至 `2.8.0`；`Makefile` 新增 `ckpt280` target；docker tag 同步。

### Fixed
- `tests/test_v27_policy.py` 一处保存当前模型后的版本断言改为对比 `__version__`
  （原为硬编码 2.7.3，跨版本升版会误报）。

### Evidence（真实运行，未编数字）
- **正式件**：eval_mse 0.0456 / naive 0.1634 / ECE 0.0565 / coverage 0.8888 /
  interval_width 0.5412 / condition_gain 15.39× / rollout_growth 2.81× / OOD 命中 0.810 /
  误报 0.065 / batch_max_diff 1.97e-06 / 训练 108.2s / 60 epochs / best_epoch 57。
- **PhysicalLoop 锚点**：loop_bit_identical_to_predict_next=true；loop_steps=[observe,
  understand, predict_action, future_state, feedback]；prediction_shape=[1,6]。
- **全量回归**：372 passed / 0 failed（v2.7.3 的 364 只增不删）。

### 被否决/降级候选（诚实保留）
- 五步在 2.8.0 仅为骨架编排：predict_action/future_state/feedback 为占位默认，
  逐阶段能力在 dev1..dev4 落地；当前默认路径与旧版逐位一致，无新数值收益宣称。

## [2.7.3] - 最终训练重建 + 全量验证 + 打包发布（2.7 线终点正式发布件）

### Added
- **正式训练重建** `scripts/build_v273_checkpoint.py`（`make ckpt273`）：同口径 seed=42 /
  n_per_kind=48 / epochs=60 / patience=12 / front / hybrid_weight=0，产出
  `checkpoints/predictor_v2.7.3.pt`（**52191 参数**）+ `benchmarks/results/training_v2.7.3.json`
  （含训练 loss 曲线、eval_mse、ECE、coverage、interval_width、condition_gain、rollout_growth、
  OOD 指标、batch_inference_max_diff、reload_consistent、features_v27 快照）。
- `scripts/verify_service_v273.py`：真实起 HTTP 服务逐接口验证（/health=2.7.3、/evaluate 含
  calibration/interval、四新端点 200/400/409、未训练 409）。
- `docs/VERIFICATION_v2.7.3.md`：完整验收报告（正反证据 + 已知限制 + 复现命令）。

### Changed
- 版本号对外升至 `2.7.3`；docker-compose / Dockerfile checkpoint 引用改为 `predictor_v2.7.3.pt`；
  `Makefile` 新增 `ckpt273` target。

### Evidence（真实运行，未编数字）
- **正式件**：eval_mse 0.0456 / naive 0.1634 / ECE 0.0565（降 6.82×）/ coverage 0.8888 /
  interval_width 0.5412 / condition_gain 15.39× / rollout_growth 2.81× / OOD 命中 0.810 /
  误报 0.065 / batch_max_diff 1.97e-06 / reload_consistent=true / save-load `torch.equal`=True。
- **全量回归**：364 passed / 0 failed / 覆盖率 92%。
- **向后兼容**：v2.1.0..v2.7.3 十件 checkpoint 全部可加载、predict [B,6] 有限、新模块不报错。
- **服务**：/health=2.7.3；/policy/select /online/adapt /active/sample /experiments 200；非法输入 400；未训练 409。

### 被否决/降级候选（诚实保留）
- 分层 rollout：单自回归头下与平铺逐位一致（`hierarchical_bit_identical_to_flat=true`），opt-in 脚手架。
- 幅值剪枝 50% 不微调掉点（条件依赖）；INT8 动态量化近似无损；蒸馏学生精度有损（27071 参数）。
- 主动学习 A/B 主动 0.264 vs 随机 0.502（有效，但增益对种子/口径敏感）。

## [2.7.2] - Patch 2：缺陷修复 + 边缘加固 + 文档精修

### Fixed
- **online**：`observe` 跳过 NaN/inf 行（不污染滑动窗口与漂移统计）；全相同（常数）数据不误报漂移；
  适配后校准器状态自洽（`is_calibrated` 存在、可再评估）。
- **active**：零样本池返回空索引/空分（不崩溃）；k>池大自动截断；单样本池正常。
- **policy**：horizon=0 显式 ValueError；空动作集 `no_valid_action=True`；scene_params=None 正常。
- **hierarchical**：horizon=1 与 coarse_factor>horizon 均退化为平铺 `predictor.rollout` 逐位一致。
- **lite**：量化输出有限；剪枝态 save/load 稀疏度保持；蒸馏学生 save/load 后 predict 形状 [B,6] 有限。
- **PredictionGuard**：挂载守卫后 policy 仍正常工作并返回有限分数（兼容 policy 输出格式）。
- `tests/test_v272_edge.py` 覆盖上述全部边界条件（16 例）。

### Changed
- 版本号对外升至 `2.7.2`。
- 文档：`docs/ROADMAP.md` 标注 v2.7 线完成 + v2.8 展望；`docs/ARCHITECTURE.md` 补 2.7 新模块表；
  `docs/DEPLOYMENT.md` 标题至 v2.7.2 + 四新端点文档；`README.md` 标题至 v2.7.2 + v2.7 特性段。

## [2.7.1] - 集成加固 + 全量回归 + 向后兼容扩展

### Added
- **跨特性集成测试** `tests/test_v27_integration.py`：
  - policy + online + active + lite + hierarchical 组合调用互不冲突、互不污染；
  - 默认路径（不挂载任何 2.7 新特性）`predict_next` 两次逐位一致；
  - 各特性调用前后 `predict_next` 逐位不变（外挂只读、无副作用）；
  - lite 量化为独立副本、近似无损、不改原模型。
- **9 checkpoint 后向兼容** `tests/test_v27_backcompat.py`：覆盖
  v2.1.0 / v2.2.1 / v2.3.1 / v2.4.0 / v2.5.0 / v2.5.2 / v2.6.0 / v2.6.2 / v2.7.0，
  每件可加载、`predict` 形状 [B,6] 有限、policy/active/hierarchical 对旧件不报错。
- **性能基准** `scripts/benchmark_v27_features.py`（`make feature-bench27`）：
  测量 policy/online/active/lite/hierarchical 推理延迟，落
  `benchmarks/results/feature_latency_v2.7.0.json`。

### Changed
- 版本号对外升至 `2.7.1`（`__init__.py`、`pyproject.toml`、`Makefile`、部署标签、测试字面量）。
- `Makefile` 新增 `feature-bench27` target。

### Evidence（CPU 小批量 32，真实跑出）
- predict_next 基线 ~77ms；policy(3 动作 MPC) ~40ms；active(池打分) ~101ms；
  online(observe+detect) ~0.04ms；hierarchical(H=8) ~25ms；lite(INT8) ~105ms。
- online/active 为前向只读外挂，不改权重；分层 rollout 与平铺逐位一致（见 dev4 负面证据）。

## [2.7.0.dev6] - 服务接口扩展（MPC / 在线 / 主动 / 实验）

### Added
- **`udos/server.py` 新增 4 个端点**（复用既有 409/400 错误语义）：
  - `POST /policy/select`：`{window, scene_params?, actions:[...], horizon?, lambda_risk?}`
    → MPC 最优动作 + 全排序；空动作集 `no_valid_action=True` 仍 200；未训练 409；缺 window/actions 400；
  - `POST /online/adapt`：`{window, enable_finetune?, finetune_epochs?}` → 推入观测、漂移触发再校准
    （默认 `enable_finetune=False` 不改权重），返回 adapted/reason/drift_score/weights_modified/log_length；未训练 409；
  - `POST /active/sample`：`{sample_pool:[N,W,RAW], k, scene_params?}` → top-K 不确定性样本索引+分数
    （k>池大自动截断）；未训练 409；空池 / k≤0 400；
  - `GET /experiments`：从 `benchmarks/results/experiment_registry.json` 读实验列表，文件不存在返回空。
- `UDOSService` 新增 `policy_select/online_adapt/active_sample/experiments` 方法（与 HTTP 层解耦，便于单测）；
  `make_handler` 的 `do_GET`/`do_POST` 路由新端点。
- `tests/test_v27_service.py`：四端点 200/400/409、返回字段完整、与离线 API（MPCActionSelector）逐字段一致、
  空动作集/空池/k>池边界、未训练实例 409 共 14 例。

### Changed
- 版本号升至 `2.7.0.dev6`（`__init__.py` + 导入四端点依赖、`pyproject.toml`、部署标签、测试字面量）。

## [2.7.0.dev5] - 实验注册与多种子 sweep 治理

### Added
- **`udos/experiment.py` `ExperimentRegistry`**：纯元数据实验治理（不触碰权重、不改预测路径）。
  - `register(name, config, metrics, seed, artifacts)` → 记录实验元数据；
  - `list()` / `get(name)` 查询；
  - `sweep_report(names=None)` 对同名不同 seed 聚合每种 metrics 的 mean/std/best
    （coverage/gain 等越大越好语义取 max，其余取 min）；
  - `save(path)` / `load(path)` JSON 原子持久化（临时文件 + rename），
    默认落 `benchmarks/results/experiment_registry.json`；
  - 同 (name, seed) 重复注册幂等覆盖；空注册表查询守卫。
- `tests/test_v27_experiment.py`：注册/查询/列表、多种子聚合 mean/std/best、
  大小两个 best 语义方向、JSON 持久化与重载、空注册表守卫、重复注册幂等、
  入参校验共 9 例。

### Changed
- 版本号升至 `2.7.0.dev5`（`__init__.py` + 导出 `ExperimentRegistry`、`pyproject.toml`、
  部署标签、测试字面量）。

## [2.7.0.dev4] - 分层 / 多尺度长时域 rollout

### Added
- **`udos/hierarchical.py` `HierarchicalRollout`**：`coarse_factor` 可配的分块粗粒度 rollout。
  - `rollout(window, horizon, scene_params)` → `{predictions:[B,H,6], coarse_points, truncated}`；
  - horizon ≤ coarse_factor 时退化为普通 rollout **逐位一致**；
  - 纯前向、确定性、不修改 predictor。
- **A/B 脚本** `scripts/ablation_hierarchical.py`：H=8/12/16 下分层 vs 平铺逐步 MSE 累积率对比，
  落 `benchmarks/results/hierarchical_ablation_v2.7.0.json`。
- `tests/test_v27_hierarchical.py`：短 horizon 退化等价、长 horizon 形状/分层标记、
  长 horizon 不退化、A/B 证据可复算共 5 例。

### Changed
- 版本号升至 `2.7.0.dev4`（`__init__.py` + 导出 `HierarchicalRollout`、`pyproject.toml`、
  部署标签、测试字面量）。**2.7 线 5 节点收尾版本号 = 2.7.0.dev4。**

### Evidence（诚实负面）
- A/B（coarse_factor=4）：H=8/12/16 下分层与平铺 `bit_identical=true`，
  flat_growth_x = hier_growth_x = 87.6 / 216.2 / 569.3。
- **结论**：本引擎 predictor 是单自回归头、无独立多步粗粒度头，分块滑窗在数学上 == 连续单步
  rollout，当前不产生误差下降（与 v2.2 SS、v2.6 hybrid 等被否决/条件依赖候选同纪律，照实记录）。
  本模块为将来接入独立粗粒度头预留的 opt-in 脚手架；接入后即可在不改短 horizon 退化契约的
  前提下叠加细粒度修正。

## [2.7.0.dev3] - 模型轻量化（剪枝 / 量化 / 蒸馏 A/B）

### Added
- **`udos/lite.py`**（全部 opt-in，默认全量模型不变）：
  - `MagnitudePruner`：全局 Linear 权重幅值剪枝，存原权重可恢复，`unprune` 复位，
    `sparsity_ratio` 报实际零参数占比；
  - `DynamicQuantizer`：`torch.ao.quantization.quantize_dynamic`（Linear→INT8，仅 CPU），
    量化后仅推理；
  - `DistillationTrainer`：teacher→小学生 CTM（d_model=32）蒸馏，
    loss = α·MSE(student, teacher.detach) + (1−α)·MSE(student, y)（回归任务的软/硬目标 KD）。
- **A/B 脚本** `scripts/ablation_lite.py`：参数-延迟-精度三维对比，落
  `benchmarks/results/lite_ablation_v2.7.0.json`。
- `tests/test_v27_lite.py`：剪枝稀疏度正确+可恢复、量化前后输出在容差内、蒸馏 loss 下降、
  A/B JSON 落盘、默认全量模型逐位不变共 5 例。

### Changed
- 版本号升至 `2.7.0.dev3`（`__init__.py` + 导出三工具、`pyproject.toml`、部署标签、测试字面量）。

### Evidence（如实记录 CPU 小模型结果）
- 全量：52191 参数 / eval_mse 0.057 / 延迟 ~5.0ms。
- 剪枝 50%：稀疏度 0.50 / eval_mse 0.584（**不微调直接幅值剪枝精度明显下降**，
  需配合微调才可用——照实记录为条件依赖）。
- 量化 INT8：eval_mse 0.066（相对全量 0.057 仅 +0.009，动态量化近似无损）。
- 蒸馏学生：27071 参数（约全量一半）/ eval_mse 0.284（小代价换一半参数，精度损失如实记录）。

## [2.7.0.dev2] - 主动学习 / 不确定性采样选点

### Added
- **`udos/active_learning.py` `UncertaintySampler`**：对未标注样本池计算信息增益分并选 top-K：
  `info_gain = α·ensemble_variance + β·interval_width_norm + γ·ood_score_norm`
  （默认 α=0.4/β=0.3/γ=0.3，权重可配）。
  - ensemble_variance：predictor 为 `DeepEnsemble` 时取成员预测间方差；否则用确定性代理
    （1 − 校准后置信，逐样本输入相关）；
  - interval_width_norm：conformal 半宽归一化（全局标量）；ood_score_norm：马氏距离/阈值；
  - `select_top_k` 返回最大 k 个索引与分数；纯前向、确定性、不改 predictor。
- **A/B 证据脚本** `scripts/ablation_active_learning.py`：同种子、同初始训练集，主动选点 vs
  随机选点的 eval_mse 对比，落 `benchmarks/results/active_learning_ablation_v2.7.0.json`。
- `tests/test_v27_active.py`：排序单调性、top-K 选择、确定性、空池/k=0 守卫、A/B JSON 落盘共 6 例。

### Changed
- 版本号升至 `2.7.0.dev2`（`__init__.py` + 导出 `UncertaintySampler`、`pyproject.toml`、
  部署文件标签、全部测试字面量）。

### Evidence
- A/B（seed=42，初始 160 条 + 各加 128 条）：seed-only 基线 eval_mse 1.668；
  **主动选点 0.264 vs 随机选点 0.502**，active_minus_random = −0.238，`active_better=true`。
  在本合成小模型上主动选点显著降低样本需求；仍标注"增益对种子/口径敏感，如实记录差值"。

## [2.7.0.dev1] - 在线增量适配与漂移触发再校准闭环

### Added
- **`udos/online.py` `OnlineAdapter`**：封装 `ood.StreamingDriftDetector`，构建
  "检测→触发→再校准→确认"闭环。
  - `observe(window)`：把新观测推入流式滑动窗口（自动展平为 [W*RAW]）；
  - `check_and_adapt(predictor, new_calibration_data)`：窗口均值马氏距离超阈值时触发，
    默认**仅用新校准集重跑 PAVA 校准 + conformal 半宽，不改模型权重**
    （`weights_modified=False`）；opt-in `enable_finetune=True` 时再做少量 epoch 增量微调
    （有 `finetune_epochs` 硬上限，默认 3，防在线灾难性遗忘）；
  - 无漂移时完全不触发、不改变状态；`reset()` 清空窗口与日志（保留参考分布）；
  - 每次触发在 `adaptation_log` 记录 `{timestamp, trigger_reason, drift_score,
    before_ece, after_ece, weights_modified}`。
- `tests/test_v27_online.py`：无漂移不触发、漂移触发再校准、默认不改权重、opt-in 微调降误差、
  日志完整、reset、与旧 OOD 检测器兼容共 8 例。

### Changed
- 版本号升至 `2.7.0.dev1`（`udos/__init__.py` + 导出 `OnlineAdapter`、`pyproject.toml`、
  部署文件标签、全部测试字面量）。

### Evidence
- 全量 pytest 全绿（节点 31 的 288 + 8 新增）。
- 实测：参考分布窗口均值马氏距离 ~2.56 < 阈值 5.88（不触发）；分布平移 +15 后窗口均值
  马氏距离 ~3120 >> 阈值（触发）。默认触发后 `weights_modified=False` 且主权重 state_dict
  逐位一致；opt-in 微调后权重确实改变。

## [2.7.0] - MPC 式候选动作优选 + 正式训练重建（2.7 线首发件）

### Added
- **`udos/policy.py` `MPCActionSelector`**：在已有 predictor 之上构建"从预测到行动"的闭环。
  给定当前 window + scene_params + 候选动作集，对每个动作做自由 rollout，按
  `score = objective_reward − λ·risk_score − (0 if safe else safety_penalty)` 打分并全排序。
  - 候选动作为调用方提供的 dict，支持 `scene_param`（场景参数覆盖）、`state_perturbation`
    （初始状态扰动）、`candidate_state`（提议目标态供安全边界判定）；
  - risk_score 来自 `decision.RiskGrader.grade()`，安全违例计数来自 `decision.safety_boundary()`；
  - `objective_reward` 可配置 callable，默认 None：给了 `reference[H,6]` 取负 MSE，否则 0.0；
  - 纯前向、确定性、不修改 predictor；空候选集返回 `no_valid_action=True`；opt-in，
    不调用时旧路径逐位一致。
- **正式件重建**：`scripts/build_v270_checkpoint.py`（同 v2.6.2 口径 seed=42 / n_per_kind=48 /
  epochs=60 / patience=12 / front / hybrid_weight=0），`Makefile` 新增 `ckpt270` target。
  产出 `checkpoints/predictor_v2.7.0.pt`（**52191 参数**）与
  `benchmarks/results/training_v2.7.0.json`。
- `tests/test_v27_policy.py`：空候选守卫、已知最优动作选择、风险惩罚单调性、安全边界扣分、
  确定性、save/load 兼容共 7 例。

### Changed
- 版本号升至 `2.7.0`：`udos/__init__.py`（文档字符串 + `__version__` + 导出 `MPCActionSelector`）、
  `pyproject.toml`、`Makefile`（docker-build/run 标签 + `ckpt270`）、`docker-compose.yml`、
  `Dockerfile`（LABEL + checkpoint 引用至 `predictor_v2.7.0.pt`）；全部测试中的版本字面量同步。

### Evidence
- 全量 pytest **288 passed**（281 基线 + 7 新增），0 skipped，0 failed。
- 正式件（同口径确定性复现）：n_params **52191** / final_loss 0.059488 / eval_mse 0.045556 /
  naive_mse 0.163415 / ECE 0.056546（raw 0.385466，降 6.817×）/ coverage 0.8888 /
  rollout_growth 2.805× / batch_inference_max_diff 1.97e-06 / reload_consistent=true。
- policy 为推理外挂，主模型结构与训练口径不变，故训练指标与 v2.6.2 逐位一致（确定性种子）。
- 8 代旧 checkpoint 仍全部向后兼容（backcompat 表保留 v2.6.2 件元数据字面量）。

## [2.6.2] - 最终训练重建 + 全量验证 + 打包发布（2.6 线终点正式发布件）

### Added
- **正式件重建**：`scripts/build_v262_checkpoint.py`（参照 build_v260），`Makefile` 新增
  `ckpt262` target。训练口径 seed=42 / n_per_kind=48 / epochs=60 / patience=12 / front /
  hybrid_weight=0。产出 `checkpoints/predictor_v2.6.2.pt`（**52191 参数**）与
  `benchmarks/results/training_v2.6.2.json`（含 ood/calibration/interval 分段 + rollout 曲线 +
  batch_inference_max_diff + reload_consistent=true）。
- **8 checkpoint 后向兼容**：`tests/test_v26_backcompat.py` 扩展覆盖 v2.6.2（共 8 件）。
- **服务逐接口验证脚本**：`scripts/verify_service_v262.py`（含未训练实例 409、非法输入 400、
  rollback 409→200、4 新端点 200）。
- `docs/VERIFICATION_v2.6.2.md` 完整验收报告（正反证据 / 被否决候选 / 已知限制 / 复现命令）。

### Changed
- `README.md` 标题至 v2.6.2 并补 v2.6.2/v2.6.1 特性段；`docs/DEPLOYMENT.md` 标题至 v2.6.2；
  `docs/ROADMAP.md` 标注 v2.6.1/v2.6.2 完成；`Dockerfile` / `docker-compose.yml` 版本号与
  checkpoint 引用至 `predictor_v2.6.2.pt`。

### Evidence
- 全量 pytest **281 passed**，`--cov=udos` 总计 **92%**（3054 stmts / 237 miss）。
- 正式件：final_loss 0.059488 / eval_mse 0.045556 / naive_mse 0.163415 / ECE 0.056546
  （raw 0.385466，降 6.817×）/ coverage 0.8888 / interval_width 0.54115 /
  condition_gain 15.393× / rollout_growth 2.805× / batch_inference_max_diff 1.97e-06。
- OOD：threshold 5.5275、ID 误报 6.46%、OOD 命中 81.04%。
- 8 checkpoint 全部可加载、predict 形状 [B,6] 有限、旧件 hybrid=None、已校准件 is_calibrated=True。
- HTTP：/health=2.6.2、/evaluate、/predict、/detect-ood、/metrics、/counterfactual、/identify、
  /risk、/diff-checkpoints 全 200；/rollback 无历史 409→load→200；/predict 缺 window 400；
  未训练实例 /counterfactual 409。
- 打包 `udos-engine-v2.6.2.zip`，独立解压到 /tmp 复跑（版本/pytest/build --quick/服务
  /health=2.6.2//evaluate 含 calibration/interval/新端点/md5 一致）全部通过。

## [2.6.1] - Patch 1：缺陷修复 + 边缘加固 + 文档精修

### Fixed / Edge-hardening
- **空 batch**：`udos/batch.py` `BatchPredictor.predict` / 便捷 `predict_batch` 对空列表与
  batch 维 0 改为返回形状 `[0, RAW_DIM]` 的空张量（此前抛 ValueError），调用方无需对空序列
  特判 try/except；非法输入类型仍显式报错。旧用例 `test_empty_batch_guarded_*` 同步改为
  锁新契约。
- **极端 scene_param**：`PhysicsPredictor.predict_next` 对 `scene_params` 做有限性自检，
  含 NaN/inf 时显式 `ValueError`（不再静默传播 inf）；1e6 经 scene_encoder LayerNorm 归一化后
  输出仍有限（实测 `torch.isfinite` 全 True）。
- **horizon=1 自适应**：`adaptive_rollout(max_horizon=1)` 正常返回 `horizon_used=1`、
  trajectory `[1,1,R]`，半宽增长率判据有 `len>=2` 与 `widths[-2]>1e-12` 双保险，不除零。
- **RiskGrader NaN OOD**：`udos/decision.py` 当 OOD score 为 NaN 时降级为 0 并在 components
  记录 `ood_nan_degraded=True`，risk_score 不再被 NaN 污染（仍为 [0,1] 有限值）。
- **反事实 NaN 干预**：`udos/counterfactual.py` 新增 `_validate_intervention`，scene_params /
  initial_state / velocity_override 干预值含 NaN/inf 时显式 `ValueError`。
- **hybrid × guard 顺序**：`predict_next` 调整为 forward → hybrid 修正 → guard 清洗，使 guard
  作用于 hybrid 修正后的最终输出；`hybrid=False`（默认）路径逐位不变。
- **校准器 hybrid 模式**：hybrid 仅改预测中位数，不触碰 certainty；已挂校准器时
  `RiskGrader.grade` 的置信分量仍经 `calibrator.transform`。

### Added
- `tests/test_v261_edge.py`（10 用例）：空 batch / 1e6 不 inf / NaN scene_param 报错 /
  horizon=1 / NaN OOD 降级 / NaN 干预报错 / hybrid+guard 兼容 / hybrid 不改校准路径 / 版本断言。

### Docs
- `docs/DEPLOYMENT.md` 标题至 v2.6.1，新增 `/counterfactual` `/identify` `/risk`
  `/diff-checkpoints` 四端点契约段。
- `README.md` 标题至 v2.6.1，新增 v2.6.1 加固段与 v2.6.0 特性摘要段。
- `docs/ARCHITECTURE.md` v2.6 模块表已覆盖 hybrid/counterfactual/identification/adaptive/
  decision/快照差分（dev7 已齐）。

### Evidence
- 全量 pytest **281 passed**（271 + 10 新）。
- 实测：1e6 scene_params 输出 `isfinite` 全 True；NaN scene_params → ValueError；
  horizon=1 → `horizon_used=1`；hybrid+guard 输出有限；NaN OOD → `ood_nan_degraded=True`、
  risk_score 有限；NaN 干预 → ValueError。

## [2.6.0+dev7] - 集成加固 + 全量回归 + 后向兼容扩展 + 性能基准 + 文档

### Added
- **跨特性集成测试**：`tests/test_v26_integration.py`（4 用例）：
  hybrid↔counterfactual（挂载后零干预仍=基线）、adaptive_rollout 推进后扩展窗口喂
  RiskGrader、identify 反演参数作 scene_params 再做反事实干预、load→predict→risk→
  counterfactual→identify 全链路。
- **后向兼容扩展**：`tests/test_v26_backcompat.py`（7 用例）覆盖全部 7 个 checkpoint
  （v2.1.0/v2.2.1/v2.3.1/v2.4.0/v2.5.0/v2.5.2/v2.6.0）：load 成功、predict_next 形状
  [B,6] 且有限、`is_calibrated` 与 `hybrid` 属性存在（旧件 hybrid=None）、RiskGrader/
  safety_boundary 对旧件诚实退化不崩、export_snapshot 全件可用。
- **性能基准**：`scripts/benchmark_v26_features.py` 测量 hybrid/counterfactual/identify/
  adaptive/risk 在小批量 32 上的推理延迟，落
  `benchmarks/results/feature_latency_v2.6.0.json`；Makefile 新增 `feature-bench` target。
- **文档**：`docs/ARCHITECTURE.md` 新增 v2.6 模块表（hybrid/counterfactual/identification/
  adaptive/decision/快照差分）；`docs/ROADMAP.md` 新增 v2.6 线段落。

### Evidence (dev7 真实运行)
- 全量 pytest **271 passed**（260 + 11 新）。
- 7 个 checkpoint 全部向后兼容加载；旧件（v2.1.0/v2.2.1）无残差分位 => safety_boundary
  诚实全 safe；v2.3.1+ 已挂半宽 => 正常计算边界。
- 特性延迟（CPU, batch=32, warmup=2, repeats=10）：predict_next≈87ms、hybrid≈90ms、
  risk≈84ms、identify(grid3)≈28ms、adaptive(h2)≈170ms、counterfactual(h2)≈360ms
  （约 2 步 rollout×2 份前向 + 干预，符合线性预期）。

## [2.6.0+dev6] - 服务接口扩展（反事实 / 辨识 / 风险 / 快照差分）

### Added
- **`udos/server.py`** 在 UDOSService 新增 4 个方法并注册到 do_POST 路由表：
  - `POST /counterfactual`：入参 window[W,R]\|[N,W,R] / horizon(1-8) / scene_params? /
    intervention?(dict)。未训练->409、非法形状->400。用 CounterfactualEngine，返回
    baseline / counterfactual(tolist) / ate_by_step / ate_mean / final_state_diff。
  - `POST /identify`：入参 window[W,R] / horizon?(默认2) / grid_size?(默认5)。
    未训练->409。用 SceneParameterIdentifier，返回 identified_params / param_names / loss_min。
  - `POST /risk`：入参 window / horizon?(默认1) / scene_params?。未训练->409。
    用 RiskGrader，返回 risk_score / risk_level / components。
  - `POST /diff-checkpoints`：入参 name_a / name_b（白名单 checkpoint 名）/ n_per_kind?。
    用 _safe_checkpoint_path 校验两路径（不存在->400）；不需已挂载预测器，直接加载对比。
  - 新增 `_parse_window` 静态助手统一解析 window/scene_params。
- **测试**：`tests/test_v26_service.py`（7 用例）：反事实 200+零干预=基线、未训练 409、
  辨识 200、风险等级∈{...}、diff-checkpoints 200、不存在件 400。
- 更新 server.py 模块文档字符串路由列表。

### Evidence (dev6 真实运行)
- 全量 pytest **260 passed**（253 + 7 新）。
- 预加载 v2.6.0 后：/counterfactual 零干预 ate_mean≈0、baseline==counterfactual；
  scene_params{2:1.2} 干预后 ate_mean>0。/identify 返回 4 维参数 + 正确 param_names。
  /risk risk_level∈{low,medium,high}、score∈[0,1]。/diff-checkpoints(v2.5.2,v2.6.0)
  200 且 params 均 52191；不存在件返回 400。
- 全新 UDOSService.preset("small").counterfactual(...) 抛 ServiceNotReady（HTTP 层 409）。

## [2.6.0+dev5] - 推理快照差分 / checkpoint 数值对比

### Added
- **`udos/persistence.py`**:
  - `diff_snapshots(snap_a, snap_b)`：结构化对比两个 export_snapshot ——
    config_diff（ctm_config / raw_dim / scene_param_dim）、calibration_diff
    （kind、calibrator state_dict 键值、residual_quantiles 逐维差）、ood_diff
    （threshold / mean / precision）。递归 JSON 差分，数值报 max_abs_diff；
    同件导出两次 => identical=True 且各段为空。
  - `compare_checkpoints(path_a, path_b, n_per_kind=16, seed=999)`：加载两 checkpoint，
    在同一份固定 seed 的新鲜测试集上分别 evaluate_predictor，并对单步预测逐位比较；
    返回 a_metrics / b_metrics / pred_diff_max / pred_diff_mean / mse_diff /
    ece_diff（取 calibrated.ece）/ coverage_diff / params_a / params_b。
    纯前向、确定性、不重训。
- **测试**：`tests/test_v26_snapshot_diff.py`（6 用例）：同件差分 identical、受控
  残差分位扰动触发 calibration_diff、v2.5.2/v2.6.0 快照一致性、字段完整、同件 pred_diff≈0。
- `udos/__init__.py` 导出 `diff_snapshots`, `compare_checkpoints`。

### Evidence (dev5 真实运行)
- 全量 pytest **253 passed**（247 + 6 新）。
- **诚实发现**：export_snapshot 不含权重也不含 hybrid 状态，故 v2.6.0 相对 v2.5.2
  （仅新增 hybrid 能力）的快照在结构化层面完全 identical=True；权重差异由
  compare_checkpoints 在张量层给出 —— 实测 v2.5.2 vs v2.6.0 pred_diff_max=0、
  mse_diff=0（二者主权重同源），同件对比 pred_diff_max≈0。
- 受控放大残差分位 1.5x 后，calibration_diff 正确落到 residual_quantiles 逐维差、
  identical=False，config_diff 仍为空（架构未动）。

## [2.6.0+dev4] - 不确定性向下游决策传播（风险分级 / 动作安全边界）

### Added
- **`udos/decision.py`**:
  - `RiskGrader(width_low=0.5, width_high=2.0, ood_threshold_factor=1.0)`：
    纯前向、只读 predictor、确定性聚合三类不确定性：
    conformal 区间平均半宽（已挂半宽时）或 certainty 倒数代理（未挂，诚实退化）、
    OOD 马氏距离（已挂检测器时，按 threshold*因子归一，否则 0）、
    校准后置信均值（已挂校准器时经 calibrator.transform，否则原始 certainty）。
    `risk_score = 0.4*width_norm + 0.3*ood_norm + 0.3*(1-confidence) ∈ [0,1]`；
    risk_level: low(<0.33)/medium(0.33-0.66)/high(>0.66)。
    返回 risk_score / risk_level / components{interval_width, ood_score, confidence}。
  - `safety_boundary(predictor, raw_window, action_candidates, ...)`：
    已挂半宽时按位置维（前 3 维）是否 ≥ 区间下界（median-half_width）判安全，
    distance_to_boundary = min_pos(cand - lower)；未挂半宽时全部 safe=True（诚实
    退化，无区间信息不过滤）。返回 {action_index, safe, distance_to_boundary, reason}。
- **测试**：`tests/test_v26_decision.py`（7 用例）：结构完整、OOD 输入抬高风险、
  同输入确定性、未挂半宽全 safe、手动大残差分位过滤越界候选。
- `udos/__init__.py` 导出 `RiskGrader`, `safety_boundary`。

### Evidence (dev4 真实运行)
- 全量 pytest **247 passed**（240 + 7 新）。
- v2.6.0 checkpoint（已挂 pava 校准 + OOD）grade 返回 risk_level∈{low,medium,high}、
  score∈[0,1]、interval_width_proxy=False。
- OOD 输入（window×5）ood_score 与 risk_score 均严格大于正常输入。
- 手动挂半宽=10：紧贴 median 候选 distance>0 安全；位置维压低 30 的候选 distance<0
  且 reason=below_interval_lower_bound。摘除半宽时两候选均 safe=True、distance=None。

## [2.6.0+dev3] - 自适应计算（自适应 iterations + horizon）

### Added
- **`udos/adaptive.py`**:
  - `AdaptiveStopper(threshold=1e-3, patience=3)`：纯逻辑/无状态，
    `should_stop(certainty_trajectory)` 在最后 patience 个相邻 certainty 变化
    均 < threshold 时返回 True；点数不足时 False。
  - `adaptive_rollout(predictor, raw_window, max_horizon, ...)`：逐步 rollout，
    已挂 conformal 半宽时按相邻步区间宽度增长率 > width_growth_threshold(默认2.0)
    提前停止；未挂半宽时退化为完整 max_horizon（与旧 rollout 逐位一致）。
    返回 trajectory / horizon_used / truncated / reason。
- **`PhysicsPredictor.predict_next_adaptive(...)`**：wrapper 方式跑满全部 CTM
    iterations，再依据 stopper 在已解码逐 tick 输出里选最早收敛 tick；stopper=None
    时返回最终 tick，与旧 predict_next 逐位一致（不真截断 CTM 内部循环）。
- **测试**：`tests/test_v26_adaptive.py`（5 用例）：disabled 等价、stopper 逻辑、
  未挂半宽跑满、宽区间触发截断。
- `udos/__init__.py` 导出 `AdaptiveStopper`, `adaptive_rollout`。

### Evidence (dev3 真实运行)
- 全量 pytest **240 passed**（235 + 5 新）。
- stopper：收敛序列 should_stop=True，波动序列=False，点数不足=False。
- 未挂半宽随机模型 adaptive_rollout horizon_used=max_horizon=5、truncated=False。
- 构造半宽 1.0->100.0（增长率 100x>2.0）：horizon_used=2、truncated=True、
  reason=interval_width_growth。stopper=None 时与 predict_next atol<1e-6。

## [2.6.0+dev2] - 场景参数辨识 / 归因

### Added
- **`udos/identification.py`**:
  - `SceneParameterIdentifier(grid_size=5)`：对
    SCENE_PARAM_NAMES=[v0, accel_a, spring_omega, other_v2] 在物理区间
    （v0∈[-2,2], accel_a∈[-1.5,1.5], spring_omega∈[0.6,1.6], other_v2∈[-0.5,0.5]）
    网格搜索；用窗口内自洽校验（前 W-horizon 帧作输入、后 horizon 帧作目标的
    rollout MSE）打分。匀速/惯性段 v0 已被窗口速度直接观测、模型对其不敏感
    （rollout MSE 在 v0 上平坦），加运动学先验 `0.05*(v0-v0_obs)^2` 打破退化。
    返回 identified_params[4]/param_names/loss_curve_min。
  - `sobol_attribution(...)`：一阶 Sobol 简化版，逐参数独立 ±σ 扰动测输出方差
    V_i，S_i=V_i/sum（和≈1），固定种子确定性。
- **测试**：`tests/test_v26_identification.py`（5 用例）：匀速 v0 反演误差<0.5、
  Sobol 指数和∈[0.9,1.1]、spring_omega 主导、同输入两次调用确定性。
- `udos/__init__.py` 导出 `SceneParameterIdentifier`, `sobol_attribution`。

### Evidence (dev2 真实运行)
- 全量 pytest **235 passed**（230 + 5 新）。
- 诚实记录：v2.6.0 训练 checkpoint 对 spring_omega 的 rollout 灵敏度偏弱
  （直接扰动各参数 Δoutput：accel_a=0.674 > v0=0.099 > v2=0.077 > omega=0.057，
  该模型特性），故 "spring_omega 主导" 用解析弹簧 mock 预测器验证归因机制本身；
  真实件仅验 Sobol 和≈1 与确定性。匀速 v0 反演在网格[-2,-1,0,1,2]上取 1.0，
  真值 1.3，误差 0.3 < 0.5。

## [2.6.0+dev1] - 因果链 / 反事实推演模块

### Added
- **`udos/counterfactual.py` — `CounterfactualEngine`**：持有已训练 predictor 引用,
  纯前向、确定性、不修改模型状态。支持四类干预:
  - `intervention=None/{}` => 基线 rollout, 与 `predictor.rollout` 逐位一致;
  - `{"scene_params": {idx: val}}` => 覆盖指定 scene_param 槽位后 rollout;
  - `{"initial_state": delta}` => 对窗口末帧加扰动后 rollout;
  - `{"velocity_override": v}` => 覆盖初始速度后 rollout。
- **`counterfactual(...)`** 返回 `baseline` / `counterfactual` / `ate_by_step[H]`
  (逐步 MSE 差) / `ate_mean` / `final_state_diff[B,R]` / `intervention`。
- **测试**：`tests/test_v26_counterfactual.py`（5 用例）：零干预逐位等价、
  spring_omega 干预轨迹发散、ATE 非负、速度覆盖改变位置斜率。
- `udos/__init__.py` 导出 `CounterfactualEngine`。

### Evidence (dev1 真实运行)
- 全量 pytest **230 passed**（225 + 5 新）。
- 零干预：`counterfactual==baseline==predictor.rollout`（atol<1e-7），ate_mean=0。
- spring_omega×1.6 干预：轨迹与基线逐位不同，且误差随 rollout 步数单调累积。
- 速度 1.0->3.0 干预：5 步后 x 轴位置末值差 >0.2（方向合理）。

## [2.6.0] - learned-residual 混合物理修正 + 正式训练重建

### Added
- **`udos/hybrid.py` — `HybridPhysicsCorrector`**：一阶欧拉匀速物理骨架
  （`euler_pos=last_pos+last_vel*dt`, `euler_vel=last_vel`）+ 极小残差 MLP
  （输入 concat[model_pred,euler_pred,last_state]=18 -> GELU -> 32 -> 6）。
  `hybrid_pred = euler_pred + residual`。参数量 **806**
  （18*32+32 + 32*6+6 = 576+32+192+6）。`enabled=False` 时原样返回 `model_pred`，
  不计算 euler/residual，逐位一致。
- **`PhysicsPredictor`**：新增 `self.hybrid=None`、`attach_hybrid()/detach_hybrid()`、
  `predict_next(..., hybrid=False, dt=0.5)`；默认 `hybrid=False` 与 v2.5.2 逐位一致。
- **`TrainConfig.hybrid_weight=0.0`**（默认关）：`_parametric_loss` 在
  `hybrid_weight>0 且 model.hybrid 非 None` 时对 hybrid 输出加 MSE（经
  `_last_hyb_loss` 属性透出，保持 3 元组签名不破坏旧调用方）。
- **持久化**：`save_predictor` 在挂 hybrid 时把 hybrid state_dict 单列
  `bundle["hybrid"]`（主 state_dict 排除 `hybrid.*` 键，strict 加载）；
  `load_predictor` 重建并 attach。旧 checkpoint 无 `hybrid` 键 => 保持 None（向后兼容）。
- **正式训练重建**：`scripts/build_v260_checkpoint.py`，落
  `checkpoints/predictor_v2.6.0.pt` 与 `benchmarks/results/training_v2.6.0.json`。
  **hybrid A/B**：`scripts/ablation_hybrid.py` ->
  `benchmarks/results/ablation_hybrid_v2.6.0.json`。
- **Makefile**：`ckpt260`、`hybrid-ablation` target。
- **测试**：`tests/test_v26_hybrid.py`（7 用例）：disabled 等价、残差降低、
  save/load 逐位、weight=0 路径等价。

### Evidence (v2.6.0 真实运行)
- 主 checkpoint（hybrid_weight=0 默认关）：n_params=**52191**，
  final_loss=0.059488, eval_mse=0.045556, ECE=0.056546, coverage=0.8888,
  condition_gain_x=15.393, rollout_growth_x=2.805, train_seconds=111.3,
  epochs_run=60, best_epoch=57, reload_consistent=true。
  OOD：ood_hit_rate=0.8104, id_false_alarm=0.0646。
- **hybrid A/B 结论（诚实）**：quick(n_per_kind=16, epochs=20, hybrid_weight=0.5)
  小模型上，同一权重切 hybrid on/off 的运动学残差从 **0.6044 -> 0.0384**，
  降幅 **93.6%**（euler 骨架对惯性段解析精确，残差 MLP 仅补小量）——
  hybrid 在运动学一致性上收益显著，主件默认关以保持旧输出逐位稳定。
- 全量 pytest **225 passed**（218 旧 + 7 新）。旧 6 个 checkpoint 全部可加载、
  预测正常、hybrid=None；新 v2.6.0 件可加载、reload 一致。

## [2.5.2] - 最终训练重建 + 全量验证（2.5 线终点）

### Added
- **正式训练重建**：`scripts/build_v252_checkpoint.py`（仿 v250），落
  `checkpoints/predictor_v2.5.2.pt`（52191 参数）与
  `benchmarks/results/training_v2.5.2.json`（final_loss/eval_mse/ece/coverage/
  interval_width/params_count/batch_inference_max_diff）。
- **旧 checkpoint 向后兼容测试**：`tests/test_v25_backcompat.py`（6 个用例）：
  v2.1.0/v2.2.1/v2.3.1/v2.4.0/v2.5.0 五件均可加载、预测正常、版本元数据正确；
  v2.3.1+ 已校准、v2.4.0+ 挂 OOD 检测器。
- **真实 HTTP 服务逐接口验证**：`scripts/verify_service_v252.py`：启动服务
  （--checkpoint v2.5.2），验证 /health=2.5.2、/evaluate 含 calibration/interval 段、
  /detect-ood 200、/metrics 200（Prometheus 文本）、/rollback 409（无历史）/200（有历史）、
  /predict guard 200、/checkpoints 200。
- 版本号最终定为 2.5.2，所有同步文件一致（__init__/pyproject/Makefile/compose/Dockerfile）。

### Evidence (v2.5.2 训练指标)
- final_loss=0.0595, eval_mse=0.0456, ECE=0.0565, coverage=0.8888,
  interval_width=0.541, params=52191, batch_inference_max_diff=1.97e-06。
- 全量 pytest **218 passed**，覆盖率 **92%**（batch 90%、cache 90%、server 90%、
  persistence 93%）。
- 服务验证：/health=200(2.5.2)、/evaluate=200、/detect-ood=200、/metrics=200、
  /rollback=409→200、/predict=200、/checkpoints=200。

### Notes
- 批量推理与缓存在 CPU 小模型上单次推理收益不明显（模型仅 52191 参数），
  主要收益在重复请求/大批量场景（诚实记录）。所有新能力 opt-in 默认关闭。

## [2.5.1] - 服务监控指标 + 轻量导出/无状态快照 + 回滚

### Added
- **`GET /metrics`**：`udos/server.py` 新增 `MetricsCollector`（纯标准库，线程安全），
  采集每端点请求计数、延迟分位（p50/p95/p99）、缓存命中率、OOD 触发率；Prometheus
  文本格式 exposition（`# HELP`/`# TYPE` + counter/gauge），不引入 prometheus_client。
  HTTP handler 在 `finally` 中自动计时。
- **无状态快照**：`udos/persistence.py` 新增 `export_snapshot(predictor)` /
  `import_snapshot(predictor, snapshot)`：导出架构配置 + 校准器 + 残差分位 + OOD 统计
  （**不含权重**），用于快速恢复推理后处理；导入时校验 ctm_config/raw_dim/scene_param_dim
  一致，不匹配拒绝。服务新增 `POST /export-snapshot` / `POST /import-snapshot`。
- **`POST /rollback`**：维护已加载 checkpoint 栈（`_load_stack`），回滚到上一已加载件；
  栈长 < 2 时返回 409。启动时 `--checkpoint` 预加载占栈底。
- **`/evaluate` 含 metrics 段**：响应新增 `metrics` 字段（当前服务指标快照）。
- 19 个 v2.5.1 回归测试（MetricsCollector 计数/分位/OOD/缓存/Prometheus 格式/空采集器；
  服务级 /metrics HTTP 200 + 计数递增；export/import 无权重往返/架构不匹配拒绝；
  rollback 无历史 409 / 两加载后回滚 / 预加载栈底 / 回滚预测一致）。

### Notes
- 指标采集为 opt-in 服务侧能力，不改变推理数值；快照不含权重（权重仍走 .pt）。
  回滚栈仅记录已加载 checkpoint 路径，不复制模型权重（回滚时重新 load_predictor）。

## [2.5.0] - 批量推理引擎 + 推理缓存 + 正式训练重建

### Added
- **`udos/batch.py`：`BatchPredictor`**：包装 `PhysicsPredictor`，支持变长序列
  `List[Tensor(W_i,RAW_DIM)]` 自动按窗口长度分组批量推理，等长 `[B,W,RAW]` 直接前向；
  大 batch 超过 `max_shard` 自动分片。结果与逐笔 `predict_next` 逐位一致（atol=1e-5，
  由 `test_v25_batch.py` 锁死）。空 batch 守卫、非法形状/scene_params 数量校验。
- **`udos/cache.py`：`InferenceCache`**：LRU 推理结果缓存，key = 输入张量哈希 +
  模型参数哈希（SHA-256），默认 `enabled=False`（opt-in）；开启后命中返回与未命中
  完全相同输出（`torch.equal`）；模型权重变化自动失效；线程安全计数 hits/misses/hit_rate。
- **`PhysicsPredictor.predict_batch`**：委托 `BatchPredictor`，可选 `cache` / `max_shard` /
  `guard`。
- **正式训练重建**：`scripts/build_v250_checkpoint.py`（仿 v240，front 默认 + PAVA 校准
  + conformal 区间 + OOD 拟合），落 `checkpoints/predictor_v2.5.0.pt` 与
  `benchmarks/results/training_v2.5.0.json`（final_loss/eval_mse/ece/coverage/
  interval_width/params_count/batch_inference_max_diff）。
- Makefile 新增 `ckpt25` target。
- 22 个 v2.5.0 回归测试（批量==逐笔×3、变长分组、分片一致、空 batch 守卫×2、形状校验×2、
  scene_params 数量、guard 透传、便捷函数、无 scene_params；缓存默认关闭、命中相同、
  miss 计算写回、LRU 淘汰、关闭透传、权重变化失效、统计计数、clear）。

### Notes
- 批量推理与缓存均为 opt-in 推理侧能力，不新增可学参数（参数量仍 ~52191），不改变
  `predict_next` / `rollout` / `predict_interval` 默认输出。CPU 小模型推理本身快，
  批量/缓存在重复请求场景收益明显，单次推理收益有限（诚实记录）。

## [2.4.16] - 2.4 线文档定稿（17 节点收口）

### Added / Changed
- `docs/ARCHITECTURE.md`：新增「v2.4 可信推演套件」模块表（ood / StreamingDriftDetector /
  DeepEnsemble / PredictionGuard / per_step_confidence）。
- `docs/DEPLOYMENT.md`：新增 `POST /detect-ood`（v2.4.8）与 `POST /predict`（v2.4.12）接口契约。
- `README.md`：标题与首屏 callout 升 v2.4.16，概述 2.4 线鲁棒性/不确定性主线与覆盖率。
- `docs/VERSION_PLAN_2.4.md`：标注 17/17 节点完成状态表。
- 本 CHANGELOG 至此凑齐 **2.4.0→2.4.16 共 17 条**。

### Notes
- 文档与代码/接口一致；不新增运行时能力。版本号同步至 2.4.16（2.4 线终点）。

## [2.4.15] - 2.4 线全量回归 + 覆盖率加固

### Notes
- 全量 `pytest` **171 passed**；`--cov=udos` 报告落 `docs/COVERAGE_v2.4.15.txt`，总计 93%。
- 新模块覆盖率门控（≥80%）全部达标：ood(含 StreamingDriftDetector) **93%**、
  ensemble **97%**、guard **93%**、calibration(含 TemperatureScaling/temp_scale) **98%**、
  persistence **98%**、server **88%**、evaluation **95%**。
- 不新增功能，仅回归确认；无回归需修复。版本号同步至 2.4.15（2.4 线收尾倒数第二站）。

## [2.4.14] - 在线漂移滑动窗口累积

### Added
- `udos/ood.py` 新增 `StreamingDriftDetector`（继承 `DistributionDriftDetector`，复用参考分布拟合/
  马氏打分/阈值，接口同构）：固定窗口 W 的环形缓冲在线累积，`update(x)` / `window_mean()` /
  `window_var()` / `window_drift_score()` / `reset()`。
- 窗口均值/方差与"对窗口内样本批处理"逐位一致（流式==批处理）；`reset()` 仅清窗口、保留参考分布。
- `udos/__init__.py` 导出 `StreamingDriftDetector`。
- 5 个 v2.4.14 回归测试（流式==批、窗口封顶弹出最旧、reset、与离线 score 逐位同语义、窗口漂移分）。

### Notes
- 流式检测器为 opt-in，离线 `DistributionDriftDetector` 行为逐位不变。版本号同步至 2.4.14。

## [2.4.13] - 集成模型 checkpoint 持久化

### Added
- `udos/persistence.py` 新增 `save_ensemble(ensemble, path, metrics)` /
  `load_ensemble(path) -> (DeepEnsemble, meta)`：保存 N 个成员 state_dict + 共享 ctm_config +
  版本元数据；与旧 `save_predictor/load_predictor` 并存、互不影响。
- `udos/__init__.py` 导出 `save_ensemble/load_ensemble`。
- 3 个 v2.4.13 回归测试（save/load 逐位一致+成员数正确、旧单模型路径不变、拒绝非集成档）。

### Notes
- 旧单模型 checkpoint（含 v2.4.0 正式件）加载路径逐位不变。版本号同步至 2.4.13。

## [2.4.12] - 退化守卫接入服务

### Added
- 新增 `POST /predict`：入参 `window: [W,RAW]|[N,W,RAW]`、可选 `scene_params`、可选 `guard`
  （默认 false 透传；true 过 `PredictionGuard` 并返回 `n_fallbacks/n_clips`）。未训练 409、非法形状 400。
- `POST /evaluate` 新增可选 `guard=true`：在测试集上跑守卫透传预测，返回 `guard` 段
  （`n_fallbacks/n_clips`）；健康模型为 0/0。默认不带 guard 段，旧响应不变。
- 6 个 v2.4.12 回归测试（409/形状/坏输入/NaN 触发回退/输出有限/evaluate guard 段）。

### Notes
- 守卫为 opt-in 后处理，`guard=False` 与旧版逐位一致；每请求全新计数，不串味。版本号同步至 2.4.12。

## [2.4.11] - 多水平区间覆盖率验证基准

### Added
- `scripts/ablation_conformal_levels.py`：80/90/95 三水平（alpha=0.2/0.1/0.05）在
  **独立测试集**上的覆盖率 + 平均宽度，多种子，落
  `benchmarks/results/conformal_levels_v2.4.11.json`；Makefile 新增 `conformal-levels`。
- 3 个 v2.4.11 回归测试（三水平结构/覆盖单调/宽度增宽/JSON 含三水平）。

### Evidence (诚实记录, 3 种子 quick)
- 三水平均**保守过覆盖**（实测 − 名义）：80%→0.847(+0.047)、90%→0.926(+0.026)、
  95%→0.966(+0.016)——split-conformal 在小校准集上不系统性 undercover。
- 覆盖率随名义水平单调不下降；平均宽度严格增宽 1.41<2.06<2.66（80<90<95）。

## [2.4.10] - 逐步逐维置信度可解释性

### Added
- `PhysicsPredictor.per_step_confidence(raw_window, horizon, ...)`：返回 `[B,H,RAW_DIM]`
  置信矩阵。步置信 = 每 rollout 步 CTM 标量 certainty（已挂校准器则先校准）；逐维用
  conformal 半宽-derived `exp(-q/median_q)` 可靠性因子调制（区间越窄置信越高），未挂半宽时
  逐维广播（`mean(dim=-1)` 与标量置信逐位一致）。输出裁剪到 [0,1]，纯前向确定性。
- `evaluate_predictor` 的返回新增 `confidence_breakdown` 段（跨测试集 `[H,RAW]` 均值矩阵、
  `mean_by_step`、`min/max`、`shape`），`/evaluate` 随之携带。
- 3 个 v2.4.10 回归测试（形状/值域/与标量一致/breakdown 段存在）。

### Notes
- 可解释性为 opt-in 读取路径，不改变任何旧指标键的数值（仅新增键）。版本号同步至 2.4.10。

## [2.4.9] - 噪声鲁棒性 A/B 基准

### Added
- `scripts/ablation_noise_robustness.py`：3×2 网格（train_sigma∈{0,0.05,0.1} × test_sigma∈{0,0.1}）
  同合同多种子单步 MSE 对比，落 `benchmarks/results/noise_robustness_v2.4.9.json`；Makefile 新增 `noise-ablation`。
- 4 个 v2.4.9 回归测试（eval_mse 结构/单调加噪/确定性/JSON 含三 train_sigma）。

### Evidence (诚实记录, 3 种子 quick)
- 训练噪声单调**缩小**鲁棒性缺口（MSE_noisy−MSE_clean：0.0044→0.0019→−0.0033；
  train_sigma>0 在 **3/3 种子**缺口均小于 sigma=0）。
- 但有真实代价：**干净集 MSE 随 train_sigma 显著上升**（0.402→0.617→0.679）——
  典型偏差-方差权衡，噪声增强以干净精度换测试噪声稳健性，**非免费午餐**。
- 故默认 `TrainConfig.noise_sigma=0.0` 不变；噪声增强保持 opt-in。

## [2.4.8] - OOD / 漂移服务接口

### Added
- `udos/server.py` 新增 `POST /detect-ood`（入参 `sequence: [N,W,RAW]`，返回每样本马氏距离/
  阈值/OOD 命中率/整批漂移摘要；未训练或未挂检测器 -> 409，非法形状 -> 400）。
- `POST /evaluate` 新增可选 `include_ood=true`（默认 false，不改变旧响应）。
- 5 个 v2.4.8 回归测试。

### Notes
- /detect-ood 为新增路由，旧接口行为不变。版本号同步至 2.4.8（2.4 线前 9 节点收口）。

## [2.4.7] - 集成不确定性 + conformal 融合

### Added
- `udos/ensemble.py` 新增 `DeepEnsemble.predict_interval`：集成 rollout 均值作中值，
  半宽 = sqrt(q² + v)（q=成员残差 conformal 半宽，v=集成批内平均认知方差）。
- N=1 时 v=0，逐位退化为普通区间；N>1 区间随成员分歧增宽。
- 4 个 v2.4.7 回归测试。

### Notes
- 区间融合为 opt-in；默认不改变单模型旧路径。版本号同步至 2.4.7。

## [2.4.6] - 校准方法 A/B 基准 (PAVA vs Temperature vs None)

### Added
- `scripts/ablation_calibration.py`：同合同多种子 A/B，比较 ECE/Spearman/覆盖率；
  落 `benchmarks/results/calibration_ablation_v2.4.6.json`；Makefile 新增 `calib-ablation`。
- 3 个 v2.4.6 回归测试。

### Evidence (诚实记录, 3 种子 quick)
- **PAVA** ECE≈0.052，相对 None 在 3/3 种子降低 ECE（Spearman≈0.13，覆盖率≈0.91）。
- **Temperature** ECE≈0.497，与 None 持平（0/3 种子更优）：本任务单调缩放未降 ECE，
  网格自动选 T≈1；**不作为默认**，仅与 PAVA 并存供对比。默认 method 仍为 pava。

## [2.4.5] - 退化守卫 / 安全护栏

### Added
- **新模块 `udos/guard.py`**：`PredictionGuard`（NaN/inf 整行回退到上一有效步或零向量、
  物理越界截断、修复/截断计数）。
- `PhysicsPredictor.predict_next(guard=False)`（默认 False 透传旧输出）；`attach_guard`。
- 6 个 v2.4.5 回归测试。

### Notes
- 守卫默认不激活，guard=False 与旧版逐位一致。版本号同步至 2.4.5。

## [2.4.4] - 多名义水平 conformal 区间

### Added
- `udos/calibration.py` 新增 `ALLOWED_ALPHAS=(0.2,0.1,0.05)` 与 `compute_conformal_halfwidths`；
  `PhysicsPredictor.predict_interval(alpha=...)` 支持 80%/90%/95%，默认 alpha=0.1 与 v2.3 逐位等价。
- `fit_predictor_calibration` 在 report 输出 `conformal_by_alpha`；persistence 持久化多水平半宽（向后兼容）。
- 6 个 v2.4.4 回归测试。

### Notes
- 默认 alpha=0.1 路径不变；宽度随 alpha 减小而严格增宽。版本号同步至 2.4.4。

## [2.4.3] - 数据增强 / 噪声鲁棒训练

### Added
- `udos/dynamics.py` 新增 `noise_augment(x, sigma, generator=None)`（零均值高斯噪声注入）。
- `TrainConfig.noise_sigma`（默认 0.0，逐位等价旧版）；训练循环对输入窗口按 sigma 加噪，确定性可复现。
- 6 个 v2.4.3 回归测试。

### Notes
- sigma=0 时原样返回输入、不改变旧训练行为；>0 为 opt-in 噪声鲁棒增强。版本号同步至 2.4.3。

## [2.4.2] - 温度缩放校准方法

### Added
- `udos/calibration.py` 新增 `TemperatureScaling`（单参数 T：logit 温度缩放 + 网格搜索最优 T），
  与 PAVA 并存；`fit_predictor_calibration` 支持 `method="pava"(默认,逐位等价旧版)|"temperature"|"none"`。
- persistence 校准 bundle 增加 `kind` 字段，按 kind 重建 PAVA/Temperature（旧档无 kind 按 PAVA，向后兼容）。
- 7 个 v2.4.2 回归测试。

### Notes
- 默认 method 仍为 pava，旧行为不变；T=1 恒等、温度缩放单调保序。版本号同步至 2.4.2。

## [2.4.1] - 深度集成不确定性 (opt-in)

### Added
- **新模块 `udos/ensemble.py`**：`DeepEnsemble`（N 个同架构、多种子 PhysicsPredictor），
  `predict_next/rollout` 返回 `{mean, variance, members}`，方差为成员间无偏样本方差。
- N=1 时方差恒 0、均值逐位等价单模型；`DeepEnsemble.create` 多种子构造与轻量 save/load。
- 6 个 v2.4.1 回归测试。

### Notes
- 集成纯包装已有 PhysicsPredictor、零额外可学参数、默认 opt-in；不改变任何旧默认输出。
- 版本号同步至 2.4.1。

## [2.4.0] - OOD/分布漂移检测核心 + 正式训练重建

### Added
- **新模块 `udos/ood.py`**：`DistributionDriftDetector`（岭正则马氏距离打分 + 经验分位阈值
  + 双样本 KS 统计量），`fit/score/is_ood/ks_drift/ks_two_sample_vs_reference`，纯 torch、零依赖。
- `PhysicsPredictor.attach_ood_detector` / `ood_score`（opt-in，未挂载显式报错不静默退化）；
  persistence bundle 增加可选 `ood` 键（旧档无此键按未挂载处理，向后兼容）。
- 正式重建 `checkpoints/predictor_v2.4.0.pt`、`scripts/build_v240_checkpoint.py`、
  `benchmarks/results/training_v2.4.0.json`；Makefile 新增 `ckpt24` target。
- 11 个 v2.4 OOD 回归测试。

### Notes
- OOD 检测为外挂后处理，默认不改变任何旧默认输出；阈值取训练马氏距离 (1-alpha) 保守上分位。
- 版本号同步 `udos/__init__.py` / `pyproject.toml` / `Makefile` / `docker-compose.yml` /
  `Dockerfile` 至 2.4.0。

## [2.3.1] - 2.3 线补丁与发布收口

### Added
- 正式发布件 `checkpoints/predictor_v2.3.1.pt`、`scripts/build_v231_checkpoint.py`、
  `scripts/ablation_horizon_weight.py`（F2 多方案/多种子复现）。
- 新增 v2.3 回归测试；冒烟扩项；`docs/VERSION_PLAN_2.3.md`、`docs/VERIFICATION_v2.3.1.md`。

### Fixed
- 校准/区间在空集、单样本、零误差、退化保序等边界的健壮性；版本三处与镜像 tag 对齐 2.3.1；
  文档与接口表对齐。

## [2.3.0] - 可校准、长时程更稳健的可信推演

### Added
- **置信度保序回归校准**（新模块 `udos/calibration.py`，PAVA 零依赖）：把 CTM 同步置信度
  校准为与经验精度对齐的置信分，输出回归 ECE/可靠性分桶/Spearman，校准器可随 checkpoint 持久化。
- **多时域均衡损失**：`TrainConfig.step_weight_scheme = front(默认,逐位复现2.2.1)/uniform/back`。
  两组同合同 A/B 显示 uniform 在"无早停"口径 3/3 远期更优、但在"早停 build 口径"被 front
  反超，**优劣对训练口径敏感、不稳健**，故沿用 SS 纪律：三方案均 opt-in、默认保持 front，
  正反结果分别落 `ablation_horizon_weight.py` 与 `sensitivity_step_weight_pipeline.py`。
- **split-conformal 经验预测区间**：`PhysicsPredictor.predict_interval`，评估覆盖率/宽度。
- 服务 `POST /calibrate`、`GET /checkpoints`、`POST /load`；`/evaluate` 增校准与区间段。
- persistence bundle 增加可选 `calibration`（旧档无此键按未校准处理，向后兼容）。

### Notes
- 校准/区间/新接口均为外挂增量、默认不改变 2.2.1 输出；多时域损失默认仍为 front（逐位复现
  2.2.1），uniform/back 为 opt-in；旧 checkpoint 正常载入。

## [2.2.1] - 2.2 线补丁与发布收口

### Added
- 正式发布件 `checkpoints/predictor_v2.2.1.pt`（52,191 参数，单步 MSE 0.0446、场景条件
  增益 14.10×、4 步累积率 2.49×）；`scripts/build_v221_checkpoint.py`、
  `scripts/ablation_scheduled_sampling.py`。
- 13 个 v2.2 回归测试（总 77）；冒烟扩到 15 项；`docs/VERSION_PLAN_2.2.md`、
  `docs/ROADMAP.md`、`docs/VERIFICATION_v2.2.1.md`。

### Fixed
- 健壮性与边界：Scheduled Sampling 在 horizon=1 时确定为无操作；评估/保存接口在未挂载
  预测器时返回明确的 409 而非 500；`/save` 文件名白名单清洗 + 最终路径不可逃逸校验。
- 空/退化数据集与评估分桶的除零保护。
- `health.version` 改为引用 `udos.__version__` 单一来源；版本号三处与镜像 tag 统一 2.2.1。
- 文档（README/DEPLOYMENT/ARCHITECTURE）对齐 2.2 能力与接口。

### Notes
- F1 Scheduled Sampling 经同合同 A/B 双种子复现，收益随种子方向反转，故保持 opt-in、
  默认关闭、不宣称稳健增益（详见验证报告 §2）。

## [2.2.0] - 长时程鲁棒推演 + 训练治理 + 模型管理

### Added
- **Scheduled Sampling**：`TrainConfig(ss_max, ss_start, ss_warmup)` 线性爬坡，多步训练中
  按概率以模型自身上一步（截断梯度）预测替代真值拼回窗口，缓解自由 rollout 的 exposure
  bias；默认 `ss_max=0`（纯 teacher-forcing，与 2.1 数值等价，完全向后兼容）。
- **训练治理**：早停（`patience/min_delta`，记录 `best_epoch/best_eval/best_state`）、
  `udos.training.set_seed` 统一确定性入口（默认行为不变）。
- **评估增强**：`rollout_growth_x`（长时程误差累积率）、`confidence_stratification`
  （按置信度三档的误差分层与高/低档误差比）。
- **模型管理服务**：`POST /evaluate`（对挂载预测器在新鲜测试集评估）、
  `POST /save`（连同指标落 checkpoint）；未训练返回 409。
- 文档：`docs/ROADMAP.md`、`docs/VERSION_PLAN_2.2.md`、本 CHANGELOG。

### Changed
- 服务 `train` 支持透传 scheduled-sampling 课程参数。
- 评估返回在保持旧键不变前提下新增上述指标键。

## [2.1.0]
### Added
- 参数化数据集 `ParametricDynamicsDataset`（历史窗口 X、未来 H 步 Y、隐藏物理参数 P）。
- `PhysicsPredictor.rollout` 多步自由滚动；teacher-forcing 多步训练与 `reason(horizon=)`。
- 场景隐藏参数经 `scene_encoder` 进入训练回路，联合训练真正激活 CTM `scene_gate`。
- `udos/evaluation.py` 统一评估：单步/场景消融/分类型/rollout 曲线/运动学一致性。
- 服务 `--checkpoint` 启动预加载；v2.0 旧 checkpoint 缺字段也可载入。
### Fixed
- 运动学一致性误用预测时刻速度积分（对加速运动系统性偏差），改用前一帧速度并降级为
  分类型诊断、默认不作全局硬损失。

## [2.0.0]
### Added
- `udos/dynamics.py`、`udos/training.py`、`udos/persistence.py`：合成动力学、物理预测
  训练闭环、checkpoint 存/载。
- CTM 场景条件化通路与 GPM `scene_embedding`（零门控，向后兼容）。
- 可解释下一时刻 pos/vel；服务 `POST /train`。

## [0.2.0]
### Added
- 契约/反例测试、性能基准与回归守卫、零依赖 HTTP 服务、Docker/compose/Makefile、冒烟。
### Fixed
- GPM 场景编码器每次前向重建导致的重复内化结果不确定，改为持久化子模块。

## [0.1.0]
### Added
- 对齐 Sakana AI CTM / Doc-to-LoRA 的双引擎机制内核与上游真实 CTM 适配。
