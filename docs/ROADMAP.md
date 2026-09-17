# UDOS 推演引擎版本路线图（ROADMAP）

原则：语义化版本；每个版本的每条能力都要有可复现实验/测试证据，详见对应
`docs/VERIFICATION_v*.md`。向前演进保持向后兼容（旧测试不删、不放宽）。

## 已发布

### v4.5.3 — 隐式思考 / Latent Reasoning（CPU 合成, 外挂零梯度, opt-in）
analogy, not reproduction：CTM 连续隐藏状态上的 K 条潜路径并行探索（latent best-of-K，
非 token 层 beam）+ 潜空间聚合/选路 + 隐式摘要；四档 effort（none/low/high/max，
none 与 v4.4.1 默认逐位等价为默认，余 opt-in）；难度信号自适应路由（CTM 收敛/集成
分歧/OOD/任务复杂度/停机判据 → effort + 是否显式化 + 可解释理由）；多专家在思考深度上
协作（三专家 Agent-as-Tool 打分、分歧大则升级 effort 与 star→chain）；high/max 显式链与
4.4 Trace 统一。HTTP `POST /reason/latent`、`POST /reason/route`。诚实账本：隐式不改善
物理 MSE（外挂不动主预测器）、不省延迟，价值在可观测/可追溯；自适应路由把算力花在难题上。
主参恒 52191、eval_mse 恒 0.045556、33 代 checkpoint 全兼容。详见
`docs/LATENT_REASONING_RESEARCH.md`、`benchmarks/results/pareto_v45.json`。

### v3.9.x — 宇树 UnifoLM-WLA 机制类比线（CPU 合成, 外挂零梯度）
analogy, not reproduction：统一具身推理头（编排 spatial/affordance/unified-head）、
动作三分组（EEF pose/joints/lower-body）、稀疏 change-mask vs 3.6 PWM 稠密 rollout A/B、
三路 RVQ 动作分词、动作-状态-任务对齐、外挂少步 flow-matching 动作解码器（被直接回归
反证→opt-in 留候选账本）、跨本体迁移 A/B、多任务统一头评测矩阵、`POST /wla/er`。
主参恒 52191、eval_mse 恒 0.045556、25→27 代 checkpoint 全兼容。详见
`docs/UNITREE_WLA_RESEARCH.md`、`docs/WLA_CANDIDATE_LEDGER.md`。

### v0.1.0 — 机制对齐内核
对齐 Sakana AI CTM 与 Doc-to-LoRA 的真实 API：PCE 物理 Token、CTM 内部时间轴/
神经元级时序/神经同步/确定性早停、GPM Perceiver 瓶颈+HyperLoRA+分块聚合+前向补丁注入、
双引擎协同、Debug 面板、上游真实 CTM 适配。18 测试。

### v0.2.0 — 测试 / 验证 / 部署
契约+反例加固到 36 测试、性能基准与回归守卫、零依赖 HTTP 服务、Docker/compose/Makefile、
端到端冒烟；修复 GPM 场景编码器每次前向重建导致的不确定性缺陷。

### v2.0.0 — 第二代：能学习 · 真耦合 · 可持久化 · 可解释
合成动力学训练闭环（MSE 降 72×、胜朴素基线）、GPM→CTM 零门控场景条件、checkpoint
存/载、可解释下一时刻 pos/vel、服务 `/train`。55 测试，覆盖率 90%。

### v2.1.0 — 多步滚动推演 + 场景条件联合训练
参数化数据集（隐藏物理参数 P）、teacher-forcing 多步训练与自由 rollout、场景条件
消融（condition_gain 9.93×，scene_gate 真正被学出）、运动学一致性诊断、评估体系、
服务 `--checkpoint` 预加载。64 测试，覆盖率 95%。

### v2.2.0 — 长时程鲁棒推演 + 训练治理 + 模型管理
- Scheduled Sampling 缓解自由 rollout 误差累积（同合同 A/B 验证）。
- 早停/最佳选择、统一确定性种子。
- 置信度分层校准、rollout 累积率指标。
- 服务 `/evaluate`、`/save` 模型管理。

### v2.2.1 — 2.2 线补丁与发布收口
边界/数值健壮性、接口错误语义（未训练 409、路径白名单）、文档与版本对齐、
QA 全量收口后发布。77 测试。

### v2.3.0 — 可校准、长时程更稳健的可信推演
- 置信度保序回归校准（回归 ECE/可靠性，确定性后处理），回应 2.2.1「certainty 未校准」。
- 多时域均衡损失（front/uniform/back 权重方案，默认 front；经两组 A/B 证为口径敏感、opt-in）。
- split-conformal 经验预测区间与覆盖率。
- 服务 `/calibrate`、`/checkpoints`、`/load`，校准器随 checkpoint 持久化。

### v2.3.1 — 2.3 线补丁与发布收口
边界/数值健壮性、版本与文档对齐、QA 全量收口后发布。

### v2.4.0–v2.4.16 — 鲁棒性 + 不确定性主线（17 节点）
在 v2.3 可校准置信之上补齐「可信物理推演」：**OOD/分布漂移检测**（马氏距离+KS，
`udos/ood.py`）、**在线流式漂移**（`StreamingDriftDetector`）、**深度集成**（`ensemble.py`）、
**退化守卫**（`guard.py`）、**温度缩放**（与 PAVA 并存，实测未降 ECE→opt-in）、**噪声鲁棒训练**
（`noise_sigma` opt-in，偏差-方差权衡）、**多水平 conformal 区间**（80/90/95，保守过覆盖）、
**逐步逐维置信矩阵**、服务端 `/detect-ood`、`/predict(guard)`、集成持久化。A/B 负面结论诚实留档。

### v2.5.0–v2.5.2 — 效率/工程化与可服务性主线（3 节点，20 节点终点）
- **v2.5.0**：`udos/batch.py` BatchPredictor（批量==逐笔，max diff 1.97e-06）、
  `udos/cache.py` InferenceCache（LRU opt-in）、正式重建 `predictor_v2.5.0.pt`。
- **v2.5.1**：`GET /metrics`（Prometheus 文本）、无状态快照 `export/import_snapshot`、
  `POST /rollback`。
- **v2.5.2（✅ 完成，对外发布）**：最终正式件 `predictor_v2.5.2.pt`（52191 参数）、
  `training_v2.5.2.json`、旧 checkpoint 兼容、真实 HTTP 逐接口验证、zip 独立解压复跑、
  `docs/VERIFICATION_v2.5.2.md`。全量 **218 passed / 92%**。批量/缓存在 CPU 小模型收益有限，
  诚实记录、保持 opt-in。

### v2.6.0–v2.6.0+dev7 — 因果/反事实/决策主线（7 节点，纯前向、不重训）
- **v2.6.0**：`udos/hybrid.py` HybridPhysicsCorrector（一阶欧拉骨架+学习残差，默认不挂、
  旧路径逐位一致）；正式件 `predictor_v2.6.0.pt`（52191 参数）。
- **dev1**：`udos/counterfactual.py` CounterfactualEngine 干预式反事实 rollout + ATE。
- **dev2**：`udos/identification.py` SceneParameterIdentifier 网格反演隐藏物理参数 +
  sobol_attribution 一阶敏感性。
- **dev3**：`udos/adaptive.py` AdaptiveStopper + adaptive_rollout（半宽增长率早退）+
  predict_next_adaptive。
- **dev4**：`udos/decision.py` RiskGrader（区间宽+OOD+置信->风险分/三档）+
  safety_boundary（区间下界过滤候选动作，无信息诚实退化）。
- **dev5**：`persistence.diff_snapshots`（结构化快照差分）+ `compare_checkpoints`
 （两 checkpoint 同测试集评估 + 逐位预测差）。
- **dev6**：服务新增 `POST /counterfactual`、`/identify`、`/risk`、`/diff-checkpoints`。
- **dev7（✅ 完成）**：跨特性集成加固、7 checkpoint 后向兼容、特性延迟基准
  `benchmarks/results/feature_latency_v2.6.0.json`、ARCHITECTURE/ROADMAP 更新。
  全量 **260+ passed**；4 个新端点可用；零重依赖（纯 torch+标准库）。

### v2.6.1（✅ 完成）— Patch 1：缺陷修复 + 边缘加固 + 文档精修
- 空 batch 返回 `[0,RAW_DIM]`、scene_params 非有限值显式 ValueError、horizon=1 自适应正常、
  RiskGrader OOD NaN 降级、反事实 NaN 干预报错、guard 移到 hybrid 之后清洗最终输出。
- `tests/test_v261_edge.py`（10 用例）；DEPLOYMENT/README/CHANGELOG 对齐。全量 **281 passed**。

### v2.6.2（✅ 完成，2.6 线终点正式发布件）
- `scripts/build_v262_checkpoint.py` 正式重建 `checkpoints/predictor_v2.6.2.pt`（seed=42、
  n_per_kind=48、epochs=60、front、hybrid_weight=0、52191 参数、reload_consistent）。
- `training_v2.6.2.json` 指标完整；8 checkpoint 后向兼容；真实 HTTP 逐接口验证（含 4 新端点
  200/400/409）；`docs/VERIFICATION_v2.6.2.md`；zip 独立解压到 /tmp 复跑通过。
- 全量 **281 passed / 92% 覆盖率**。

### v2.7.0–v2.7.3（✅ 完成，从预测到行动的闭环）
- **v2.7.0**：`udos/policy.py` MPCActionSelector（候选动作 rollout + 风险/安全打分优选）；
  正式件 `predictor_v2.7.0.pt`（52191 参数）。
- **dev1**：`udos/online.py` OnlineAdapter 流式漂移触发再校准（默认只重跑 PAVA、不改权重）。
- **dev2**：`udos/active_learning.py` UncertaintySampler 集成方差+区间宽+OOD 合成信息增益选点
  （A/B：主动 0.264 vs 随机 0.502，有效）。
- **dev3**：`udos/lite.py` 幅值剪枝 / 动态 INT8 量化 / 蒸馏学生（opt-in，默认全量模型不变）。
- **dev4**：`udos/hierarchical.py` HierarchicalRollout 多尺度长时域 rollout
  （**负面证据**：单自回归头下与平铺逐位一致，opt-in 脚手架）。
- **dev5**：`udos/experiment.py` ExperimentRegistry 实验元数据注册 + 多种子 sweep（mean/std/best）。
- **dev6**：服务新增 `POST /policy/select`、`/online/adapt`、`/active/sample`、`GET /experiments`。
- **2.7.1**：跨特性集成加固、9 checkpoint 后向兼容、`feature_latency_v2.7.0.json`。
- **2.7.2**：边缘加固（空动作/零池/NaN 防护/horizon=1 退化/剪枝与蒸馏 save-load 一致/guard 兼容）。
- **2.7.3（✅ 2.7 线终点正式发布件）**：正式重建 `predictor_v2.7.3.pt`，全量验证 + zip 发布。

### v2.8.0–v2.8.3（✅ 完成，Physical Loop 统一闭环 + 共享 backbone 多任务头）
- **v2.8.0**：`udos/physical_loop.py` PhysicalLoopRunner 五步显式编排骨架；正式重建
  `predictor_v2.8.0.pt`（52191 参数，eval_mse 0.0456）。
- **dev1**：observe（obs_encoder 特征提取）+ understand（结构化理解向量 [B,8]：目标状态/风险标记）。
- **dev2**：predict_action（MPCActionSelector，候选动作调用方提供/默认，动作历史累积）。
- **dev3**：future_state（rollout 多步轨迹 + 不确定性区间 + 与目标状态偏差信号）。
- **dev4**：feedback（偏差 + StreamingDriftDetector；默认仅记录不改权重，opt-in PAVA 再校准）。
- **dev5**：`udos/multitask.py` MultiTaskHead 共享 backbone 接口 + 头注册机制（零新参数）。
- **dev6**：SpatialCoordHead（[B,N_pts,3]）+ ActionTrajectoryHead（[B,H,6]）。
- **2.8.1**：FutureStateHead（[B,H,6]+不确定性）；三头 A/B（共享 2.34× 延迟优势），默认 opt-in。
- **2.8.2**：loop+multitask 集成加固、backcompat 扩到 11 件、服务新增 `/loop/step` `/multitask/predict`、
  `feature_latency_v2.8.0.json` 延迟基准。
- **2.8.3（✅ 2.8 线终点补丁件）**：边缘精修（空 window / 头维度不匹配 / horizon=0 / 极端偏差率 /
  未训练态守卫），文档对齐。

### v2.9.0–v2.9.3（✅ 完成，形态无关动作重定向 + 空间可供性）
- **v2.9.0**：`udos/retargeting.py` MorphologyConfig + ActionRetargeter（DOF 映射/关节限幅）；
  正式重建 `predictor_v2.9.0.pt`（52191 参数，eval_mse 0.0456）。
- **dev1**：MorphologyLibrary 三预设（prime_u_60dof / arm_7dof / gripper_4dof）+ 序列化/兼容检查。
- **dev2**：resample 跨控制频率线性/Catmull-Rom 插值，端点一致、非整数比、梯形能量守恒检查。
- **dev3**：zero_shot_transfer + A/B（重定向 vs 截断，1.68 vs 3.98，2.37×），`retarget_ab_v2.9.0.json`。
- **dev4**：`udos/affordance.py` AffordanceScorer（距离/速度/可达性合成打分，归一化）。
- **dev5**：SpatialRelationHead（[B,N,N,5] 左右/上下/接触，对称性约束）。
- **dev6**：AffordanceActionPlanner（低 affordance 不动作，loop predict_action opt-in hook）。
- **2.9.1**：retarget+affordance 汇总 A/B（引导成功率 1.00 vs 无引导 0.28），`v29_feature_ab.json`，
  MorphologyConfig NaN/inf 校验加固。
- **2.9.2**：跨特性集成加固、backcompat 扩到 12 件、服务新增 `/retarget/convert` `/affordance/score`、
  `feature_latency_v2.9.0.json`。
- **2.9.3（✅ 2.9 线终点补丁件）**：边界精修（dof=0 / 零物体 / 单物体 / 极端频率比 / 未训练态），
  文档对齐。**analogy, not reproduction**——全部为合成代理，不涉及真机/VLM。

### v3.0.0–v3.0.3（✅ 完成，未来状态多模态预测 + UDOS 内部五维评测）
- **v3.0.0**：`udos/future_multimodal.py` FutureMultimodalHead（共享 latent 的 RGB/深度/mask
  三模态低维代理）；正式重建 `predictor_v3.0.0.pt`（52191 参数，eval_mse 0.0456）。
- **dev1**：RGBProxyHead（颜色统计 8 维）+ DepthProxyHead（深度排序 4 维），独立损失+联合推理。
- **dev2**：MaskProxyHead（scene_params 阈值软 mask [0,1]，与 RGB/深度同 H 对齐）。
- **dev3**：CrossModalAlignmentLoss（跨模态相关性 opt-in 辅助损失，推理不计算，NaN 防护）。
- **dev4**：多模态 vs 单模态 A/B（2112 vs 924 参数，2.29×；无状态 MSE 增益 => opt-in 默认关）。
- **dev5**：`udos/eval_suite.py` FiveDimensionEvaluator（五维 UDOS 内部基准，非 PhysBrain）。
- **dev6**：composite_score 加权平均 + 四代演化 `five_dim_evolution_v3.0.0.json` + `eval5d`。
- **3.0.1**：跨特性集成、backcompat 扩到 13 件、服务新增 `/future/predict` `/eval/5d`、
  `feature_latency_v3.0.0.json`。
- **3.0.2（✅ 3.0 线补丁件）**：边界精修（H=0 / mask_dim=0 / 零维度 / alignment NaN /
  未训练态 / 权重和≠1），文档对齐。**analogy, not reproduction**——不碰真实 RGBD 图像。

### v3.1.0–v3.1.3（✅ 完成，ActionPiece 离散动作 token 化与序列建模）
- **v3.1.0**：`udos/action_piece.py` ActionPieceTokenizer（k-means 码本把连续动作量化为
  离散 token）；正式重建 `predictor_v3.1.0.pt`（52191 参数，eval_mse 0.0456）。
- **dev1**：ActionPieceCodec（粗+细两级残差量化）+ 困惑度/利用率指标。
- **dev2**：TokenizedActionPredictor（n-gram 自回归 next-token，held-out 0.918）。
- **dev3**：TokenActionDecoder（线性插值平滑 + 关节限位，4× 降跳变）+ retarget 集成。
- **dev4**：tokenized vs 连续 A/B（K=8/16/32/64 扫描，量化 MSE 随 K 递减），opt-in。
- **dev5**：SequenceCurriculum（easy→hard 纯数据调度，与随机持平 => 诚实记录无显著增益）。
- **dev6**：InContextActionPrompter（few-shot：1/3/5-shot = 0.725/0.975/1.000）。
- **3.1.1**：与 PhysicalLoop opt-in 集成（use_tokenized，默认路径逐位不变）。
- **3.1.2**：backcompat 15 件、服务 `/action/tokenize` `/action/detokenize`、
  `feature_latency_v3.1.0.json`。
- **3.1.3（✅ 3.1 线补丁件）**：边界精修（K=1 / 空序列 / 极端值 / 未训练态 / 超长上下文），
  文档对齐。**analogy, not reproduction**——合成动作 token，非 PhysBrain ActionPiece 复现。

### v3.2.0–v3.2.3（✅ 完成，Ego360 启发数据增强 · 长上下文 · 时间记忆）
- **v3.2.0**：`udos/ego_data.py` SyntheticEgoAugmenter（合成状态序列多视角/扰动/噪声/时间缩放）；
  正式重建 `predictor_v3.2.0.pt`（52191 参数，eval_mse 0.0456）。
- **dev1**：MultiViewGenerator（已知 R_k 多视角 + 对应矩阵 C[k,j]=R_j R_k^T，一致性 <1e-5）。
- **dev2**：ExtendedContextWindow（window>6 + 正弦位置编码 + 截断/填充；W=6 逐位等价锚点）。
- **dev3**：TemporalMemory（环形缓冲 + EMA 摘要；update/reset/summary；与 predictor 集成）。
- **dev4**：数据增强 A/B（in-distribution 无一致提升，OOD 鲁棒提升 => **opt-in 默认关**）。
- **dev5**：InContextLearner（0/1/3-shot；冻结小模型无 ICL 收益，如实记录 opt-in）。
- **dev6**：LongHorizonRollout（H=8/12/16；use_memory 累积上下文；H=4 逐位等价）。
- **3.2.1**：与 PhysicalLoop opt-in 集成（use_memory / icl_examples，默认路径逐位不变）。
- **3.2.2**：backcompat 16 件、服务 `/augment/generate` `/icl/predict`、
  `feature_latency_v3.2.0.json`。
- **3.2.3（✅ 3.2 线补丁件）**：边界精修（极端增强参数 / W=0 / 记忆溢出 / ICL 维度不匹配 /
  未训练态 / H=0），文档对齐。**analogy, not reproduction**——合成状态视角代理，非 Ego360/真机视频复现。

### v3.3.0–v3.3.3（✅ 完成，架构精炼 · 效率优化 · 鲁棒性收口）
- **v3.3.0**：`udos/ctm_engine.py` 可选残差连接 / LayerNorm / 可配置初始化（opt-in 默认关，
  与旧架构逐位等价）；正式重建 `predictor_v3.3.0.pt`（52191 参数，默认旧架构）。
- **dev1**：`udos/moe.py` LightweightMoE（N 专家线性层 + top-k 门控路由，推理外挂）。
- **dev2**：`lite.DistillationTrainerV2`（温度软标签 + 中间特征匹配 + 学生 d_model 自动减半/四分之一）。
- **dev3**：`lite.StructuredPrunerV2`（通道/注意力头级结构化剪枝 + mask 冻结微调；vs v1 幅值）。
- **dev4**：效率 Pareto A/B（参数-延迟-eval_mse，落 `efficiency_pareto_v3.3.0.json`；
  剪枝/学生不重训精度退化 => 推荐 full，opt-in）。
- **dev5**：`TrainConfig.gradient_checkpointing` + `ActivationFp16Cache`（默认全关）。
- **dev6**：`udos/robustness.py` RobustnessEvaluator（噪声/OOD/外推/FGSM 代理，分数 0-100）。
- **3.3.1**：全特性集成 + 五维/鲁棒性综合报告 + backcompat 17 件。
- **3.3.2（✅ 3.3 线补丁件）**：边界精修（MoE 专家数=1 / 蒸馏温度=0 / 剪枝稀疏度=1.0 /
  内存空态 / 极端噪声 / 服务未训练态 409），文档对齐。
- **3.3.3（✅ 对外发布）**：最终正式重建 `predictor_v3.3.3.pt` + 全部新特性离线 A/B +
  backcompat 18 件。**analogy, not reproduction**。

### v3.4.x — ICM 逆模型/好奇心（零梯度外挂）
`udos/icm.py` 三路线（前向/逆/对比）+ 事件切分 + 跨本体 + 上下文预算；`predictor_v3.4.5.pt`。

### v3.5.x — SFM 空间基础模型
`spatial.py`/`scene_graph.py`/`occupancy.py`/`collision.py`/`spatial_query.py`（SFM）；`predictor_v3.5.0.pt`。

### v3.6.x — PWM 潜在世界模型
`world_model.py` LatentWorldModel（外挂 2310 参数，H=1 逐位锚定主预测）+ 守恒检验；`predictor_v3.6.0.pt`。

### v3.7.x — 分层神经控制
`neural_control.py` 大脑/小脑/脊髓三层（CortexPlanner/CerebellumTracker/SpinalReflex，PID/反射无可训参数）；`predictor_v3.7.0.pt`。

### v3.8.x — 全域调度 / 数字孪生 / 多体协同与收口（✅ 终点 v3.8.6）
- **3.8.0**：`multi_agent.py` MultiAgentScene + AgentCoordinator（非学习优先级让行/速度调节）+ 正式件 `predictor_v3.8.0.pt`。
- **dev1**：`wm_scheduler.py` WMScheduler 多体 WM 想象预算分配。
- **dev2**：`closed_loop.py` ClosedLoopOrchestrator（大脑→小脑→脊髓→WM反馈→感知更新一步闭环）。
- **dev3**：`digital_twin.py` DigitalTwinScene（合成多体+障碍参数化场景，快照/回放）。
- **3.8.1**：多体 A/B（N=2/4/8，碰撞降但让行减速到达率下降，opt-in）。
- **3.8.2**：HTTP `POST /twin/step`、`POST /twin/scene`。
- **3.8.3**：24 代 checkpoint 兼容 + `feature_latency_v3.8.0.json`。
- **3.8.4**：跨 3.4-3.8 全特性综合评测（同 predictor 叠加不冲突、零梯度）。
- **3.8.5**：边界加固 + 文档对齐。
- **3.8.6（✅ 对外发布）**：最终正式重建 `predictor_v3.8.6.pt` + 25 代兼容 + 全量验证收口。**多体为合成参数化代理，数字孪生为合成场景**。

## 规划中（未承诺时间）
- **真实/仿真数据接入**：替换/补充合成动力学，验证真实物理精度（当前最大能力边界）。
- **GPM 超网络与预测器端到端联合微调**：v2.1 二者表示维度不同、未联合更新。
- **更长时域推演**：闭环再训练、scheduled-sampling 课程自适应、不确定性传播。
- **容器实建与 CI**：在具备 docker daemon / CI runner 的环境构建镜像并跑流水线。
- **正式概率预测**：v2.3 已落地保序校准与经验区间；后续走向参数化分布/分位数回归与
  分布外共形覆盖保证。
- **v2.6 候选**：批量/缓存在真实负载下的吞吐基准与默认化评估；OOD/漂移阈值自动校准；
  分位数回归替代经验半宽。

## v3.4 线 — ICM 上下文记忆（In-Context Memory）

- **3.4.0**：ICM 核心 `udos/icm.py`（DemonstrationEpisode/Memory/零梯度 ICMAggregator），首训正式件 `predictor_v3.4.0.pt`；复现 naive few-shot 退化（0.09→4.19），证明检索+聚合不退化（0.045→0.014）。
- **3.4.0.dev1**：事件级切分三流对齐 `udos/icm_events.py`。
- **3.4.0.dev2**：跨本体演示归一化 `udos/icm_cross.py`（复用 retargeting）。
- **3.4.0.dev3**：PCE 物理提示词包 `pce_format.DemonstrationPrompt/PCEPromptParser`。
- **3.4.0.dev4**：上下文预算/检索压缩 `udos/icm_budget.py` + 延迟-收益 A/B。
- **3.4.0.dev5**：k-shot scaling 曲线 + 权重逐位不变锚点。
- **3.4.0.dev6**：数据/思维链/上下文三路线对照（上下文 scaling 单位成本收益最高）。
- **3.4.1**：PhysicalLoop opt-in ICM 集成 + HTTP `/icm/predict`、`/icm/demo/register`。
- **3.4.2**：19 checkpoint backcompat + 组件 latency 基准。
- **3.4.3**：边界测试 + 文档对齐。
- **3.4.4 / 3.4.5**：全特性集成 + 最终训练重建 `predictor_v3.4.5.pt`（20 代兼容）。

## v3.5 线 — SFM 空间基础模型（Spatial Foundation Model）

> 纯推理外挂、零梯度、opt-in；合成低维 3D 代理，analogy not reproduction；不改主 52191 参数。

- **3.5.0**：`udos/spatial.py`（SpatialObject/SpatialScene/SpatialTransform，可逆性可验）+ 正式训练 `predictor_v3.5.0.pt`（21 代兼容）。
- **3.5.0.dev1**：`udos/scene_graph.py`（above/below/left/right/near/far/inside 解析关系）。
- **3.5.0.dev2**：`udos/occupancy.py`（OccupancyGrid 体素占据 + 有符号距离场）。
- **3.5.0.dev3**：`udos/collision.py`（球-球/球-盒碰撞 + 最近邻）。
- **3.5.0.dev4**：正交多视角投影 + 跨视角坐标变换一致性。
- **3.5.0.dev5**：`udos/spatial_query.py`（射线-球/视线遮挡/范围/盒检索）。
- **3.5.0.dev6**：SFM vs affordance A/B（语义 IoU≈0.33，默认 opt-in 关）。
- **3.5.1**：HTTP `POST /spatial/query`、`POST /spatial/collision`（400/409/404）。
- **3.5.2**：21 checkpoint backcompat + SFM 组件 latency 基准。
- **3.5.3**：边界精修 + 文档对齐。


## v4.3 完全自进化（自进化飞轮 + 基础设施自优化缩微类比）终点 v4.3.9

> 收口智谱"完全自训练"三维之基础设施自我优化（数据自产=4.2 / 环境自造=4.1 / 基础设施自优化=4.3）。analogy, not reproduction；CPU 合成、外挂零梯度、opt-in。

- **4.3.0**：`udos/self_evolution.py`（ConfigSpec/ConfigEvaluator/ConfigSearcher：batch 分片/缓存/集成权重/控制频率/码本规模 旋钮，固定负载上"输出保真硬门 + 确定性成本代理"）+ 正式件 `predictor_v4.3.0.pt`（第 32 代）。
- **4.3.0.dev1**：SearchVerifySelectLoop（搜索→保真验证→成本择优闭环）。
- **4.3.0.dev2**：ConfigAB（搜索后 vs 默认同合同 A/B，仅保真且更优才采纳）。
- **4.3.0.dev3**：SelfEvolutionOrchestrator（4.1 课程→4.2 数据→4.3 配置 串联成一代）。
- **4.3.0.dev4**：MultiGenerationRunner（多代成本曲线，诚实记录 improving/drifting/collapsed）。
- **4.3.0.dev5**：GlobalStopCorrectCriterion（收益递减→停；保真击穿/成本突增→回滚）。
- **4.3.0.dev6**：LongHorizonLoop（任务拆解→主预测器 rollout→错误恢复→验证）。
- **4.3.1**：HTTP `/self-evolution/{search,ab,long-horizon}`（400/409/404）。
- **4.3.2**：调研 SELF_TRAINING_EVOLUTION_RESEARCH.md + 边界精修。
- **4.3.9**：正式件 `predictor_v4.3.9.pt`（第 33 代）+ 全量回归 + 打包收口。

## v4.4 多智能体协作&协同&协调线（终点 v4.4.1，零正式训练）
- **4.4.0**：四拓扑统一接口（star/chain/tool/mesh）+ TopologySelector 图1 决策树（"控制需求不足拒绝自治"，默认 star）+ TransferBundle 五要素（Goal/Context/Done/Todo/Trace，缺关键要素拒交接）+ 治理三件套 Owner/Trace/Stop + ClaimLock 去重。
- **4.4.0.dev1**：CapabilityRegistry 把 9 个 UDOS 能力注册为 Agent-as-Tool（Tool Schema name/input/output/confidence/error_type）。
- **4.4.0.dev2**：星型 StarOrchestrator（拆分→Worker 分配→结果收集器去重/校验→统一收口）。
- **4.4.0.dev3**：链式 ChainHandoff（Triage→Specialist→Return + Bundle 交接 + 回退补救）。
- **4.4.0.dev4**：网状 MeshSwarm（**默认关 opt-in**：能力发现/局部协商/再委派/冲突-未对齐检测 + 治理开销统计）。
- **4.4.0.dev5**：三类失败 RED→GREEN 回归（状态丢失/重复劳动/责任不清）。
- **4.4.0.dev6**：拓扑 A/B（`make collab-ab` → collab_ab_v44.json）。
- **4.4.1**：HTTP `/collab/{select,run,handoff}` + `/collab/trace/{id}` + 加固 + 文档对齐 + 全量回归收口。
