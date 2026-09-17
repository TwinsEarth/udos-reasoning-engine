# UDOS 推演引擎架构说明 (v0.1.0)

## 1. 总体数据流

```
                     ┌─────────────────────────── PCE-Format 数据层 ───────────────────────────┐
                     │ PhysicalToken: position(3)+velocity(3)+force(3)+attributes+causal_parents │
                     │ PhysicsScene: 一段时序的 Token 集合 + 约束            (udos/pce_format.py) │
                     └───────────────────────────────────┬───────────────────────────────────────┘
                                                         │ PhysicsSceneEncoder (任意属性维 -> d_model)
                    ┌────────────────────────────────────┴───────────────────────────────────┐
                    │                                                                        │
                    ▼                                                                        ▼
   ┌─────────────────────────────── GPM 场景内化 (对齐 Doc-to-LoRA) ───────────────┐  ┌──────────── CTM 时序推演 (对齐 CTM) ───────────┐
   │ PhysicsContextEncoder: 场景 Token -> features [B,S,F]                        │  │ 内部时间轴 iterations 个 tick 循环:             │
   │ PerceiverBottleneck: latent-query 交叉注意力 + 自注意力块 -> LoRA 槽位        │  │  sync_action -> 对物理序列 Attention            │
   │ HyperLoRA head: 归一化 -> A/B, A[L,r,d_in], B[L,r,d_out]                     │  │  -> Synapse 跨神经元混合                       │
   │ forward_chunked: 多 chunk 平均聚合 (D2L combine_lora / mean-pooling)         │  │  -> trace 滚动 + NeuronLevelModel(每神经元私有) │
   │ LoRAInjector: 前向补丁 y=Linear(x)+B(Ax)·s, 不改权重, reset 零误差           │  │  -> activated -> sync_out -> prediction        │
   └───────────────────────────────┬─────────────────────────────────────────────┘  │  -> certainty=1-归一化熵 (自适应早停)           │
                                   │ 注入基座 (TinyBaseModel / 真实 LLM)              └──────────────────────────┬────────────────────┘
                                   └──────────────────────────────────────────────►  在"记住场景"的模型上时序推演 │
                                                                                                     ▼
                                                                              ReasoningResult: prediction + certainty 轨迹
                                                                                               + 因果链 + LoRA 参数量
```

## 2. 关键张量形状

### CTM (`udos/ctm_engine.py`)
| 张量 | 形状 | 含义 |
|------|------|------|
| physical_tokens | `[B, S, d_input]` | 物理 Token 嵌入序列 |
| state trace | `[B, D, M]` | 每神经元长度 M 的 pre-activation 历史 |
| activated_state | `[B, D]` | 当前 tick 神经元 post-activation |
| sync_action/out | `[B, n_synch_*]` | random-pairing 神经元对同步表示 |
| predictions | `[B, out_dims, T]` | 每个内部 tick 一版预测 |
| certainties | `[B, 2, T]` | (归一化熵, 1-熵) |

同步递推 (对齐上游 `compute_synchronisation`):
`alpha←r·alpha+a_i·a_j; beta←r·beta+1; sync=alpha/sqrt(beta); r=exp(-decay)`。

### GPM (`udos/gpm_engine.py`)
| 张量 | 形状 | 含义 |
|------|------|------|
| features | `[B, S, feature_dim]` | 场景编码特征 |
| perceiver out | `[B, n_layers·n_modules·r, latent]` | LoRA 槽位向量 |
| LoRA A / B | `[n_layers, r, d_in]` / `[n_layers, r, d_out]` | 每目标模块的低秩矩阵 |
| 前向增量 | `[., ., d_out]` | `B(Ax)·scaling` |

### v2.3 校准与经验区间 (`udos/calibration.py`，外挂、不入 `state_dict`)
| 张量/对象 | 形状 | 含义 |
|------|------|------|
| conf | `[N]` | 校准集每样本原始 certainty（同步 1−熵） |
| step_err | `[N]` | 单步逐样本误差均值，经验精度 `a=exp(−e/scale)` |
| PAVA 块 | `blocks ≤ N` | 单调非降的 `置信→经验精度` 阶梯映射（坏排序时退化为常量，`num_segments=1`） |
| resid | `[N,H,RAW]` | 自由 rollout 相对真值的有符号残差 |
| residual_quantiles | `List[Tensor(2)]` 长 H | 每步 (q_lo,q_hi) 保守分位（下取整/上取整，避免小样本偏窄） |
| predict_interval | median/lower/upper `[B,H,RAW]` | 中值=rollout 点预测，界=中值+该步分位 |

张量流：`collect_predictor_outputs → (conf,err,resid)`；`fit` 得 PAVA 校准器、按步统计
残差分位；`attach_calibration` 外挂到预测器；持久化时随 bundle 的可选 `calibration` 键存取，
旧档缺该键则 `is_calibrated=False`（向后兼容）。F1/F3 均为**确定性后处理、零可学参数、
不重训**，故其增益不依赖训练随机性；F2 多时域损失在 `training.py` 内，仅改各步损失加权。

### v2.4 可信推演套件（外挂、opt-in、不入 `state_dict`）

| 模块 | 文件 | 职责 | 关键 API |
|------|------|------|----------|
| OOD/漂移 | `udos/ood.py` | 岭正则马氏距离打分 + 经验分位阈值 + 双样本 KS；纯统计 | `DistributionDriftDetector.fit/score/is_ood/ks_drift` |
| 流式漂移 | `udos/ood.py` | 固定窗口 W 在线均值/方差，继承离线接口，`reset()` 清窗口 | `StreamingDriftDetector.update/window_mean/window_var/window_drift_score` |
| 深度集成 | `udos/ensemble.py` | N 个同架构多种子成员，成员分歧近似认知不确定性 | `DeepEnsemble.predict_next/rollout/predict_interval`（N=1 退化为单模型） |
| 退化守卫 | `udos/guard.py` | NaN/inf 整行回退 + 越界截断，记录 `n_fallbacks/n_clips` | `PredictionGuard.sanitize/stats`；`predict_next(guard=True)` |
| 置信矩阵 | `udos/training.py` | 逐步逐维 `[B,H,RAW]` 置信（步 certainty × 逐维半宽可靠性因子） | `PhysicsPredictor.per_step_confidence`；`/evaluate` 的 `confidence_breakdown` |

设计纪律：全部**外挂后处理、默认关闭或等价旧版**；`attach_*` 挂载、`opt-in` 参数激活；
不改变任何旧默认输出。集成持久化走 `save_ensemble/load_ensemble`（多成员 state_dict + 共享
ctm_config），与旧单模型 `save_predictor/load_predictor` 并存。多名义水平 conformal 区间
`alpha∈{0.2,0.1,0.05}`（80/90/95），默认 0.1 逐位不变。

### v2.5 效率与服务化套件（推理侧增量、opt-in、零可学参数）

| 模块 | 文件 | 职责 | 关键 API |
|------|------|------|----------|
| 批量推理 | `udos/batch.py` | 变长序列按窗口长度分组、等长直接前向、超大 batch 自动分片；结果与逐笔逐位一致 | `BatchPredictor`；`PhysicsPredictor.predict_batch` |
| 推理缓存 | `udos/cache.py` | LRU，key=输入张量 SHA-256 + 模型参数哈希；权重变化自动失效；默认关闭 | `InferenceCache`（enabled 默认 False） |
| 服务指标 | `udos/server.py` | `MetricsCollector`（纯标准库、线程安全）：请求计数/延迟 p50/p95/p99/缓存命中/OOD 触发率，Prometheus 文本 | `GET /metrics` |
| 无状态快照 | `udos/persistence.py` | 导出配置+校准器+残差分位+OOD 统计（**不含权重**），导入校验架构一致 | `export_snapshot` / `import_snapshot` |
| 加载栈回滚 | `udos/server.py` | 维护已加载 checkpoint 栈，回滚上一；栈长<2 返回 409 | `POST /rollback` |

设计纪律：v2.5 全部**不新增可学参数**（参数量仍 52191）、**不改变** `predict_next`/`rollout`/
`predict_interval` 默认输出；批量与缓存均 opt-in。CPU 小模型（52191 参数）单次推理本身快，
批量并行/缓存收益主要在大批量与重复请求场景（诚实记录）。

### v2.6 因果/反事实/决策套件（推理侧增量、纯前向、确定性、不重训）

| 模块 | 文件 | 职责 | 关键 API |
|------|------|------|----------|
| 混合物理修正 | `udos/hybrid.py` | 一阶欧拉匀速骨架 + 极小 MLP(≈806 参数) 学残差；默认不挂（predictor.hybrid is None）走旧路径逐位一致 | `HybridPhysicsCorrector`；`predict_next(hybrid=True)` |
| 反事实推演 | `udos/counterfactual.py` | 干预式 rollout（scene_params / initial_state / velocity_override），零干预=基线逐位；ATE=逐步均方差 | `CounterfactualEngine.counterfactual` |
| 场景辨识/归因 | `udos/identification.py` | 网格搜索反推 4 维隐藏物理参数（窗口内自洽校验 + v0 运动学先验）；一阶 Sobol 简化归因 | `SceneParameterIdentifier.identify`；`sobol_attribution` |
| 自适应计算 | `udos/adaptive.py` | `AdaptiveStopper`（certainty 收敛早退）；`adaptive_rollout` 按 conformal 半宽增长率截断，未挂半宽跑满 | `adaptive_rollout`；`predict_next_adaptive` |
| 风险分级/安全边界 | `udos/decision.py` | 聚合区间宽+OOD+校准置信为 [0,1] 风险分与三档；按区间下界过滤候选下一动作；无信息时诚实退化 | `RiskGrader.grade`；`safety_boundary` |
| 快照差分/对比 | `udos/persistence.py` | 结构化 diff 两快照（config/calibration/ood）；两 checkpoint 同测试集 evaluate + 逐位预测差 | `diff_snapshots`；`compare_checkpoints` |

服务侧新增端点：`POST /counterfactual`、`POST /identify`、`POST /risk`、
`POST /diff-checkpoints`。设计纪律同 v2.5：全部 opt-in、纯前向、不改旧输出、
7 个旧 checkpoint 全部向后兼容（旧件无 hybrid => None，新后处理对旧件诚实退化）。

**v2.6.1 边缘加固（不改架构，只收紧失败路径）**：空 batch 返回 `[0,RAW_DIM]`、scene_params
非有限值显式 ValueError、RiskGrader OOD NaN 降级、反事实 NaN 干预报错、`predict_next` 中
guard 移到 hybrid 之后清洗最终输出。详见 `tests/test_v261_edge.py`。

### v2.7 从预测到行动的闭环套件（推理侧外挂、纯前向、确定性、不重训主模型）

| 模块 | 文件 | 职责 | 关键 API |
|------|------|------|----------|
| MPC 动作优选 | `udos/policy.py` | 对候选动作逐个 rollout，按 奖励 − λ·风险 − 安全违例 打分排序，返回最优安全动作 + 全排序；空候选集 `no_valid_action=True` | `MPCActionSelector.select` |
| 在线适配 | `udos/online.py` | `StreamingDriftDetector` 流式监测；漂移触发时默认仅重跑 PAVA（不改权重），opt-in 才少量增量微调；NaN 观测跳过 | `OnlineAdapter.observe / check_and_adapt` |
| 主动学习 | `udos/active_learning.py` | 信息增益 = α·集成方差 + β·区间宽 + γ·OOD，对未标注池选 top-K；空池/k>池/k≤0 守卫 | `UncertaintySampler.score_samples / select_top_k` |
| 模型轻量化 | `udos/lite.py` | 全局幅值剪枝（可恢复 mask）/ 动态 INT8 量化（Linear，仅 CPU）/ 蒸馏小学生 CTM；全 opt-in | `MagnitudePruner / DynamicQuantizer / DistillationTrainer` |
| 分层 rollout | `udos/hierarchical.py` | coarse_factor 分块粗粒度跳步；horizon≤coarse 退化为平铺逐位一致（单自回归头下为 opt-in 脚手架，无误差下降） | `HierarchicalRollout.rollout` |
| 实验治理 | `udos/experiment.py` | 纯元数据 ExperimentRegistry：config/metrics/seed/artifacts 注册 + 多种子 sweep（mean/std/best）+ JSON 原子持久化；不碰权重 | `ExperimentRegistry.register / sweep_report / save / load` |

服务侧新增端点（v2.7.0.dev6）：`POST /policy/select`、`POST /online/adapt`、
`POST /active/sample`、`GET /experiments`（未训练 409、非法输入 400，复用既有错误语义）。

**v2.7.2 边缘加固（不改架构，只收紧失败路径）**：online 跳过 NaN/inf 行、active 空池返回空、
policy horizon=0 拒绝、hierarchical horizon=1 / coarse>horizon 退化为平铺、剪枝与蒸馏学生 save/load
自洽、`PredictionGuard` 挂载后 policy 仍工作。详见 `tests/test_v272_edge.py`。

### v2.8 Physical Loop 统一闭环 + 共享 backbone 多任务头（推理侧编排，不改主模型权重）

| 模块 | 文件 | 职责 | 关键 API |
|------|------|------|----------|
| 物理闭环编排 | `udos/physical_loop.py` | observe→understand→predict_action→future_state→feedback 五步固定顺序显式编排，每步可插拔 hook；默认未挂 hook 时 `run()` 末步 `predict_next` 与直接调用逐位一致；`loop_state` 记录每步形状/耗时 + action_history + correction | `PhysicalLoopRunner.run` |
| 多任务头 | `udos/multitask.py` | 共享 backbone 接口（复用 obs_encoder 时间维均值池化→裁剪/零填充对齐 latent_dim，零新参数）+ 头注册路由；默认 enable=False 主路径不变 | `MultiTaskHead.encode / register_head / forward` |
| 结构化头 | 同上 | `SpatialCoordHead`[B,N_pts,3]、`ActionTrajectoryHead`[B,H,6]、`FutureStateHead`[B,H,6]+不确定性 | 各 `*.forward` |

A/B（`benchmarks/results/multitask_ab_v2.8.0.json`）：共享 backbone encode 一次 vs 独立三次，
**2.34× 延迟优势**；头为随机初始化线性投影，故默认 opt-in。服务侧新增端点（v2.8.2）：
`POST /loop/step`、`POST /multitask/predict`（未训练 409、非法输入 400）。

**v2.8.3 边缘加固**：空 window / 非有限值显式 ValueError、horizon=0 拒绝、头 latent 维度不匹配显式报错、
极端偏差阈值仅记录不改权重、未训练端点 409。详见 `tests/test_v283_edge.py`。

## 3. 与上游源码的对应关系

| 本工程 | 上游文件 (third_party/) | 继承的机制 |
|--------|--------------------------|-----------|
| `ctm_engine.NeuronLevelModel` | `ctm/models/modules.py::SuperLinear` + `ctm.py::get_neuron_level_models` | 每神经元私有权重处理历史 (einsum 等价 grouped-linear) |
| `ctm_engine.CTMPhysicsEngine._synchronise` | `ctm/models/ctm.py::compute_synchronisation` | random-pairing + alpha/beta 衰减递推 |
| 内部 tick 循环 + certainty 早停 | `ctm.py::forward` + `compute_certainty` | 内部时间轴、1-归一化熵 |
| `gpm_engine.PerceiverBottleneck` | `d2l/.../aggregator.py::Perceiver`, `idefics2.py` | latent-query 瓶颈 |
| `gpm_engine.PhysicsHypernetwork` | `d2l/.../hypernet.py::HyperLoRA` | 归一化 + head 生成 A/B、scaler_A=1/scaler_B=0 |
| `forward_chunked`/`aggregate_loras` | `hypernet.py::combine_lora` | 分块平均聚合 |
| `gpm_engine.LoRAInjector` | `d2l/.../lora_layer.py::lora_forward/apply_lora_to_layers` | 前向补丁、reset 还原 |
| `reasoning.internalize_scene/reset_scene` | `hypernet.py::ModulatedPretrainedModel.internalize/reset` | 场景内化/移除 API |
| `adapters/sakana_ctm_adapter.py` | `ctm/models/ctm.py::ContinuousThoughtMachine` | **直接加载真实上游代码**做交叉验证 |

## 4. 为什么用"前向补丁"而非改权重

D2L 原版通过 `partial` 替换目标 `nn.Linear.forward`, 在其中叠加 `B(Ax)`,
原始权重保持不变; `reset()` 只需还原 `forward_orig`。这样:
1. 无浮点回滚误差; 2. 可瞬时切换多个场景; 3. 与 PEFT/LoRA 前向数学一致。
本工程严格沿用该方式。

## 2.9 线：形态无关动作重定向 + 空间可供性（v2.9.0–v2.9.3）

在推理外挂层新增两个独立模块，不参与主模型训练（52191 参数不变）：

- `udos/retargeting.py`：`MorphologyConfig`（DOF/控制频率/关节限位/运动学占位）+
  `ActionRetargeter`（端点对齐 DOF 映射 `map_dof` + 跨频率线性/Catmull-Rom 重采样
  `resample` + 关节限幅 clamp + `zero_shot_transfer`）+ `MorphologyLibrary`（三预设）。
- `udos/affordance.py`：`AffordanceScorer`（状态/物体部位代理向量 -> [B,N_parts] 归一化打分，
  距离/速度/可达性合成特征）+ `AffordanceActionPlanner`（面向可操作部位生成动作序列，
  低 affordance 不动作）。
- `udos/multitask.py` 新增 `SpatialRelationHead`：共享 latent -> [B,N_obj,N_obj,5] 物体对
  空间关系矩阵（right/left/up/down/contact），对称性约束，经 register_head 独立开关。

全部为推理时外挂、opt-in，默认 `predict_next` 路径逐位不变。**analogy, not reproduction**：
用不同维动作向量代理不同机器人形态，用状态向量子集代理物体部位，不涉及真机/RGBD/VLM。

## 3.0 线：未来状态多模态预测 + UDOS 内部五维评测（v3.0.0–v3.0.3）

在推理外挂层新增两个独立模块，不参与主模型训练（52191 参数不变）：

- `udos/future_multimodal.py`：`FutureMultimodalHead`（共享 latent -> 未来 H 步三模态低维
  代理：RGB `[B,H,8]` / 深度 `[B,H,4]` / 对象 mask `[B,H,4]`）；`RGBProxyHead`（颜色统计）、
  `DepthProxyHead`（深度排序）、`MaskProxyHead`（scene_params 阈值软 mask ∈[0,1]）；
  `CrossModalAlignmentLoss`（跨模态相关性 opt-in 训练辅助损失，推理不计算，NaN 防护）。
- `udos/eval_suite.py`：`FiveDimensionEvaluator`（**UDOS 内部基准，非 PhysBrain 榜单分数**，
  metadata 显式标注）——①状态重建精度 ②scene_params 辨识 ③policy 动作优选 ④affordance
  准确率 ⑤rollout 轨迹精度；`composite_score` 可配置加权平均（自动归一化）。
- 服务：`POST /future/predict`（多模态未来）、`GET /eval/5d`（五维+composite，未训练 409）。

全部为推理时外挂、opt-in，默认 `predict_next` 路径逐位不变。**analogy, not reproduction**：
用 latent 的不同线性投影代理三种模态，不碰真实 RGBD 图像/VLM。

## 3.1 线：ActionPiece 离散动作 token 化与序列建模（v3.1.0–v3.1.3）

在推理外挂层新增 `udos/action_piece.py`，不参与主模型训练（52191 参数不变）：

- `ActionPieceTokenizer`：k-means 码本把连续动作向量 [N,action_dim] 量化为离散 token id；
  k-means++/random/uniform_grid 初始化；encode/decode、码本学习、覆盖率/利用率、往返误差。
- `ActionPieceCodec`：粗粒度 + 细粒度残差两级量化；困惑度/利用率质量指标；两级误差 <= 粗级。
- `TokenizedActionPredictor`：n-gram 自回归 next-token（纯计数，短历史向低阶回退）；
  `generate()` 自回归生成；`decode_to_actions()` 回连续动作。
- `TokenActionDecoder`：token 序列 -> 连续动作；`smooth_decode()` 线性插值平滑（步间跳变 4×↓）；
  可选关节限位后处理 clamp；与 `ActionRetargeter` 集成。
- `SequenceCurriculum`：按轨迹复杂度 easy->hard 纯数据调度（不改模型权重）。
- `InContextActionPrompter`：few-shot 示例 + 查询即时建 n-gram，in-context 预测。
- 集成：`PhysicalLoopRunner(use_tokenized=True, tokenizer=..., token_predictor=...)` opt-in；
  默认关时 `predict_action` 走原 MPC，整环输出与 `predict_next` 逐位一致。
- 服务：`POST /action/tokenize`（连续->token）、`POST /action/detokenize`（token->连续）。

全部为推理时外挂、opt-in，默认 `predict_next`/loop 路径逐位不变。**analogy, not reproduction**：
k-means 码本在合成动作上拟合，受 PhysBrain ActionPiece/长上下文启发的轻量化类比，非复现，
不碰真机动作/视频/VLM。

## 3.2 线：Ego360 启发数据增强 + 长上下文 + 时间记忆（v3.2.0–v3.2.3）

在推理外挂层新增五个模块，不参与主模型训练（52191 参数不变）：

- `udos/ego_data.py`：`SyntheticEgoAugmenter`（xy 平面刚体旋转/平移 + 轨迹扰动 + 噪声 +
  时间缩放；(X,Y) 成对增强保证标签对齐）；`MultiViewGenerator`（已知 R_k 多视角 + 对应矩阵 C[k,j]=R_j R_k^T）。
- `udos/extended_context.py`：`ExtendedContextWindow`（window>6 长历史；正弦位置编码 opt-in；
  历史截断/填充；W=6 默认路径逐位等价）。
- `udos/temporal_memory.py`：`TemporalMemory`（容量 C 环形缓冲 + EMA 历史摘要；update/reset/summary）。
- `udos/incontext.py`：`InContextLearner`（few-shot 示例窗口 + 任务描述 + 查询拼接为扩展上下文）。
- `udos/longhorizon.py`：`LongHorizonRollout`（H=8/12/16；use_memory 时逐步累积记忆；H=4 默认逐位等价）。
- 集成（3.2.1）：`PhysicalLoopRunner(use_memory=..., memory=..., icl_examples=...)` opt-in；
  默认关时整环输出与 `predict_next` 逐位一致。
- 服务（3.2.2）：`POST /augment/generate`（无状态增强）、`POST /icl/predict`（few-shot 预测）。

全部为推理时外挂、opt-in，默认 `predict_next`/loop 路径逐位不变。**analogy, not reproduction**：
受 Ego360/PhysBrain 长上下文启发的轻量化类比，非复现，不碰真机视频/第一人称 RGB/VLM。

## 3.3 线：架构精炼 · 效率优化 · 鲁棒性收口（v3.3.0–v3.3.3）
- `udos/ctm_engine.py`：`CTMConfig.residual/act_norm/init_mode`（残差/LayerNorm/可配置初始化，
  opt-in 默认关，与旧架构逐位等价）。
- `udos/moe.py`：`LightweightMoE`（N 个 batched 专家线性层 + 门控 top-k 路由，推理外挂）。
- `udos/lite.py`：`DistillationTrainerV2`（温度软标签 + 中间特征匹配 + 学生 d_model 自动收缩）、
  `StructuredPrunerV2`（通道/注意力头级结构化剪枝 + mask 冻结微调）。
- `udos/training.py`：`TrainConfig.gradient_checkpointing`、`ActivationFp16Cache`（默认全关）。
- `udos/robustness.py`：`RobustnessEvaluator`（噪声/OOD/外推/FGSM 代理，综合分 0-100）。
- 效率 Pareto：`benchmarks/results/efficiency_pareto_v3.3.0.json`（推荐 full，其余 opt-in）。
全部 opt-in，正式件 `predictor_v3.3.3.pt` 仍用旧架构默认。**analogy, not reproduction**。



## v3.4 ICM 上下文记忆架构（零梯度，opt-in）

```
查询窗口 [W,6] ──► DemonstrationMemory 余弦检索 top-k ──► ICMAggregator
                                                            │ 残差 r_i=result_i−predict(in_i) 注册期缓存
演示 episode ──► DemonstrationEpisode(输入→动作→结果因果块)  │ p_icm = p0 + λ·Σ softmax(s_i)·r_i
                                                            └► 输出 [6] (不改主权重)
```

- **零梯度**：全程 no_grad，聚合器可训参数=0，不入主 state_dict；state_dict md5 推理前后逐位一致。
- **检索+聚合而非朴素拼接**：在输出残差空间聚合，避免 naive 拼接稀释 52k 注意力（naive 0.09→4.19 vs ICM 0.045→0.014）。
- 配套：`icm_events.py`（事件切分三流对齐）、`icm_cross.py`（跨本体归一化，复用 retargeting）、`icm_budget.py`（预算/压缩）、`pce_format.DemonstrationPrompt`（HTTP 提示词包）。
- 默认 opt-in：不挂 ICM 时 `predictor.predict_next` 与旧路径逐位一致。

## v3.5 SFM 空间基础模型架构（纯几何，零梯度，opt-in）

```
SpatialScene(多物体) ──► SceneGraph(above/below/near/far/inside 解析边)
        │             ├─► OccupancyGrid(体素占据) ─► DistanceField(有符号距离)
        ├─► CollisionDetector(球-球/球-盒) + NearestNeighbor
        ├─► OrthographicView(正交投影, 可逆)  多视角一致性
        └─► SpatialQueryEngine(射线-球 / 视线遮挡 / 范围 / 盒)
                    │
        HTTP POST /spatial/query、/spatial/collision  (独立, 不需主 predictor)
```

- **analogy not reproduction**：合成低维 3D 代理，不复现 SpatialVLM/真实 3D/点云/真实相机。
- **纯前向、确定性**：关系/碰撞/查询全部解析几何（numpy），可学参数=0，不改主 52191 权重，state_dict md5 推理前后逐位一致。
- **可逆性可验**：SpatialTransform 用 3×3 线性矩阵解析求逆；OrthographicView 正交投影-反投影往返 ~机器精度。
- 默认 opt-in：不引入 SFM 时，主推理与服务旧端点逐位等价；SFM vs affordance 语义 IoU≈0.33，不互相替代。


## v3.8 全域调度 / 数字孪生 / 多体协同（纯推理外挂层）

```
MultiAgentScene (N 体状态容器 [pos3+vel3])  ── AgentCoordinator (优先级让行/速度调节, 非学习)
        │
        ├── WMScheduler      ── 多体 WM 想象预算分配 (复用 LatentWorldModel, H=1 锚定真实)
        ├── ClosedLoopOrchestrator ── 大脑→小脑→脊髓→WM想象反馈→空间感知更新 (一步闭环)
        └── DigitalTwinScene ── 合成多体+障碍参数化场景 (seed 确定性, 快照/回放)

        HTTP POST /twin/scene (创建/查询孪生), /twin/step (一步闭环)
```

- **零外挂**：四个模块均为确定性算法或复用既有外挂，不进主 state_dict、不改主 52191 权重。
- **opt-in**：不构造这些对象时，旧推理路径与全部旧端点逐位一致。
- **analogy not reproduction**：多体为合成参数化代理，数字孪生为合成场景，非真机多机器人/实时总线。

## v4.3 完全自进化线（基础设施自优化缩微类比）

```
        ConfigSpec (旋钮: max_shard / cache / ensemble_weights / cortex_hz / codebook_size)
              │
        ConfigEvaluator ── 固定基准负载上:  保真轴 = 输出 max_abs_diff(vs 默认) <= atol 硬门
              │                            效率轴 = 确定性成本代理 (forward 当量+规划调用+码本)
        ConfigSearcher  ── 网格枚举, 仅在"保真且可接受"配置里取成本最小 (默认永远在候选内)
              │
   SearchVerifySelectLoop (search→verify→select 闭环, archive 账本)
              │
        SelfEvolutionOrchestrator  ── 一代 = 4.1 课程 → 4.2 三元组 → 4.3 配置
              │
        MultiGenerationRunner ── 多代成本曲线 (诚实 improving/drifting/collapsed)
        GlobalStopCorrectCriterion ── 收益递减→停 / 保真击穿→回滚
        LongHorizonLoop ── 拆解→主预测器 rollout→错误恢复→验证

        HTTP POST /self-evolution/search, /self-evolution/ab, /self-evolution/long-horizon
```

- **零外挂**：全部为确定性编排/搜索，不进主 state_dict、不改主 52191 权重；主预测器全程只读 eval。
- **质量硬门**：任何候选配置不允许"靠降质换速度"——输出保真不达标即 REJECT 留候选账本。
- **诚实纪律**：多代曲线照实记录真改进 vs 退化，不宣称自进化飞轮必然提升。
- **analogy, not reproduction**：真实系统优化算子/内核/调度/缓存/服务栈，UDOS 优化其 CPU 等价物。

## v4.4 多智能体协作线（终点 v4.4.1）

- **分层**：`transfer_bundle`(五要素交接) → `collab_governance`(Owner/Trace/Stop/ClaimLock) → `collab_agents`(CapabilityRegistry/Agent-as-Tool) → 四拓扑执行器 `collab_orchestrator`(star) / `collab_handoff`(chain) / `collab_swarm`(mesh, 默认关)。`collab_topology.TopologySelector` 按图1 四问选拓扑。
- **数据流**：任务 → 决策树选拓扑 → 治理硬门(无 owner/无 stop 拒启动) → 拆/路由/协商 → Bundle 交接(每棒 validate) → 结果收集/共识检测 → 沿 TraceChain 强制收口唯一 owner。
- **零外挂**：全部为纯前向编排算法，无可训参数、零梯度、不进主 state_dict、不改 52191；opt-in（swarm 默认关）。
- **analogy, not reproduction**：多智能体为内存纯函数能力包装，非真实多机器人集群；Tool Schema 思想对标 MCP function-calling，A2A 仅概念借鉴，不做线上字节协议。

## v4.5 隐式思考线（终点 v4.5.3）

- **分层**：`latent_reasoner`（CTM 隐藏状态 K 条潜路径并行探索 + best-of-K 聚合 + 隐式摘要）→ `reasoning_router`（难度信号合成 → effort 档位 + 是否显式化 + 可解释理由）→ `latent_collab`（三专家 Agent-as-Tool 打分、分歧大则升级 effort 与 star→chain）。
- **数据流**：场景 → 编码物理 token → (none 零扰动直出 / opt-in 加 K 条确定性扰动分支) → 各分支独立 CTM 前向 → 按终态 certainty 软选路 → 隐式摘要(路径数/收敛分/分歧/置信) → high/max 投影为可读显式链(与 4.4 TraceChain 统一: 探索路径数/选择理由/切换点/专家分歧/收口人)。
- **四档 effort**：none(K=1,σ=0,逐位等价默认) / low(K=2,σ=0.02,不显式化) / high(K=4,σ=0.05,部分可读链) / max(K=8,σ=0.08,完整可读链)。
- **零外挂**：隐式探索为对编码 token 的确定性扰动分支 + 规则聚合，无可训参数、零梯度、不进主 state_dict、不改 52191；none 默认逐位等价，low/high/max 与自适应路由 opt-in。
- **诚实账本**：Pareto A/B 实测隐式不改善物理 MSE、不省延迟（见 `benchmarks/results/pareto_v45.json`），价值在可观测/可追溯；自适应路由把算力按难度分桶花在难题上。
- **analogy, not reproduction**：分支=对编码 token 的确定性扰动，非 Coconut 的 hidden-state 回喂 / token 层 beam；不宣称复刻大模型潜空间推理。
