# UDOS 引擎 v2.7 版本规划 —— 从预测到行动的闭环：MPC 动作优选、在线自适应与实验治理

> 起点：v2.6.2（2.6 线终点，281 passed / 0 skipped / 92% 覆盖率，正式件 predictor_v2.6.2.pt）
> 终点：**v2.7.3**（对外发布，正式件 `predictor_v2.7.3.pt`）
> 训练节点：**2.7.0** 与 **2.7.3** 各重建一次正式件；其余 8 节点不重训，用确定性算法 + 轻量单测推进。
> 铁律：v2.6.2 的 281 测试持续全绿、只增不删旧契约；新能力默认不改变旧默认输出；版本号同步且有断言。
> 主线：在 v2.6「因果可解释与物理一致性」之上，构建**从预测到行动的闭环**——MPC 式候选动作优选、在线漂移触发适配、主动学习采样、模型轻量化、分层长时域 rollout、实验注册治理，最终把预测能力转化为可执行的安全决策。

## 2.7 线 10 节点清单

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 31 | **2.7.0** | MPC 式候选动作优选 + 正式训练重建 | `udos/policy.py`：`MPCActionSelector`（给定当前状态 + 候选动作集，对每个动作做 rollout 预测，按目标函数 + 风险惩罚 + 安全边界打分，返回最优安全动作 + 全排序）；基于 `decision.RiskGrader` + `safety_boundary`；opt-in 默认不改变旧路径；正式训练重建 `predictor_v2.7.0.pt` + `training_v2.7.0.json`（同口径 52191 参数） | `tests/test_v27_policy.py`：空候选守卫、已知最优动作选择正确、风险惩罚单调性、安全边界过滤、save/load 兼容 |
| 32 | 2.7.0+dev1 | 在线增量适配与漂移触发再校准闭环 | `udos/online.py`：`OnlineAdapter`（封装 `StreamingDriftDetector`，检测到漂移时触发 PAVA 再校准 + 可选增量微调 opt-in；**默认只再校准不改权重**；适配后记录 `adaptation_log` 含触发原因/时间/指标差）；检测→触发→适配→确认闭环 | `tests/test_v27_online.py`：无漂移不触发、漂移触发再校准、默认不改权重、opt-in 微调降误差、适配日志完整、与旧 OOD 检测器兼容 |
| 33 | 2.7.0+dev2 | 主动学习 / 不确定性采样选点 | `udos/active_learning.py`：`UncertaintySampler`（用集成方差 + conformal 区间宽 + OOD 距离合成信息增益分，对未标注样本池排序选 top-K）；A/B 证据：主动选样 vs 随机选样在同样本量下的 eval_mse 差，证明主动选样降样本需求 | `tests/test_v27_active.py`：不确定性排序单调性、top-K 选择正确、A/B 主动 vs 随机 JSON 落盘、空池守卫、与 ensemble/conformal 接口一致 |
| 34 | 2.7.0+dev3 | 模型轻量化（剪枝 / 量化 / 蒸馏 A/B） | `udos/lite.py`：`MagnitudePruner`（按权重幅值剪枝 + 可恢复 mask）、`DynamicQuantizer`（`torch.ao.quantization` dynamic INT8）、`DistillationTrainer`（teacher-student KL 蒸馏，小学生模型）；A/B：参数-延迟-精度三维对比落 `benchmarks/results/lite_ablation_v2.7.0.json`；**默认全量模型不变，lite 模型 opt-in** | `tests/test_v27_lite.py`：剪枝稀疏度正确、量化前后输出接近（容差内）、蒸馏 loss 下降、A/B JSON 落盘、默认全量模型逐位不变 |
| 35 | 2.7.0+dev4 | 分层 / 多尺度长时域 rollout | `udos/hierarchical.py`：`HierarchicalRollout`（粗粒度长时域规划 + 细粒度短时域修正，减少长 horizon 误差累积；coarse_factor 可配置）；A/B：H=8/12/16 下分层 vs 平铺 rollout 的误差累积率对比 | `tests/test_v27_hierarchical.py`：短 horizon（≤coarse_factor）退化等价、长 horizon 误差下降、分层标记正确、A/B 证据可复算、与 predictor rollout 接口一致 |
| 36 | 2.7.0+dev5 | 实验注册与多种子 sweep 治理 | `udos/experiment.py`：`ExperimentRegistry`（记录实验 config / metrics / artifacts / seed，支持多种子 sweep 自动对比报告——均值/标准差/best，JSON 持久化到 `benchmarks/results/experiment_registry.json`）；纯确定性、不改变预测路径 | `tests/test_v27_experiment.py`：注册/查询/列表、多种子聚合统计（mean/std/best）、JSON 持久化与重载、空注册表守卫、重复注册幂等 |
| 37 | 2.7.0+dev6 | 服务接口扩展（MPC / 在线 / 主动 / 实验） | `udos/server.py` 新增 `POST /policy/select`（MPC 动作优选）、`POST /online/adapt`（触发在线适配）、`POST /active/sample`（不确定性采样）、`GET /experiments`（实验注册表列表）；未训练 409、非法输入 400；`/evaluate` 可选含 policy 段 | `tests/test_v27_service.py`：四端点 200/400/409、返回字段完整、与离线 API 结果一致、未训练实例 409 |
| 38 | **2.7.1** | 集成加固 + 全量回归 + 向后兼容扩展 | 跨特性集成测试（policy+online+active+lite+hierarchical 组合不冲突）；扩展 `test_v27_backcompat.py` 覆盖 8 旧 checkpoint + v2.7.0（共 9 件）；性能基准 `benchmarks/results/feature_latency_v2.7.0.json`；文档骨架对齐（ROADMAP/ARCHITECTURE 补 2.7 模块） | `tests/test_v27_integration.py`：组合特性默认路径全绿；backcompat 9 件全加载；latency JSON 落盘；全量回归无退化 |
| 39 | **2.7.2** | Patch 2：缺陷修复 + 边缘加固 + 文档精修 | 修复集成阶段暴露的边界问题（空动作集、极端漂移率、零样本主动学习池、量化 NaN 防护、horizon=1 分层退化、online 适配后校准器状态一致性）；`PredictionGuard` 兼容 policy 输出；校准器在 online 适配后的 ECE 验证；ROADMAP/ARCHITECTURE/DEPLOYMENT 更新至 2.7 线 | `tests/test_v272_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |
| 40 | **2.7.3** | 最终训练重建 + 全量验证 + 打包发布 | 正式训练重建 `predictor_v2.7.3.pt`（52191 参数量级，默认全量模型）+ `training_v2.7.3.json`（含全部新特性离线评估指标：policy/online/active/lite/hierarchical A/B）；全量 pytest 全绿报总数/覆盖率；9 checkpoint（含 v2.7.0/v2.7.3）向后兼容；真实起 HTTP 服务逐接口验证（含新增 4 端点 200/400/409）；`docs/VERIFICATION_v2.7.3.md`；打包 `udos-engine-v2.7.3.zip`（顶层 udos-engine/ 目录）；从 zip 独立解压到 /tmp 复跑验证（版本/pytest/build --quick/服务 /health=2.7.3 /evaluate 含 calibration/interval/新增端点/md5 一致） | 全量测试通过；zip 内 checkpoint md5 与工程一致；/health=2.7.3；/evaluate 含 calibration/interval 段；新增 4 端点可用；zip 内复跑全绿 |

## 设计约束

1. **MPC 动作优选**：候选动作由调用方提供（本引擎不生成动作空间）；对每个动作将其作为 scene_params 覆盖或初始状态扰动输入 predictor 做 rollout；打分 = 目标奖励（可配置，默认负 MSE）− λ·risk_score − 安全违例惩罚；空候选集返回 None 并标记 no_valid_action；默认不挂载时旧路径逐位一致。
2. **在线适配**：复用 `ood.StreamingDriftDetector` 做检测；触发后默认仅用新校准集重跑 PAVA（不改模型权重）；增量微调为 opt-in（`enable_finetune=True`）且有 epochs 上限；适配前后指标差记入日志；无漂移时完全不触发、不改变状态。
3. **主动学习**：信息增益分 = α·ensemble_variance + β·interval_width_norm + γ·ood_score_norm（权重可配置，默认等权）；对未标注样本池排序选 top-K；A/B 用同种子、同训练集大小对比主动选样 vs 随机选样的 eval_mse；若收益不显著则照实写 opt-in/条件依赖。
4. **模型轻量化**：剪枝用全局幅值阈值 + mask，推理时 mask 相乘；量化用 `torch.ao.quantization.quantize_dynamic`（Linear 层 INT8），仅 CPU；蒸馏用 KL 散度 + MSE 混合损失，学生模型为 d_model 减半的小 CTM；**默认全量模型不变**，lite 模型需显式调用；A/B 必须含参数量/延迟/eval_mse 三列。
5. **分层 rollout**：coarse_factor=N 时，每 N 步做一次粗粒度预测（跳步），其余步用细粒度修正；短 horizon（≤N）退化为普通 rollout 逐位一致；粗粒度用 scene_params 条件下的多步直接预测；A/B 对比 H=8/12/16 的逐步 MSE 累积率。
6. **实验注册**：纯元数据记录，不触碰模型权重；`register(name, config, metrics, seed, artifacts)` → `list()` / `get(name)` / `sweep_report(names)` 聚合多种子 mean/std/best；JSON 原子写入；不改变任何预测路径。
7. **零重依赖**：同 2.6 线约束，纯 torch + 标准库；量化用 torch 内置，不引入额外包。
8. **证据诚实**：每个特性若在 CPU 小模型上收益不明显或不成立，照实写 opt-in/混合/负面，保留被否决候选（参照 v2.2 SS、v2.3 权重、v2.5 温度/噪声、v2.6 hybrid/Sobol 纪律）。
9. **版本号**：每节点同步 `udos/__init__.py`、`pyproject.toml`、`Makefile`、`docker-compose.yml`、`Dockerfile` 及所有测试中的版本断言；dev 节点版本号写为 `2.7.0.devN`（内部），对外发布节点写 `2.7.0/2.7.1/2.7.2/2.7.3`。
10. **训练纪律**：仅 2.7.0 与 2.7.3 正式训练（~100s 每次）；其余 8 节点不重训，测试保持快速（全量 < 60s）；训练口径同 v2.6.2（seed=42 / n_per_kind=48 / epochs=60 / patience=12 / front / hybrid_weight=0）。

## 训练节点说明

- **2.7.0**：正式训练，同 v2.6.2 口径，产出 `predictor_v2.7.0.pt`（52191 参数）+ `training_v2.7.0.json`。policy 模块为推理时外挂，不影响训练。
- **2.7.3**：正式训练，同口径，产出 `predictor_v2.7.3.pt` + `training_v2.7.3.json`（含全部 2.7 新特性的离线评估 A/B 指标）。
- 其余 8 节点：不重训，用确定性算法 + 轻量单测推进。
