# UDOS 引擎 v2.6 版本规划 —— 因果可解释与物理一致性增强主线

> 起点：v2.5.2（2.4/2.5 线 20 节点终点，218 passed / 92%）
> 终点：**v2.6.2**（对外发布，正式件 `predictor_v2.6.2.pt`）
> 训练节点：**2.6.0** 与 **2.6.2** 各重建一次正式件；其余 8 节点不重训，用确定性算法 + 轻量单测推进。
> 铁律：v2.5.2 的 218 测试持续全绿、只增不删旧契约；新能力默认不改变旧默认输出；版本号同步且有断言。
> 主线：在「物理一致性 / 因果与可解释 / 推理正确性」上深做——learned-residual 混合物理修正为基线，向上构建反事实推演、场景参数辨识、自适应计算、不确定性决策传播。

## 2.6 线 10 节点清单

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 21 | **2.6.0** | learned-residual 混合物理修正 + 正式训练重建 | `udos/hybrid.py`：`HybridPhysicsCorrector`（一阶欧拉运动学校正项 + 学习残差 MLP，opt-in 默认关，关闭时与纯数据驱动逐位一致）；`PhysicsPredictor` 新增 `attach_hybrid` / `predict_next_hybrid`；`TrainConfig` 新增 `hybrid_weight`（默认 0.0）；正式训练重建 `predictor_v2.6.0.pt` + `training_v2.6.0.json`（同口径 52191 参数，混合模式 A/B 证据） | `tests/test_v26_hybrid.py`：关闭时逐位等价锚点、开启时运动学残差下降、save/load 持久化、混合 vs 纯数据 A/B |
| 22 | 2.6.0+dev1 | 因果链/反事实推演模块 | `udos/counterfactual.py`：`CounterfactualEngine`（给定干预 intervention={scene_param 覆盖/初始状态扰动}，输出反事实轨迹 + 影响度量 ATE：干预 vs 基线轨迹差的逐步 MSE + 终态偏移）；纯前向、确定性、不改变旧路径 | `tests/test_v26_counterfactual.py`：零干预=基线逐位一致、已知干预方向正确、ATE 度量非负、save/load 兼容 |
| 23 | 2.6.0+dev2 | 场景参数辨识/归因 | `udos/identification.py`：`SceneParameterIdentifier`（从观测窗口用梯度-free 网格搜索反演 scene_params，最小化多步 rollout 误差）+ `sobol_attribution`（一阶 Sobol 指数：各 scene_param 对输出方差的贡献占比，纯扰动采样）；确定性、可单测 | `tests/test_v26_identification.py`：已知参数反演误差在容差内、Sobol 指数和≈1、零扰动退化、与训练数据 API 一致 |
| 24 | 2.6.0+dev3 | 自适应计算（自适应 iterations + horizon） | `udos/adaptive.py`：`AdaptiveStopper`（按 CTM certainty 收敛动态停步：连续 k tick 变化 < ε 则提前返回，默认关闭 opt-in）+ `adaptive_horizon`（按区间宽度增长率动态截断 horizon，超过阈值则停步并标记 truncated）；默认关闭时与旧版逐位一致 | `tests/test_v26_adaptive.py`：关闭时逐位等价、开启时 ticks_used ≤ iterations、高确定性样本提前停、horizon 截断标记正确 |
| 25 | 2.6.0+dev4 | 不确定性向下游决策传播（风险分级/动作安全边界） | `udos/decision.py`：`RiskGrader`（综合 conformal 区间宽度 + OOD score + 校准后置信，输出 low/medium/high 三级风险 + 风险分数）+ `safety_boundary`（给定动作候选集，筛选区间下界不越安全阈值的动作）；纯后处理、不改变预测输出 | `tests/test_v26_decision.py`：风险分级单调性、已知 OOD 样本升为 high、安全边界筛选正确、空候选守卫 |
| 26 | 2.6.0+dev5 | 推理快照差分 / checkpoint 数值对比 | `udos/persistence.py` 扩展：`diff_snapshots`（对比两个 export_snapshot 的配置/校准器/分位差异，输出结构化 diff）+ `compare_checkpoints`（加载两个 checkpoint，在同测试集上对比预测差异分布与指标差）；纯前向、确定性 | `tests/test_v26_snapshot_diff.py`：同件 diff 为空、异件 diff 非空且字段正确、跨版本 checkpoint 可对比、指标差可复算 |
| 27 | 2.6.0+dev6 | 服务接口扩展（反事实/辨识/风险/快照差分） | `udos/server.py` 新增 `POST /counterfactual`（干预推演）、`POST /identify`（场景参数反演）、`POST /risk`（风险分级）、`POST /diff-checkpoints`（两 checkpoint 对比）；未训练 409、非法输入 400；`/evaluate` 含 risk 段（可选） | `tests/test_v26_service.py`：四端点 200/400/409、返回字段完整、与离线 API 结果一致 |
| 28 | 2.6.0+dev7 | 集成加固 + 全量回归 + 向后兼容扩展 | 跨特性集成测试（hybrid+counterfactual+adaptive 组合）；扩展 `test_v25_backcompat.py` → `test_v26_backcompat.py` 覆盖 7 个旧 checkpoint（v2.1.0/v2.2.1/v2.3.1/v2.4.0/v2.5.0/v2.5.2/v2.6.0）；性能基准 `benchmarks/results/ablation_hybrid_v2.6.0.json`；文档骨架对齐 | `tests/test_v26_integration.py`：组合特性不冲突、默认路径全绿；backcompat 7 件全加载；ablation JSON 落盘 |
| 29 | **2.6.1** | Patch 1：缺陷修复 + 边缘加固 + 文档精修 | 修复集成阶段暴露的边界问题（空 batch、极端 scene_param、horizon=1 自适应、NaN 防护）；`PredictionGuard` 兼容 hybrid 输出；校准器在 hybrid 模式下的 ECE 验证；ROADMAP/ARCHITECTURE/DEPLOYMENT 更新至 2.6 线 | 新增 `tests/test_v261_edge.py`：边界条件全绿；全量回归无回归 |
| 30 | **2.6.2** | 最终训练重建 + 全量验证 + 打包发布 | 正式训练重建 `predictor_v2.6.2.pt`（hybrid opt-in 默认关，52191 参数量级）+ `training_v2.6.2.json`（训练/校准/区间/hybrid A/B/反事实/辨识指标）；全量 pytest 全绿报总数/覆盖率；8 个 checkpoint（含 v2.6.0/v2.6.2）向后兼容；真实起 HTTP 服务逐接口验证（含新增 4 端点 200/400/409）；`docs/VERIFICATION_v2.6.2.md`；打包 `udos-engine-v2.6.2.zip`；从 zip 独立解压到 /tmp 复跑验证 | 全量测试通过；zip 内 checkpoint md5 与工程一致；/health=2.6.2；/evaluate 含 calibration/interval/risk 段；新增 4 端点可用 |

## 设计约束

1. **混合物理修正**：默认关闭（`hybrid_weight=0`），关闭时 `predict_next` 与 v2.5.2 逐位一致；开启时校正项 = 一阶欧拉预测 + 学习残差，残差 MLP 为新增可学参数（需说明参数量变化）。
2. **反事实推演**：干预通过 scene_params 覆盖或初始状态扰动实现；零干预必须与基线 rollout 逐位一致；ATE 度量为逐步 MSE 差 + 终态偏移向量。
3. **场景参数辨识**：网格搜索范围由 SCENE_PARAM_NAMES 的物理合理区间决定；Sobol 指数用 Saltelli 采样的简化版（纯扰动、确定性种子）。
4. **自适应计算**：`certainty_threshold` 已有但训练时强制 0.0；自适应停步为推理时 opt-in，不影响训练；默认关闭逐位等价。
5. **风险分级**：三级阈值由校准集统计确定（可配置）；不修改预测值，仅附加风险标签。
6. **快照差分**：仅对比非权重状态（配置/校准器/分位/OOD 统计）；checkpoint 对比在同测试集上跑前向。
7. **零重依赖**：同 2.5 线约束，纯 torch + 标准库。
8. **证据诚实**：每个特性若在 CPU 小模型上收益不明显或不成立，照实写 opt-in/混合/负面，保留被否决候选（参照 v2.2 SS、v2.3 权重、v2.5 温度/噪声纪律）。
9. **版本号**：每节点同步 `udos/__init__.py`、`pyproject.toml`、`Makefile`、`docker-compose.yml`、`Dockerfile` 及所有测试中的版本断言。

## 训练节点说明

- **2.6.0**：正式训练，hybrid_weight=0（默认关），与 v2.5.2 同口径训练 + 校准 + OOD + conformal，产出 `predictor_v2.6.0.pt`。另跑 hybrid_weight>0 的 A/B 消融落 `benchmarks/results/ablation_hybrid_v2.6.0.json`。
- **2.6.2**：正式训练，同口径，产出 `predictor_v2.6.2.pt` + `training_v2.6.2.json`（含全部新特性的离线评估指标）。
- 其余 8 节点：不重训，用确定性算法 + 轻量单测推进，测试保持快速（< 30s 全量）。
