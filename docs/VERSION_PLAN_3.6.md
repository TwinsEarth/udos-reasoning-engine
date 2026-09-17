# UDOS v3.6 版本规划 —— PWM 物理世界模型（阶段三·物理）

> 起点：v3.5.3
> 终点：v3.6.3
> 训练节点：3.6.0（1 次）
> 迭代总数：恰好 10 节点
> 红线：analogy, not reproduction；不宣称复现 V-JEPA/Cosmos/Genie/视频世界模型；潜在空间为合成低维代理；零梯度外挂优先。

## 主线
在 hybrid（物理骨架+残差）、counterfactual（反事实 rollout）、future_multimodal（未来多模态代理头）、identification（场景参数辨识）之上，构建潜在空间前向世界模型：多步"想象"rollout、接触/碰撞事件预测、物理守恒一致性、世界模型与预测器 A/B。

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **3.6.0** | 潜在空间前向世界模型核心 + 正式训练 | `udos/world_model.py`：`LatentWorldModel`（将 predictor 中间激活作为潜在状态 z_t，学习/代理 z_{t+1}=f(z_t, action) 前向转移；若可学则为外挂小 MLP 不入主 state_dict，参照 hybrid 先例）；`imagine()` 多步潜在 rollout；正式训练 `predictor_v3.6.0.pt` | `tests/test_v36_wm_core.py`：潜在状态提取、前向转移形状、imagine rollout、零梯度锚点、空输入守卫 |
| 2 | 3.6.0.dev1 | 多步"想象"rollout | `LatentWorldModel.imagine_rollout()`（H 步潜在轨迹生成 + 解码回物理状态）；与 predictor.rollout 对比；H=1 逐位等价锚点 | `tests/test_v36_imagine.py`：多步想象形状、H=1 锚点、与真实 rollout 对比、空守卫 |
| 3 | 3.6.0.dev2 | 接触/碰撞事件预测 | `udos/wm_events.py`：`ContactPredictor`（从潜在状态预测接触/碰撞事件发生概率，基于碰撞几何代理而非学习）、事件序列标注 | `tests/test_v36_contact.py`：接触检测、事件序列、无接触场景、与 collision 模块一致 |
| 4 | 3.6.0.dev3 | 物理守恒一致性 | `udos/wm_conservation.py`：`ConservationChecker`（动量代理量 m*v、能量代理量 0.5*m*v^2 + 势能在 rollout 中的一致性检验；输出违反量）；合成可验 | `tests/test_v36_conservation.py`：动量/能量计算、守恒检验、违反检测、空序列守卫 |
| 5 | 3.6.0.dev4 | 世界模型与预测器一致性 A/B | A/B：WM imagine rollout vs predictor 真实 rollout 的逐步 MSE / 守恒违反量 / 延迟对比；落 `benchmarks/results/wm_consistency_v3.6.0.json`；收益不稳 opt-in | `tests/test_v36_consistency_ab.py`：A/B JSON 落盘、逐步对比、opt-in、被否决候选 |
| 6 | 3.6.0.dev5 | 世界模型不确定性/置信度 | `LatentWorldModel` 扩展：集成式不确定性（多次想象的方差，复用 ensemble 思想）、置信度校准；高不确定时回退 predictor | `tests/test_v36_uncertainty.py`：不确定性估计、置信度、回退机制、空守卫 |
| 7 | 3.6.0.dev6 | 想象 vs 真实 rollout 对比实验 | 综合实验：想象 rollout 在长 horizon(H=8/16/32) 下的误差累积 vs 真实 rollout；潜在空间压缩率；落 `benchmarks/results/wm_imagine_v3.6.0.json` | `tests/test_v36_imagine_ab.py`：长 horizon 对比、JSON 落盘、标注合成类比 |
| 8 | **3.6.1** | 集成 + HTTP 端点 | WM 作为 PhysicalLoopRunner opt-in 想象模块；server 新增 `POST /wm/imagine`（潜在空间多步想象）、`POST /wm/conservation`（守恒检验） | `tests/test_v361_integration.py`：loop+wm 组合、默认逐位一致、两新端点 200/400/409、全特性兼容 |
| 9 | **3.6.2** | 加固 + 22 checkpoint 兼容 + 性能 | backcompat 扩展至 22 件；性能基准 `feature_latency_v3.6.0.json`；全量回归 | `tests/test_v362_service.py`：backcompat 22 件、latency JSON、端点验证、全量回归 |
| 10 | **3.6.3** | Patch 精修 + 文档对齐 | 边界测试（零 horizon、NaN 潜在状态、守恒极端值、WM 未初始化）；更新文档至 3.6 线；全量 pytest | `tests/test_v363_edge.py`：边界全绿、全量回归、文档有效 |
