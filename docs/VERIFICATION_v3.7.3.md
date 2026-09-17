# UDOS v3.7.3 验证报告 —— 全域 WM 统一 + 分层神经控制（大脑/小脑/脊髓）

> 起点 v3.6.3（1018 测试）→ 终点 v3.7.3。analogy, not reproduction：分层控制为合成可测延迟的轻量化类比，不宣称复现真机 whole-body control。

## 1. 交付概览

| 节点 | 版本 | 主题 | 关键交付 |
|---|------|------|----------|
| 1 | 3.7.0 | 三层控制核心 + 正式训练 | `udos/neural_control.py`、`predictor_v3.7.0.pt` |
| 2 | 3.7.0.dev1 | 大脑慢规划 | CortexPlanner 封装 MPCActionSelector |
| 3 | 3.7.0.dev2 | 小脑跟踪 | CerebellumTracker PID+前馈+低通 |
| 4 | 3.7.0.dev3 | 脊髓反射弧 | SpinalReflex 碰撞/越界/超速 |
| 5 | 3.7.0.dev4 | 多频率调度 | run_counts / scheduler_summary |
| 6 | 3.7.0.dev5 | 安全边界与优先级 | safety_state 机 / priority_matrix |
| 7 | 3.7.0.dev6 | 延迟 A/B | neural_latency_v3.7.0.json |
| 8 | 3.7.1 | HTTP | /neural/step、/neural/reflex/log |
| 9 | 3.7.2 | 加固+基准 | 23 checkpoint、feature_latency_v3.7.0.json |
| 10 | 3.7.3 | Patch+文档 | test_v373_edge.py、本报告 |

## 2. 架构

- **大脑 CortexPlanner**：慢频率（默认 2Hz），封装 `policy.MPCActionSelector` 多步候选 rollout 评估，输出目标轨迹点；非规划步复用上一步缓存。
- **小脑 CerebellumTracker**：快频率（默认 20Hz），PID 代理 + 前馈 + 一阶低通，把目标点转为平滑控制信号。
- **脊髓 SpinalReflex**：最高频（默认 50Hz），碰撞→零速制动、越界→截断、超速→减速；同步当步完成（反射延迟 <1 步），先制动再上报。
- **优先级**：脊髓(0) > 小脑(1) > 大脑(2)；反射触发时 final command = 脊髓命令，覆盖上层（可证）。

## 3. 实测指标（CPU 2 线程）

- 分层单次延迟 p50：**脊髓 ≈0.005ms ≪ 小脑 ≈0.035ms ≪ 大脑 ≈3.2ms**；均在合成预算（0.5/5/40ms）内。
- 单步 `neural_full_step` ≈3.44ms（大脑规划主导）。
- 正式件 `predictor_v3.7.0.pt`：主参 **52191**（不变），eval_mse **0.045556**（与 v3.6.0 同口径）。

## 4. 兼容性与正确性

- **23 代 checkpoint**（v2.1.0..v3.7.0）全部可加载，主参恒 52191，输出有限。
- 默认输出逐位等价：未构造 `HierarchicalController` 时旧推理路径零改动；HTTP `/loop/step` 不受影响。
- 零梯度外挂：三层为确定性算法（PID/反射无可训参数），step 前后主权重 md5 不变。
- HTTP 错误语义：ServiceNotReady→409、ValueError→400、未知路由→404、异常→500。

## 5. 产物

- 代码：`udos/neural_control.py`
- 脚本：`scripts/build_v370_checkpoint.py`（`make ckpt370`）、`neural_latency_v370.py`、`neural_feature_latency_v37.py`
- 基准：`benchmarks/results/{training,neural_latency,feature_latency}_v3.7.0.json`
- 测试：`tests/test_v37{_neural_core,_cortex,_cerebellum,_spinal,_scheduler,_safety,_latency_ab}.py`、`test_v371_integration.py`、`test_v372_service.py`、`test_v373_edge.py`
