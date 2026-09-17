# UDOS v3.7 版本规划 —— 全域 WM 统一 + 分层神经控制（阶段四 + 大脑/小脑/脊髓）

> 起点：v3.6.3
> 终点：v3.7.3
> 训练节点：3.7.0（1 次）
> 迭代总数：恰好 10 节点
> 红线：三层控制为合成延迟预算可测的轻量化类比；不宣称复现真机 whole-body control；反射弧"先制动/保护再上报"必须可证。

## 主线
构建机器人神经系统三层架构：大脑=慢规划（复用 policy MPC，多步候选评估）、小脑=边缘轨迹平滑/跟踪协调（低通滤波/前馈补偿代理）、脊髓=本地反射弧（碰撞/越界即时制动，不等中央）。三回路多频率分层调度，量化各回路延迟预算。

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **3.7.0** | 三层控制架构核心 + 正式训练 | `udos/neural_control.py`：`HierarchicalController`（大脑/小脑/脊髓三层容器，统一 step 接口）、`ControlLayer` 基类（频率/延迟预算/优先级）；正式训练 `predictor_v3.7.0.pt` | `tests/test_v37_neural_core.py`：三层结构、step 接口、频率配置、空守卫、与 policy/loop 接口一致 |
| 2 | 3.7.0.dev1 | 大脑慢规划（复用 policy MPC） | `CortexPlanner`（封装 MPCActionSelector，慢频率 1-5Hz，多步候选 rollout 评估，输出目标轨迹点）；规划结果缓存供小脑消费 | `tests/test_v37_cortex.py`：规划输出、慢频率、候选评估、空候选守卫、与 policy 一致 |
| 3 | 3.7.0.dev2 | 小脑边缘轨迹平滑/跟踪协调 | `CerebellumTracker`（轨迹跟踪：PID 代理控制器 + 前馈补偿 + 低通平滑，快频率 10-50Hz；将大脑目标点转为平滑控制信号） | `tests/test_v37_cerebellum.py`：跟踪误差、平滑性、PID 参数、阶跃响应、空目标守卫 |
| 4 | 3.7.0.dev3 | 脊髓本地反射弧 | `SpinalReflex`（即时反射：碰撞检测→制动、越界→截断、速度超限→减速；"先制动/保护再上报"，反射延迟 <1 步，不等待大脑/小脑）；反射事件日志 | `tests/test_v37_spinal.py`：碰撞制动、越界截断、反射优先级、事件日志、无触发场景 |
| 5 | 3.7.0.dev4 | 三回路多频率分层调度 | `HierarchicalController.step()` 实现多频率调度：大脑每 N 步规划一次、小脑每步跟踪、脊髓每步反射；频率比可配；调度正确性验证 | `tests/test_v37_scheduler.py`：多频率调度、频率比、大脑不每步调用、脊髓每步检查、空守卫 |
| 6 | 3.7.0.dev5 | 安全边界与反射优先级 | 安全边界集成（复用 decision.safety_boundary）；反射优先级矩阵（脊髓 > 小脑 > 大脑，安全违例时脊髓覆盖上层指令）；安全状态机 | `tests/test_v37_safety.py`：优先级覆盖、安全违例制动、状态机、正常场景不触发 |
| 7 | 3.7.0.dev6 | 分层延迟预算量化 A/B | 量化各回路延迟：大脑(规划)/小脑(跟踪)/脊髓(反射)的实测延迟；频率-精度-延迟三维 A/B；落 `benchmarks/results/neural_latency_v3.7.0.json` | `tests/test_v37_latency_ab.py`：延迟 JSON 落盘、三维对比、频率扫描、opt-in |
| 8 | **3.7.1** | 集成 + HTTP 端点 | HierarchicalController 作为 PhysicalLoopRunner opt-in 控制层；server 新增 `POST /neural/step`（一步分层控制）、`POST /neural/reflex/log`（反射日志查询） | `tests/test_v371_integration.py`：loop+neural 组合、默认逐位一致、两新端点 200/400/409、全特性兼容 |
| 9 | **3.7.2** | 加固 + 23 checkpoint 兼容 + 性能 | backcompat 扩展至 23 件；性能基准 `feature_latency_v3.7.0.json`；全量回归 | `tests/test_v372_service.py`：backcompat 23 件、latency JSON、端点验证、全量回归 |
| 10 | **3.7.3** | Patch 精修 + 文档对齐 | 边界测试（零频率、反射冲突、PID 发散、大脑无候选）；更新文档至 3.7 线；全量 pytest | `tests/test_v373_edge.py`：边界全绿、全量回归、文档有效 |
