# UDOS v3.8 版本规划 —— 全域调度/数字孪生/多体协同与收口（终点 v3.8.6）

> 起点：v3.7.3
> 终点：**v3.8.6**（对外发布，正式件 predictor_v3.8.6.pt）
> 训练节点：3.8.0、3.8.6（2 次）
> 迭代总数：恰好 10 节点
> 红线：多体为合成参数化代理，非真机多机器人；数字孪生为合成场景；全域闭环可测可复算。

## 主线
整合 SFM(3.5) + PWM(3.6) + 分层神经控制(3.7) + ICM(3.4)，构建多体协同与世界模型调度、规划-执行-反馈全域闭环、合成数字孪生场景，最终大版本收口至 v3.8.6。

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **3.8.0** | 多体协同核心 + 正式训练 | `udos/multi_agent.py`：`MultiAgentScene`（N 个智能体状态容器，每个有独立目标/形态）、`AgentCoordinator`（简单冲突消解：优先级让行/速度调节，非学习）；正式训练 `predictor_v3.8.0.pt` | `tests/test_v38_multi_core.py`：多体状态、协调器、冲突消解、单体退化、空场景守卫 |
| 2 | 3.8.0.dev1 | 世界模型调度器 | `udos/wm_scheduler.py`：`WMScheduler`（为多体分配世界模型想象预算：谁需要想象、想象几步、何时回退真实预测；复用 LatentWorldModel）；预算-收益 A/B | `tests/test_v38_wm_scheduler.py`：调度分配、预算控制、回退机制、单体守卫 |
| 3 | 3.8.0.dev2 | 规划-执行-反馈全域闭环 | `udos/closed_loop.py`：`ClosedLoopOrchestrator`（大脑规划→小脑执行→脊髓反射→WM 想象反馈→空间感知更新，一步完整闭环；复用全部既有模块） | `tests/test_v38_closed_loop.py`：闭环 step、各模块调用、反馈更新、空场景守卫、与全部模块接口一致 |
| 4 | 3.8.0.dev3 | 合成数字孪生场景 | `udos/digital_twin.py`：`DigitalTwinScene`（合成多体+多物体+障碍的参数化场景生成器，可配置智能体数/障碍数/场景参数；场景快照/回放） | `tests/test_v38_twin.py`：场景生成、快照/回放、可配置性、空场景守卫、与 spatial/multi_agent 一致 |
| 5 | **3.8.1** | 多体协同 A/B + 冲突消解 | A/B：有协调 vs 无协调的碰撞率/到达率/延迟对比；智能体数 2/4/8 扫描；落 `benchmarks/results/multi_agent_ab_v3.8.0.json`；收益不稳 opt-in | `tests/test_v381_ab.py`：A/B JSON 落盘、智能体数扫描、opt-in、被否决候选 |
| 6 | **3.8.2** | 系统集成 + HTTP 端点 | ClosedLoopOrchestrator 作为服务级 opt-in 全域控制；server 新增 `POST /twin/step`（数字孪生一步闭环）、`POST /twin/scene`（创建/查询孪生场景） | `tests/test_v382_integration.py`：全模块组合、默认逐位一致、两新端点 200/400/409、全特性兼容 |
| 7 | **3.8.3** | 加固 + 24 checkpoint 兼容 + 性能 | backcompat 扩展至 24 件（v2.1.0..v3.8.0）；性能基准 `feature_latency_v3.8.0.json`；全量回归 | `tests/test_v383_service.py`：backcompat 24 件、latency JSON、端点验证、全量回归 |
| 8 | **3.8.4** | 综合评测 | 跨 3.4-3.8 全部特性综合评测（ICM+SFM+PWM+分层控制+多体+数字孪生组合不冲突）；五维评测 + 专项评测综合报告；backcompat 24 件 | `tests/test_v384_eval.py`：全特性组合、综合评测 JSON、backcompat 24 件、默认逐位一致 |
| 9 | **3.8.5** | Patch + 文档对齐 | 边界测试（多体零智能体、孪生场景极端配置、闭环模块缺失、调度预算为零）；更新 ROADMAP/ARCHITECTURE/DEPLOYMENT/README 至 3.8 线；全量 pytest | `tests/test_v385_edge.py`：边界全绿、全量回归、文档有效 |
| 10 | **3.8.6** | 最终训练重建 + 全量验证 + 25 代兼容 + 收尾 | 正式训练 `predictor_v3.8.6.pt` + `training_v3.8.6.json`（含全部新特性离线 A/B）；backcompat 扩展至 25 件（v2.1.0..v3.8.6）；全量 pytest 全绿报总数/覆盖率；CHANGELOG 40 条收尾；`docs/VERIFICATION_v3.8.6.md` | 全量测试通过；25 代 checkpoint 兼容；正式件指标齐全；CHANGELOG 40 条（3.5-3.8 线）；零梯度可证 |
