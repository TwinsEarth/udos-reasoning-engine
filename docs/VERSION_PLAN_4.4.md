# UDOS v4.4 版本计划 — 多智能体协作&协同&协调线（终点 v4.4.1）

> 起点 v4.3.9（1359 passed / 33 checkpoint），终点 v4.4.1。训练点从严控制（若框架本身零梯度外挂则不训练；确需训练仅 4.4.0 一次并说明）。全部 CPU 合成数据机制类比，默认 opt-in（swarm 尤其默认关），外挂优先零梯度。

| # | 版本 | 主题 |
|---|---|---|
| 1 | 4.4.0 | 四拓扑统一接口 + 拓扑选择器决策树 + Transfer Bundle 五要素 + 治理三件套（Owner/Trace/Stop）+ 正式训练（若需） |
| 2 | 4.4.0.dev1 | 能力注册表：UDOS 既有能力注册为 Agent-as-Tool（Tool Schema input/output/confidence/error_type） |
| 3 | 4.4.0.dev2 | 星型 Orchestrator：任务拆分→Worker 分配→结果收集器（去重/校验/一致性），复用 3.8 wm_scheduler/closed_loop |
| 4 | 4.4.0.dev3 | 链式 Handoff：Triage→Specialist→Return + Bundle 交接 + 可回退/补救 |
| 5 | 4.4.0.dev4 | 网状 Swarm（默认关闭 opt-in）：peer 能力声明/发现/局部协商/共识或冲突-未对齐检测 |
| 6 | 4.4.0.dev5 | 三类失败 RED→GREEN 回归：状态丢失（残缺 bundle 被拒）、重复劳动（认领锁去重）、责任不清（无 owner/stop 不得启动） |
| 7 | 4.4.0.dev6 | 拓扑 A/B（完成率/重复工作量/trace 完整率/冲突数/收敛跳数/延迟/消息量）落 JSON + Makefile |
| 8 | 4.4.1 | HTTP 端点集成（/collab/select、/collab/run、/collab/handoff、/collab/trace/{id}）+ 加固 + 性能基准 + 文档对齐 + 全量回归收口 |
