# UDOS v4.5 版本计划 — 隐式思考 / Latent Reasoning（终点 v4.5.3）

> 起点 v4.4.1（1422 passed / 33 checkpoint），终点 v4.5.3。训练点从严控制（优先零梯度外挂，能不训则不训；确需训练仅 4.5.0 一次并说明）。全部 CPU 合成数据机制类比，none 档逐位等价默认，low/high/max 与自适应路由 opt-in。

| # | 版本 | 主题 |
|---|---|---|
| 1 | 4.5.0 | latent_reasoner：CTM 隐藏状态内部 tick + K 条潜路径并行探索（latent best-of-K）+ 潜空间聚合/选择 + 隐式摘要 + 四档 effort 骨架（none 逐位等价锚点）+ 正式训练（若需） |
| 2 | 4.5.0.dev1 | reasoning_router：难度信号（CTM 收敛/ensemble 分歧/calibration/ood 不确定度/任务类型/4.3 停机判据）→ 隐式 tick 数/路径数/是否显式化 + 可解释切换理由 |
| 3 | 4.5.0.dev2 | 四档 effort 量化（none/low/high/max：tick 数/K/显式化/显式链长度/集成规模/延迟）+ none 逐位等价锚点测试 |
| 4 | 4.5.0.dev3 | 多专家在思考深度上协作（与 4.4 耦合：Agent-as-Tool 候选/评分、Orchestrator 聚合、分歧大则升级隐式→显式/star→handoff/mesh）+ "难度×拓扑×effort"联合策略 |
| 5 | 4.5.0.dev4 | 精度-延迟-token Pareto A/B（四档 + 自适应路由 vs 固定档位/全隐式/全显式，按难度分桶）落 JSON + Makefile |
| 6 | 4.5.0.dev5 | 可追溯：high/max 显式链与 4.4 Trace 统一（探索路径数/选择理由/切换点/专家分歧/收口人）+ 隐式摘要不黑盒 |
| 7 | 4.5.1 | HTTP 端点集成（/reason/latent effort=none|low|high|max、/reason/route 难度评估+推荐档位）+ 加固 |
| 8 | 4.5.2 | 性能基准 + 边界精修 + 文档对齐（ROADMAP/ARCHITECTURE/DEPLOYMENT/README） |
| 9 | 4.5.3 | 全量回归收口 + 最终验证 + 打包 + 全新解压 /tmp 复跑 |
