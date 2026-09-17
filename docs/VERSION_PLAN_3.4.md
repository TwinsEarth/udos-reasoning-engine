# UDOS 引擎 v3.4 版本规划 —— ICM 上下文记忆（In-Context Memory）大版本

> 起点：v3.3.4（766 passed / 0 skipped / ~93% 覆盖率，正式件 predictor_v3.3.3.pt，52191 参数，18 代 checkpoint）
> 终点：**v3.4.5**（对外发布，正式件 `predictor_v3.4.5.pt`）
> 训练节点：**3.4.0、3.4.5** 各重建一次正式件（共 2 次）；其余 10 节点不重训。
> 迭代总数：**恰好 12 节点**（3.4.0 大版本 + 6 dev + 5 patch）。
> 铁律：v3.3.4 的 766 测试持续全绿、只增不删；新能力默认 opt-in、不改默认输出；ICM 推理路径零梯度（权重逐位不变）；版本号同步且有断言。
> 红线：所有 ICM 借鉴均为 "analogy, not reproduction"，CPU-only 合成数据验证，不宣称复现视频 VLA/10-50 万小时预训练/真机。

## 主线逻辑

ICM（In-Context Learning for Manipulation）定义物理AI第三阶段：任务从"权重"转移到"上下文"，零梯度、人做一遍机器人看一遍当场执行。v3.2 线已有 `InContextLearner` 雏形，但实测 0-shot MSE=0.0464 而 1-shot=0.6686/3-shot=1.1076 **反而更差**——根因是朴素拼接原始窗口稀释了 52k 小模型的注意力信号。v3.4 线的 ICM 上下文记忆不是重复 naive few-shot，而是体系化增强：**检索式演示记忆库 + 零梯度原型聚合 + 事件级切分 + PCE 物理提示词接口 + 上下文预算管理**，在合成数据上证明"检索+聚合"路径优于"朴素拼接"，且权重逐位不变。

## 12 迭代节点

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **3.4.0** | ICM 上下文记忆核心 + 正式训练 | `udos/icm.py`：`DemonstrationEpisode`（输入轨迹→动作→结果的 PCE 因果块）、`DemonstrationMemory`（相似度检索 top-k）、`ICMAggregator`（零梯度原型编码+跨演示加权聚合，输出演示条件化修正）；先复现并解释既有 few-shot 退化（0.046→1.108，根因=朴素拼接稀释信号）；ICM 检索聚合路径在合成数据上验证 k-shot 不退化；正式训练重建 `predictor_v3.4.0.pt` + `training_v3.4.0.json`（scripts/build_v340_checkpoint.py，Makefile ckpt340） | `tests/test_v34_icm_core.py`：episode 结构、memory 检索正确性、aggregator 零梯度（权重 hash 前后一致）、k-shot MSE 不爆炸、复现 naive ICL 退化、空记忆守卫 |
| 2 | 3.4.0.dev1 | 事件级切分与三流对齐（WALL-WM analogy） | `udos/icm_events.py`：`EventSegmenter`（状态/动作变点检测切分事件边界）、`ThreeStreamAligner`（物理token/动作/结果三流在事件边界对齐）；合成可验：事件边界对应物理状态突变 | `tests/test_v34_events.py`：事件边界检测、三流对齐形状、空序列守卫、边界退化、与 icm 接口一致 |
| 3 | 3.4.0.dev2 | 跨本体演示归一化后 ICL | 复用 `retargeting.py`：`CrossEmbodimentICM`（源本体演示→DOF映射+时间重采样→目标本体归一化→条件化预测）；验证重定向后演示 ICL vs 原始演示 ICL | `tests/test_v34_cross_embodiment.py`：跨本体归一化、重定向后 ICL 可用、DOF 不匹配守卫、与 retargeting 接口一致 |
| 4 | 3.4.0.dev3 | PCE 物理提示词数据包接口 | 扩展 `pce_format.py`：`DemonstrationPrompt`（因果块 action→result + 适配块 cross-embodiment，序列化为上下文）、`PCEPromptParser`（解析物理提示词包为 ICM 可用的 episode 列表）；HTTP-ready JSON 序列化 | `tests/test_v34_pce_prompt.py`：prompt 结构、解析往返、因果块/适配块、空包守卫、与 pce_format 接口一致 |
| 5 | 3.4.0.dev4 | 上下文预算/检索压缩与延迟-收益 A/B | `udos/icm_budget.py`：`ContextBudgetManager`（窗口预算上限、top-k 检索截断、原型压缩；复用 cache/batch 模式）；延迟-精度 A/B：budget=4/8/16/32 扫描，落 `benchmarks/results/icm_budget_ab_v3.4.0.json`；呼应"8000步token不能全上云" | `tests/test_v34_budget.py`：预算截断、压缩正确性、A/B JSON 落盘、opt-in 默认关、被否决候选保留 |
| 6 | 3.4.0.dev5 | k-shot scaling 曲线 + 权重逐位不变锚点 | ICM k-shot scaling：0/1/3/5/10-shot MSE 曲线（合成基准）；权重不变锚点：`state_dict` md5 在 ICM 推理前后逐位一致；与 naive InContextLearner 对比（ICM 检索聚合 vs 朴素拼接） | `tests/test_v34_shot_scaling.py`：k-shot 曲线有限、权重 hash 锚点、vs naive ICL 对比、空示例守卫 |
| 7 | 3.4.0.dev6 | 三路线对照实验（数据/思维链/上下文 Scaling） | `scripts/icm_three_route_ab.py`：同合成基准上 ①数据Scaling（增加训练数据量→eval_mse）②思维链Scaling（CTM 多 tick iterations→精度/延迟）③上下文Scaling（ICM 演示数→精度/延迟）；样本-性能-延迟三维对照，落 `benchmarks/results/icm_three_route_v3.4.0.json`（明确合成类比） | `tests/test_v34_three_route.py`：三路线 JSON 落盘可复算、维度齐全、标注合成类比、opt-in |
| 8 | **3.4.1** | ICM 与 Physical Loop 集成 + 服务端点 | ICM 作为 `PhysicalLoopRunner` 的 opt-in 上下文记忆模块；server 新增 `POST /icm/predict`（演示条件化预测，避开既有 /icl/predict）、`POST /icm/demo/register`（注册演示到记忆库）；错误语义：客户端 400、未挂载 409、未知 404、异常 500 不崩 | `tests/test_v341_integration.py`：loop+icm 组合不冲突、默认路径逐位一致、两新端点 200/400/409、与全部 2.8-3.3 特性兼容 |
| 9 | **3.4.2** | 集成加固 + 19 checkpoint 兼容 + 性能基准 | backcompat 扩展至 v2.1.0..v3.4.0 共 19 件；性能基准 `benchmarks/results/feature_latency_v3.4.0.json`；ICM 端点延迟测量；全量回归无退化 | `tests/test_v342_service.py`：backcompat 19 件全加载、latency JSON、ICM 端点 200/400/409、全量回归 |
| 10 | **3.4.3** | Patch 精修 + 边界测试 + 文档对齐 | 边界测试（空记忆、零预算、跨本体 DOF=0、事件边界退化、ICM 未注册演示、服务未训练态）；更新 ROADMAP/ARCHITECTURE/DEPLOYMENT/README 至 3.4 线（含 ICM 运维与 PCE 接口）；全量 pytest | `tests/test_v343_edge.py`：边界条件全绿、全量回归无回归、文档链接有效 |
| 11 | **3.4.4** | 全特性集成 + 综合评测 | 跨全部 2.8-3.4 特性集成测试（loop+icm+memory+retarget+action_piece+events+budget 组合不冲突）；五维评测 + ICM 专项评测综合报告；backcompat 19 件 | `tests/test_v344_integration.py`：全特性组合默认路径逐位一致、opt-in 开启后各特性可用、综合评测 JSON、backcompat 19 件 |
| 12 | **3.4.5** | 最终训练重建 + 全量验证 + 20代兼容 + 收尾 | 正式训练重建 `predictor_v3.4.5.pt` + `training_v3.4.5.json`（含 ICM 全特性离线评估 A/B；scripts/build_v345_checkpoint.py，Makefile ckpt345）；backcompat 扩展至 20 件（v2.1.0..v3.4.5）；全量 pytest 全绿报总数/覆盖率；CHANGELOG 12 条收尾；`docs/VERIFICATION_v3.4.5.md` | 全量测试通过；20 代 checkpoint 兼容；正式件指标齐全；CHANGELOG 12 条（3.4 线）；零梯度可证 |

## 设计约束

1. **ICM 零梯度原则**：主 predictor 权重不动；ICMAggregator 若需可学演示编码器，作为外挂模块（不入主 state_dict，参照 hybrid 残差 MLP 先例）单独小训并说明参数量；3.4.0/3.4.5 重训主件须说明理由并给前后指标，且证明 ICM 推理路径本身零梯度。
2. **兼容铁律**：v3.3.4 的 766 测试持续全绿、只增不删；除被证明的 bug 修复外默认输出/数值逐位等价（加锚点测试）；新能力默认 opt-in、不改默认；18 代旧 checkpoint 全部仍可加载。
3. **先复现再超越**：必须先复现 v3.2 InContextLearner 的 0-shot=0.0464/1-shot=0.6686/3-shot=1.1076 退化并解释根因，再证明 ICM 检索聚合路径不退化（或收益不稳则 opt-in 并留候选账本）。
4. **证据诚实**：每个特性若收益不明显或不成立，照实写 opt-in/混合/负面，保留被否决候选（参照历代纪律：多头共享、旋转增强、few-shot ICL 变差、hierarchical 逐位等价等）。
5. **零重依赖**：纯 torch + 标准库，CPU-only 2 线程；不下载大权重/视频、不用 GPU、无 docker。
6. **术语**：第二引擎统一 GPM；交付简体中文；所有 ICM 启发处标注 "analogy, not reproduction"。
7. **训练纪律**：仅 3.4.0/3.4.5 正式训练（同口径 seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）；其余 10 节点不重训。
8. **日志纪律**：新模块必须沿用 `udos/logging_config.py`，默认走 stderr，绝不污染 stdout/HTTP 体/ /metrics 纯文本。

## 版本号同步

每节点升版时运行：
```bash
python3 scripts/bump_version.py <old> <new>
```
dev 节点版本号写为 `3.4.0.dev1` 等（内部），对外发布节点写 `3.4.0/3.4.1/.../3.4.5`。
