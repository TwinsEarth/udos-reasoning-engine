# UDOS 引擎 v3.1→v3.3 版本规划 —— ActionPiece 离散动作、Ego360 启发数据增强、架构精炼与效率鲁棒性收口

> 起点：v3.0.3（542 passed / 0 skipped / 92% 覆盖率，正式件 predictor_v3.0.3.pt，52191 参数，14 代 checkpoint）
> 终点：**v3.3.3**（对外发布，正式件 `predictor_v3.3.3.pt`）
> 训练节点：**3.1.0、3.2.0、3.3.0、3.3.3** 各重建一次正式件（共 4 次，每次 ~100s）；其余 26 节点不重训。
> 迭代总数：**恰好 30 节点**（3.1 线 10 + 3.2 线 10 + 3.3 线 10）。
> 铁律：v3.0.3 的 542 测试持续全绿、只增不删旧契约；新能力默认不改变旧默认输出；版本号同步且有断言。
> 红线：所有 PhysBrain 启发均为 "analogy, not reproduction"，CPU-only 合成数据验证，不宣称复现 VLM/视频预训练/真机。

## 主线逻辑

在 3.0 线（Physical Loop + 多任务头 + 形态无关动作 + 可供性 + 多模态未来预测 + 五维评测）基础上，进一步深化三个方向：
- **3.1 线**：受 PhysBrain ActionPiece 离散动作 token 启发，将连续动作转化为离散 token 序列，探索 tokenized action 建模、序列课程学习与 in-context action prompting。
- **3.2 线**：受 PhysBrain Ego360 全景人类数据启发，在合成数据上构建多视角/多情境数据增强、上下文窗口扩展与时间记忆机制，探索 in-context learning。
- **3.3 线**：架构精炼与效率鲁棒性收口——轻量 MoE 路由、蒸馏 v2、结构化剪枝 v2、效率 Pareto A/B、内存优化、鲁棒性加固，最终 3.3.3 全面验证。

---

## 3.1 线：ActionPiece 离散动作 token 化与序列建模（10 节点）

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 31 | **3.1.0** | ActionPiece tokenizer（连续动作→离散 token）+ 正式训练 | `udos/action_piece.py`：`ActionPieceTokenizer`（k-means 码本将连续动作向量量化为离散 token，含 encode/decode、码本学习、token 覆盖率统计；analogy: 受 PhysBrain ActionPiece 启发，在合成动作上验证）；正式训练重建 `predictor_v3.1.0.pt` + `training_v3.1.0.json`（scripts/build_v310_checkpoint.py，Makefile ckpt310） | `tests/test_v31_tokenizer.py`：码本学习收敛、encode/decode 往返误差、token 覆盖率、空动作守卫、save/load 码本 |
| 32 | 3.1.0.dev1 | 动作码本增强与多尺度码本 | `ActionPieceCodec`（支持多尺度码本：粗粒度+细粒度两级量化；码本初始化策略：k-means++ / 随机 / 均匀网格；码本大小可配置）；码本质量指标（困惑度/利用率） | `tests/test_v31_codec.py`：多尺度量化误差递减、码本利用率、初始化策略对比、save/load |
| 33 | 3.1.0.dev2 | Tokenized 动作序列预测（自回归 next-token） | `TokenizedActionPredictor`（给定历史动作 token 序列，自回归预测下一个 token；用简单 n-gram / 线性投影在合成数据上验证；不替换主 CTM，为推理外挂）；token 序列 → 连续动作解码 | `tests/test_v31_token_predict.py`：next-token 预测形状有限、序列生成、token→连续解码、与 policy 动作接口一致 |
| 34 | 3.1.0.dev3 | Token ↔ 连续动作解码器与平滑 | `TokenActionDecoder`（token 序列→连续动作，含线性插值平滑、关节限位后处理；对比硬解码 vs 平滑解码的动作连续性）；与 retargeting 集成 | `tests/test_v31_decoder.py`：解码形状有限、平滑后连续性提升、限位合规、与 retargeting 接口一致 |
| 35 | 3.1.0.dev4 | Tokenized vs 连续动作 A/B | A/B：tokenized action 预测 vs 直接连续动作预测的 eval_mse/延迟/参数对比；码本大小 8/16/32/64 扫描；落 `benchmarks/results/action_piece_ab_v3.1.0.json`；收益不稳 opt-in | `tests/test_v31_ab.py`：A/B JSON 落盘可复算、码本大小扫描、opt-in 默认关时旧路径逐位一致、被否决候选保留 |
| 36 | 3.1.0.dev5 | 序列课程学习（easy→hard） | `SequenceCurriculum`（按轨迹长度/复杂度排序训练样本，从短简单轨迹逐步过渡到长复杂轨迹；在合成数据上验证课程 vs 随机顺序的收敛速度/最终精度；纯数据调度，不改模型） | `tests/test_v31_curriculum.py`：课程排序正确、收敛速度对比、空数据集守卫、与训练流程集成（不改正式件口径） |
| 37 | 3.1.0.dev6 | In-context action prompting（few-shot） | `InContextActionPrompter`（给定 few-shot 示例轨迹+查询状态，拼接为上下文序列，用 tokenized predictor 预测后续动作；在合成数据上验证 1-shot/3-shot/5-shot 精度；analogy: 受 PhysBrain 长上下文启发） | `tests/test_v31_prompting.py`：few-shot 拼接正确、1/3/5-shot 精度对比、空示例守卫、与 token_predict 接口一致 |
| 38 | **3.1.1** | ActionPiece 与 Physical Loop 集成 | 将 tokenized action 预测作为 PhysicalLoopRunner.predict_action 的 opt-in 替代路径（`use_tokenized=True`）；in-context prompting 作为 loop 的上下文注入选项；集成测试确认默认路径不变 | `tests/test_v31_integration.py`：loop+tokenized 组合不冲突、默认路径逐位一致、opt-in 开启后 tokenized 路径可用、与全部 2.8-3.0 特性兼容 |
| 39 | **3.1.2** | 集成加固 + 15 checkpoint 向后兼容 + 服务端点 | backcompat 扩展至 v2.1.0..v3.1.0 共 15 件；server 新增 `POST /action/tokenize`（连续→token）、`POST /action/detokenize`（token→连续）；性能基准 `feature_latency_v3.1.0.json` | `tests/test_v312_service.py`：两新端点 200/400/409；backcompat 15 件全加载；latency JSON；全量回归无退化 |
| 40 | **3.1.3** | Patch 精修 + 文档对齐 | 边界测试（码本大小=1、空 token 序列、极端动作值、服务未训练态、in-context 示例超长）；更新 ROADMAP/ARCHITECTURE/DEPLOYMENT/README 至 3.1 线；全量 pytest | `tests/test_v313_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |

---

## 3.2 线：Ego360 启发合成数据增强与上下文扩展（10 节点）

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 41 | **3.2.0** | 合成多视角数据增强 + 正式训练 | `udos/ego_data.py`：`SyntheticEgoAugmenter`（受 Ego360 启发，在合成参数化数据上做多视角增强：视角旋转/平移、轨迹扰动、噪声注入、时间缩放；生成"第一人称视角"代理数据；analogy, not reproduction）；正式训练重建 `predictor_v3.2.0.pt` + `training_v3.2.0.json`（scripts/build_v320_checkpoint.py，Makefile ckpt320） | `tests/test_v32_augment.py`：增强输出形状正确、视角旋转几何一致、扰动可控、增强前后标签对齐、空数据集守卫 |
| 42 | 3.2.0.dev1 | 多视角合成数据生成与视图一致性 | `MultiViewGenerator`（从同一场景参数生成多个"视角"的状态序列，验证跨视图物理一致性：同一物体在不同视角下的状态变换可互逆）；视图间对应关系矩阵 | `tests/test_v32_multiview.py`：多视图生成形状、视图变换可逆性、跨视图一致性误差、空场景守卫 |
| 43 | 3.2.0.dev2 | 上下文窗口扩展（更长历史） | `ExtendedContextWindow`（支持 window>6 的长历史输入，含位置编码扩展、历史截断/填充策略；在合成数据上验证 W=6/12/24 的预测精度；不改默认 window=6） | `tests/test_v32_context.py`：长窗口输入形状、位置编码扩展、W=6 逐位等价锚点、W=12/24 精度对比、空窗口守卫 |
| 44 | 3.2.0.dev3 | 时间记忆机制（滑动窗口+摘要） | `TemporalMemory`（滑动窗口记忆 + 历史摘要向量，支持超过 window 的长时域信息保留；记忆更新/重置/查询；纯推理外挂，不改模型权重）；与 predictor 集成 | `tests/test_v32_memory.py`：记忆更新正确、历史查询、长时域信息保留、记忆重置、与 predictor 接口一致 |
| 45 | 3.2.0.dev4 | 数据增强 A/B（增强 vs 原始） | A/B：增强数据训练 vs 原始数据训练的 eval_mse/OOD 鲁棒性/噪声鲁棒性对比；增强类型消融（视角/扰动/噪声/时间缩放）；落 `benchmarks/results/ego_augment_ab_v3.2.0.json`；收益不稳 opt-in | `tests/test_v32_augment_ab.py`：A/B JSON 落盘、增强类型消融、opt-in 默认关、被否决候选保留、全量回归 |
| 46 | 3.2.0.dev5 | In-context learning（任务描述+示例） | `InContextLearner`（给定任务描述向量 + few-shot 示例 + 查询，拼接为扩展上下文输入 predictor；在合成数据上验证 0/1/3-shot 精度提升；analogy: 受 PhysBrain 262K 上下文启发） | `tests/test_v32_icl.py`：上下文拼接正确、0/1/3-shot 精度对比、任务描述向量、空示例守卫、与 extended_context 接口一致 |
| 47 | 3.2.0.dev6 | 长时域 rollout 与记忆协同 | `LongHorizonRollout`（H>4 的长时域预测，结合 TemporalMemory 逐步累积上下文，减少误差累积；与 hierarchical rollout 对比；H=8/12/16 验证） | `tests/test_v32_longhorizon.py`：长 horizon 形状、记忆累积、H=4 逐位等价、H=8/12/16 误差对比、与 hierarchical 接口一致 |
| 48 | **3.2.1** | 数据增强与 Physical Loop 集成 | 将 SyntheticEgoAugmenter 作为训练时数据增强 opt-in（不改正式件口径）；TemporalMemory 作为 PhysicalLoopRunner 的可选记忆模块；in-context learning 作为 loop 的上下文注入 | `tests/test_v32_integration.py`：loop+memory+ICL 组合不冲突、默认路径逐位一致、opt-in 开启后可用、与全部 2.8-3.1 特性兼容 |
| 49 | **3.2.2** | 集成加固 + 16 checkpoint 向后兼容 + 服务端点 | backcompat 扩展至 v2.1.0..v3.2.0 共 16 件；server 新增 `POST /augment/generate`（生成增强数据）、`POST /icl/predict`（in-context 预测）；性能基准 `feature_latency_v3.2.0.json` | `tests/test_v322_service.py`：两新端点 200/400/409；backcompat 16 件全加载；latency JSON；全量回归无退化 |
| 50 | **3.2.3** | Patch 精修 + 文档对齐 | 边界测试（增强参数极端值、长窗口 W=0、记忆溢出、ICL 示例维度不匹配、服务未训练态、long horizon H=0）；更新 ROADMAP/ARCHITECTURE/DEPLOYMENT/README 至 3.2 线；全量 pytest | `tests/test_v323_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |

---

## 3.3 线：架构精炼、效率优化与鲁棒性收口（10 节点）

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 51 | **3.3.0** | 架构精炼（CTM 残差连接+层归一化）+ 正式训练 | `udos/ctm_engine.py` 增量改进：可选残差连接（residual=True opt-in）、层归一化（LayerNorm opt-in）、改进初始化（Xavier/He 可配置）；**默认全关，与旧架构逐位等价**；正式训练重建 `predictor_v3.3.0.pt` + `training_v3.3.0.json`（scripts/build_v330_checkpoint.py，Makefile ckpt330，默认配置=旧架构） | `tests/test_v33_arch.py`：默认配置逐位等价锚点、残差连接前向、LayerNorm 前向、初始化可配置、save/load 新配置、训练不崩溃 |
| 52 | 3.3.0.dev1 | 轻量 MoE 任务路由 | `udos/moe.py`：`LightweightMoE`（N 个专家子网络+门控路由，按输入特征选择 top-k 专家；专家为小线性层，总参数可控；analogy: 受混合专家架构启发，在合成数据上验证；opt-in 默认关） | `tests/test_v33_moe.py`：MoE 前向形状、门控路由、top-k 选择、参数计数、空输入守卫、opt-in 默认关逐位等价 |
| 53 | 3.3.0.dev2 | 知识蒸馏 v2（温度+软标签+特征匹配） | `udos/lite.py` 扩展 `DistillationTrainerV2`（温度可调软标签 + 中间特征匹配损失 + 学生模型自动架构搜索 d_model 减半/四分之一；对比 v1 蒸馏的精度/参数） | `tests/test_v33_distill.py`：蒸馏 loss 下降、温度可调、特征匹配、学生参数计数、vs v1 对比、save/load 学生 |
| 54 | 3.3.0.dev3 | 结构化剪枝 v2（通道/头级剪枝+微调） | `udos/lite.py` 扩展 `StructuredPrunerV2`（按通道/注意力头级结构化剪枝，而非逐元素幅值；剪枝后可选少量微调恢复；对比 v1 幅值剪枝的稀疏度/精度） | `tests/test_v33_prune.py`：结构化剪枝稀疏度、通道级裁剪、微调恢复、vs v1 对比、save/load 剪枝态、opt-in |
| 55 | 3.3.0.dev4 | 效率 Pareto A/B（参数/延迟/精度三维） | 综合 A/B：全量模型 / MoE / 蒸馏学生 / 剪枝模型 的参数-延迟-eval_mse 三维 Pareto 对比；落 `benchmarks/results/efficiency_pareto_v3.3.0.json`；确定推荐配置（收益不稳则 opt-in） | `tests/test_v33_efficiency.py`：Pareto JSON 落盘、四维对比、推荐配置标注、opt-in 默认关、被否决候选保留 |
| 56 | 3.3.0.dev5 | 内存优化（梯度检查点+激活压缩） | `udos/training.py` 扩展：可选梯度检查点（gradient checkpointing，用时间换内存）、激活量化缓存（推理时中间激活 FP16 压缩）；**默认全关，不改旧路径**；CPU 内存占用对比 | `tests/test_v33_memory.py`：梯度检查点训练不崩溃、激活压缩推理、内存占用对比、默认关逐位等价、opt-in |
| 57 | 3.3.0.dev6 | 鲁棒性加固（噪声/OOD/外推/对抗扰动） | `udos/robustness.py`：`RobustnessEvaluator`（综合评估：噪声鲁棒性 train_sigma×test_sigma、OOD 检测命中率/误报率、外推 extrapolation 误差、对抗扰动 FGSM 代理；输出鲁棒性分数）；与现有 ood/noise 模块集成 | `tests/test_v33_robustness.py`：四项鲁棒性指标、分数 0-100、与 ood/noise 模块一致、空模型守卫、可复现 |
| 58 | **3.3.1** | 全特性集成 + 综合评测 | 跨全部 2.8-3.3 特性集成测试（loop+multitask+retarget+affordance+multimodal+eval5d+action_piece+ego_data+moe+robustness 组合不冲突）；五维评测 + 鲁棒性评测综合报告；backcompat 17 件 | `tests/test_v331_integration.py`：全特性组合默认路径逐位一致、opt-in 开启后各特性可用、综合评测 JSON、backcompat 17 件 |
| 59 | **3.3.2** | Patch 精修 + 文档对齐 | 边界测试（MoE 专家数=1、蒸馏温度=0、剪枝稀疏度=1.0、内存优化 OOM 防护、鲁棒性极端噪声、服务未训练态）；更新 ROADMAP/ARCHITECTURE/DEPLOYMENT/README 至 3.3 线；全量 pytest | `tests/test_v332_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |
| 60 | **3.3.3** | 最终训练重建 + 全量验证 + 代码收尾 | 正式训练重建 `predictor_v3.3.3.pt` + `training_v3.3.3.json`（含全部新特性离线评估 A/B：action_piece/ego_augment/moe/distill_v2/prune_v2/efficiency/robustness；scripts/build_v333_checkpoint.py，Makefile ckpt333）；backcompat 扩展至 18 件（v2.1.0..v3.3.3）；全量 pytest 全绿报总数/覆盖率；CHANGELOG 最后一条。**不做 zip 打包和独立 /tmp 复跑（由后续 QA 代理完成）** | 全量测试通过；18 代 checkpoint 兼容；正式件指标齐全；CHANGELOG 30 条（3.1-3.3） |

---

## 设计约束

1. **ActionPiece**：k-means 码本在合成动作上学习，token 为离散索引；tokenized 预测为推理外挂，不替换主 CTM；多尺度码本/课程学习/in-context prompting 均 opt-in；A/B 收益不显著则保留为被否决候选。
2. **Ego360 数据增强**：在合成参数化数据上做视角旋转/扰动/噪声/时间缩放，不涉及真实视频；多视图一致性用已知几何变换验证；上下文扩展/记忆/ICL 均不改默认 window=6；数据增强 A/B 不改正式件训练口径（正式件仍用原始数据）。
3. **架构精炼**：残差/LayerNorm/初始化改进均为 opt-in，默认全关与旧架构逐位等价；MoE 为轻量专家路由，总参数可控；蒸馏/剪枝 v2 对比 v1；效率 Pareto 确定推荐配置；内存优化默认关；鲁棒性评测为只读评估。
4. **兼容铁律**：v3.0.3 的 542 测试持续全绿、只增不删；新能力默认不改旧默认输出；改默认须同合同多口径/多种子证据+旧行为 opt-in+逐位等价锚点测试；每升版同步版本号。
5. **证据诚实**：每个特性若收益不明显或不成立，照实写 opt-in/混合/负面，保留被否决候选（参照历代纪律）。
6. **零重依赖**：纯 torch + 标准库，CPU-only 2 线程；不下载大权重、不用 GPU、无 docker。
7. **术语**：第二引擎统一 GPM；交付简体中文；所有 PhysBrain 启发处标注 "analogy, not reproduction"。
8. **训练纪律**：仅 3.1.0/3.2.0/3.3.0/3.3.3 正式训练（~100s 每次，同口径 seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）；其余 26 节点不重训。

## 训练节点说明

- **3.1.0**：正式训练，同口径，产出 `predictor_v3.1.0.pt`（52191 参数）+ `training_v3.1.0.json`。ActionPiece 为推理外挂。
- **3.2.0**：正式训练，同口径（原始数据，增强仅 opt-in），产出 `predictor_v3.2.0.pt` + `training_v3.2.0.json`。
- **3.3.0**：正式训练，同口径（默认旧架构配置），产出 `predictor_v3.3.0.pt` + `training_v3.3.0.json`。
- **3.3.3**：正式训练，同口径，产出 `predictor_v3.3.3.pt` + `training_v3.3.3.json`（含全部新特性离线评估 A/B）。
- 其余 26 节点：不重训，用确定性算法 + 轻量单测推进。

## 版本号同步

每节点升版时运行：
```bash
python3 scripts/bump_version.py <old> <new>
```
dev 节点版本号写为 `3.1.0.dev1` 等（内部），对外发布节点写 `3.1.0/3.1.1/.../3.3.3`。
