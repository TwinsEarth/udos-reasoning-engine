# UDOS 引擎 v2.8→v3.0 版本规划 —— 受 PhysBrain 1.5 启发的 Physical Loop 统一闭环、形态无关动作与多模态未来预测

> 起点：v2.7.3（364 passed / 0 skipped / 92% 覆盖率，正式件 predictor_v2.7.3.pt，52191 参数）
> 终点：**v3.0.3**（对外发布，正式件 `predictor_v3.0.3.pt`）
> 训练节点：**2.8.0、2.9.0、3.0.0、3.0.3** 各重建一次正式件（共 4 次，每次 ~100s）；其余 26 节点不重训，用确定性算法 + 轻量单测推进。
> 迭代总数：**恰好 30 节点**（2.8 线 10 + 2.9 线 10 + 3.0 线 10）。
> 铁律：v2.7.3 的 364 测试持续全绿、只增不删旧契约；新能力默认不改变旧默认输出；版本号同步且有断言；PhysBrain 外部数字与 UDOS 自测数字严格分开。
> 红线：UDOS 是 CPU-only、~52191 参数、合成参数化动力学小模型，**不可能也不得宣称复现 2B/8B VLM、Ego360 全景视频预训练或真机控制**。所有借鉴均为 "analogy, not reproduction"，在代码注释/文档中显式标注。

## 主线逻辑

受 PhysBrain 1.5 "Physical Loop"（观察→理解→预测动作→未来状态→反馈修正）统一架构启发，在 UDOS 现有 policy/online/adaptive/predictor 模块之上构建显式闭环 runner；进一步借鉴其统一自回归多任务头、Human-as-Humanoid 形态无关动作重定向、未来状态三模态预测、可供性打分，以及五维评测体系。所有机制在合成参数化动力学数据上做轻量化类比验证，A/B 证据不成立则 opt-in 或保留为被否决候选。

---

## 2.8 线：Physical Loop 统一闭环与多任务统一头（10 节点）

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **2.8.0** | Physical Loop 五步闭环 runner 骨架 + 正式训练重建 | `udos/physical_loop.py`：`PhysicalLoopRunner`（observe→understand→predict_action→future_state→feedback 五步显式编排，整合现有 predictor/reasoning/policy/online/adaptive 而非新建重复件；每步可插拔、默认不挂载时旧路径逐位一致；loop_state 记录每步输入输出与耗时）；正式训练重建 `predictor_v2.8.0.pt` + `training_v2.8.0.json`（同口径 52191 参数） | `tests/test_v28_loop.py`：空循环守卫、五步顺序正确、默认不挂载逐位等价、loop_state 完整、save/load 兼容 |
| 2 | 2.8.0.dev1 | observe / understand 阶段实现 | `PhysicalLoopRunner.observe`（接收 window+scene_params，调用 predictor 特征提取）；`understand`（调用 reasoning 做空间/任务理解，输出结构化理解向量含目标状态/风险标记）；理解结果作为后续步骤输入 | `tests/test_v28_observe_understand.py`：observe 输出形状有限、understand 含目标/风险字段、空输入守卫、与旧 reasoning 接口一致 |
| 3 | 2.8.0.dev2 | predict_action 阶段（MPC 集成进 loop） | `predict_action`（调用 policy.MPCActionSelector，候选动作可由调用方提供或默认生成；输出最优动作+全排序+风险分）；动作历史累积进 loop_state | `tests/test_v28_predict_action.py`：空候选 no_valid_action、已知最优选择正确、风险惩罚单调性、动作历史累积、与旧 policy 逐位一致 |
| 4 | 2.8.0.dev3 | future_state 阶段（rollout 预测集成） | `future_state`（对选中动作做 predictor.rollout 多步预测，输出未来 H 步状态轨迹+不确定性区间）；预测结果与 understand 的目标状态对比生成偏差信号 | `tests/test_v28_future_state.py`：rollout 形状 [B,H,6] 有限、偏差信号计算正确、horizon=1 退化、与旧 predictor rollout 逐位一致 |
| 5 | 2.8.0.dev4 | feedback / correction 阶段（online + adaptive 集成） | `feedback`（用 future_state 偏差 + online.StreamingDriftDetector 检测是否需要修正；默认仅记录不触发权重变更；opt-in 时触发 PAVA 再校准）；correction 信号回写 loop_state 供下一步 observe 使用 | `tests/test_v28_feedback.py`：无偏差不触发、偏差超阈记录、默认不改权重、opt-in 再校准、与旧 online/adaptive 接口一致 |
| 6 | 2.8.0.dev5 | 统一多任务头骨架（共享 backbone 接口） | `udos/multitask.py`：`MultiTaskHead`（定义共享 backbone 接口 `encode(window, scene_params) -> latent`，以及头注册机制 `register_head(name, head)`；默认不挂载时预测路径不变；latent 维度可配置）；纯接口层+确定性路由，不改变现有 CTM 权重 | `tests/test_v28_multitask_base.py`：头注册/查询/列表、空头守卫、encode 形状有限、默认路径逐位不变、save/load 头配置 |
| 7 | 2.8.0.dev6 | 空间坐标头 + 动作轨迹头 | `SpatialCoordHead`（从 shared latent 输出结构化空间坐标 [B, N_pts, 3] 代理目标，在合成数据里用状态向量的前 3 维代理 xyz 坐标）；`ActionTrajectoryHead`（输出动作轨迹 [B, H, action_dim]，与 policy 动作空间对齐）；两头共享 backbone，可独立/联合推理 | `tests/test_v28_heads.py`：坐标头形状有限、动作头形状有限、共享 backbone 梯度不冲突（推理模式）、头单独关闭时旧路径不变、A/B 共享 vs 独立头延迟对比 |
| 8 | **2.8.1** | 未来状态头 + 多任务 A/B 证据 + opt-in 门控 | `FutureStateHead`（输出未来状态代理 [B, H, 6] + 不确定性，与 predictor rollout 对齐）；多任务 A/B：共享 backbone 三头 vs 独立头的 eval_mse/延迟/参数对比，落 `benchmarks/results/multitask_ab_v2.8.0.json`；**收益不稳则 opt-in 默认关**；`MultiTaskHead.enable` 开关 | `tests/test_v28_multitask_ab.py`：未来头形状有限、A/B JSON 落盘可复算、opt-in 默认关时旧路径逐位一致、enable 后三头联合推理、被否决候选保留 |
| 9 | **2.8.2** | 集成加固 + 11 checkpoint 向后兼容 + 服务端点 | 跨特性集成测试（loop+multitask 组合不冲突）；扩展 backcompat 覆盖 v2.1.0..v2.8.0（共 11 件）；`udos/server.py` 新增 `POST /loop/step`（单步 Physical Loop）、`POST /multitask/predict`（多任务头联合推理）；未训练 409、非法输入 400；性能基准 `feature_latency_v2.8.0.json` | `tests/test_v28_integration.py`：组合默认路径全绿；backcompat 11 件全加载；两新端点 200/400/409；latency JSON 落盘；全量回归无退化 |
| 10 | **2.8.3** | Patch 精修 + 文档对齐 + 边缘加固 | 修复集成阶段边界问题（loop 空 window、multitask 头维度不匹配、future_state horizon=0、feedback 极端偏差率、服务端点未训练态）；ROADMAP/ARCHITECTURE/DEPLOYMENT/README 更新至 2.8 线；Makefile 新增 `ckpt280` target | `tests/test_v283_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |

---

## 2.9 线：形态无关动作生成/重定向与空间可供性（10 节点）

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 11 | **2.9.0** | Human-as-Humanoid 形态无关动作重定向适配器 + 正式训练 | `udos/retargeting.py`：`MorphologyConfig`（DOF 数、控制频率、关节限位、运动学参数）+ `ActionRetargeter`（将源形态动作轨迹重定向到目标形态，含 DOF 映射、时间重采样、关节限幅；**analogy: not reproduction**，在合成数据里用不同维度的动作向量代理不同机器人形态）；正式训练重建 `predictor_v2.9.0.pt` + `training_v2.9.0.json` | `tests/test_v29_retarget.py`：形态配置验证、DOF 映射正确、控制频率重采样、关节限幅、空动作守卫、save/load 兼容 |
| 12 | 2.9.0.dev1 | 形态配置体系与预设形态库 | `MorphologyLibrary`（预设 3 种合成形态：prime_u_60dof 代理 60 自由度、arm_7dof、gripper_4dof；每种含 DOF/控制频率/限位）；形态配置序列化/反序列化；形态间兼容性检查 | `tests/test_v29_morphology.py`：预设形态参数正确、序列化往返一致、兼容性检查、未知形态守卫、与 retargeter 接口一致 |
| 13 | 2.9.0.dev2 | 动作时间重采样（跨控制频率对齐） | `ActionRetargeter.resample`（线性/三次样条插值将源频率动作重采样到目标频率，保持端点一致；支持频率比非整数）；重采样前后动作能量守恒检查；与 loop.predict_action 集成 | `tests/test_v29_resample.py`：频率比 2x/0.5x/1.5x 重采样形状正确、端点一致、插值单调性、能量近似守恒、与旧 policy 动作接口一致 |
| 14 | 2.9.0.dev3 | 零样本形态切换合成验证 | `ActionRetargeter.zero_shot_transfer`（源形态训练的动作策略直接重定向到未见目标形态，在合成数据上验证重定向后动作有限且满足目标限位）；A/B：重定向 vs 直接截断的 eval_mse/限位违例率对比，落 `benchmarks/results/retarget_ab_v2.9.0.json` | `tests/test_v29_zeroshot.py`：零样本切换后动作有限、限位违例率为 0、A/B JSON 落盘、未见形态守卫、收益不显著则 opt-in |
| 15 | 2.9.0.dev4 | 可供性 affordance 打分模块 | `udos/affordance.py`：`AffordanceScorer`（给定状态+物体代理向量，输出可操作部位打分 [B, N_parts] + 操作建议；在合成数据里用状态维度的子集代理"物体部位"，**analogy**）；打分基于距离/速度/可达性合成特征 | `tests/test_v29_affordance.py`：打分形状有限、归一化正确、空物体守卫、可达性过滤、与 loop.understand 集成 |
| 16 | 2.9.0.dev5 | 空间关系头（物体间空间关系推理） | `SpatialRelationHead`（multitask 注册新头，从 shared latent 输出物体对空间关系矩阵 [B, N_obj, N_obj, rel_dim]，代理"左/右/上/下/接触"关系；在合成数据里用状态向量间几何关系代理）；关系头可独立开关 | `tests/test_v29_spatial_relation.py`：关系矩阵形状有限、对称性约束（左右互为逆）、接触检测正确、头关闭时旧路径不变、与 multitask 框架兼容 |
| 17 | 2.9.0.dev6 | affordance + 动作联合推理 | `AffordanceActionPlanner`（结合 affordance 打分 + spatial relation + retargeting，生成面向可操作部位的动作序列；affordance 低的部位不生成动作）；与 PhysicalLoopRunner.predict_action 集成（opt-in） | `tests/test_v29_affordance_action.py`：联合推理输出动作有限、低 affordance 部位被过滤、空物体守卫、opt-in 默认关时旧 policy 逐位一致、与 loop 集成 |
| 18 | **2.9.1** | 重定向 A/B + affordance A/B 证据汇总 | 汇总 retargeting A/B（重定向 vs 截断）+ affordance A/B（有 affordance 引导 vs 无引导的动作成功率代理指标），落 `benchmarks/results/v29_feature_ab.json`；**收益不稳 opt-in**；被否决候选保留；`MorphologyConfig` 校验加固 | `tests/test_v29_ab.py`：A/B JSON 落盘可复算、opt-in 默认关、被否决候选记录、全量回归无退化 |
| 19 | **2.9.2** | 集成加固 + 12 checkpoint 向后兼容 + 服务端点 | 跨特性集成（retargeting+affordance+spatial_relation+loop+multitask 组合）；backcompat 扩展至 v2.1.0..v2.9.0（共 12 件）；server 新增 `POST /retarget/convert`（动作重定向）、`POST /affordance/score`（可供性打分）；性能基准 `feature_latency_v2.9.0.json` | `tests/test_v29_integration.py`：组合默认路径全绿；backcompat 12 件全加载；两新端点 200/400/409；latency JSON；全量回归无退化 |
| 20 | **2.9.3** | Patch 精修 + 文档对齐 + 边缘加固 | 修复边界（retargeting DOF=0、affordance 零物体、spatial_relation 单物体、频率比极端值、服务端点未训练态）；ROADMAP/ARCHITECTURE/DEPLOYMENT/README 更新至 2.9 线；Makefile 新增 `ckpt290` | `tests/test_v293_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |

---

## 3.0 线：未来状态多模态预测 + 五维评测套件 + 3.0 收口（10 节点）

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 21 | **3.0.0** | 未来状态多模态代理目标（RGB/深度/mask 低维向量）+ 正式训练 | `udos/future_multimodal.py`：`FutureMultimodalHead`（multitask 注册新头，输出未来 1 秒（H 步）的三模态代理：RGB 代理 [B,H,rgb_dim]、深度代理 [B,H,depth_dim]、对象 mask 代理 [B,H,mask_dim]；**在合成数据里用状态向量的不同线性投影代理三种模态，analogy not reproduction**）；三模态共享 backbone；正式训练重建 `predictor_v3.0.0.pt` + `training_v3.0.0.json` | `tests/test_v30_multimodal.py`：三模态输出形状有限、各模态独立可开关、默认关时旧路径不变、save/load 兼容、与 multitask 框架兼容 |
| 22 | 3.0.0.dev1 | RGB 代理头 + 深度代理头细化 | `RGBProxyHead`（用状态向量→低维"颜色统计"代理：均值/方差/直方图 bin，维度可配置默认 8）；`DepthProxyHead`（用状态向量位置→"深度排序"代理：相对距离/深度梯度，维度默认 4）；两头独立损失 + 联合推理 | `tests/test_v30_rgb_depth.py`：RGB 头形状 [B,H,8] 有限、深度头 [B,H,4] 有限、投影可逆性检查（合成数据）、头单独关闭、与旧 predictor 输出不冲突 |
| 23 | 3.0.0.dev2 | 对象 mask 代理头 | `MaskProxyHead`（用状态向量的场景参数→"对象存在性 mask"代理：二值/软 mask [B,H,mask_dim]，维度默认 N_obj；基于 scene_params 的阈值生成软 mask）；mask 与 RGB/深度空间对齐（同 H 步） | `tests/test_v30_mask.py`：mask 形状 [B,H,N_obj] 有限、值在 [0,1]、与 RGB/深度 H 对齐、空场景守卫、头单独关闭 |
| 24 | 3.0.0.dev3 | 跨模态空间对齐一致性损失 | `CrossModalAlignmentLoss`（约束三模态在同一时间步的空间一致性：RGB 统计与深度排序应相关、mask 应与 RGB 对象区域对应；在合成数据里用已知投影关系验证一致性损失下降）；一致性损失作为 multitask 训练的辅助损失（opt-in，推理时不计算） | `tests/test_v30_alignment.py`：一致性损失有限且可下降、三模态对齐检查、opt-in 推理时不计算、与训练流程集成（不影响正式件训练口径） |
| 25 | 3.0.0.dev4 | 未来预测 A/B 证据 | A/B：多模态未来头（三模态联合）vs 单模态未来头（仅状态预测）的 eval_mse/延迟/参数对比；跨模态一致性损失开 vs 关的对比；落 `benchmarks/results/future_multimodal_ab_v3.0.0.json`；**收益不稳 opt-in 默认关**；被否决候选保留 | `tests/test_v30_future_ab.py`：A/B JSON 落盘可复算、opt-in 默认关时旧路径逐位一致、被否决候选记录、全量回归无退化 |
| 26 | 3.0.0.dev5 | 五维评测套件（UDOS 内部基准） | `udos/eval_suite.py`：`FiveDimensionEvaluator`（参照 PhysBrain EvalKit 五维度为 **UDOS 自身**建多维评测：①视觉空间感知代理=状态重建精度；②多视角空间理解代理=scene_params 辨识精度；③具身认知与规划代理=policy 动作优选质量；④空间指向与可供性代理=affordance 打分准确率；⑤视觉轨迹推理代理=rollout 轨迹精度；**明确标注是 UDOS 内部基准，不是 PhysBrain 榜单分数**） | `tests/test_v30_eval_suite.py`：五维度各输出分数 0-100、综合均分计算正确、空模型守卫、与现有 evaluation 模块不重复、分数可复现 |
| 27 | 3.0.0.dev6 | 综合均分 + 多维 benchmark JSON + 历史对比 | `FiveDimensionEvaluator.composite_score`（五维加权平均，权重可配置默认等权）；对 v2.7.3/v2.8.0/v2.9.0/v3.0.0 四代正式件跑五维评测，落 `benchmarks/results/five_dim_evolution_v3.0.0.json`；Makefile 新增 `eval5d` target | `tests/test_v30_eval_evolution.py`：综合均分计算正确、四代对比 JSON 落盘、分数单调合理性检查（不强制提升）、eval5d target 可运行、与 PhysBrain 分数严格分开标注 |
| 28 | **3.0.1** | 3.0 集成加固 + 13 checkpoint 向后兼容 + 服务端点 | 跨特性集成（future_multimodal+eval_suite+全部 2.8/2.9 特性组合）；backcompat 扩展至 v2.1.0..v3.0.0（共 13 件）；server 新增 `POST /future/predict`（多模态未来预测）、`GET /eval/5d`（五维评测）；性能基准 `feature_latency_v3.0.0.json` | `tests/test_v30_integration.py`：组合默认路径全绿；backcompat 13 件全加载；两新端点 200/400/409；latency JSON；全量回归无退化 |
| 29 | **3.0.2** | Patch 精修 + 文档对齐 + 边缘加固 | 修复边界（multimodal H=0、eval_suite 零维度、mask 维度=0、alignment_loss NaN 防护、服务端点未训练态、五维权重和不为 1）；ROADMAP/ARCHITECTURE/DEPLOYMENT/README 更新至 3.0 线；Makefile 新增 `ckpt300` | `tests/test_v302_edge.py`：边界条件全绿；全量回归无回归；文档链接有效 |
| 30 | **3.0.3** | 最终训练重建 + 全量验证 + 打包发布 | 正式训练重建 `predictor_v3.0.3.pt`（52191 参数量级）+ `training_v3.0.3.json`（含全部新特性离线评估 A/B：multitask/retarget/affordance/future_multimodal/eval5d）；全量 pytest 全绿报总数/覆盖率；14 代 checkpoint（v2.1.0..v3.0.3）向后兼容；真实起 HTTP 逐接口验证（含全部新增端点 200/400/409）；`docs/VERIFICATION_v3.0.3.md`；`docs/PHYSBRAIN15_ANALYSIS.md`（调研子代理产出）；打包 `udos-engine-v3.0.3.zip`（顶层 udos-engine/ 目录）；从 zip 独立解压 /tmp 复跑（版本/pytest/build --quick/服务 /health=3.0.3 /evaluate 含 calibration/interval/新端点/md5 一致） | 全量测试通过；zip 内 checkpoint md5 与工程一致；/health=3.0.3；/evaluate 含 calibration/interval；全部新增端点可用；zip 内复跑全绿 |

---

## 设计约束

1. **Physical Loop**：五步为显式编排层，每步调用现有模块（predictor/reasoning/policy/online/adaptive），不重复造轮子；默认不挂载时 `predict_next` 两次逐位一致；loop_state 记录每步输入输出/耗时/偏差，纯元数据无副作用。
2. **多任务头**：共享 backbone 为接口层，默认不挂载时预测路径不变；三头（spatial_coord/action_trajectory/future_state）+ 2.9 的 spatial_relation + 3.0 的 future_multimodal 均通过 `MultiTaskHead.register_head` 注册；A/B 收益不稳则 opt-in 默认关，保留被否决候选。
3. **动作重定向**：MorphologyConfig 为合成形态配置（DOF/频率/限位），不涉及真实机器人；重定向含 DOF 映射+时间重采样+关节限幅；零样本切换在合成数据上验证；A/B 收益不显著则 opt-in。
4. **可供性**：用状态向量子集代理"物体部位"，打分基于距离/速度/可达性合成特征；与 spatial_relation 头联合推理；默认 opt-in。
5. **未来多模态**：RGB/深度/mask 均为低维向量代理（状态向量的不同线性投影），不碰真实图像；跨模态一致性损失为训练辅助损失 opt-in，推理不计算；三模态空间对齐（同 H 步）。
6. **五维评测**：明确标注是 UDOS 内部基准，不是 PhysBrain 榜单分数；五维度用合成数据代理；综合均分加权平均；与 PhysBrain 外部数字严格分开。
7. **兼容铁律**：v2.7.3 的 364 测试持续全绿、只增不删；新能力默认不改旧默认输出；改默认须同合同多口径/多种子证据+旧行为 opt-in+逐位等价锚点测试；每升版同步版本号（用 `scripts/bump_version.py old new`）且有断言。
8. **证据诚实**：每个特性若在 CPU 小模型上收益不明显或不成立，照实写 opt-in/混合/负面，保留被否决候选（参照历代 uniform 权重/SS/温度/噪声/hybrid/Sobol/hierarchical 纪律）。
9. **零重依赖**：纯 torch + 标准库；CPU-only 2 线程；不下载大权重、不跑大模型推理、不用 GPU、无 docker。
10. **训练纪律**：仅 2.8.0/2.9.0/3.0.0/3.0.3 正式训练（~100s 每次，seed=42/n_per_kind=48/epochs=60/patience=12/front/hybrid_weight=0）；其余 26 节点不重训，全量测试 < 60s。
11. **术语**：第二引擎统一 GPM（不写 CPM）；交付简体中文；所有 PhysBrain 借鉴处标注 "analogy, not reproduction"。

## 训练节点说明

- **2.8.0**：正式训练，同 v2.7.3 口径，产出 `predictor_v2.8.0.pt`（52191 参数）+ `training_v2.8.0.json`。Physical Loop 与 multitask 为推理时外挂/接口层，不影响训练。
- **2.9.0**：正式训练，同口径，产出 `predictor_v2.9.0.pt` + `training_v2.9.0.json`。retargeting/affordance 为推理时外挂。
- **3.0.0**：正式训练，同口径，产出 `predictor_v3.0.0.pt` + `training_v3.0.0.json`。future_multimodal 为推理时外挂头。
- **3.0.3**：正式训练，同口径，产出 `predictor_v3.0.3.pt` + `training_v3.0.3.json`（含全部新特性离线评估 A/B 指标）。
- 其余 26 节点：不重训，用确定性算法 + 轻量单测推进。

## 版本号同步

每节点升版时运行：
```bash
python3 scripts/bump_version.py <old> <new>
```
同步更新 `udos/__init__.py`、`pyproject.toml`、`Makefile`、`docker-compose.yml`、`Dockerfile`、所有 `tests/*.py` 中的版本断言、`scripts/verify_service_*.py`。dev 节点版本号写为 `2.8.0.dev1` 等（内部），对外发布节点写 `2.8.0/2.8.1/.../3.0.3`。
