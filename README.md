# UDOS Reasoning Engine · UDOS 推演引擎

**当前版本：v5.4.4** ｜ 双引擎（CTM 连续思维机 + GPM 场景内化）｜ 纯 CPU 可跑 ｜ 1572 个测试全绿 ｜ License: Apache-2.0

UDOS 推演引擎是 UDOS 物理世界数字化基础设施的**认知架构内核**：以 PCE-Format 物理 Token 为数据层，融合 Sakana AI 连续思维机（CTM）的时序同步推理与 Doc-to-LoRA（GPM）的场景内化；并在其上以“外挂、零梯度、opt-in、可证伪”的方式，叠加可信推演、因果决策、具身闭环、空间/世界模型、多智能体协作、自进化，以及 v5.x 的安全治理、AGI/ASI 情报、KV Cache 分层、类脑树突与精细生物物理数值核。

> **定位纪律（analogy, not reproduction）**：受前沿工作启发的模块只在 CPU 上实现可验证的**机制对齐**，不复刻论文规模、不伪造效应量；缺少 GPU / QAT / NEURON / 在线 LLM 等条件时显式返回 `ENV_BLOCKED` / 503。主模型 `PhysicsPredictor` 恒为 **52191 个可学习参数、eval_mse = 0.045556**，`checkpoints/` 内 33 个自训练 checkpoint 全部向后兼容。

## 能力矩阵

| 层 | 能力 | 版本 | 启用方式 |
|---|---|---|---|
| 双引擎核心 | CTM 时序同步推理、GPM 场景内化（HyperLoRA 前向补丁）、PCE-Format | v0.1 | 默认 |
| 可信推演 | PAVA 保序校准、split-conformal 区间、OOD/漂移、深度集成、退化守卫 | v2.3–v2.4 | 默认 + opt-in |
| 效率服务 | 批量推理、LRU 缓存、Prometheus 指标、快照/回滚 | v2.5 | 默认 |
| 因果决策 | learned-residual 混合修正、反事实、辨识归因、风险分级、MPC、主动学习 | v2.6–v2.7 | opt-in |
| 具身/世界模型 | 物理闭环、形态重定向、可供性、SFM、PWM、三层控制、数字孪生、WLA 动作线 | v2.8–v3.9 | opt-in |
| 自治/协作 | 自规划课程、自训练、配置自进化、四拓扑多智能体、隐式思考 best-of-K | v4.x | opt-in |
| 安全治理 | PBKDF2 认证、RBAC、受控联网、加密备份、自治 kill-switch、调试面板 | v5.0.1 | `UDOS_AUTH` / `UDOS_DEBUG` |
| AGI/ASI 情报 | EWMA 系数、奇点门、能力向量、S 曲线 ETA、Brier 记分、RSI 间隔 | v5.0.2 | `/intel/*`（状态端点常开） |
| KV Cache 类比 | HBM→DDR→SSD→remote 多级、分页/前缀共享、无损压缩、成本账本、fuse/cascade/infinity | v5.1 | `UDOS_KVCACHE` |
| 类脑树突 | 区室模型、树突门控、多模态对齐、DHS 分层调度（对拍 Hines） | v5.3 | `UDOS_BRAIN` |
| 精细生物物理 | 被动电缆、Hodgkin–Huxley、NMDA 时序抑制（可证伪）、Payeur 四类、突触位置鲁棒性、NGRAD 假设 | v5.4.3 | `UDOS_FINESIM` |

## 快速开始

```bash
pip install -r requirements.txt

# 全量回归（应全部通过）
python -m pytest -q -p no:warnings

# 启动主引擎（认证默认关），加载自训练 checkpoint
UDOS_AUTH=off python -m udos.server --port 8000 --preset small
curl localhost:8000/health

# 开启 v5.4.3 精细生物物理核（不开时 /finesim/* 返回 503；/intel/finesim 始终 200）
UDOS_FINESIM=on python -m udos.server --port 8000 --preset small
curl -X POST localhost:8000/finesim/hh -d '{"I":10}'
curl localhost:8000/intel/finesim

# 不开服务器，直接复算关键数值（电缆/HH/NMDA/Hines-DHS/鲁棒性）
python - <<'PY'
from udos.finesim import cable, hh, nmda, payeur, hines_dhs
print(cable.cable_params()); print(hh.simulate(10.0)); print(hh.f_I_curve())
print(nmda.falsifiable_control()); print(hines_dhs.compare())
print(payeur.synaptic_position_robustness())
PY
```

容器化：`docker compose up`（镜像 `python:3.12-slim`）；常用任务见 `Makefile`（`make test / cov / bench / train`）。

## 文档索引

- [架构总览](docs/ARCHITECTURE.md)
- [v5.4.3 内部结构与全版本更新记录](docs/UDOS引擎v5.4.3-内部结构与全版本更新记录.md)
- 全版本变更：[CHANGELOG.md](CHANGELOG.md)（v0.1–v5.0.1）与 `docs/VERIFICATION_v*.md`（v5.0.2 起逐版验证报告）
- [v5.4.3 精细生物物理核](docs/BRAIN_FINE_SIMULATION_v5.4.3.md) ｜ [v5.3 树突/DHS](docs/BRAIN_DENDRITIC_v5.3.md) ｜ [v5.1 KV Cache 分层](docs/KVCACHE_HIERARCHY_v5.1.md)
- [安全说明](docs/SECURITY.md) ｜ [部署说明](docs/DEPLOYMENT.md) ｜ [情报 API 契约](docs/INTEL_API_CONTRACT.md)
- [开源资源目录（第三方模型/数据集/仿真器元数据）](docs/OPENSOURCE_CATALOG.md)

## 与上游、第三方的关系

本仓库**不包含**任何上游或第三方的源代码与预训练权重：CTM / Doc-to-LoRA 仅为机制对齐，可选适配器在你显式启用时经 `huggingface_hub` 运行时拉取；`udos/connectors/` 与 `docs/opensource_catalog.*` 仅保存第三方模型/数据集/仿真器的元数据与许可信息。内置第三方资产仅有 Apache-2.0 的 ECharts（`web/echarts.min.js`）。详见 [NOTICE](NOTICE)。`checkpoints/` 为本项目自有合成动力学训练管线产出的小模型，不含任何上游预训练权重。

## 能力边界（请先阅读）

- `udos/finesim`、`udos/dendrite`、`udos/kvcache` 均为 **CPU 合成机制原型**，不是 NEURON/CoreNEURON/GPU/QAT 的复现，也不代表论文效应量。
- “最多 16 线程 / 约 10× / 100–1000× / 8 GPU 5 万神经元 / O(N³)→O(2N)” 以及 Intel QAT、TTFT≈5× 等数字为**论文或厂商特定条件口径，标 `[UNVERIFIED]`**；本仓库自测只报告本机 CPU 可复算的步数、层数、命中率与盈亏比。
- v5.3.1 规划中的 MuJoCo 因果虚拟小鼠**尚未实现**（依赖中无 mujoco、无对应模块）。
- 在线多 LLM 交叉打分、GPU + vLLM KV offload 实测、NEURON/DeepDendrite 对拍均未启动（需凭证 / GPU 预算）。
- **双引擎耦合范围（避免被名字夸大）**：`reason()` 内是三条相互独立的支路——① GPM 生成的 LoRA 前向补丁只注入演示用 `TinyBaseModel`（`udos/gpm_engine.py`），当前 `reason()` **不调用该基座前向**，LoRA 可注入/撤销（`internalize`/`reset` 零误差）但尚未端到端改变物理轨迹；② GPM 的 `scene_embedding` 仅作为场景条件送入内部轻量 `CTMPhysicsEngine` 支路，产出机制演示用 `prediction`；③ 对外的下一状态与多步轨迹 `predicted_state`/`future_states` 来自独立训练、经 `attach_predictor` 挂载的 `PhysicsPredictor.rollout`，其输入只有位置/速度观测窗口，**不接收** GPM 场景嵌入或四维场景参数。
- 自然语言 `query` 在 `reason()` 中仅被记录与回显，**不进入**任何预测计算；多模态为 RGB/深度/mask 的**低维代理头**（opt-in，A/B 显示不降低状态 MSE）；“自进化”是在**冻结主预测器**前提下对缓存/分片等运行配置做搜索；资源注册表中“列出某模型”仅代表元数据/懒加载契约就绪，**不等于已加载权重或已实际运行**。

## License

[Apache License 2.0](LICENSE)，第三方归属见 [NOTICE](NOTICE)。

## 引用

```bibtex
@software{udos_reasoning_engine_2026,
  title   = {UDOS Reasoning Engine},
  author  = {UDOS Authors},
  year    = {2026},
  version = {5.4.4},
  license = {Apache-2.0}
}
```

---

# 历史版本详细说明（v4.5.3 及更早）

> 以下为仓库累积保留的历史版本说明，原貌留存；v5.x（安全 / 情报 / KV Cache / 树突 / 精细核）的变更请看 `docs/VERIFICATION_v5.*.md` 与上方文档索引。

## （历史）UDOS 推演引擎 v4.5.3（隐式思考 / Latent Reasoning：CTM 潜空间分支探索 + 四档 effort + 自适应路由）

> **v4.5.3（当前）**在 v4.4.1 之上新增隐式思考线（外挂零梯度、opt-in）：CTM 连续隐藏状态
> K 条潜路径并行探索（latent best-of-K）、四档 effort（none/low/high/max，none 与默认逐位
> 等价）、难度自适应路由、多专家在思考深度上协作、high/max 显式链与 4.4 Trace 统一。
> HTTP `POST /reason/latent`、`POST /reason/route`。诚实账本：隐式不改善物理 MSE、不省延迟，
> 价值在可观测/可追溯。主参恒 52191、eval_mse 恒 0.045556、33 代 checkpoint 全兼容。详见
> `docs/LATENT_REASONING_RESEARCH.md`、`docs/VERIFICATION_v4.5.3.md`。

> **历史**：v3.3.4（hardening 补丁件）在 v3.3.3 之上做加固、不重训、不改权重（正式件仍
> `predictor_v3.3.3.pt`，md5 `f993bcbdd476473c28dd4604bbbe11d6`）：修复 6 项真缺陷（未训练 409、
> Docker checkpoint 代际、非法输入 400、health 线程安全、checkpoint 目录锚定工程根）；全量切换
> 标准库 `logging`（42 模块、默认 WARNING、stderr、`UDOS_LOG_LEVEL` 可调）；性能 2 项 ACCEPT
> （physical_loop 单步 −11%、bs=64 批量 −41%）。全量测试 766 passed / 覆盖率 93%。详见
> `docs/QA_HARDENING_v3.3.4.md` 与 `docs/VERIFICATION_v3.3.4.md`。
>
> **v3.3.2（3.3 线补丁件）**在 v3.2 之上新增：`udos/ctm_engine.py` 可选残差/LayerNorm/
> 可配置初始化（默认关，逐位等价）、`udos/moe.py` LightweightMoE、`lite.DistillationTrainerV2`/
> `StructuredPrunerV2`、`TrainConfig.gradient_checkpointing`+`ActivationFp16Cache`、
> `udos/robustness.py` RobustnessEvaluator。效率 Pareto（`efficiency_pareto_v3.3.0.json`）判定
> 剪枝/学生不重训精度退化 => 推荐 full、其余 opt-in。正式件 `predictor_v3.3.0.pt` 52191 参数；
> backcompat 17 件（3.3.1）。**analogy, not reproduction**。
>
> **v3.2.3（3.2 线终点）**在 v3.1 之上新增 `udos/ego_data.py`（Ego360 启发合成多视角数据增强
> SyntheticEgoAugmenter + MultiViewGenerator，analogy not reproduction）、`udos/extended_context.py`
> （ExtendedContextWindow 长历史窗口 + 位置编码）、`udos/temporal_memory.py`（TemporalMemory 滑动窗口记忆 + EMA 摘要）、
> `udos/incontext.py`（InContextLearner few-shot）、`udos/longhorizon.py`（LongHorizonRollout）。
> 正式重建 `checkpoints/predictor_v3.2.0.pt`（**52191 参数**，eval_mse 0.0456）。服务新增
> `/augment/generate` `/icl/predict`。**16 个 checkpoint 全部向后兼容**。
> A/B：数据增强 in-distribution 无一致提升（OOD 鲁棒性提升）故 **opt-in 默认关**；ICL/长窗口增益不显著如实保留。
> （analogy, not reproduction：受 Ego360/PhysBrain 长上下文启发的轻量化类比，非复现，
> 不涉及真机视频/第一人称 RGB/VLM；第二引擎统一称 GPM。）
>
> **v3.0.2（未来状态多模态预测 + UDOS 内部五维评测）**在 v2.9 之上新增
> **`udos/future_multimodal.py`**（FutureMultimodalHead：共享 latent 的 RGB/深度/对象 mask
> 三模态低维代理；RGBProxyHead/DepthProxyHead/MaskProxyHead；CrossModalAlignmentLoss
> opt-in 训练辅助损失）与 **`udos/eval_suite.py`**（FiveDimensionEvaluator 五维评测——
> **UDOS 内部基准，非 PhysBrain 榜单分数**）。正式重建 `checkpoints/predictor_v3.0.0.pt`
> （**52191 参数**，eval_mse 0.0456）。服务新增 `/future/predict` `/eval/5d`。**13 个
> checkpoint 全部向后兼容**。A/B：多模态头不提升状态 MSE（仅增 2.29× 参数），故 opt-in
> 默认关。全量 **542 passed / 覆盖率 ~92%**。（analogy, not reproduction：用 latent 的
> 不同线性投影代理三种模态，非复现，不涉及真实 RGBD 图像/VLM；第二引擎统一称 GPM。）
>
> **v2.8.3（统一物理闭环 + 多任务头）**在 v2.7 动作/自适应之上，新增**显式五步物理闭环编排**
> （`udos/physical_loop.py`，observe→understand→predict_action→future_state→feedback，每步可插拔、
> 默认未挂 hook 时与 `predict_next` 逐位一致）与**共享 backbone 多任务头**（`udos/multitask.py`，
> `MultiTaskHead` + SpatialCoord/ActionTrajectory/FutureState 三头，纯接口层、零新参数、opt-in）。
> 正式重建 `checkpoints/predictor_v2.8.0.pt`（**52191 参数**，eval_mse 0.0456）。服务新增
> `/loop/step` `/multitask/predict` 两端点。共享 backbone 三头 2.34× 延迟优势。
> （analogy, not reproduction：受 PhysBrain 1.5 启发的轻量化类比，非复现。）
>
> **v2.7.2（从预测到行动的闭环）**在 v2.6 因果/决策能力之上，把**预测转化为可执行的安全决策**：
> **MPC 动作优选**（`udos/policy.py`，候选动作 rollout + 风险/安全打分排序）、**在线漂移触发再校准**
> （`udos/online.py`，默认只重跑 PAVA、不改权重）、**主动学习选点**（`udos/active_learning.py`，
> A/B：主动 0.264 vs 随机 0.502）、**模型轻量化**（`udos/lite.py`，幅值剪枝 / 动态 INT8 / 蒸馏学生，
> 全 opt-in）、**分层 rollout**（`udos/hierarchical.py`，单自回归头下与平铺逐位一致，opt-in 脚手架）、
> **实验治理**（`udos/experiment.py`，多种子 sweep mean/std/best）。服务新增
> `/policy/select` `/online/adapt` `/active/sample` `/experiments` 四端点。**9 个 checkpoint 全部向后兼容**，
> 跨特性集成 + 边缘加固测试全绿。全量 **350+ passed / 覆盖率 ~92%**。
>
> **v2.6.2（2.6 线终点正式发布件）**在 v2.6.1 边缘加固代码上**正式训练重建 + 全量验证 + 打包发布**：
> `checkpoints/predictor_v2.6.2.pt`（**52191 参数**，seed=42 / n_per_kind=48 / epochs=60 / front /
> hybrid_weight=0，reload_consistent）、`training_v2.6.2.json`（eval_mse 0.0456、ECE 0.0565、
> coverage 0.8888、OOD 命中 81%/误报 6.5%、ECE 降 6.8×）。**8 个 checkpoint 全部向后兼容**，
> 真实 HTTP 逐接口验证通过（含 `/counterfactual` `/identify` `/risk` `/diff-checkpoints` 四新端点
> 200、非法输入 400、未训练 409），zip 独立解压到 /tmp 复跑通过。全量 **281 passed / 覆盖率 92%**。
>
> **v2.6.1（补丁 1，缺陷修复 + 边缘加固）**在 v2.6.0 之上做**边缘加固 + 文档精修**：
> **空 batch 加固**（`predict_batch([])` 返回 `[0,RAW_DIM]` 空张量而非报错）、**极端 scene_param
> 防护**（1e6 不产生 inf；NaN/inf 显式 ValueError，不静默传播）、**horizon=1 自适应 rollout** 正常返回、
> **RiskGrader NaN 降级**（OOD score 为 NaN 时降级 0 并标 `ood_nan_degraded`）、**反事实 NaN 干预**
> 显式 ValueError、**hybrid×guard 顺序修正**（guard 清洗 hybrid 修正后的最终输出）、**hybrid 不改
> certainty 路径**（校准器 transform 仍可用）。`tests/test_v261_edge.py`（10 用例）。
>
> **v2.6.0（因果/反事实/决策套件，纯前向、opt-in、不改旧输出）**在 v2.5 效率与可服务性之上补齐
> **可解释决策**：**混合物理修正**（`udos/hybrid.py`，一阶欧拉匀速骨架 + 极小 MLP≈806 参数学残差，
> 默认不挂载走旧路径逐位一致；A/B 残差降约 93.6%）、**干预式反事实**（`udos/counterfactual.py`，
> scene_params/初始状态/速度干预，零干预=基线逐位，ATE 逐步均方差）、**场景辨识/归因**
> （`udos/identification.py`，网格搜索反推 4 维隐藏物理参数 + 一阶 Sobol）、**自适应计算**
> （`udos/adaptive.py`，certainty 收敛早退 + conformal 半宽增长率截断）、**风险分级/安全边界**
> （`udos/decision.py`，区间宽+OOD+置信合成风险分三档）、**快照差分/对比**（`udos/persistence.py`
> `diff_snapshots`/`compare_checkpoints`）。服务端新增 **`/counterfactual`、`/identify`、`/risk`、
> `/diff-checkpoints`** 四端点。全部外挂 opt-in、默认不改变旧输出。
>
> **v2.5.2（2.4/2.5 线 20 节点终点）**在 v2.4 可信推演套件之上补齐**效率与可服务性**：
> **批量推理**（`udos/batch.py` BatchPredictor，变长分组/自动分片，与逐笔逐位一致 max diff 1.97e-06）、
> **LRU 推理缓存**（`udos/cache.py`，key=输入+权重哈希，默认关闭 opt-in）、服务端
> **`GET /metrics`**（Prometheus 文本）、**无状态快照**（`/export-snapshot`、`/import-snapshot`，不含权重）、
> **`POST /rollback`**（已加载 checkpoint 栈回滚）。正式件 `predictor_v2.5.2.pt`（**52191 参数**）、
> `training_v2.5.2.json`（eval_mse 0.0456、ECE 0.0565、coverage 0.8888、OOD 命中 81%/误报 6.5%）。
> 全量 **218 passed / 覆盖率 92%**，旧 checkpoint（v2.1.0–v2.5.0）全兼容，zip 独立解压复跑通过。
> **诚实记录**：CPU 52191 参数小模型单次推理快，批量/缓存收益主要在大批量与重复请求，两项均 opt-in 默认关闭。
>
> **v2.4.16（2.4 线收尾）**在 v2.3 可校准置信之上，把"可信物理推演"主线补齐为
> **鲁棒性 + 不确定性**：**OOD/分布漂移检测**（岭正则马氏距离 + KS，`udos/ood.py`）、
> **在线流式漂移**（固定窗口 Welford 统计，`StreamingDriftDetector`）、**深度集成**（多成员
> 分歧近似认知不确定性，N=1 退化为单模型）、**退化守卫**（NaN/inf 回退 + 越界截断）、
> **温度缩放校准**（与 PAVA 并存）、**噪声鲁棒训练**（`noise_sigma` opt-in）、
> **多名义水平 conformal 区间**（80/90/95）、**逐步逐维置信矩阵**（`per_step_confidence`）、
> 服务端 **`/detect-ood`、`POST /predict`（guard）**、**集成 checkpoint 持久化**。
> 全部外挂 opt-in、默认不改变旧输出；v2.4 线仅 2.4.0 训练一次，其余用确定性算法推进，总测试
> **171 passed**、`--cov` 总计 **93%**（新模块 ood 93% / ensemble 97% / guard 93% /
> calibration 98%，均 ≥80%）。A/B 证据诚实留档：噪声增强以干净精度换稳健缺口、温度缩放
> 未降 ECE、三水平区间均保守过覆盖（见 `benchmarks/results/*.json`）。
>
> **v2.3.1**针对 v2.2.1 实测的两个短板（置信未校准、长时程误差累积）做证据驱动
> 升级：新增 **保序回归置信校准 + 回归 ECE/可靠性**（确定性后处理，独立集 ECE 降约 4.5×）、
> **多时域均衡损失**（`front/uniform/back`；两组同合同 A/B 证实方案优劣对训练口径敏感、
> 不稳健，故默认仍为 `front` 逐位复现 2.2.1、其余 opt-in，正反结果全留档）、
> **split-conformal 经验预测区间与覆盖率**（90% 名义区间实测覆盖≈90%）、服务端
> **`/calibrate`、`GET /checkpoints`、`POST /load`**，校准器随 checkpoint 持久化。
> v2.2.1 的 77 测试原样全绿，总测试 91。
>
> **v2.2.1**在 v2.1 多步推演之上补齐工程化短板：新增 **Scheduled Sampling**
> （opt-in，默认关闭；A/B 实测收益随种子变化，故不设默认、如实披露）、**早停/统一
> 确定性**、**rollout 累积率与置信度分层校准诊断**、服务端 **`/evaluate`、`/save`**
> 模型管理（未训练 409、落盘路径白名单）。
>
> v2.1 在 v2.0「能学习 · 真耦合 · 可持久化 · 可解释」之上，把单步预测扩展为
> **多步滚动推演**，让 v2.0 中恒为 0 的场景门控**真正进入训练回路**（消融实证增益），
> 并新增**运动学一致性诊断**与开箱预加载 checkpoint。下文保留历代说明。

基于 **Sakana AI 两大开源代码库**实现的双引擎物理因果推演内核：

- **CTM**（Continuous Thought Machine，时序推演）—— 对齐 `SakanaAI/continuous-thought-machines`
- **GPM**（Generative Physics，场景内化）—— 对齐 `SakanaAI/doc-to-lora`（Doc-to-LoRA）

GPM 在**单次前向、零反向传播**内把物理场景压缩为 LoRA 并注入模型；CTM 在
**内部时间轴**上逐 tick 展开因果推演，以**神经元同步**作为表示、以 certainty
做自适应早停。二者协同形成「物理数据 → 场景记忆 → 因果决策」闭环。

> 两个上游代码库已随工程克隆到 `third_party/ctm`、`third_party/d2l`，
> 测试会**真实加载上游 CTM** 做交叉验证（非仅文字引用）。

## 目录结构

```
udos-engine/
├── udos/
│   ├── pce_format.py        # PCE-Format 物理 Token/场景、编解码、任意属性维自适应
│   ├── ctm_engine.py        # CTM 物理推演: 内部时间轴+神经元级时序+同步表示+早停
│   ├── gpm_engine.py        # GPM: Perceiver 瓶颈+HyperLoRA 生成+分块聚合+前向补丁注入
│   ├── reasoning.py         # 双引擎协同 + GPM场景条件化 + 可解释物理预测
│   ├── dynamics.py          # v2 单步数据集 + v2.1 参数化多步数据集/运动学残差
│   ├── training.py          # PhysicsPredictor+训练器(v2.2 SS/早停; v2.3 多时域权重/区间)
│   ├── calibration.py       # v2.3 保序回归校准(PAVA)+回归ECE/可靠性+残差分位
│   ├── evaluation.py        # 评估体系: 消融/分类/rollout/一致性 + v2.3 校准段/区间覆盖率
│   ├── persistence.py       # checkpoint 存/载 (兼容 v2.0/v2.1/v2.2 旧档, 校准器随件)
│   ├── debug.py             # 分级 Debug 面板 (默认关闭, UDOS_DEBUG=1 开启, L0-L3)
│   ├── server.py            # 零依赖 HTTP 服务 (/evaluate /save /calibrate /checkpoints /load)
│   └── adapters/
│       └── sakana_ctm_adapter.py  # 直接加载 third_party/ctm 真实实现
├── demos/                   # 5 个 Demo (demo5=v2.1 多步/场景条件/一致性)
├── checkpoints/             # 预置 predictor_v2.3.1.pt (并保留 v2.2.1/v2.1 历史件)
├── benchmarks/benchmark.py  # 性能基准 + 回归守卫 (--guard), 结果落 results/
├── scripts/smoke_test.py    # 真实起服务的端到端冒烟 (含全部 v2.3 接口 + checkpoint)
├── scripts/build_v231_checkpoint.py      # v2.3.1 可复现: 训练+校准+评估+落 checkpoint
├── scripts/ablation_horizon_weight.py    # v2.3 F2 多方案/多种子同合同 A/B
├── scripts/ablation_scheduled_sampling.py  # v2.2 F1 可复现 A/B (SS vs teacher-forcing)
├── tests/                   # 91 个测试: 单元/契约/学习/多步/消融/持久化/服务/鲁棒性/校准/上游
├── third_party/ctm          # SakanaAI/continuous-thought-machines (depth-1 克隆)
├── third_party/d2l          # SakanaAI/doc-to-lora (depth-1 克隆)
├── Dockerfile / docker-compose.yml / Makefile
├── docs/ARCHITECTURE.md     # 架构/张量形状/上游映射
├── docs/DEPLOYMENT.md       # 部署手册: API/容器/systemd/资源/排障
├── docs/BUGFIX_REPORT.md    # v0.1 缺陷闭环
├── docs/VERIFICATION_v0.2.0.md  # v0.2 测试/验证/部署报告
└── docs/VERIFICATION_v2.0.0.md  # v2 第二代验证报告 (训练前后真实 MSE)
```

## 第二代 v2.0.0：从"机制骨架"到"能学习"
1. **能学习**：`dynamics` 合成四类运动，`training` 训练 CTM 预测下一状态；
   实测 test MSE 从未训练 6.05→0.084（**降 72×、胜"状态不变"基线 2.49×**），
   certainty 0.005→0.833。`make train` 可复现。
2. **真耦合**：GPM 的 Perceiver 场景嵌入经**零初始化门控**条件化 CTM 每个内部 tick；
   门控初始为 0，开启即与一代逐元素等价（完全向后兼容），训练后场景才调制推演。
3. **可持久化**：`persistence` 存/载整套预测器或引擎，载入前向逐元素一致。
4. **可解释**：`reason` 输出下一时刻 `position/velocity`；服务新增 `POST /train`。

## v2.1.0：从"单步外推"到"多步推演 + 场景真正起作用"

v2.0 报告自陈短板：场景门控零初始化、且预测器单独训练，门控始终≈0，"场景改变推演"
并未被联合训练激活；且只能预测下一帧。v2.1 针对性补齐（本沙箱实测，见
`docs/VERIFICATION_v2.1.0.md`，脚本 `scripts/build_v21_checkpoint.py` 可复现）：

1. **多步滚动推演（M1）**：新增参数化数据集 `ParametricDynamicsDataset`
   （观测窗口 W + 隐藏物理参数 P + 未来 H 步目标）；训练用 **teacher-forcing** 多步监督，
   推理用 `rollout` 自由滚动。`reason(horizon=H)` 一次给出 H 步 pos/vel。
2. **场景条件联合训练 + 消融实证（M2）**：隐藏参数（初速/加速度/弹簧角频率/被撞速度）
   经 `scene_encoder` 进入 CTM 每 tick，门控从 0 真正学到非零。**消融对照**：训练后
   给隐藏参数 conditioned MSE=0.082 vs 不给 unconditioned=0.813，**场景条件增益 9.93×**；
   分类型 accel 增益最大（仅凭窗口无法估加速度）。测试以 `gain>1.3` 锁死该能力。
3. **物理一致性（M3）**：`kinematic_residual` 以**前一帧速度**做一阶欧拉
   x(t+1)=x(t)+v(t)·dt，分运动类型诊断；匀速段残差最低，加速/振动段保留 O(a·dt²)
   固有偏离（诚实标注，不做全局硬损失）。
4. **评估体系与开箱预加载**：`evaluation.evaluate_predictor` 统一产出单步/消融/分类/
   rollout 曲线/一致性；`POST /train` 返回全套指标；服务支持 `--checkpoint` 启动即带
   训练态（预置 `checkpoints/predictor_v2.1.0.pt`，222KB）。
- **向后兼容**：v2.0 全部 55 测试原样全绿；旧单步数据集、旧 checkpoint（`.get` 缺省）、
  零门控初始等价无条件等契约全部保留；总测试增至 **64**、覆盖率 **95%**。

## v2.2.x：长时程鲁棒训练 · 训练治理 · 校准诊断 · 模型管理

针对 v2.1 报告自陈短板（自由 rollout 误差累积、缺早停/确定性、缺置信校准、预测器不能
在线评估/落盘）补齐，设计判据见 `docs/VERSION_PLAN_2.2.md`，证据见
`docs/VERIFICATION_v2.2.1.md`：

1. **Scheduled Sampling（F1，opt-in）**：多步训练中按线性爬坡概率以模型自身上一步
   （截断梯度）预测替代真值拼回滚动窗口，缓解 exposure bias。**默认 `ss_max=0`，此时与
   v2.1 teacher-forcing 逐位元数值等价（有测试锁定）**。同合同 A/B（同种子/数据/初始
   权重）显示：其降低长时程 rollout 误差的收益**在本沙箱小模型+合成数据下随种子方向
   不稳**（一种子改善、另一种子恶化），因此保持 opt-in、不设默认、不宣称确定增益；
   `scripts/ablation_scheduled_sampling.py` 可复现全部对照。
2. **训练治理（F2）**：`set_seed` 统一确定性入口；`patience/min_delta` 早停并恢复最佳
   权重（`best_epoch/best_eval/stopped_early`），默认 `patience=None` 跑满，等价 v2.1。
3. **评估增强（F3，纯增量）**：`rollout_growth_x`（末步/首步误差累积率）与
   `confidence_stratification`（按置信度三档的误差分层）。诊断如实暴露当前 certainty
   **并非校准置信**（最高置信档误差反而更高），为后续正式概率校准提供动机。
4. **模型管理服务（F4）**：`POST /evaluate`（新鲜测试集在线评估）、`POST /save`
   （连同指标落 checkpoint，文件名清洗 + 路径不可逃逸的根本校验）；未挂载预测器统一
   返回 **409**；`/train` 可透传 SS 课程参数。
- **2.2.1 补丁**：边界/除零健壮性、版本号三处单一来源对齐、冒烟补新接口、QA 收口。
- **向后兼容**：v2.1 全部 64 测试原样全绿，总测试 **75**；旧接口字段只增不删不改语义。

## v2.3.x：可校准置信 · 多时域均衡 · 经验预测区间

针对 v2.2.1 实测短板（C1 certainty 未校准、C2 长时程误差累积且远期步损失欠加权），设计判据见
`docs/VERSION_PLAN_2.3.md`，证据见 `docs/VERIFICATION_v2.3.1.md`：

1. **置信度保序回归校准（F1，确定性后处理，新模块 `udos/calibration.py`）**：连续回归先以
   校准集误差均值把逐样本误差映射为 (0,1] 的经验精度，再用零依赖 PAVA 学一个单调非降的
   `原始置信→经验精度` 映射；评估输出**校准前/后回归 ECE、可靠性分桶、置信-误差 Spearman**。
   校准在独立测试集也显著降 ECE。**诚实标注**：当原始 certainty 与精度无单调一致关系时
   （v2.2.1 上确为坏排序），保序会退化为常量映射——它修正置信的**数值水平**，但不凭空制造
   排序能力，报告以 `ranking_informative/num_segments` 显式披露，不宣称"高置信必然更准"。
2. **多时域均衡损失（F2，无随机性的长时程杠杆）**：`TrainConfig.step_weight_scheme` 支持
   `front`（默认，=2.2.1 的 linspace(1,0.5)，逐位复现）/`uniform`（等权）/`back`。两组同合同
   A/B：口径A（无早停、3 种子）uniform 远期 tail 3/3 更低（均值 −60%）；但口径B（早停、正式
   build 数据划分）front 反超 uniform（单步 +50%、tail +29%）。**跨口径方向不一致 → 方案优劣
   对训练口径敏感、不稳健**，故默认保持 front、uniform/back 为 opt-in（脚本
   `ablation_horizon_weight.py`、`sensitivity_step_weight_pipeline.py` 可复现正反两面）。
3. **split-conformal 经验预测区间（F3）**：`PhysicsPredictor.predict_interval` 用校准集每步
   残差分位给出中值预测 + 区间，评估覆盖率/宽度；90% 名义区间在同分布独立集实测覆盖率达标，
   宽度随 horizon 单调不减。属交换性假设下的经验区间，不保证分布外覆盖。
4. **服务与持久化（F4）**：`POST /calibrate`（在线拟合并同时给独立集对照，未训练 409）、
   `GET /checkpoints`（列件）、`POST /load`（白名单载回，含校准器）；校准器与残差分位随
   checkpoint 存取，旧档无该键按未校准处理（向后兼容）。
- **2.3.1 补丁**：边界/退化健壮性、版本三处与镜像 tag 对齐、冒烟补 4 项、QA 收口。
- **向后兼容**：v2.2.1 全部 77 测试原样全绿，总测试 **91**；评估旧键只增不删；训练默认仍
  为 front（逐位复现 2.2.1），校准/区间/接口均为外挂增量，uniform/back 显式 opt-in。

## 安装

```bash
pip install -r requirements.txt          # torch / numpy / huggingface_hub
# 上游代码库已内置; 如需重新获取:
# git clone --depth 1 https://github.com/SakanaAI/continuous-thought-machines third_party/ctm
# git clone --depth 1 https://github.com/SakanaAI/doc-to-lora third_party/d2l
```

CPU 即可运行，无需 GPU、无需下载 LLM 权重。

## 快速开始

```bash
python -m demos.demo1_ctm_physics        # CTM 时序推演
python -m demos.demo2_gpm_internalize    # GPM 场景内化 + 无损 reset
python -m demos.demo3_dual_engine        # 双引擎协同 + 上游真实 CTM 对照
python -m demos.demo4_learn_to_predict   # v2 训练闭环: 学会预测下一状态
python demos/demo5_multistep_reasoning.py  # v2.1 多步/场景条件消融/一致性
python -m pytest tests/                  # 全部测试 (91 项)
make train                               # v2 训练演示 (EPOCHS=45 可调)
make ckpt23                              # v2.3.1 训练+校准+落评估JSON+预置checkpoint
make horizon-ablation                    # v2.3 F2 多方案/多种子同合同 A/B
make test guard smoke                    # 测试线/性能守卫/服务冒烟
# 启动即带训练+校准好的多步物理推演 (也可不带 --checkpoint 从零起)
python -m udos.server --port 8000 --checkpoint checkpoints/predictor_v2.3.1.pt
```

## 服务化与部署（v0.2.0）

零额外依赖的 REST 服务（Python 标准库），接口与部署见 `docs/DEPLOYMENT.md`：

```bash
python -m udos.server --host 0.0.0.0 --port 8000 --preset small
curl -s localhost:8000/health
curl -s localhost:8000/demo | python -m json.tool        # 一键全链路
# v2.1 场景条件多步训练并挂载 (可透传 ss_* 开 SS、step_weight_scheme 选权重)
curl -XPOST localhost:8000/train -d '{"epochs":45,"n_per_kind":32,"horizon":4}'
# v2.2 训练后: 新鲜测试集在线评估 / 落盘 checkpoint (未训练返回 409)
curl -XPOST localhost:8000/evaluate -d '{"n_per_kind":32,"horizon":4}'
# v2.3 保序置信校准(给独立集校准前/后ECE与区间覆盖) / 列件 / 载回
curl -XPOST localhost:8000/calibrate -d '{"n_per_kind":32,"horizon":4}'
curl -s localhost:8000/checkpoints | python -m json.tool
curl -XPOST localhost:8000/load -d '{"name":"predictor_run1.pt"}'
curl -XPOST localhost:8000/save -d '{"name":"predictor_run1.pt"}'
# 多步推演: reason 带 horizon, 返回 future_states(H 步)
curl -XPOST localhost:8000/reason -d '{"scene":{...},"horizon":4}'
# 容器化 (镜像默认带 --checkpoint 预加载 v3.3.3 正式件)
docker build -t udos-reasoning-engine:3.3.4 .
docker compose up -d
```

**日志配置（v3.3.4）**：服务与库统一走标准库 `logging`，默认级别 `WARNING`，全部输出到
`stderr`（不污染 stdout / 不串入 `/metrics` 与 HTTP 响应体）。用环境变量 `UDOS_LOG_LEVEL`
调整（`DEBUG`/`INFO`/`WARNING`/`ERROR`），例如：

```bash
UDOS_LOG_LEVEL=INFO python -m udos.server --host 0.0.0.0 --port 8000 --preset small \
    --checkpoint checkpoints/predictor_v3.3.3.pt
# INFO 记录 internalize/save/load/calibration/在线自适应摘要；DEBUG 记访问日志与 loop 每步
# /metrics 始终是纯 Prometheus 文本 (text/plain)，与日志流分离
```

| 接口 | 作用 |
|------|------|
| GET `/health` | 健康检查、已内化场景、预测器训练状态、版本 |
| POST `/internalize` | GPM 把 PCE 场景内化为 LoRA |
| POST `/reason` | CTM 时序推演（GPM 场景条件化），`horizon=H` 返回下一时刻及 H 步 future_states |
| POST `/reset` | 无损移除场景记忆 |
| POST `/train` | 参数化场景条件多步训练（v2.2 可透传 `ss_max/ss_start/ss_warmup`），返回 loss 曲线 + 全套评估 |
| POST `/evaluate` | v2.2 对挂载预测器在新鲜测试集在线评估；未训练 409 |
| POST `/save` | v2.2 把预测器连同指标落 `checkpoints/<name>.pt`；未训练 409，名称清洗防穿越 |
| POST `/calibrate` | v2.3 在独立校准集拟合保序校准+残差分位并挂载；返回独立集校准前/后 ECE 与区间；未训练 409 |
| GET `/checkpoints` | v2.3 列出 `checkpoints/` 下可用件（名/字节/修改时间） |
| POST `/load` | v2.3 按白名单名加载 checkpoint（含校准器）并挂载；不存在/穿越名 400 |
| GET `/demo` | 内置场景跑完整闭环 |

### 最小调用

```python
from udos import (PhysicalToken, PhysicsScene, CTMConfig, GPMConfig,
                  UDOSReasoningEngine)

scene = PhysicsScene(scene_id="demo")
for t in range(8):
    scene.add(PhysicalToken("arm-1", t, position=[0.1*t, 0, 0],
                            velocity=[0.1, 0, 0], attributes={"mass": 4.5}))

eng = UDOSReasoningEngine(CTMConfig(d_input=64, d_model=128, iterations=16),
                          GPMConfig(feature_dim=64, latent_size=64,
                                    init_scaler_b_zero=False))
print(eng.internalize_scene(scene))      # GPM: 场景 -> LoRA -> 注入
res = eng.reason(scene, query="下一时刻状态?")
print(res.summary())                    # ticks / certainty / 因果边数
print(eng.reset_scene("demo"))          # 无损移除场景记忆
```

## 与上游实现的机制对齐（不是照抄名字）

| 机制 | 上游出处 | 本工程实现 |
|------|----------|-----------|
| 内部时间轴 | CTM `forward` 的 iterations 循环 | tick 循环，每 tick 产出预测+certainty |
| 神经元级时序模型 | CTM `SuperLinear`（每神经元私有权重处理历史） | `NeuronLevelModel`，einsum per-neuron 权重 |
| 同步即表示 | CTM `compute_synchronisation`（random-pairing + 衰减递推） | `_synchronise` 同公式 |
| 自适应计算时长 | CTM `compute_certainty`（1-归一化熵） | certainty 阈值早停 |
| 上下文→LoRA | D2L `HyperLoRA` + Perceiver | `PerceiverBottleneck` + head 出 A/B |
| 分块超长上下文 | D2L `combine_lora`（chunk + mean） | `forward_chunked`/`aggregate_loras` |
| 无损注入/移除 | D2L `lora_forward` 前向补丁 + `internalize/reset` | `LoRAInjector` 前向补丁，reset 零误差 |

## 相对初版示意 Demo 的关键修复

1. **注入方式**：初版直接改 `weight.data` 再减法回滚（有累积误差、非上游做法）
   → 改为前向补丁，实测 reset 最大还原误差 `0.000e+00`。
2. **维度错配**：初版场景向量维度与超网络输入写死不匹配 → 统一编码器自适应。
3. **LoRA 槽位宽度**：修正为 `d_in+d_out`（对齐 D2L），否则 B 矩阵被截断。
4. **CTM 拼接宽度**：补齐 `d_input→d_model` 投影，对齐上游 LazyLinear 自适应。
5. 完整清单见 `docs/BUGFIX_REPORT.md`。

## v0.2.0 新增修复
- **场景编码器确定性**：v0.1 中 GPM 每次前向都新建随机初始化的场景编码器，导致同场景
  重复内化结果漂移、服务化后不可复现；v0.2 将其持久化为子模块（随 state_dict 保存/训练），
  并以确定性/批次独立/重复注入不叠加等契约测试锁死。
- 新增 HTTP 推理服务、性能基准+回归守卫、Docker/compose/Makefile、端到端冒烟。

## 验证结果（本沙箱实测，v2.3.1）

- **91 passed**（v2.2.1 的 77 项原样全绿 + v2.3 新增 14：PAVA/Spearman、校准单调/clip/
  退化/边界、多时域权重与 2.2.1 回退锚点、预测区间、校准不入 state_dict、存载往返与旧档
  兼容、`/calibrate`/`/checkpoints`/`/load` 服务流与 400/409）。
- **F2 多时域权重（诚实的口径敏感结论）**：口径A（无早停/3 种子）uniform 远期 tail 3/3
  低于 front（−60.2%）；口径B（早停/build 数据划分）front 反超 uniform。跨口径不一致，故
  默认保持 front、uniform/back opt-in，两个脚本与 JSON 留全部正反结果，不宣称稳健增益。
- **F1 校准 / F3 区间**：正式件（52,191 参数）独立测试集回归 ECE 0.335→0.075（降约 4.5×），
  但校准后置信-误差 Spearman 仍为正（原始同步置信排序方向是反的），保序只修正置信数值
  刻度、不凭空恢复逐样本排序，以 `ranking_informative` 如实披露；90% 区间实测覆盖 0.898、
  宽度随步单调。详见 `benchmarks/results/training_v2.3.1.json`。
- 覆盖率：calibration 99%、persistence 100%、training 96%、evaluation 95%；`--guard` 与
  端到端冒烟（含 4 个 v2.3 新接口）收口见 `docs/VERIFICATION_v2.3.1.md`。

## 验证结果（本沙箱实测，v2.2.1）

- **75 passed**（v2.1 的 64 项原样全绿 + v2.2 新增 11：SS 机制等价性/课程/horizon1
  无操作/可反传、早停、确定性、评估新指标、`/evaluate`/`/save` 与 409、HTTP 状态码）。
- **F1 A/B（诚实结论）**：激进 SS(0.6) 损害单步；温和 `ss0.3/s15/w25` 在 seed42 改善
  step3-4 约 17.6%，但 seed7、horizon6 对照方向相反，故**收益不稳健、保持 opt-in 默认
  关闭**，原始结果落 `benchmarks/results/ss_ablation_v2.2.0.json`。
- **F3 校准诊断**：最高置信档误差并非最低（2.2.1 件高/低档误差比 1.80、非单调），
  如实暴露 certainty 未校准。
- **发布件**：`predictor_v2.2.1.pt`（52,191 参数、228KB），单步 MSE 0.0446、胜未训练
  103.9×、胜朴素基线 4.04×、场景条件增益 14.10×、4 步累积率 2.49×，落盘/重载逐位一致。
- 核心模块覆盖率约 **95%**；`--guard` 通过（e2e 中位 8.35ms、峰值 RSS 285.5MB，未退化）；
  端到端冒烟 **15/15**。完整证据见 `docs/VERIFICATION_v2.2.1.md` 与
  `benchmarks/results/training_v2.2.1.json`。

## 验证结果（本沙箱实测，v2.1.0）

- **64 passed**（v2.0 的 55 项原样全绿 + v2.1 新增 9：参数化数据/一致性公式/rollout/
  场景条件消融/评估体系/带场景编码器存载/多步 reason/checkpoint 预加载），覆盖率 **95%**。
- **多步 + 场景条件**（seed42，52,191 参数，训练 56.8s）：单步 MSE 5.47→0.082；
  消融 conditioned 0.082 vs unconditioned 0.813，**增益 9.93×**，scene_gate 0→−0.29；
  4 步自由 rollout MSE 0.082→0.104→0.157→0.293（误差随步数累积，长时程行为可见）。
- **物理一致性**：一阶欧拉残差 overall 0.189，匀速段 0.122 最低（符合物理）。
- 服务冒烟 **11/11**（含多步 reason 与 `--checkpoint` 启动即训练态）；`--guard` 通过。
- 完整对照与"为何不用预测时刻速度积分"的纠错过程见 `docs/VERIFICATION_v2.1.0.md`。

## 验证结果（本沙箱实测，v2.0.0）

- **55 passed**（一代 36 项原样全绿 + 二代新增 19：学习收敛/梯度/条件化/持久化/服务训练）。
- **能学习**：默认训练 test MSE 6.05→0.084（降 72×、胜朴素基线 2.49×），certainty 0.005→0.833。
- **真耦合**：零门控初始与无条件逐元素一致、同场景一致、不同场景门控打开后不同。
- **可持久化**：存/载前向逐元素一致（atol=1e-7）。
- 覆盖率：dynamics/persistence 100%、training 96%、CTM 96%、GPM 99%、协同 94%、**总 90%**。
- 性能（CPU 中位数）：预测器前向 2.84ms、CTM-small 7.1ms、GPM 内化 2.7ms、
  端到端 reason 8.0ms；峰值 RSS 285MB；`--guard` 通过（未较一代退化）。
- 服务冒烟 10/10 通过（含 /train 学习闭环与训练后物理预测）；一代结果见 `VERIFICATION_v0.2.0.md`。

## 接入真实 LLM / 训练的路径

- **真实 LLM 内化**：把 `TinyBaseModel` 换成 HF 因果模型（layers 命名与 D2L 一致：
  `model.layers[*].mlp.{gate,up,down}_proj`），`GPMConfig.dims` 用
  `infer_dims_from_model` 自动探测；`PhysicsContextEncoder` 可替换为 D2L 的冻结
  LLM 编码器（`third_party/d2l/src/ctx_to_lora/modeling/ctx_encoder.py`）。
- **训练超网络/CTM**：本工程提供机制与前向，训练循环可复用上游
  `third_party/ctm/tasks/*/train.py` 与 `third_party/d2l/train.py`。
- **原版 D2L** 依赖 transformers/peft/einops 等重依赖，请在其目录按上游 README 安装。

## 限制说明

- v2.1 已在**可解析合成动力学**上证明多步可学习、且隐藏场景参数经联合训练真正起作用
  （消融 9.93×）；但合成数据不等于真实物理精度，接入实测/仿真数据训练才能获得真实预测力。
- rollout 误差随步数累积。v2.3 试验了 uniform/back 多时域权重：其相对默认 front 的优势
  对是否早停/数据划分口径敏感、不能跨口径复现，故保持 opt-in、默认 front 复现 2.2.1；
  v2.2 的 Scheduled Sampling 同样因收益随种子不稳保持 opt-in 默认关闭。运动学一致性为
  一阶近似诊断，强加速/振动段不把残差压到 0。
- v2.3 的保序校准把 CTM certainty 校准到与经验精度同尺度、独立集 ECE 下降；但当原始
  certainty 与误差无单调一致关系时会退化为常量映射（`ranking_informative=false`），
  **只修正置信数值、不凭空恢复排序能力**。split-conformal 区间依赖校准/测试同分布
  （交换性），不保证分布外覆盖；正式参数化分布/分位数回归留待后续。
- GPM 仍保持零反向传播；协同引擎 `reason` 的多步 rollout 走观测窗口外推（GPM 场景
  嵌入维度与预测器隐藏参数维度不同，不混用），端到端联合微调留待后续版本。
- 未下载 ImageNet/LLM 级预训练权重（沙箱轻量化考虑）；上游预训练权重见
  HuggingFace `SakanaAI/ctm-imagenet` 等。

## v3.4 ICM 上下文记忆（In-Context Memory）

v3.4 线引入 **ICM 上下文记忆**——任务从权重转移到上下文、零梯度、人做一遍机器当场执行。核心是"检索+聚合"而非"朴素拼接"：

- `udos/icm.py`：`DemonstrationEpisode`（输入→动作→结果因果块）、`DemonstrationMemory`（相似度检索 top-k）、`ICMAggregator`（零梯度残差空间聚合，`p_icm = p0 + λ·Σ softmax·r`）。
- **先复现再超越**：naive 朴素拼接 few-shot 反而退化（0.09→4.19）；ICM 检索聚合 k-shot 不退化且单调改善（0.045→0.014）。
- 配套：`icm_events.py`（事件切分三流对齐）、`icm_cross.py`（跨本体归一化）、`icm_budget.py`（预算/压缩）、`pce_format.DemonstrationPrompt`（HTTP 提示词包）。
- **服务端点**：`POST /icm/demo/register`、`POST /icm/predict`。
- **零梯度**：主权重不动，state_dict md5 推理前后逐位一致；默认 opt-in，不挂 ICM 时旧路径逐位一致。analogy, not reproduction。

## v3.5 SFM 空间基础模型（Spatial Foundation Model）

v3.5 线在 affordance / pce_format 之上体系化构建**纯解析空间推理**——零梯度、opt-in、合成低维 3D 代理（analogy, not reproduction，不复现真实 3D/点云）：

- `udos/spatial.py`：`SpatialObject`/`SpatialScene`/`SpatialTransform`（平移/旋转/缩放，解析逆可逆）+ 正交多视角投影一致性。
- `udos/scene_graph.py`：above/below/left/right/near/far/inside 解析关系边。
- `udos/occupancy.py`：3D 体素占据网格 + 有符号距离场。
- `udos/collision.py`：球-球/球-盒碰撞 + 最近邻。
- `udos/spatial_query.py`：射线-球相交、视线遮挡、范围/盒检索。
- **服务端点**：`POST /spatial/query`、`POST /spatial/collision`（400/409/404 语义）。
- **零外挂**：不改主 52191 参数；正式件 `predictor_v3.5.0.pt`（eval_mse=0.045556，与 v3.4.5 逐位一致）；backcompat 21 件。

## v3.6 PWM 物理世界模型（Physical World Model）

v3.6 线在 hybrid / counterfactual / future_multimodal / identification 之上构建**潜在空间前向世界模型**——纯前向、零梯度、外挂、合成低维潜在代理（analogy, not reproduction，不复现 V-JEPA/Cosmos/Genie/视频世界模型）：

- `udos/world_model.py`：`LatentWorldModel`（从 `obs_encoder` 末帧激活提取潜在状态 `z_t`；外挂小 MLP 前向转移 + 线性读出；`imagine()`/`imagine_rollout()` 多步想象，第 0 步逐位锚定主单步）；`imagine_uncertain()` 集成方差不确定性与高不确定回退。
- `udos/wm_events.py`：`ContactPredictor`（球-球几何代理接触/碰撞事件序列，非学习）。
- `udos/wm_conservation.py`：`ConservationChecker`（动量 `m*v` / 能量 `0.5*m*v²+势能` 一致性检验与违反量）。
- **服务端点**：`POST /wm/imagine`、`POST /wm/conservation`（400/409/404 语义）。
- **零外挂**：主 predictor 仍 **52191 参数**；世界模型外挂自身 2310 参数不入主 state_dict；正式件 `predictor_v3.6.0.pt`（eval_mse=0.045556，与 v3.5.0 同口径）；backcompat **22 件**。

## v3.8 全域调度 / 数字孪生 / 多体协同与收口（终点 v3.8.6）

- `udos/multi_agent.py`：`MultiAgentScene`（N 体状态容器）+ `AgentCoordinator`（非学习优先级让行/速度调节）。
- `udos/wm_scheduler.py`：`WMScheduler`（多体世界模型想象预算分配，复用 `LatentWorldModel`，H=1 回退真实）。
- `udos/closed_loop.py`：`ClosedLoopOrchestrator`（大脑→小脑→脊髓→WM反馈→感知更新一步闭环）。
- `udos/digital_twin.py`：`DigitalTwinScene`（合成多体+障碍参数化场景，seed 确定性，快照/回放）。
- **服务端点**：`POST /twin/scene`、`POST /twin/step`（400/409/404/500 语义）。
- **零外挂**：主 predictor 仍 **52191 参数**；四模块纯推理、零梯度、opt-in；正式件 `predictor_v3.8.6.pt`（eval_mse=0.045556，同口径）；backcompat **25 件**。**analogy, not reproduction**。

## v4.3 完全自进化（自进化飞轮 + 基础设施自优化缩微类比，终点 v4.3.9）

- `udos/self_evolution.py`：`ConfigSpec`/`ConfigEvaluator`/`ConfigSearcher`（系统配置自优化：batch 分片/缓存/集成权重/控制频率/码本规模，固定负载上"输出保真硬门 + 确定性成本代理"）；`SearchVerifySelectLoop`/`ConfigAB`/`SelfEvolutionOrchestrator`/`MultiGenerationRunner`/`GlobalStopCorrectCriterion`/`LongHorizonLoop`。
- **三维收口**：环境自造=4.1（课程）、数据自产=4.2（三元组）、基础设施自优化=4.3（配置搜索），串成一代自进化 orchestrator。
- **服务端点**：`POST /self-evolution/{search,ab,long-horizon}`（400/409/404 语义）。
- **零外挂**：主 predictor 仍 **52191 参数**；全线纯前向零梯度、opt-in；正式件 `predictor_v4.3.0.pt`（32 代）/`predictor_v4.3.9.pt`（33 代，eval_mse=0.045556）；backcompat **33 件**。**analogy, not reproduction**；多代曲线诚实记录真改进 vs 退化，不宣称飞轮必然提升。

## v4.4.1 多智能体协作&协同&协调线（零正式训练，外挂零梯度 opt-in）

四拓扑（star/chain/tool/mesh）+ 图1 决策树选型 + Transfer Bundle 五要素 + 治理三件套（Owner/Trace/Stop）。HTTP：`POST /collab/select|run|handoff`、`GET /collab/trace/{id}`。mesh/swarm 默认关（opt-in）。拓扑 A/B：`make collab-ab` → `benchmarks/results/collab_ab_v44.json`。详见 `docs/MULTI_AGENT_COLLAB_RESEARCH.md` 与 `docs/VERIFICATION_v4.4.1.md`。analogy, not reproduction——内存纯函数能力包装，非真实多机器人集群/A2A 线上协议。
