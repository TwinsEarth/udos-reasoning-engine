# UDOS 推演引擎 v2.5.2 最终验收与发布验证报告

> 验收工程师：UDOS 推演引擎最终验证与发布工程师
> 验收日期：2026-09-13（Asia/Shanghai）
> 运行环境：Python 3.12.11 · torch 2.14.0+cpu · CPU 2 线程 · pytest
> 术语统一：第二引擎一律称 **GPM**（Generative Physics），不出现 CPM。
> 本报告**诚实呈现**正反证据，不夸大收益、不隐瞒限制。

---

## 0. 验收结论（TL;DR）

| 硬验收项 | 结果 |
|---|---|
| 全量回归 | **218 passed, 0 failed, 0 skipped, 0 xfailed**（60.03s） |
| 总体覆盖率 | **92%**（2590 stmts / 195 miss） |
| 正式件 checkpoint | `checkpoints/predictor_v2.5.2.pt`，可训练参数 **52191**（state_dict 含 48 个持久化 buffer 共 52239） |
| 旧 checkpoint 兼容 | v2.1.0 / v2.2.1 / v2.3.1 / v2.4.0 / v2.5.0 五件全部可加载、预测正常 |
| 服务逐接口 | /health=2.5.2、/evaluate、/detect-ood、/metrics、/rollback(409→200)、/predict(guard)、/checkpoints 全通过 |
| 独立解压复跑 | /tmp 解压后版本=2.5.2、pytest 218 全绿、build --quick 可复现、服务起得来、md5 与工程一致（见 §8） |
| 打包 | `udos-engine-v2.5.2.zip`（排除 pycache/pytest_cache/.git/.coverage） |

**结论：v2.5.2 通过硬验收，准予发布。**

---

## 1. 全量回归与覆盖率

命令：`python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing`

- 通过：**218 / 218**，exit code 0，耗时 60.03s。
- 无 `pytest.mark.skip` / `xfail` / `skipif` 使用（grep 全 tests/ 为空），不存在不当跳过。
- 总覆盖率：**92%**（Stmts 2590 / Miss 195）。

### 各模块覆盖率

| 模块 | Stmts | Miss | Cover | 说明 |
|---|---|---|---|---|
| udos/__init__.py | 16 | 0 | 100% | |
| adapters/sakana_ctm_adapter.py | 49 | 7 | 86% | 上游适配分支 |
| batch.py | 115 | 11 | 90% | v2.5.0 批量推理 |
| cache.py | 83 | 8 | 90% | v2.5.0 LRU 缓存 |
| calibration.py | 246 | 4 | 98% | PAVA + Temperature + 多 alpha conformal |
| ctm_engine.py | 152 | 6 | 96% | CTM 内核 |
| **debug.py** | 58 | 58 | **0%** | Debug 面板，交互式 UI 路径，**不在单测覆盖范围**（已知限制，见 §7） |
| dynamics.py | 148 | 1 | 99% | |
| ensemble.py | 76 | 2 | 97% | v2.4.1 深度集成 |
| evaluation.py | 83 | 4 | 95% | |
| gpm_engine.py | 251 | 3 | 99% | GPM 引擎 |
| guard.py | 44 | 3 | 93% | v2.4.5 退化守卫 |
| ood.py | 150 | 11 | 93% | v2.4.0 OOD + v2.4.14 流式漂移 |
| pce_format.py | 108 | 6 | 94% | PCE 物理 Token |
| persistence.py | 123 | 9 | 93% | 存/载 + 集成档 + 快照 |
| reasoning.py | 123 | 7 | 94% | |
| server.py | 457 | 45 | 90% | 服务端路由（部分错误分支/管理路径） |
| training.py | 306 | 10 | 97% | |
| **TOTAL** | **2590** | **195** | **92%** | |

> 注：除 `debug.py`（0%，交互式调试面板，刻意不纳入单测）外，业务模块均 ≥86%，新模块 batch/cache/ood/ensemble/guard 均 ≥90%。

---

## 2. v2.5.2 正式件指标

来源：`benchmarks/results/training_v2.5.2.json`（与 `checkpoints/predictor_v2.5.2.pt` 配套）。
训练口径：seed=42、n_per_kind=48、horizon=4、step_weight_scheme=front、epochs=60（best_epoch=57）、未早停、CPU 2 线程。

| 指标 | 值 | 说明 |
|---|---|---|
| final_loss | **0.059488** | 训练最终 loss |
| eval_mse | **0.045556** | 独立测试集单步 MSE |
| naive_mse | 0.163415 | 朴素 persistence 基线（X_t 当 X_{t+1}） |
| untrained_single_mse | 4.415056 | 未训练随机模型单步 MSE |
| ECE（校准后） | **0.056546** | 独立测试集期望校准误差 |
| raw ECE（校准前） | 0.385466 | 校准前 raw ECE |
| ECE 下降倍数 | 6.817× | 0.3855 → 0.0565 |
| coverage（90% 名义） | **0.8888** | conformal 区间整体覆盖 |
| interval_width | **0.54115** | 各步平均半宽 |
| params_count | **52191** | 可训练参数（=model.parameters()） |
| condition_gain_x | 15.393× | 场景条件消融增益 |
| rollout_growth_x | 2.805× | 4 步自由滚动误差累积率 |
| batch_inference_max_diff | **1.97e-06** | 批量推理 vs 逐笔最大绝对差 |

**rollout MSE 曲线（步 1→4）**：0.045556 → 0.058896 → 0.087502 → 0.127804（随步数单调增长，符合长时程误差累积预期）。

**分步覆盖 / 半宽**：

| 步 | coverage | width |
|---|---|---|
| 1 | 0.8929 | 0.4103 |
| 2 | 0.8924 | 0.4755 |
| 3 | 0.8887 | 0.5839 |
| 4 | 0.8813 | 0.6949 |

半宽随步**严格非减**（`width_non_decreasing=true`），长时程不确定性合理增长。

**OOD 检测**：threshold=5.5275，ID 均值 2.640 / max 8.20，OOD 均值 13.10 / max 41.72；**OOD 命中率 81.04%，ID 误报率 6.46%**。

> checkpoint 核对：`torch.load` 解析 `state_dict` 张量总 52239，其中可训练参数 52191，另含 48 个持久化 buffer（非可学习，如归一化 running 统计等）——`params_count=52191` 指可训练参数口径，与 `model.parameters()` 一致。

---

## 3. 20 版本清单（2.4.0 → 2.5.2，编号自洽）

2.4 线 17 节点（2.4.0–2.4.16）+ 2.5 线 3 节点（2.5.0–2.5.2）= **20 节点**，终点 2.5.2。
「新增测试」来自各版本 CHANGELOG 与 tests/ 目录；累计终点为 218。

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|---|---|---|---|
| 1 | 2.4.0 | OOD/漂移检测核心 + 正式重建 | `udos/ood.py` 马氏距离+KS；`predictor_v2.4.0.pt` | 11 |
| 2 | 2.4.1 | 深度集成不确定性(opt-in) | `ensemble.py` DeepEnsemble | 6 |
| 3 | 2.4.2 | 温度缩放校准方法 | `TemperatureScaling`（与 PAVA 并存） | 7 |
| 4 | 2.4.3 | 噪声鲁棒训练(opt-in) | `noise_augment` / `TrainConfig.noise_sigma` | 6 |
| 5 | 2.4.4 | 多水平 conformal 区间 | alpha∈{0.2,0.1,0.05}（80/90/95） | 6 |
| 6 | 2.4.5 | 退化守卫 | `guard.py` PredictionGuard（NaN/越界） | 6 |
| 7 | 2.4.6 | 校准方法 A/B | `ablation_calibration.py`（PAVA/Temp/None） | 3 |
| 8 | 2.4.7 | 集成+conformal 融合 | `DeepEnsemble.predict_interval` | 4 |
| 9 | 2.4.8 | OOD 服务接口 | `POST /detect-ood` | 5 |
| 10 | 2.4.9 | 噪声鲁棒 A/B | `ablation_noise_robustness.py` | 4 |
| 11 | 2.4.10 | 逐步逐维置信 | `per_step_confidence` + confidence_breakdown | 3 |
| 12 | 2.4.11 | 多水平覆盖基准 | `ablation_conformal_levels.py` | 3 |
| 13 | 2.4.12 | 守卫接入服务 | `POST /predict(guard)`、/evaluate guard 段 | 6 |
| 14 | 2.4.13 | 集成持久化 | `save_ensemble/load_ensemble` | 3 |
| 15 | 2.4.14 | 流式漂移累积 | `StreamingDriftDetector`（窗口 Welford） | 5 |
| 16 | 2.4.15 | 全量回归+覆盖率加固 | 无新功能，回归确认（171 passed / 93%） | 0 |
| 17 | 2.4.16 | 2.4 线文档定稿 | ARCHITECTURE/DEPLOYMENT/README/VERSION_PLAN 对齐 | 0 |
| 18 | 2.5.0 | 批量推理 + 推理缓存 + 正式重建 | `batch.py` BatchPredictor、`cache.py` InferenceCache、`predict_batch`、`predictor_v2.5.0.pt` | 22 |
| 19 | 2.5.1 | 服务指标 + 无状态快照 + 回滚 | `GET /metrics`、`export/import_snapshot`、`POST /rollback` | 19 |
| 20 | **2.5.2** | 最终训练重建 + 全量验证 + 打包发布 | `predictor_v2.5.2.pt`、`training_v2.5.2.json`、backcompat、服务逐接口验证、zip 独立复跑 | 6 |

> 编号与 `docs/VERSION_PLAN_2.4.md`（节点 1–17）、`docs/VERSION_PLAN_2.5.md`（节点 18–20）严格对齐。

---

## 4. 大版本特性的正反证据（诚实记录）

### 4.1 v2.4 线：校准 / 噪声 / 区间 / OOD

**(a) 温度缩放 A/B —— 负面结论（保留 opt-in，不作默认）**
来源 `benchmarks/results/calibration_ablation_v2.4.6.json`（3 种子 quick）：

| 方法 | 平均 ECE | 比 None 更低的种子 | 覆盖率 |
|---|---|---|---|
| **PAVA** | **0.0517** | 3/3 | 0.9058 |
| Temperature | 0.4968 | **0/3** | 0.9058 |
| None（不校准） | 0.4968 | — | 0.9058 |

结论：温度缩放（单调 logit 缩放）在本任务上 **ECE 与不校准持平（≈0.497），网格自动选 T≈1**；PAVA 把 ECE 压到 ≈0.052。故默认 method 仍为 **pava**，Temperature 仅与 PAVA 并存供对比，**不启用为默认**。

**(b) 噪声鲁棒训练 —— 偏差-方差权衡（默认 sigma=0）**
来源 `benchmarks/results/noise_robustness_v2.4.9.json`（train_sigma × test_sigma，3 种子）：

| train_sigma | 干净集 MSE | 加噪集 MSE | 鲁棒缺口(noisy-clean) | 缺口比 sigma=0 更小 |
|---|---|---|---|---|
| 0.0 | 0.4023 | 0.4067 | +0.0044 | — |
| 0.05 | 0.6169 | 0.6188 | +0.0019 | 3/3 |
| 0.1 | 0.6793 | 0.6761 | −0.0033 | 3/3 |

正面：训练噪声单调**缩小**对测试噪声的鲁棒缺口（3/3 种子）。
**负面（诚实）**：干净集 MSE 随 train_sigma 显著上升（0.402→0.617→0.679）——这是典型**偏差-方差权衡**，噪声增强以干净精度换噪声稳健性，**非免费午餐**。故默认 `noise_sigma=0.0` 不变，噪声增强保持 opt-in。

**(c) 多水平区间 —— 保守过覆盖（小校准集 split-conformal）**
来源 `benchmarks/results/conformal_levels_v2.4.11.json`：

| 名义水平 | 实测覆盖 | 覆盖−名义 | 平均半宽 |
|---|---|---|---|
| 80% | 0.8466 | +0.0466 | 1.407 |
| 90% | 0.9256 | +0.0256 | 2.063 |
| 95% | 0.9655 | +0.0155 | 2.657 |

三水平均**保守过覆盖**（不系统性 undercover），覆盖率随名义水平单调不下降，半宽严格增宽（1.41 < 2.06 < 2.66）。

**(d) OOD 检测 —— 命中率 81% / 误报 6.5%**
正式件 `training_v2.5.2.json`：OOD（`X*5.0` 分布偏移）命中率 **81.04%**，ID 样本误报率 **6.46%**，阈值 5.53（训练马氏距离 95% 上分位）。

### 4.2 v2.5 线：批量推理 / 缓存 / CPU 小模型诚实说明

**(a) 批量推理与逐笔逐位一致**
正式件自检 `batch_inference_max_diff = 1.97e-06`（< 1e-5 门限）；`test_v25_batch.py` 锁死「批量==逐笔」「变长分组」「大 batch 分片」「空 batch 守卫」「形状校验」。

**(b) 缓存 opt-in**
`InferenceCache` 默认 `enabled=False`；key = 输入张量 SHA-256 + 模型参数哈希；开启后命中与未命中 `torch.equal` 逐位相同；权重变化自动失效；`test_v25_cache.py` 锁死 LRU 淘汰 / 关闭透传 / 权重失效。

**(c) 诚实说明：CPU 小模型上批量/缓存收益有限**
模型仅 **52191 参数**，CPU 2 线程下单次推理本身极快，**批量并行与 LRU 缓存在单次/小请求场景收益不明显**；其主要价值在**大批量**与**重复请求**场景。本版不夸大吞吐收益，两项均为 **opt-in、默认关闭**，不改变 `predict_next`/`rollout`/`predict_interval` 默认输出。

---

## 5. 被证据否决 / 回退的候选清单

| 候选 | 证据结论 | 处置 |
|---|---|---|
| Temperature scaling 作默认校准 | ECE 0.497 ≈ None，0/3 种子更优（v2.4.6） | 保留 opt-in，默认仍 PAVA |
| 噪声增强（sigma>0）作默认 | 干净 MSE 显著上升（偏差-方差权衡，v2.4.9） | 默认 sigma=0，opt-in |
| 多时域损失 uniform/back 作默认 | 优劣对训练口径敏感、不稳健（v2.3 两组 A/B） | 默认 front，其余 opt-in |
| Scheduled Sampling 作默认 | A/B 收益随种子方向反转（v2.2） | 默认关闭（ss_max=0） |
| 批量推理/缓存作默认 | CPU 52191 参数小模型单次推理收益有限（v2.5.0） | opt-in，默认关闭 |
| 集成 N>1 收益外推 | N=1 退化单模型；N>1 区间随分歧增宽，认知方差非免费精度提升 | 保留 opt-in，不宣称精度增益 |

---

## 6. 旧 checkpoint 兼容性

`torch.load` 实测全部可加载、版本元数据正确：

| checkpoint | 内部 udos_version | 可训练参数 | 可加载 |
|---|---|---|---|
| predictor_v2.1.0.pt | 2.1.0 | 52191 | ✅ |
| predictor_v2.2.1.pt | 2.2.1 | 52191 | ✅ |
| predictor_v2.3.1.pt | 2.3.1 | 52191 | ✅ |
| predictor_v2.4.0.pt | 2.4.0 | 52191 | ✅ |
| predictor_v2.5.0.pt | 2.5.0 | 52191 | ✅ |
| predictor_v2.5.2.pt | 2.5.2 | 52191 | ✅ |

`tests/test_v25_backcompat.py`（6 用例）断言：v2.3.1+ 已挂校准器、v2.4.0+ 已挂 OOD 检测器，预测正常。旧单模型加载路径逐位不变；集成档走独立 `load_ensemble`。

---

## 7. 服务验证结果（真实 HTTP 逐接口）

脚本 `scripts/verify_service_v252.py`：`python -m udos.server --checkpoint checkpoints/predictor_v2.5.2.pt`，逐接口断言全部通过：

| 接口 | 方法 | 状态码 | 验证点 |
|---|---|---|---|
| /health | GET | **200** | version=**2.5.2** |
| /evaluate | POST | **200** | 含 `calibration` 段 + `interval` 段 + `service_metrics` 段 |
| /detect-ood | POST | **200** | 返回 ood 段 / ood_rate / threshold |
| /metrics | GET | **200** | Prometheus 文本（含 udos_requests_total / udos_latency_seconds，约 1016 字节） |
| /rollback（无历史） | POST | **409** | 加载栈长度 < 2 |
| /load（v2.5.0） | POST | **200** | |
| /rollback（有历史） | POST | **200** | 回滚至 predictor_v2.5.2.pt |
| /predict（guard=True） | POST | **200** | guard.enabled=true，健康模型 n_fallbacks=0/n_clips=0 |
| /checkpoints | GET | **200** | 返回 6 个可用 checkpoint |

---

## 8. 独立解压复跑（硬验收，真实执行）

将发布 zip 独立解压到 `/tmp/udos-verify-v252` 后复跑，全部真实执行（非模拟）：

| 步骤 | 命令 | 结果 |
|---|---|---|
| 版本 | `python3 -c "import udos; print(udos.__version__)"` | 输出 **2.5.2** ✅ |
| 全量测试 | `python3 -m pytest tests/ -q` | **218 passed in 58.79s**，exit 0 ✅ |
| build --quick | `python3 scripts/build_v252_checkpoint.py --quick` | exit 0，`reload_consistent=true`，产出 checkpoint（udos_version=2.5.2）✅ |
| 服务 /health | `curl GET /health` | 200，`version=2.5.2`、`predictor_trained=true` ✅ |
| 服务 /evaluate | `curl POST /evaluate` | 200，响应 `metrics` 含 `calibration` 段与 `interval` 段 ✅ |
| 服务 /metrics、/detect-ood | `curl GET /metrics`、`curl POST /detect-ood` | 均 **200** ✅ |
| md5 | `md5sum .../predictor_v2.5.2.pt`（/tmp 解压件 vs 工程正式件） | 两边均为 **`003f2b6e46a44c6eec5165d467ce9150`**，完全一致 ✅ |

> 说明：build --quick 会覆盖 /tmp 内 checkpoint（小规模 15 epochs），故 md5 核对在 build --quick **之前**对刚解压的正式件执行；服务验证在 md5 核对后，将工程正式件重新拷入 /tmp 再拉起，确保验证的是对外发布的正式件。

---

## 9. 复现命令

```bash
# 全量回归 + 覆盖率
make test          # = python3 -m pytest tests/ -q  → 218 passed
make cov           # = ... --cov=udos --cov-report=term-missing → 92%

# 正式件训练重建（确定性, seed=42）
make ckpt252       # = python3 scripts/build_v252_checkpoint.py
python3 scripts/build_v252_checkpoint.py --quick   # 小规模快速复现(15 epochs)

# A/B 基准复跑
make calib-ablation       # PAVA vs Temperature vs None
make noise-ablation       # train_sigma × test_sigma
make conformal-levels     # 80/90/95 覆盖基准

# 服务
make serve CHECKPOINT=checkpoints/predictor_v2.5.2.pt
python3 scripts/verify_service_v252.py   # 逐接口真实验证

# 打包
make docker-build   # tag udos-reasoning-engine:2.5.2
```

---

## 10. 已知限制（不隐瞒）

1. **仅合成数据**：训练/校准/区间均在合成参数化动力学上，未接入真实物理数据（当前最大能力边界）。
2. **校准不修坏排序**：PAVA 是保序回归——若原始置信度与经验精度**完全无相关性（坏排序）**，PAVA 退化为输出常量均值，无法凭空恢复排序信息；它只在「排序本身有信息」前提下把置信分对齐到经验精度。
3. **Conformal 仅在交换性/同分布下有覆盖保证**：split-conformal 的覆盖保证依赖校准集与测试集**可交换（近似同分布）**；分布漂移（如 OOD）下覆盖率不保证，故需配合 `/detect-ood`。
4. **CPU 小模型不外推**：52191 参数、合成动力学，能力不外推到真实系统或域外工况。
5. **批量/缓存收益在小模型有限**：CPU 小模型单次推理快，批量并行与 LRU 缓存主要价值在大批量/重复请求，不夸大单次吞吐。
6. **debug.py 0% 覆盖率**：交互式 Debug 面板（带密码分级面板）为人工交互路径，未纳入自动化单测。
7. **服务指标为 opt-in 观测**：`/metrics` 纯标准库实现、不引入 prometheus_client；快照 `/export-snapshot` 不含权重（权重仍走 .pt）。

---

## 11. 本次验收发现的问题与处置

| # | 发现 | 处置 |
|---|---|---|
| 1 | 旧 v2.5.2 件内部 `udos_version` / 训练 JSON `version` 字段为 **2.5.1**（构建早于最终版本号 bump），与文件名/发布号 2.5.2 不一致 | 以当前 `__version__=2.5.2` **确定性重建正式件**，使 checkpoint 与 JSON 元数据自洽为 2.5.2（训练确定性：seed=42、固定数据种子、CPU 2 线程，指标与 v2.5.0 一致） |
| 2 | `Makefile` 缺 `ckpt252` target（仅有 ckpt25=v2.5.0） | 新增 `ckpt252` → `scripts/build_v252_checkpoint.py`，并补入 .PHONY |
| 3 | `docs/ARCHITECTURE.md` 未含 v2.5 模块（batch/cache） | 补充「v2.5 效率与服务化」模块表 |
| 4 | `docs/DEPLOYMENT.md` 未含 v2.5.1 四接口（/metrics、/rollback、/export-snapshot、/import-snapshot），标题仍写 v0.2.0 | 补接口契约并升标题版本 |
| 5 | `README.md` 标题/特性段停在 v2.4.16 | 升标题至 v2.5.2 并补 v2.5 特性段 |
| 6 | `docs/ROADMAP.md` 已发布线停在 v2.3.1 | 补 v2.4 / v2.5 线并标注 v2.5.2 完成 |
| 7 | `docs/VERSION_PLAN_2.5.md` 验收清单复选框未勾 | 勾定为完成 |

---

*报告完。v2.5.2 为 2.4/2.5 线（20 节点）终点正式发布件。*
