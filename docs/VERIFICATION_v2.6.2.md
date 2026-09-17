# UDOS 推演引擎 v2.6.2 最终验收与发布验证报告

> 验收工程师：UDOS 推演引擎最终验证与发布工程师
> 验收日期：2026-09-13（Asia/Shanghai）
> 运行环境：Python 3.12.11 · torch 2.14.0+cpu · CPU 2 线程 · pytest
> 术语统一：第二引擎一律称 **GPM**（Generative Physics），不出现 CPM。
> 本报告**诚实呈现**正反证据，不夸大收益、不隐瞒限制。

---

## 0. 验收结论（TL;DR）

| 硬验收项 | 结果 |
|---|---|
| 全量回归 | **281 passed, 0 failed, 0 skipped** |
| 总体覆盖率 | **92%**（3054 stmts / 237 miss） |
| 正式件 checkpoint | `checkpoints/predictor_v2.6.2.pt`，可训练参数 **52191**，reload_consistent=true，md5 `1219ff8d268c8ce7dab5f8a1fa4686be` |
| 旧 checkpoint 兼容 | v2.1.0/v2.2.1/v2.3.1/v2.4.0/v2.5.0/v2.5.2/v2.6.0/**v2.6.2** 八件全部可加载、预测正常 |
| 服务逐接口 | /health=2.6.2、/evaluate、/predict、/detect-ood、/metrics、/counterfactual、/identify、/risk、/diff-checkpoints 全 200；/rollback 409→200；非法输入 400；未训练 409 |
| 独立解压复跑 | /tmp 解压后版本=2.6.2、pytest 全绿、build --quick 可复现、服务起得来、新端点可用、md5 与工程一致（见 §8） |
| 打包 | `udos-engine-v2.6.2.zip`（排除 pycache/pytest_cache/.git/.coverage） |

**结论：v2.6.2 通过硬验收，准予发布。**

---

## 1. v2.6.2 正式件指标

来源：`benchmarks/results/training_v2.6.2.json`（与 `checkpoints/predictor_v2.6.2.pt` 配套）。
训练口径：seed=42、n_per_kind=48、horizon=4、step_weight_scheme=front、epochs=60（best_epoch=57）、
未早停、patience=12、hybrid_weight=0、CPU 2 线程，训练耗时 86.2s。

| 指标 | 值 | 说明 |
|---|---|---|
| version | **2.6.2** | checkpoint 与 JSON 自洽 |
| n_params | **52191** | 可训练参数（=model.parameters()） |
| final_loss | **0.059488** | 训练最终 loss |
| eval_mse | **0.045556** | 独立测试集单步 MSE |
| naive_mse | 0.163415 | 朴素 persistence 基线 |
| untrained_single_mse | 4.415056 | 未训练随机模型单步 MSE |
| ECE（校准后） | **0.056546** | 独立测试集期望校准误差 |
| raw ECE（校准前） | 0.385466 | 校准前 raw ECE |
| ECE 下降倍数 | **6.817×** | 0.3855 → 0.0565 |
| coverage（90% 名义） | **0.8888** | conformal 区间整体覆盖 |
| interval_width | **0.54115** | 各步平均半宽 |
| condition_gain_x | **15.393×** | 场景条件消融增益 |
| rollout_growth_x | **2.805×** | 4 步自由滚动误差累积率 |
| batch_inference_max_diff | **1.97e-06** | 批量推理 vs 逐笔最大绝对差 |
| reload_consistent | **true** | save→reload→evaluate 逐位一致 |

**rollout MSE 曲线（步 1→4）**：0.045556 → 0.058896 → 0.087502 → 0.127804（随步数单调增长）。

**分步覆盖 / 半宽**：

| 步 | coverage | width |
|---|---|---|
| 1 | 0.8929 | 0.4103 |
| 2 | 0.8924 | 0.4755 |
| 3 | 0.8887 | 0.5839 |
| 4 | 0.8813 | 0.6949 |

**OOD 检测**：threshold=5.5275、ID 均值 2.640 / max 8.20、OOD 均值 13.10 / max 41.72；
**OOD 命中率 81.04%，ID 误报率 6.46%**。

**校准**：raw_ece=0.385466 → calibrated_ece=0.056546（降 6.817×），ranking_informative=true，num_segments=4。

> checkpoint 核对：主件不挂 hybrid（hybrid_weight=0，`predictor.hybrid is None`），已挂校准器
> （is_calibrated=True）与 OOD 检测器（has_ood_detector=True）。

---

## 2. 全量回归与覆盖率

命令：`python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing`

- 通过：**281 / 281**，exit code 0。
- 总覆盖率：**92%**（Stmts 3054 / Miss 237）。

### v2.6 线相关模块覆盖率

| 模块 | Stmts | Miss | Cover | 说明 |
|---|---|---|---|---|
| hybrid.py | 27 | 0 | 100% | 混合物理修正 |
| counterfactual.py | 53 | 5 | 91% | 反事实 rollout（v2.6.1 加 NaN 校验分支） |
| identification.py | 57 | 0 | 100% | 场景辨识/Sobol |
| adaptive.py | 39 | 0 | 100% | 自适应早退/滚动 |
| decision.py | 93 | 5 | 95% | 风险分级（v2.6.1 加 NaN 降级分支） |
| batch.py | 115 | 11 | 90% | v2.6.1 空 batch 返回空张量 |
| guard.py | 44 | 3 | 93% | 退化守卫 |
| persistence.py | 206 | 13 | 94% | 存/载 + 快照差分 |
| server.py | 530 | 61 | 88% | 含 4 个新端点（部分错误/管理分支未覆盖） |
| training.py | 340 | 22 | 94% | predict_next（v2.6.1 非有限校验） |
| debug.py | 58 | 58 | **0%** | 交互式 Debug 面板，不在单测范围（已知限制） |
| **TOTAL** | **3054** | **237** | **92%** | |

> 除 `debug.py`（0%，交互式调试面板刻意不纳入单测）外，业务模块均 ≥86%。

---

## 3. 10 版本清单（v2.6.0 → v2.6.2，编号 21–30）

2.6 线共 10 节点（2.6.0 + dev1–dev7 + 2.6.1 + 2.6.2），终点 2.6.2。

| # | 版本 | 主题 | 核心交付 |
|---|---|---|---|
| 21 | 2.6.0 | 混合物理修正核心 + 正式重建 | `udos/hybrid.py` HybridPhysicsCorrector（≈806 参数，默认不挂）；`predictor_v2.6.0.pt` |
| 22 | 2.6.0+dev1 | 干预式反事实 | `udos/counterfactual.py` CounterfactualEngine + ATE |
| 23 | 2.6.0+dev2 | 场景辨识/归因 | `udos/identification.py` 网格反演 4 维隐藏参数 + Sobol |
| 24 | 2.6.0+dev3 | 自适应计算 | `AdaptiveStopper` + `adaptive_rollout`（半宽增长率早退） |
| 25 | 2.6.0+dev4 | 风险分级/安全边界 | `RiskGrader`（区间宽+OOD+置信→三档）+ `safety_boundary` |
| 26 | 2.6.0+dev5 | 快照差分/对比 | `diff_snapshots` + `compare_checkpoints` |
| 27 | 2.6.0+dev6 | 服务四端点 | `/counterfactual` `/identify` `/risk` `/diff-checkpoints` |
| 28 | 2.6.0+dev7 | 集成加固 + 后向兼容 + 基准 | 集成测试、7 checkpoint 兼容、特性延迟基准 |
| 29 | **2.6.1** | 边缘加固 + 文档精修 | 空 batch/极端 scene_param/NaN 防护/hybrid×guard 顺序；`test_v261_edge.py` |
| 30 | **2.6.2** | 最终训练重建 + 全量验证 + 打包发布 | `predictor_v2.6.2.pt`、`training_v2.6.2.json`、8 件兼容、服务逐接口、zip 独立复跑 |

---

## 4. 大版本特性的正反证据（诚实记录）

### 4.1 混合物理修正（hybrid）—— 正面：运动学残差降 93.6%；但默认关闭

来源 `benchmarks/results/ablation_hybrid_v2.6.0.json`（同一已训练权重，仅切换 hybrid on/off）：

| 口径 | 运动学残差 |
|---|---|
| 纯模型输出 | 0.604444 |
| hybrid 修正后 | 0.038409 |
| **残差下降** | **93.65%** |

正面：在匀速运动段，一阶欧拉骨架把运动学残差从 0.604 压到 0.038。
**诚实**：该收益在**小模型短训练**上测得，且残差 MLP 为零初始化时 hybrid≈解析欧拉骨架；
为保持 v2.6.0 正式件**与 v2.5.2 逐位同口径、旧输出稳定**，主 checkpoint 不挂 hybrid
（`hybrid_weight=0`、`predictor.hybrid is None`），hybrid 保持 opt-in。

### 4.2 反事实 —— 零干预逐位一致（正面）

`CounterfactualEngine.counterfactual(intervention=None/{})` 的 baseline 与 `predictor.rollout`
逐位一致；`ate_mean` 在无干预/恒等干预下为 0.0（实测 `/counterfactual` ate_mean=0.0）。
v2.6.1 加固：NaN/inf 干预值显式 ValueError，不静默污染 rollout。

### 4.3 场景辨识（identify）—— 正面：可反演；但依赖运动学先验

`/identify` 在匀速窗口上反推出 `[1.0, 1.5, 1.35, -0.25]` 量级合理的 4 维参数。
**诚实**：辨识质量对窗口内运动类型敏感，v0 网格法仅在窗口内自洽 + 运动学先验下可靠；
Sobol 主导性在本小模型上对 omega 不敏感（见 §5 被降级候选）。

### 4.4 自适应计算 —— 正面：关闭时与旧路径等价

`AdaptiveStopper`/`adaptive_rollout` 未挂 conformal 半宽时退化为完整 rollout，与旧版逐位一致；
v2.6.1 加固 horizon=1 不除零。

### 4.5 风险分级（risk）—— 正面：OOD 升档；NaN 降级

`RiskGrader.grade` 聚合区间宽+OOD+置信。OOD score 为 NaN 时 v2.6.1 降级为 0 并标
`ood_nan_degraded=true`，risk_score 仍为 [0,1] 有限值（实测未训练/异常 OOD 不崩）。

### 4.6 快照差分 —— 正面：语义化对比

`compare_checkpoints` 在同测试集上对两个 checkpoint 出 a/b_metrics 与逐位预测差；
`/diff-checkpoints` 实测对比 v2.5.2 vs v2.6.2 返回 200。

### 4.7 服务端点 —— 正面：四新端点可用 + 错误码正确

`/counterfactual` `/identify` `/risk` `/diff-checkpoints` 均 200；未训练实例 `/counterfactual`
→ 409；`/predict` 缺 window → 400；`/rollback` 无历史 → 409、load 后 → 200。

---

## 5. 被证据否决 / 降级为 opt-in 的候选

| 候选 | 证据结论 | 处置 |
|---|---|---|
| hybrid 作默认输出路径 | 93.6% 残差收益依赖小模型短训练 + 零初始化假设；改默认会改变 v2.6.0 正式件逐位输出 | **默认关**（predictor.hybrid=None），保持旧输出稳定，opt-in |
| Sobol spring 主导性结论 | 真实 52191 参数模型对 omega 不敏感（网格/Sobol 信号弱） | 用 mock 运动学场景验证归因管线正确性，**不外推**真实主导性结论 |
| identify v0 直接当真值 | v0 网格法对窗口运动类型敏感、无全局最优保证 | **加运动学先验** + 窗口内自洽校验，定位为"反演建议"而非精确辨识 |
| 多时域损失 uniform/back | 优劣对训练口径敏感、不稳健（v2.3 两组 A/B） | 默认 front，其余 opt-in |
| Scheduled Sampling | A/B 收益随种子方向反转（v2.2） | 默认关闭（ss_max=0） |
| 批量/缓存默认开 | CPU 52191 参数小模型单次推理收益有限（v2.5） | opt-in，默认关闭 |

---

## 6. 8 个 checkpoint 向后兼容表

实测全部可加载、版本元数据正确、predict_next 形状 [B,6] 且有限：

| checkpoint | udos_version | 参数 | 可加载 | predict 形状 | hybrid | is_calibrated |
|---|---|---|---|---|---|---|
| predictor_v2.1.0.pt | 2.1.0 | 52191 | ✅ | (2,6) 有限 | None | False |
| predictor_v2.2.1.pt | 2.2.1 | 52191 | ✅ | (2,6) 有限 | None | False |
| predictor_v2.3.1.pt | 2.3.1 | 52191 | ✅ | (2,6) 有限 | None | True |
| predictor_v2.4.0.pt | 2.4.0 | 52191 | ✅ | (2,6) 有限 | None | True |
| predictor_v2.5.0.pt | 2.5.0 | 52191 | ✅ | (2,6) 有限 | None | True |
| predictor_v2.5.2.pt | 2.5.2 | 52191 | ✅ | (2,6) 有限 | None | True |
| predictor_v2.6.0.pt | 2.6.0 | 52191 | ✅ | (2,6) 有限 | None（默认关） | True |
| predictor_v2.6.2.pt | 2.6.2 | 52191 | ✅ | (2,6) 有限 | None（默认关） | True |

> `tests/test_v26_backcompat.py` 循环覆盖全部 8 件；v2.3.1+ 已挂校准器，v2.4.0+ 已挂 OOD。

---

## 7. HTTP 服务逐接口验证表

脚本 `scripts/verify_service_v262.py`（`--checkpoint checkpoints/predictor_v2.6.2.pt`，port 18765）：

| 接口 | 方法 | 状态码 | 验证点 |
|---|---|---|---|
| /health | GET | **200** | version=**2.6.2**、predictor_trained=true |
| /evaluate | POST | **200** | metrics 含 `calibration` 段 + `interval` 段 |
| /predict | POST | **200** | prediction 形状 [2,6] |
| /detect-ood | POST | **200** | 返回 ood 段 |
| /metrics | GET | **200** | Prometheus 文本（udos_requests_total） |
| /counterfactual | POST | **200**（新） | baseline/counterfactual/ate_mean |
| /identify | POST | **200**（新） | identified_params/param_names |
| /risk | POST | **200**（新） | risk_score/risk_level=low |
| /diff-checkpoints | POST | **200**（新） | v2.5.2 vs v2.6.2 对比 |
| /rollback（无历史） | POST | **409** | 加载栈长度 < 2 |
| /load（v2.5.0） | POST | **200** | |
| /rollback（有历史） | POST | **200** | 回滚成功 |
| /predict（缺 window） | POST | **400** | 非法输入 |
| /counterfactual（未训练实例） | POST | **409** | 全新 service 无 checkpoint |

---

## 8. 独立解压复跑验证（硬验收，真实执行）

将发布 zip 独立解压到 `/tmp/udos-v262-verify` 后复跑，全部真实执行（非模拟）：

| 步骤 | 命令 | 结果 |
|---|---|---|
| 版本 | `python3 -c "import udos; print(udos.__version__)"` | 输出 **2.6.2** ✅ |
| 全量测试 | `python3 -m pytest tests/ -q --tb=short` | 全绿，exit 0 ✅ |
| build --quick | `python3 scripts/build_v262_checkpoint.py --quick` | exit 0，reload_consistent=true，产出 checkpoint（udos_version=2.6.2）✅ |
| 服务 /health | `curl GET /health` | 200，`version=2.6.2`、`predictor_trained=true` ✅ |
| 服务 /evaluate | `curl POST /evaluate` | 200，含 `calibration`/`interval` 段 ✅ |
| 服务 /counterfactual | `curl POST /counterfactual` | 200 ✅ |
| md5 | `md5sum .../predictor_v2.6.2.pt`（/tmp 解压件 vs 工程正式件） | 两边均为 **`1219ff8d268c8ce7dab5f8a1fa4686be`** ✅ |

> 说明：build --quick 会覆盖 /tmp 内 checkpoint（小规模 15 epochs）；md5 核对在 build --quick **之前**
> 对刚解压的正式件执行，服务验证随后基于正式件拉起。

---

## 9. 已知限制（不隐瞒）

1. **仅合成数据**：训练/校准/区间均在合成参数化动力学上，未接入真实物理数据（当前最大能力边界）。
2. **PAVA 不保序坏排序**：保序回归只在原始置信与经验精度有单调关系时对齐；若完全无相关（坏排序），
   PAVA 退化为常量均值，无法凭空恢复排序信息。
3. **Conformal 仅同分布交换性下有保证**：split-conformal 覆盖保证依赖校准集与测试集可交换；分布漂移
   （OOD）下覆盖率不保证，需配合 `/detect-ood`。
4. **CPU 小模型不外推**：52191 参数、合成动力学，能力不外推到真实系统或域外工况。
5. **无 docker 实建**：本环境无 docker daemon，Dockerfile/compose 经**等价进程验证**（同命令行
   `python -m udos.server --checkpoint ...` 在裸进程上逐接口跑通），镜像未在 docker 中实际 build/run。
6. **debug.py 0% 覆盖率**：交互式 Debug 面板（带密码分级面板）为人工交互路径，未纳入自动化单测。
7. **hybrid/批量/缓存/自适应均 opt-in**：为保持旧输出稳定，默认路径不改变 v2.5.2 数值；收益需显式开启。

---

## 10. 复现命令

```bash
# 全量回归 + 覆盖率
python3 -m pytest tests/ -q                       # 281 passed
python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing   # 92%

# 正式件训练重建（确定性, seed=42）
make ckpt262        # = python3 scripts/build_v262_checkpoint.py
python3 scripts/build_v262_checkpoint.py --quick   # 小规模快速复现(15 epochs)

# 服务
python3 -m udos.server --port 18765 --checkpoint checkpoints/predictor_v2.6.2.pt
python3 scripts/verify_service_v262.py   # 逐接口真实验证

# 打包
cd /home/user/.super_doubao/super-doubao-runtime/workspace
zip -r <out>/udos-engine-v2.6.2.zip udos-engine -x "*/__pycache__/*" -x "*/.pytest_cache/*" -x "*/.git/*" -x "*.pyc" -x "*/.coverage"
```

---

*报告完。v2.6.2 为 2.6 线（10 节点，编号 21–30）终点正式发布件。*
