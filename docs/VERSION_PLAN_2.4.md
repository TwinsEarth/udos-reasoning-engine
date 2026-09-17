# UDOS 引擎 v2.4 版本规划 —— 鲁棒性与不确定性主线

> 起点：v2.3.1（92 passed，calibration99/persistence100/training96/evaluation95）
> 终点：v2.4.16（2.4 线收尾，全量回归绿）

## ✅ 完成状态总览（v2.4.16 收口）

**17/17 节点全部完成，总测试 171 passed，`--cov` 总计 93%。**

| # | 版本 | 状态 | 证据/落点 |
|---|------|------|-----------|
| 1 | 2.4.0 | ✅ | `checkpoints/predictor_v2.4.0.pt`、`training_v2.4.0.json`、test_v24_ood |
| 2 | 2.4.1 | ✅ | `ensemble.py` DeepEnsemble、test_v24_ensemble |
| 3 | 2.4.2 | ✅ | `TemperatureScaling`、test_v24_temp_scale（实测 T≈1，不降 ECE） |
| 4 | 2.4.3 | ✅ | `noise_augment`/`noise_sigma`、test_v24_noise_aug |
| 5 | 2.4.4 | ✅ | 三水平 conformal、test_v24_conformal_levels |
| 6 | 2.4.5 | ✅ | `PredictionGuard`、test_v24_guard |
| 7 | 2.4.6 | ✅ | `ablation_calibration.py`、`calibration_ablation_v2.4.6.json` |
| 8 | 2.4.7 | ✅ | `predict_interval` 融合、test_v24_ensemble_interval |
| 9 | 2.4.8 | ✅ | `POST /detect-ood`、test_v24_ood_service |
| 10 | 2.4.9 | ✅ | `ablation_noise_robustness.py`、`noise_robustness_v2.4.9.json` |
| 11 | 2.4.10 | ✅ | `per_step_confidence` + `confidence_breakdown`、test_v24_conf_breakdown |
| 12 | 2.4.11 | ✅ | `ablation_conformal_levels.py`、`conformal_levels_v2.4.11.json` |
| 13 | 2.4.12 | ✅ | `/evaluate` guard 段 + `POST /predict`、test_v24_guard_service |
| 14 | 2.4.13 | ✅ | `save_ensemble/load_ensemble`、test_v24_ensemble_persistence |
| 15 | 2.4.14 | ✅ | `StreamingDriftDetector`、test_v24_streaming_drift |
| 16 | 2.4.15 | ✅ | 全量回归 171 绿 + `docs/COVERAGE_v2.4.15.txt`（新模块≥80%） |
| 17 | 2.4.16 | ✅ | ARCHITECTURE/DEPLOYMENT/CHANGELOG/README/本计划定稿 |


> 训练节点：仅 **2.4.0** 重建正式件 `checkpoints/predictor_v2.4.0.pt`；其余 patch 用确定性算法/轻量单测推进，不重训。
> 铁律：v2.3.1 的 92 测试持续全绿；新增能力默认不改变旧默认输出；版本号每升一版同步 `udos/__init__.py` / `pyproject.toml` / `Makefile` / `docker-compose.yml` / `Dockerfile` 且有测试断言。

## 2.4 线 17 节点清单

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **2.4.0** | OOD/分布漂移检测核心 + 正式训练重建 | `udos/ood.py`：`DistributionDriftDetector`（基于训练集特征均值/协方差的马氏距离 + KS 检验），`PhysicsPredictor.predict` 可选返回 `ood_score`；`build_parametric_dataset` 暴露特征统计钩子；正式训练重建 `predictor_v2.4.0.pt` + `training_v2.4.0.json` | `tests/test_v24_ood.py`：检测器拟合/评分/阈值/OOD 样本高分；版本断言 |
| 2 | 2.4.1 | 深度集成不确定性（opt-in） | `udos/ensemble.py`：`DeepEnsemble`（N 个同架构 PhysicsPredictor，多种子），`predict` 返回均值+方差+逐成员预测；默认 N=1 等价单模型 | `tests/test_v24_ensemble.py`：N=1 等价性、N>1 方差>0、save/load |
| 3 | 2.4.2 | 温度缩放校准方法 | `udos/calibration.py` 新增 `TemperatureScaling`（单参数 logit 温度，闭式/网格搜索最优 T），与 PAVA 并存；`fit_predictor_calibration` 支持 `method="pava"\|"temperature"\|"none"` | `tests/test_v24_temp_scale.py`：T=1 等价原始、T 优化降低 ECE、极端情况守卫 |
| 4 | 2.4.3 | 数据增强/噪声鲁棒训练 | `udos/dynamics.py` 新增 `noise_augment`（高斯噪声注入，可选 sigma）；`TrainConfig.noise_sigma`（默认 0.0 等价旧版）；训练循环支持 | `tests/test_v24_noise_aug.py`：sigma=0 等价、sigma>0 输出分布展宽、确定性 |
| 5 | 2.4.4 | 多名义水平 conformal 区间 | `udos/calibration.py` 的 `ConformalPredictor`（或现有区间逻辑）支持 `alpha∈{0.2,0.1,0.05}`（即 80%/90%/95%）；`predict_interval(alpha=...)`；默认 alpha=0.1 不变 | `tests/test_v24_conformal_levels.py`：三水平覆盖率、宽度随 alpha 减小而增宽、非减性 |
| 6 | 2.4.5 | 退化守卫/安全护栏 | `udos/guard.py`：`PredictionGuard`（NaN/inf 检测、物理量越界检测、回退到上一有效步或零向量）；`PhysicsPredictor.predict` 可选 `guard=True`（默认 False 不改变旧行为） | `tests/test_v24_guard.py`：NaN 输入回退、越界截断、guard=False 透传 |
| 7 | 2.4.6 | 校准方法对比 A/B 基准 | `scripts/ablation_calibration.py`：PAVA vs Temperature vs None，同合同多种子，ECE/Spearman/覆盖率，落 `benchmarks/results/calibration_ablation_v2.4.6.json`；Makefile 加 `calib-ablation` target | 基准脚本可一键复跑；结果 JSON 含三方法各指标 |
| 8 | 2.4.7 | 集成不确定性 + conformal 融合 | `udos/ensemble.py` 新增 `predict_interval`（集成方差缩放 conformal 半宽）；证据：集成区间 vs 单模型区间覆盖率/宽度；若收益不稳则标记 opt-in | `tests/test_v24_ensemble_interval.py`：区间包含真值比例、宽度有限、N=1 退化为普通区间 |
| 9 | 2.4.8 | OOD/漂移服务接口 | `udos/server.py` 新增 `POST /detect-ood`（入参场景序列，返回 ood_score + 是否漂移）；未训练 409；`/evaluate` 可选含 ood 段 | `tests/test_v24_ood_service.py`：200/409、OOD 样本高分、ID 样本低分 |
| 10 | 2.4.9 | 噪声鲁棒性 A/B 基准 | `scripts/ablation_noise_robustness.py`：noise_sigma=0 vs 0.05 vs 0.1，测试时加噪 vs 不加噪，MSE 对比，落 `benchmarks/results/noise_robustness_v2.4.9.json`；Makefile `noise-ablation` | 基准脚本可复跑；JSON 含三 sigma × 两测试条件 |
| 11 | 2.4.10 | 逐步逐维置信度可解释性 | `udos/calibration.py` / `evaluation.py` 新增 `per_step_confidence`（返回 [H, RAW_DIM] 置信矩阵）；`/evaluate` 含 `confidence_breakdown` 段 | `tests/test_v24_conf_breakdown.py`：形状正确、值在 [0,1]、与标量置信一致 |
| 12 | 2.4.11 | 多水平区间覆盖率验证基准 | `scripts/ablation_conformal_levels.py`：80/90/95 三水平独立测试集覆盖率 + 宽度，落 `benchmarks/results/conformal_levels_v2.4.11.json`；Makefile `conformal-levels` | 基准脚本可复跑；JSON 含三水平 coverage/width |
| 13 | 2.4.12 | 退化守卫接入 server | `udos/server.py` 的 `/evaluate` 新增 `guard` 段（触发次数、回退次数）；`POST /predict`（如不存在则新增）支持 `guard=True` 参数 | `tests/test_v24_guard_service.py`：guard 段存在、异常输入触发回退 |
| 14 | 2.4.13 | 集成模型 checkpoint 持久化 | `udos/persistence.py` 新增 `save_ensemble` / `load_ensemble`（多成员 state_dict + 配置）；旧 `save_predictor`/`load_predictor` 不变 | `tests/test_v24_ensemble_persistence.py`：save/load 逐位一致、成员数正确、旧单模型加载不受影响 |
| 15 | 2.4.14 | 在线漂移滑动窗口累积 | `udos/ood.py` 新增 `StreamingDriftDetector`（滑动窗口 Welford 均值/方差更新，`update(x)` / `score(x)` / `reset()`）；与离线检测器共享接口 | `tests/test_v24_streaming_drift.py`：流式统计与批处理一致、窗口重置、漂移检测 |
| 16 | 2.4.15 | 2.4 线全量回归 + 覆盖率加固 | 全量 pytest 全绿；`--cov` 报告；新增测试覆盖 ood/ensemble/guard/temp_scale 等新模块；修复任何回归 | 全量测试通过 + 覆盖率报告 |
| 17 | 2.4.16 | 2.4 线文档定稿 | 更新 `docs/ARCHITECTURE.md`（新增模块说明）、`docs/DEPLOYMENT.md`（新接口）、根 `CHANGELOG.md`（17 条）、`README.md`（2.4 特性段）；`docs/VERSION_PLAN_2.4.md` 标注完成状态 | 文档与代码一致 |

## 设计约束

1. **零重依赖**：不引入 GPU/联网/大权重依赖；所有新模块纯 torch+numpy+标准库。
2. **默认不变**：所有新能力默认关闭或等价旧版；opt-in 才激活。
3. **证据诚实**：A/B 基准若收益不稳或负面，照实写入 JSON 与文档，不硬凑结论。
4. **确定性**：所有算法路径可复现（set_seed）；测试不依赖随机通过。
5. **术语统一**：第二组件一律 GPM，不出现 CPM。
