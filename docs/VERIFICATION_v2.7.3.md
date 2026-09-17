# UDOS 推演引擎 v2.7.3 最终验收与发布验证报告

> 验收工程师：UDOS 推演引擎最终验证与发布工程师
> 验收日期：2026-09-13（Asia/Shanghai）
> 运行环境：Python 3.12.11 · torch 2.14.0+cpu · CPU 2 线程 · pytest · 无 GPU / 无 docker 实建
> 术语统一：第二引擎一律称 **GPM**（Generative Physics），不出现 CPM。
> 本报告**诚实呈现**正反证据，不夸大收益、不隐瞒限制。

---

## 0. 验收结论（TL;DR）

| 硬验收项 | 结果 |
|---|---|
| 全量回归 | **364 passed, 0 failed, 0 skipped** |
| 总体覆盖率 | **92%**（3554 stmts / 279 miss） |
| 正式件 checkpoint | `checkpoints/predictor_v2.7.3.pt`，可训练参数 **52191**，reload_consistent=true，md5 `dbcc111c62c0b53c72b1f44bd9212187`（zip 内件与工程件逐字节一致） |
| 旧 checkpoint 兼容 | v2.1.0/v2.2.1/v2.3.1/v2.4.0/v2.5.0/v2.5.2/v2.6.0/v2.6.2/v2.7.0/**v2.7.3** 十件全部可加载、预测形状 [B,6] 有限 |
| 服务逐接口 | /health=2.7.3、/evaluate（含 calibration/interval）、/predict、/policy/select、/online/adapt、/active/sample、/experiments 全 200；非法输入 400；未训练 409 |
| save/load 逐位一致 | predictor_v2.7.3.pt save→reload 后预测 `torch.equal == True`，max_abs_diff=0.0 |
| 打包 | `udos-engine-v2.7.3.zip`（顶层 udos-engine/，排除 pycache/pytest_cache/.git/.coverage/htmlcov） |
| 独立解压复跑 | /tmp 解压后版本=2.7.3、pytest 全绿、build --quick 可复现、服务起得来、/health=2.7.3、新端点可用、zip 内 checkpoint md5 与工程一致（见 §8） |

**结论：v2.7.3 通过硬验收，准予发布。**

---

## 1. v2.7.3 正式件指标

来源：`benchmarks/results/training_v2.7.3.json`（与 `checkpoints/predictor_v2.7.3.pt` 配套）。
训练口径：seed=42、n_per_kind=48、horizon=4、step_weight_scheme=front、epochs=60（best_epoch=57）、
未早停、patience=12、hybrid_weight=0、CPU 2 线程，训练耗时 **102.5s**（与 v2.7.0/v2.6.2 同口径）。

| 指标 | 值 | 说明 |
|---|---|---|
| version | **2.7.3** | checkpoint 与 JSON 自洽 |
| n_params | **52191** | 可训练参数（=model.parameters()） |
| final_loss | **0.059488** | 训练最终 loss（60 epoch 曲线完整落 JSON） |
| eval_mse | **0.045556** | 独立测试集单步 MSE |
| naive_mse | 0.163415 | 朴素 persistence 基线 |
| untrained_single_mse | 4.415056 | 未训练随机模型单步 MSE |
| ECE（校准后） | **0.056546** | 独立测试集期望校准误差 |
| raw ECE（校准前） | 0.385466 | 校准前 raw ECE |
| ECE 下降倍数 | **6.817×** | 0.3855 → 0.0565 |
| coverage（90% 名义） | **0.8888** | split-conformal 区间整体覆盖 |
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

**OOD 诊断**：阈值 5.5275；ID score 均值 2.640 / 最大 8.20；OOD（×5 扰动）score 均值 13.10；
OOD 命中率 0.8104；ID 误报率 0.0646。

---

## 2. 2.7 各特性正反证据

| 特性 | 模块 | 证据 | 结论 |
|---|---|---|---|
| MPC 动作优选 | `policy.MPCActionSelector` | 空候选集 no_valid_action；已知最优动作被选；风险惩罚单调；安全边界越界扣分；服务 `/policy/select` 200 | **有效**（纯前向外挂，不改主模型） |
| 在线适配 | `online.OnlineAdapter` | 无漂移不触发；漂移触发仅重跑 PAVA（不改权重）；NaN 观测跳过；A/B 适配日志完整 | **有效**（默认不改权重，微调 opt-in） |
| 主动学习 | `active_learning.UncertaintySampler` | A/B：**主动 0.264 vs 随机 0.502**（active_minus_random=−0.238）；top-K 降序 | **有效**（合成小模型上降样本需求；增益对种子/口径敏感，如实记录差值） |
| 轻量化 | `lite.*` | INT8 动态量化 eval_mse 0.066（相对全量 0.057 近似无损）；蒸馏学生 27071 参数 | **量化近似无损；剪枝 50% 不微调掉点（条件依赖）；蒸馏精度有损**（照实记录） |
| 分层 rollout | `hierarchical.HierarchicalRollout` | H=8 与平铺 `bit_identical=true`（本件实测 `hierarchical_bit_identical_to_flat=true`） | **负面/脚手架**：单自回归头无独立粗粒度头，分块==连续单步，当前无误差下降 |
| 实验治理 | `experiment.ExperimentRegistry` | 多种子 mean/std/best；JSON 原子持久化；幂等覆盖；`/experiments` 200 | **有效**（纯元数据，不碰权重） |

---

## 3. 被证据否决 / 降级 opt-in 的候选（诚实保留）

- **分层 rollout（负面）**：单自回归头下分块滑窗在数学上 == 连续单步 rollout，A/B 逐位一致，
  不产生误差下降。保留为将来接入独立（非自回归）粗粒度头的 opt-in 脚手架。
- **幅值剪枝 50%（条件依赖）**：不微调直接剪枝 eval_mse 明显下降（0.057→0.584），需配合微调才可用；
  动态 INT8 量化才是"近似无损"的轻量化手段。
- **蒸馏学生（精度有损）**：27071 参数换一半参数，但 eval_mse 0.284，如实记录为"小代价换参数，精度损失"。
- **Scheduled Sampling / hybrid 等更早候选**：与 v2.2/v2.6 同纪律，均记录为条件依赖/opt-in，不伪造收益。

---

## 4. 已知限制

1. **合成数据**：所有精度/校准/OOD 指标均在合成参数化动力学上测得，不代表真实物理系统精度。
2. **PAVA 不修坏排序**：保序回归只做单调映射，若原始置信排序本身不 informative 则不改善。
3. **conformal 仅同分布交换性保证**：split-conformal 区间覆盖率在测试分布与校准分布可交换时成立；
   分布外（OOD）不提供覆盖保证。
4. **CPU 小模型不外推**：52191 参数 / CPU 2 线程，指标不外推到大模型或 GPU 场景。
5. **无 docker 实建**：本环境无 docker daemon，容器镜像为等价进程验证（同一命令行 `python -m udos.server`
   起停逐接口测通），Dockerfile/docker-compose 仅静态校验引用正确。

---

## 5. 复现命令

```bash
make test                      # 全量 pytest
make cov                       # 全量 pytest + 覆盖率 (目标 ~92%)
make ckpt273                   # 正式训练重建 predictor_v2.7.3.pt (~100s)
make feature-bench27           # 2.7 特性推理延迟基准
python3 scripts/verify_service_v273.py   # 真实 HTTP 逐接口验证
make serve CHECKPOINT=checkpoints/predictor_v2.7.3.pt
```

## 6. 服务逐接口验证摘要（真实起停）

- `GET /health` → 200, version=2.7.3, predictor_trained=true
- `POST /evaluate` → 200，metrics 含 calibration / interval 段
- `POST /policy/select` → 200（best_index=0）；缺 actions → 400；未训练 → 409
- `POST /online/adapt` → 200（分布内 adapted=False）
- `POST /active/sample` → 200（返回 3 索引）；空池 → 400
- `GET /experiments` → 200（注册表不存在时 count=0）

## 7. 向后兼容

`tests/test_v27_backcompat.py` 覆盖十件 checkpoint（v2.1.0..v2.7.3）：每件 load 成功、
元数据版本正确、predict 形状 [B,6] 有限、policy/active/hierarchical 对旧件不报错。

## 8. 独立解压复跑与 md5

见发布流程：zip 解压到 `/tmp` 独立目录后，版本断言=2.7.3、pytest 全绿、
`build_v273_checkpoint.py --quick` 可复现、服务 `/health`=2.7.3 且新端点可用、
zip 内 `predictor_v2.7.3.pt` 的 md5 与工程内一致。
