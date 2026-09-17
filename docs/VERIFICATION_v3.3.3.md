# UDOS 推演引擎 v3.3.3 最终验收与发布验证报告

> 验收工程师：UDOS 推演引擎最终验证与发布工程师
> 验收日期：2026-09-14（Asia/Shanghai）
> 运行环境：Python 3.12.11 · torch 2.14.0+cpu · CPU 2 线程 · pytest · 无 GPU / 无 docker 实建
> 术语统一：第二引擎一律称 **GPM**（Generative Physics），不出现 CPM。
> 本报告**诚实呈现**正反证据，不夸大收益、不隐瞒限制；所有数字均为真实运行/可复算结果，未编造。

---

## 0. 验收结论（TL;DR）

| 硬验收项 | 结果 |
|---|---|
| 全量回归（工程内） | **738 passed, 0 failed, 0 skipped**（进度点逐一点数，无 F/E/s） |
| 总体覆盖率 | **93%**（5731 stmts / 402 miss） |
| 正式件 checkpoint | `checkpoints/predictor_v3.3.3.pt`，可训练参数 **52191**，udos_version=**3.3.3**，reload_consistent=true，md5 `f993bcbdd476473c28dd4604bbbe11d6`（241867 字节） |
| save/load 逐位一致 | save→reload 后 `torch.equal == True`，max_abs_diff=**0.0** |
| 旧 checkpoint 兼容 | **18 件**（v2.1.0…v3.3.3）全部可加载、元数据版本正确、predict 形状 [B,6] 有限 |
| 服务逐接口 | /health=3.3.3、/evaluate（含 calibration/interval）、/loop/step、/multitask/predict、/retarget/convert、/affordance/score、/future/predict、/eval/5d、/action/tokenize、/action/detokenize、/augment/generate、/icl/predict 全 **200**；非法输入 **400**；未训练 **409** |
| 打包 | `udos-engine-v3.3.3.zip`（顶层 `udos-engine/`，排除 `__pycache__`/`*.pyc`/`.pytest_cache`/`.git`，无泄漏） |
| 独立 /tmp 解压复跑 | 版本=3.3.3、pytest 738 全绿、`build_v333_checkpoint.py --quick` 可复现、服务 `/health`=3.3.3、/evaluate 含 calibration/interval、新端点 /loop/step=200、zip 内 checkpoint md5 与工程一致 |

**结论：v3.3.3 通过硬验收，准予发布。**

---

## 1. v3.3.3 正式件关键指标

来源：`benchmarks/results/training_v3.3.3.json`（与 `checkpoints/predictor_v3.3.3.pt` 配套）。
训练口径与 v3.2.0/v3.3.0 同口径：seed=42、n_per_kind=48、horizon=4、step_weight_scheme=front、
epochs=60（best_epoch=57）、未早停、patience=12、hybrid_weight=0、CPU 2 线程，训练耗时 **102.5s**。

| 指标 | 值 | 说明 |
|---|---|---|
| version | **3.3.3** | checkpoint 与 JSON 自洽 |
| n_params | **52191** | 可训练参数（=sum(p.numel())，实测） |
| epochs_run / best_epoch | 60 / 57 | stopped_early=false |
| final_loss | **0.059488** | 训练最终 loss |
| eval_mse | **0.045556** | 独立测试集单步 MSE |
| naive_mse | 0.163415 | 朴素 persistence 基线 |
| untrained_single_mse | 4.415056 | 未训练随机模型单步 MSE |
| ECE（校准后） | **0.056546** | 独立测试集期望校准误差 |
| coverage（90% 名义） | **0.8888** | split-conformal 区间整体覆盖 |
| batch_inference_max_diff | **1.97e-06** | 批量推理 vs 逐笔最大绝对差 |
| reload_consistent | **true** | save→reload→evaluate 逐位一致 |
| **condition_gain_x / interval_width / rollout_growth** | **未记录** | v3.3.3 快照 JSON 未输出该字段，不编造；与 v2.7.3 报告不同口径 |

**OOD 诊断**（`ood` 段）：阈值 5.5275；ID score 均值 2.640；OOD（×5 扰动）score 均值 13.098；
OOD 命中率 **0.8104**；ID 误报率 **0.0646**。

**3.3 新特性离线 A/B 快照**（`features_v333_offline_ab`，真实运行）：

| 子项 | 值 |
|---|---|
| action_piece_roundtrip | 0.008166 |
| ego_augment_finite | true |
| moe_out_finite / moe_params | true / 2244 |
| prune_v2 channel_sparsity | 0.5 |
| **pruned_no_retrain_mse** | **2.840944**（vs full 0.0456，不重训明显劣化） |
| student_v2 params / student_v2_mse | 15027 / **2.630763** |
| robustness_score | 55.18 |
| analogy_not_reproduction | true |

> 注：`benchmarks/results/integration_report_v3.3.1.json` 另给出综合 robustness_score=48.6（含噪声/OOD/外推/FGSM 四子项），
> 五维内部评测：visual_spatial 97.12 / multiview 49.34 / cognition_planning 100.0 / affordance 100.0 / trajectory_reasoning 94.57。
> 该五维为 **UDOS 内部基准，非 PhysBrain 榜单分数**。

---

## 2. 18 代 checkpoint 向后兼容

独立脚本遍历 `checkpoints/predictor_v*.pt` 共 **18 件**，每件 `load_predictor` 后 `predict_next` 输出
形状 [2,6] 且全部有限、元数据 `udos_version` 与文件名版本一致。实测结果：

| checkpoint | 元数据版本 | 输出形状 | 有限 |
|---|---|---|---|
| predictor_v2.1.0.pt | 2.1.0 | (2,6) | ✓ |
| predictor_v2.2.1.pt | 2.2.1 | (2,6) | ✓ |
| predictor_v2.3.1.pt | 2.3.1 | (2,6) | ✓ |
| predictor_v2.4.0.pt | 2.4.0 | (2,6) | ✓ |
| predictor_v2.5.0.pt | 2.5.0 | (2,6) | ✓ |
| predictor_v2.5.2.pt | 2.5.2 | (2,6) | ✓ |
| predictor_v2.6.0.pt | 2.6.0 | (2,6) | ✓ |
| predictor_v2.6.2.pt | 2.6.2 | (2,6) | ✓ |
| predictor_v2.7.0.pt | 2.7.0 | (2,6) | ✓ |
| predictor_v2.7.3.pt | 2.7.3 | (2,6) | ✓ |
| predictor_v2.8.0.pt | 2.8.0 | (2,6) | ✓ |
| predictor_v2.9.0.pt | 2.9.0 | (2,6) | ✓ |
| predictor_v3.0.0.pt | 3.0.0 | (2,6) | ✓ |
| predictor_v3.0.3.pt | 3.0.3 | (2,6) | ✓ |
| predictor_v3.1.0.pt | 3.1.0 | (2,6) | ✓ |
| predictor_v3.2.0.pt | 3.2.0 | (2,6) | ✓ |
| predictor_v3.3.0.pt | 3.3.0 | (2,6) | ✓ |
| predictor_v3.3.3.pt | 3.3.3 | (2,6) | ✓ |

**18/18 通过。**

---

## 3. HTTP 服务逐接口验证（真实起停）

脚本 `scripts/verify_service_v333.py` 以 `--checkpoint checkpoints/predictor_v3.3.3.pt` 真实起服务后逐接口请求。

| 端点 | 状态码 | 说明 |
|---|---|---|
| GET /health | **200** | version=3.3.3, predictor_trained=true |
| POST /evaluate | **200** | metrics 含 calibration / interval 段 |
| POST /loop/step | **200** | v2.8 Physical Loop |
| POST /multitask/predict | **200** | v2.8 共享 backbone 三头 |
| POST /retarget/convert | **200** | v2.9 prime_u_60dof→gripper_4dof |
| POST /affordance/score | **200** | v2.9 object_proxy [B,N_parts,6] |
| POST /future/predict | **200** | v3.0 多模态未来头 |
| GET /eval/5d | **200** | v3.0 内部五维（非 PhysBrain 榜单） |
| POST /action/tokenize | **200** | v3.1 连续→离散 token |
| POST /action/detokenize | **200** | v3.1 离散→连续 |
| POST /augment/generate | **200** | v3.2 Ego360 增强 |
| POST /icl/predict | **200** | v3.2 few-shot 注入 |
| POST /loop/step（horizon=99 越界） | **400** | 非法输入 |
| POST /multitask/predict（缺 window） | **400** | 非法输入 |
| POST /retarget/convert（缺 source） | **400** | 非法输入 |
| POST /action/tokenize（缺 actions） | **400** | 非法输入 |
| POST /loop/step（未训练实例） | **409** | 未训练态 |
| POST /multitask/predict（未训练实例） | **409** | 未训练态 |
| POST /icl/predict（未训练实例） | **409** | 未训练态 |

---

## 4. 每特性正反证据与被回退/降级候选（诚实保留）

| 特性 | 结论 | 真实证据 |
|---|---|---|
| **多任务三头（multitask）** | 降级 opt-in | 共享 backbone 省 encode 次数（1 vs 3，延迟 0.22ms vs 0.53ms，2.38×）；但头为随机初始化线性投影，`future_head_proxy_mse_vs_rollout=8.10`，无训练精度收益 → 默认关 |
| **分层 rollout（hierarchical）** | 负面/脚手架 | H=8/12/16 与平铺曲线逐位 `bit_identical=true`（误差增长倍数完全相同）；单自回归头下分块==连续单步，无误差下降 |
| **数据增强（ego_augment）** | in-dist 变差，降级 opt-in | 分布内 eval_mse：raw 0.498 → augmented 1.845（`augment_mse_gain=-1.347`）；OOD 有改善（ood_mse 17.8→3.4，gain +14.4）；无一致分布内收益 → 不进正式训练 |
| **ICL 上下文学习** | 无收益，opt-in | 0-shot MSE=0.0464；1-shot=0.6686；3-shot=1.1076 —— few-shot 反而更差，纯推理外挂不改权重 |
| **课程学习（curriculum）** | 无增益，opt-in | 合成 next-token 任务：课程 0.6258 vs 随机 0.6258 持平；纯数据调度，不进正式训练 |
| **结构化剪枝 v2（prune_v2）** | 不重训劣化，opt-in | 通道/头级 50% 稀疏，`pruned_no_retrain_mse=2.84`（vs full 0.046）；不重训精度大幅下降 |
| **蒸馏学生 v2（distill_v2）** | 精度有损，opt-in | 学生 15027 参数（约 full 29%）但 `student_mse=2.63`；换参数不换精度 |
| **多模态未来头（future multimodal）** | 无状态增益，opt-in | 仅输出代理向量不输出未来状态，无状态 MSE 增益；参数开销 2.29×；一致性损失 opt-in 不进主损失 |
| **动作重定向（retarget）** | 合成代理有效，类比非复现 | endpoint_aligned+clamp 相对 naive 截断 MSE 降 2.37×；`analogy_not_reproduction=true`，synthetic proxy only |
| **ActionPiece token 化** | 有效但 opt-in | 码本扫描 8/16/32/64，tokenized 量化 MSE 0.029 < 连续基线；默认仍为连续动作主路径 |
| **轻量 MoE（moe）** | 推理外挂 opt-in | 2244 参数路由头，输出有限；不并入主干 |
| **效率 Pareto** | **推荐 full** | full 52191 参数 / 6.70ms / mse 0.0475；pruned 6.46ms/mse 2.94；student 4.55ms/mse 2.30；无候选精度劣化≤5% → 默认维持 full |

**一句话**：6 条 minor 线（2.8/2.9/3.0/3.1/3.2/3.3）共 60 个迭代节点中，所有"看似更强"的候选凡未在合成小模型上跑出**一致、可复现**的精度/校准收益者，一律降级为 opt-in（默认关），正式件默认路径与 v2.7.3 逐位不变。

---

## 5. PhysBrain 资料抓取覆盖度

来源 `docs/PHYSBRAIN15_ANALYSIS.md`（38600 字节，非空，已确认存在）。其 §13 自报覆盖度：

| 资料 | 覆盖度 |
|---|---|
| S1 Project Page | 100%（静态可抓部分） |
| S5 EvalKit 仓库 | 高（核心评测逻辑已读，各 benchmark/*.py 未逐行） |
| S7 Human-as-Humanoid 论文 | 高（方法主线已读，消融表未逐字） |
| S8 PhysBrain 1.0 技术报告 | 低（仅 abstract） |
| S9 ZAKER/量子位新闻稿 | 100% |
| S10 DataNorth 报道 | 100% |
| S2/S3/S4 HuggingFace | 0% [FETCH FAILED] |
| S6 技术报告 1.5 | 0% [FULL TEXT NEEDED] |
| S11 HF Demo | 0%（JS 渲染未访问） |

未决项已在该文档标注 `[FULL TEXT NEEDED]` / `[FETCH FAILED]` / `[CITATION NEEDED]` / `[UNVERIFIED]`，未编造任何论文标题/作者/DOI/分数。**UDOS 内部五维评测明确标注"非 PhysBrain 榜单分数"。**

---

## 6. 已知限制

1. **合成数据**：所有精度/校准/OOD 指标均在合成参数化动力学上测得，不代表真实物理/机器人系统精度。
2. **理念类比非复现**：retarget / affordance / future-multimodal / ego-augment / action-piece 均为对 PhysBrain/PrimeU/Ego360/Human-as-Humanoid 等上游工作的**合成代理类比**（`analogy_not_reproduction=true`），非论文复现。
3. **CPU 小模型不外推**：52191 参数 / CPU 2 线程，指标不外推到大模型或 GPU 场景。
4. **conformal 仅同分布交换性保证**：OOD 不提供覆盖保证；robustness 外推子项得分低（scale=2.0 时 degradation 28.1×）如实记录。
5. **condition_gain / rollout 曲线未入 v3.3.3 快照**：本次训练 JSON 未输出该字段，本报告不沿用旧版数字凑表，标注为未记录。
6. **无 docker 实建**：本环境无 docker daemon，容器为**等价进程验证**（同一命令行 `python -m udos.server` 起停逐接口测通）；Dockerfile/docker-compose 仅静态校验引用正确。

---

## 7. 复现命令

```bash
# 全量回归 + 覆盖率
python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing   # 738 passed, 93%

# 正式件重建 (~100s) / 快速复现
python3 scripts/build_v333_checkpoint.py          # 正式口径
python3 scripts/build_v333_checkpoint.py --quick  # 小规模可复现

# 真实 HTTP 逐接口验证
python3 scripts/verify_service_v333.py

# 独立复跑
python3 -c "import udos; print(udos.__version__)"  # 3.3.3
md5sum checkpoints/predictor_v3.3.3.pt            # f993bcbdd476473c28dd4604bbbe11d6
```

---

## 8. 独立 /tmp 解压复跑与 md5

将 `udos-engine-v3.3.3.zip` 解压到 `/tmp/udos-verify-v333/udos-engine` 后：

| 步骤 | 结果 |
|---|---|
| `import udos; __version__` | **3.3.3** ✓ |
| `pytest tests/ -q` | **738 passed, 0 failed, 0 skipped** ✓ |
| `build_v333_checkpoint.py --quick` | 成功复现：version=3.3.3, n_params=52191, backcompat=18 ✓ |
| 起服务 `--port 8898 --checkpoint …v3.3.3.pt` | /health version=3.3.3, predictor_trained=true ✓ |
| POST /evaluate | 含 calibration / interval（均 True）✓ |
| POST /loop/step（新端点） | **200** ✓ |
| checkpoint md5（解压后、build 前） | **f993bcbdd476473c28dd4604bbbe11d6** —— 与工程内逐字节一致 ✓ |

> 说明：`build --quick` 会以小口径（n_per_kind=16/epochs=15）重写同名 checkpoint 文件，故服务验证与最终 md5 核对前，
> 以工程内正式件（md5 已证与 zip 解压件逐字节相同）恢复该文件；这是对"快速复现可跑 + 正式件逐位一致"两项要求的并集，
> 不改动正式件本身。

---

## 9. 最终门禁结果

- [x] 全量回归 0 failed（738/0/0）
- [x] 覆盖率 93%
- [x] 正式件 n_params=52191 / version=3.3.3 / predict [1,6] 有限 / save-load 逐位一致
- [x] 18 代 checkpoint 全部兼容
- [x] HTTP 端点 200/400/409 符合预期
- [x] zip 顶层结构正确、无 pycache 泄漏、含正式件
- [x] /tmp 独立解压复跑通过、md5 一致
- [x] CHANGELOG 2.8–3.3 六条 minor 线共 **60** 个迭代节点
- [x] docs/PHYSBRAIN15_ANALYSIS.md 存在且非空

**v3.3.3 通过最终质量门禁，准予发布。**
