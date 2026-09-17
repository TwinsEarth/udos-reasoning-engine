# UDOS 推演引擎部署手册 (v3.3.4)

## 1. 部署形态

| 形态 | 适用 | 启动方式 |
|------|------|----------|
| 裸机进程 | 开发 / 单机 | `python -m udos.server --host 0.0.0.0 --port 8000 --preset small` |
| Docker | 生产单机 | `docker build -t udos-reasoning-engine:0.2.0 . && docker run -p 8000:8000 udos-reasoning-engine:0.2.0` |
| Compose | 带资源限制/重启策略 | `docker compose up -d` |
| systemd | 常驻主机 | 见 §5 |

规格预设：`small`（width=64 / d_model=128 / 16 ticks / 2 层 / rank=4）、
`medium`（width=128 / d_model=256 / 32 ticks / 4 层 / rank=8）。

## 2. API 契约

Content-Type 均为 `application/json`；PCE 场景结构：
```json
{"scene_id":"s1","duration":8,"constraints":[],"metadata":{},
 "tokens":[{"object_id":"arm","timestamp":0,"position":[0,0,0],
            "velocity":[0.1,0,0],"force":[0,0,0],
            "attributes":{"mass":4.5},"causal_parents":[]}]}
```

### GET /health
返回 `{status, service, version, predictor_trained, internalized_scenes[]}`。

### POST /internalize
请求 `{scene, chunks?}`；GPM 单次前向把场景内化为 LoRA。
返回 `{status, scene_id, lora_params, lora_kb_fp32, target_modules, message}`。

### POST /reason
请求 `{scene, query?, horizon?=1}`（horizon 1–16）；在已内化场景上做 GPM 场景条件化的
CTM 时序推演。返回 `{summary, ticks_used, scene_conditioned, horizon, final_certainty,
certainty_trajectory[], prediction_vector[], causal_chain[], internalized,
predicted_next_state?, future_states?}`；`predicted_next_state`（position/velocity）在
挂载预测器后出现；`horizon>1` 时额外返回 `future_states`（H 步 pos/vel 滚动推演）。

### POST /reason/latent （v4.5.1）
请求 `{scene, effort=none|low|high|max, horizon?=1, query?}`（未挂载预测器→409；非法 effort→400）。
隐式思考：在 CTM 连续隐藏状态跑 K 条潜路径并行探索（latent best-of-K），按终态 certainty 聚合。
返回 `{effort, n_paths, internal_ticks, externalized, latent_summary{explored_paths/
convergence_*/path_disagreement/selection_reason/switch_point/closer}, explicit_chain?,
prediction_vector?, latency_ms, main_params_untouched, analogy_not_reproduction}`。
**none 档与 `/reason` 默认逐位等价**；low/high/max opt-in，high/max 投影可读显式链。

### POST /reason/route （v4.5.1）
请求 `{scene, task_complexity?∈[0,1], needs_deeper?∈[0,1]}`（未挂载预测器→409）。
用 CTM 收敛 + 任务提示合成难度分，推荐 effort 档位与是否显式化。返回
`{recommended_effort, difficulty, externalize, rationale[]}`。

### POST /train （v2.1.0）
请求 `{epochs?=45, n_per_kind?=32, horizon?=4, phys_weight?=0.0}`
（范围 epochs 1–300、n_per_kind 1–512、horizon 1–8，越界 400）。
在**参数化场景条件多步**动力学上训练带 scene_encoder 的 PhysicsPredictor 并挂载。
返回 `{horizon, untrained_mse, naive_mse, trained_mse, reduction_*_x,
evaluation{single_step_mse, ablation{conditioned/unconditioned/condition_gain_x},
per_kind_mse, rollout_mse_curve, kinematic_residual}, train_loss_curve[],
eval_mse_curve[], certainty_curve[], phys_violation_curve[], predictor_params}`。
CPU 上默认参数约 1 分钟，属按需触发的重操作。

### 启动即带训练态：--checkpoint（v2.1.0）
`python -m udos.server --checkpoint checkpoints/predictor_v3.3.3.pt` 启动时预加载
预置预测器，`/health` 立即 `predictor_trained=true`，无需先调 `/train`；不带该参数
则与 v2.0 一致（初始未训练）。镜像与 docker-compose 默认已带此预加载（**v3.3.4 FIX-002 已修正为
v3.3.3 正式件**，不再误加载 v3.2.0）。

### POST /evaluate（v2.2.0）/ POST /save（v2.2.0）
`/evaluate` 请求 `{n_per_kind?=32, horizon?=4, seed?}`，对**已挂载**预测器在新鲜
测试集在线评估，未训练返回 **409**；预测器已校准时结果额外含 `calibration`/`interval`。
`/save` 请求 `{name}`，连同指标落 `checkpoints/<name>.pt`（名称清洗、路径不可逃逸），
未训练 409。

### POST /calibrate（v2.3.0）
请求 `{n_per_kind?=32, horizon?=4, calibration_seed?=2031, test_seed?=2042}`（范围同
/train，越界 400；未训练 **409**）。在独立校准集拟合保序校准器+每步残差分位并挂载，
返回 `{status, fitted_on_calibration_set{raw/calibrated ECE、scale、num_segments、
ranking_informative}, independent_test{calibration, interval}}`。纯确定性后处理、不重训。

### GET /checkpoints（v2.3.0）
列出 `checkpoints/` 下文件：`{count, checkpoints:[{name,bytes,modified}]}`；目录不存在
返回 `count:0`，不报错。

### POST /load（v2.3.0）
请求 `{name}`；白名单 + commonpath 校验，仅允许 `checkpoints/` 下普通文件名，
不存在或含 `../`/绝对路径返回 **400**。成功返回 `{status, source_version, calibrated,
saved_at}`，载回的校准器随件挂载。

### POST /detect-ood（v2.4.8）
请求 `{sequence: [W,RAW]|[N,W,RAW]}`；对窗口序列做 OOD/漂移检测。未训练或未挂检测器
返回 **409**，形状/维不符返回 **400**。返回 `{threshold, scores[], ood[], ood_rate,
score_mean, score_max}`。阈值为训练集马氏距离的 (1−alpha) 保守上分位。
`/evaluate` 加 `include_ood=true` 时结果额外含 `ood` 段（逐维 KS 漂移摘要）。

### POST /predict（v2.4.12）
请求 `{window: [W,RAW]|[N,W,RAW], scene_params?: [N,P], guard?=false}`；在线单步下一状态
预测，未训练 **409**、形状/维不符 **400**。返回 `{prediction, shape, guard{enabled,
n_fallbacks, n_clips}}`。`guard=true` 时过 `PredictionGuard`（NaN/inf 回退 + 越界截断），
并返回触发/回退计数；每请求全新计数。`/evaluate` 加 `guard=true` 时结果额外含 `guard` 段。

### GET /metrics（v2.5.1）
无请求体。返回 **Prometheus 文本格式** exposition（纯标准库实现，不引入 prometheus_client）：
`udos_requests_total{route,method}`、`udos_latency_seconds`（p50/p95/p99）、`udos_cache_hit_ratio`、
`udos_ood_trigger_ratio`。线程安全，HTTP handler 在 `finally` 中自动计时。**200**。

### POST /rollback（v2.5.1）
无请求体。维护已加载 checkpoint 栈（启动 `--checkpoint` 预加载占栈底，`POST /load` 压栈）；
回滚到上一已加载件并重新 `load_predictor`。栈长 < 2（无历史）返回 **409**；成功 **200**
并返回 `restored_path`。回滚不复制权重，仅记录路径。

### POST /export-snapshot（v2.5.1）
无请求体。导出**无权重**快照 JSON：架构配置 + 校准器 + 残差分位 + OOD 统计，用于快速恢复
推理后处理。权重仍走 `.pt`。**200**。

### POST /import-snapshot（v2.5.1）
请求 `{snapshot: <export-snapshot 返回对象>`。导入前校验 ctm_config/raw_dim/scene_param_dim
与当前模型一致，不匹配 **400**；匹配则恢复校准器/分位/OOD 统计。**200**。

### POST /counterfactual（v2.6.0）
请求 `{window:[W,R]|[N,W,R], horizon?(1-8, 默认2), scene_params?, intervention?}`。
`intervention` 形如 `{"scene_params":{idx:val}, "initial_state":delta, "velocity_override":v}`；
None/空 = 基线 rollout。返回 baseline / counterfactual / ate_by_step / ate_mean / final_state_diff。
未训练 **409**；非法形状或干预值含 NaN/inf **400**。**200**。

### POST /identify（v2.6.0）
请求 `{window:[W,R], horizon?(默认2, ≤W-1), grid_size?(默认5, 2-12)}`。仅凭观测窗口网格
反推 4 维隐藏场景物理参数。返回 identified_params / param_names / loss_min / n_grid。
未训练 **409**；越界参数 **400**。**200**。

### POST /risk（v2.6.0）
请求 `{window, horizon?(1-4, 默认1), scene_params?}`。聚合 conformal 区间宽 + OOD 马氏距离 +
校准置信为 [0,1] 风险分与三档等级（low/medium/high）。返回 risk_score / risk_level / components。
OOD score 为 NaN 时降级为 0 并标 `ood_nan_degraded=true`（v2.6.1）。未训练 **409**。**200**。

### POST /diff-checkpoints（v2.6.0）
请求 `{name_a, name_b, n_per_kind?(1-128, 默认16)}`。在白名单内两个 checkpoint 上同测试集
evaluate + 逐位预测数值对比（不需当前预测器在线）。返回两侧 metrics 与逐位差统计。
未列名 **400**。**200**。

### POST /policy/select（v2.7.0.dev6）
请求 `{window, scene_params?, actions:[{scene_param?|state_perturbation?|candidate_state?}],
horizon?(1-8, 默认4), lambda_risk?(默认1.0)}`。MPC 对每个候选动作 rollout 打分排序，返回
`best_action/best_score/best_index/ranked_actions/no_valid_action`。空动作集仍 **200** 且
`no_valid_action=true`；缺 window/actions **400**；未训练 **409**。

### POST /online/adapt（v2.7.0.dev6）
请求 `{window, enable_finetune?(默认false), finetune_epochs?(0-10, 默认3)}`。把观测推入流式
漂移检测器，漂移触发时默认仅重跑 PAVA 校准（不改权重）；opt-in 才少量微调。返回
`adapted/reason/drift_score/weights_modified/log_length`。NaN/inf 观测行自动跳过；未训练 **409**。

### POST /active/sample（v2.7.0.dev6）
请求 `{sample_pool:[N,W,RAW], k, scene_params?}`。对样本池算信息增益分并选 top-K，返回
`indices/scores`（k>池大自动截断）。空池 / k≤0 **400**；未训练 **409**。

### GET /experiments（v2.7.0.dev6）
从 `benchmarks/results/experiment_registry.json` 读实验注册表列表；文件不存在返回空
`count=0, experiments=[]`。**200**。

### POST /loop/step（v2.8.2）
请求 `{window:[B,W,RAW], scene_params?, horizon?[1..8], actions?[dict]}`。对一条窗口跑
PhysicalLoopRunner 五步闭环，返回 `prediction`、`steps`、`action_history_len`、`correction`。
非法 horizon / 缺 window **400**；未训练 **409**。

### POST /multitask/predict（v2.8.2）
请求 `{window:[B,W,RAW], scene_params?}`。懒建共享 backbone 多头（spatial/action/future）联合推理，
返回三头 `spatial`[B,N,3] / `action`[B,H,6] / `future.trajectory`[B,H,6] + 不确定性。
未训练 **409**。

### POST /reset
请求 `{scene_id}`；无损移除场景 LoRA 前向补丁。

### GET /demo
用内置工厂场景跑 internalize→reason→reset，便于联调与探活。

错误约定：客户端输入问题返回 HTTP 400 + `{status:"error",message}`；
需先训练/校准但未挂载预测器返回 **409**（ServiceNotReady）；未知路由 404；
服务端异常 500 且**不中断进程**。

### curl 示例
```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/train -H 'Content-Type: application/json' -d '{}'
curl -s localhost:8000/demo | python -m json.tool | head
# 内化/推演/重置: body 为上面的 PCE JSON
curl -s -X POST localhost:8000/internalize -H 'Content-Type: application/json' \
  -d '{"scene": <PCE_JSON>}'
```

## 3. 实测资源与性能 (CPU, 2 线程, torch 2.14)

来源 `benchmarks/results/baseline_v0.2.0.json`（warmup 后 10 次采样中位数）：

| workload | median | 参数量/产物 |
|----------|-------:|-------------|
| CTM small 前向 | 7.1 ms | 216,192 |
| CTM medium 前向 | 17.8 ms | 691,424 |
| GPM small 内化 | 2.7 ms | LoRA 30 KB |
| GPM medium 内化 | 3.7 ms | LoRA 240 KB |
| 端到端 reason | 8.0 ms | — |
| 上游真实 CTM 对照 | 9.1 ms | 134,146 |
| v2 物理预测器前向 | 2.84 ms | 48,798 |
| v2.1 场景条件多步预测器 | 与 v2 同量级 | 52,191 |
| 进程峰值 RSS | 285 MB | — |

> 数字随 CPU 型号波动；`make guard` 用工程自定宽松门槛捕获显著回归，
> 不用于性能宣传。LoRA 产物为 KB 级，替代 GB 级 KV-Cache 是该架构核心收益。

## 4. 上线前验证清单（三线协同）

```bash
make test     # 单元+契约+学习/多步/消融/持久化+服务HTTP 测试 (64 项)
make cov      # 覆盖率 (总 95%, persistence 100%)
make demo5    # v2.1 多步/场景条件消融/一致性演示
make ckpt     # v2.1 训练并落评估 JSON + 预置 checkpoint
make guard    # 性能回归守卫
make smoke    # 真实拉起服务打 19 项接口冒烟 (含 v2.3 calibrate/checkpoints/load 与 checkpoint 预加载)
```
全部通过再发布。CI 中按相同顺序执行即可。

## 5. systemd 单元（裸机常驻示例）

```ini
[Unit]
Description=UDOS Reasoning Engine
After=network.target
[Service]
WorkingDirectory=/opt/udos-engine
# 如需开箱即带训练态, 末尾追加: --checkpoint checkpoints/predictor_v2.1.0.pt
ExecStart=/usr/bin/python3 -m udos.server --host 0.0.0.0 --port 8000 --preset small
Restart=always
RestartSec=3
Environment=OMP_NUM_THREADS=2
MemoryMax=1G
[Install]
WantedBy=multi-user.target
```

## 6. 故障排查

| 现象 | 根因 | 处理 |
|------|------|------|
| `No module named torch/pytest` | 运行时环境被重置/解释器不符 | 按 `requirements.txt` 重装；CPU 用 `--index-url .../whl/cpu` |
| 同场景重复内化结果漂移 | （v0.2.0 已修）场景编码器曾每次随机新建 | 升级到 0.2.0，编码器已持久化为子模块 |
| reset 后输出未还原 | 误用直接改权重的注入方式 | 使用 `LoRAInjector` 前向补丁，reset 零误差 |
| 400 缺少字段 scene/scene_id | 请求体不符契约 | 按 §2 补字段 |
| 端口占用 | 8000 被占 | `--port` 换端口 |
| 容器内不含上游对照 | 镜像按需只打入 `third_party/ctm` | 需要 d2l 源码时移除 `.dockerignore` 中对应排除项 |

## 7. 扩缩与接入真实 LLM

- 无状态 HTTP + 单引擎内 `threading.Lock` 串行化推演；水平扩容直接多副本 + 负载均衡，场景内化请求带粘性或在 reason 前重新 internalize。
- 接真实 LLM：用 HF 因果模型替换 `TinyBaseModel`（层命名 `model.layers[*].mlp.*_proj`），
  `GPMConfig.dims` 由 `infer_dims_from_model` 自动探测；详见 README「接入真实 LLM」。

## 2.9 线部署补充（v2.9.0–v2.9.3）

- 正式件：`checkpoints/predictor_v2.9.0.pt`（52191 参数，口径同 v2.8.0）；构建命令
  `make ckpt290`（或 `python3 scripts/build_v290_checkpoint.py`，--quick 小规模复现）。
- 新增服务端点（均依赖已加载预测器，未训练 409、非法输入 400）：
  - `POST /retarget/convert`：body `{actions, source, target}`，source/target 为预设名
    （arm_7dof/gripper_4dof/prime_u_60dof）或形态 dict；
  - `POST /affordance/score`：body `{state, objects, reach_radius?}`，返回 scores/best_part。
- A/B 与基准产物：`benchmarks/results/retarget_ab_v2.9.0.json`、
  `v29_feature_ab.json`、`feature_latency_v2.9.0.json`。
- 向后兼容：v2.1.0..v2.9.0 共 12 件 checkpoint 全量可加载。

## 3.0 线部署补充（v3.0.0–v3.0.3）

- 正式件：`checkpoints/predictor_v3.0.0.pt`（52191 参数，口径同 v2.9.0）；构建命令
  `make ckpt300`（或 `python3 scripts/build_v300_checkpoint.py`，--quick 小规模复现）。
- 新增服务端点（均依赖已加载预测器，未训练 409、非法输入 400）：
  - `POST /future/predict`：body `{window, scene_params?, horizon?}`，返回
    rgb/depth/mask 三模态代理及 shapes；
  - `GET /eval/5d`：返回五维分数 + composite（**UDOS 内部基准，非 PhysBrain**）。
- 评测与基准产物：`benchmarks/results/future_multimodal_ab_v3.0.0.json`、
  `five_dim_evolution_v3.0.0.json`、`feature_latency_v3.0.0.json`；五维演化命令 `make eval5d`。
- 向后兼容：v2.1.0..v3.0.0 共 13 件 checkpoint 全量可加载。

## 3.1 线（ActionPiece 离散动作 token 化）
- 正式件：`checkpoints/predictor_v3.1.0.pt`（52191 参数，口径同 v3.0.3）；构建命令
  `make ckpt310`（或 `python3 scripts/build_v310_checkpoint.py`，--quick 小规模复现）。
- `udos/action_piece.py`：ActionPieceTokenizer / Codec（粗+细两级）/ TokenizedActionPredictor
  （n-gram next-token）/ TokenActionDecoder（线性插值平滑+限位）/ SequenceCurriculum /
  InContextActionPrompter；**全部为推理外挂，opt-in 默认关，主 CTM 52191 参数不变**。
- PhysicalLoop opt-in：`PhysicalLoopRunner(..., use_tokenized=True, tokenizer=...,
  token_predictor=...)`；默认 `use_tokenized=False` 时与历史逐位一致。
- 新增服务端点（未训练 409、非法输入 400）：
  - `POST /action/tokenize`：body `{actions:[N,D]}`，返回 `{tokens, shape, codebook_size}`；
  - `POST /action/detokenize`：body `{tokens:[]}`，返回 `{actions, shape}`。
- 基准产物：`benchmarks/results/action_piece_ab_v3.1.0.json`、
  `feature_latency_v3.1.0.json`、`training_v3.1.0.json`。
- 向后兼容：v2.1.0..v3.1.0 共 15 件 checkpoint 全量可加载。
- **analogy, not reproduction**：k-means 码本在合成动作上拟合，非 PhysBrain ActionPiece 复现。

## 3.2 线（Ego360 启发数据增强 · 长上下文 · 时间记忆）
- 正式件：`checkpoints/predictor_v3.2.0.pt`（52191 参数，口径同 v3.1.0）；构建命令
  `make ckpt320`（或 `python3 scripts/build_v320_checkpoint.py`，--quick 小规模复现）。
- 新增模块：`udos/ego_data.py`（SyntheticEgoAugmenter / MultiViewGenerator）、
  `udos/extended_context.py`（ExtendedContextWindow）、`udos/temporal_memory.py`（TemporalMemory）、
  `udos/incontext.py`（InContextLearner）、`udos/longhorizon.py`（LongHorizonRollout）；
  **全部为推理外挂，opt-in 默认关，主 CTM 52191 参数不变**。
- PhysicalLoop opt-in：`PhysicalLoopRunner(..., use_memory=True, memory=..., icl_examples=...)`；
  默认关时整环输出与 `predict_next` 逐位一致。
- 新增服务端点（未训练 409、非法输入 400）：
  - `POST /augment/generate`：body `{window:[W,6], view_rotate_deg, ...}`，无状态不需模型；
  - `POST /icl/predict`：body `{window, examples:[], scene_params}`，few-shot 预测。
- 基准产物：`benchmarks/results/ego_augment_ab_v3.2.0.json`、
  `feature_latency_v3.2.0.json`、`training_v3.2.0.json`。
- 性能（CPU 2 线程, n_iters=30）：predict 3.19ms / rollout(4) 12.28ms / augment 0.05ms / icl 3.26ms。
- 向后兼容：v2.1.0..v3.2.0 共 16 件 checkpoint 全量可加载。
- **analogy, not reproduction**：合成状态序列上的视角/增强代理，非 Ego360/真机视频复现。

## 3.3 线（架构精炼 · 效率优化 · 鲁棒性收口）
- 正式件：`checkpoints/predictor_v3.3.0.pt`（`make ckpt330`）与 `predictor_v3.3.3.pt`
  （`make ckpt333`），均 52191 参数、默认旧架构；新能力全部 opt-in。
- 基准产物：`benchmarks/results/training_v3.3.0.json`、`training_v3.3.3.json`、
  `efficiency_pareto_v3.3.0.json`、`integration_report_v3.3.1.json`。
- 向后兼容：v2.1.0..v3.3.3 共 18 件 checkpoint 全量可加载。
- **analogy, not reproduction**：残差/MoE/蒸馏/剪枝/鲁棒性均为合成数据上的机制类比验证。

## 3.3.4 hardening 运维补充（v3.3.4）

### 日志运维配置
- 全量切换标准库 `logging`（`udos/logging_config.py`），**42 个 udo 模块**接入；
  默认级别 **WARNING**，输出到 **stderr**，ISO 时间格式 `%(asctime)s %(levelname)s %(name)s %(message)s`。
- 用环境变量调整级别：`UDOS_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR`；服务进程 `main()` 固定 INFO。
- **红线**：日志只打 shape/摘要统计，不打完整张量 repr / 请求体 / 用户路径；日志走 stderr，
  不进 stdout、不串入 `/metrics` 与 HTTP 响应体。
- `/metrics` 始终为纯 Prometheus 文本（`Content-Type: text/plain`），与日志流分离。
- 运维取日志：容器 `docker logs <container>` 即得（stderr 合并到容器日志流）；
  调 `UDOS_LOG_LEVEL=INFO` 看 internalize/save/load/calibration/在线自适应摘要。

### 健康检查
- `GET /health` 返回 `status`/`version`（v3.3.4）/`predictor_trained`/`internalized_scenes`；
  `Dockerfile` 与 `docker-compose.yml` 已配置 `HEALTHCHECK`（15s/5s/start-period 20s/retries 3）。
- v3.3.4 起 `/health` 读 `internalized_scenes` 走服务锁（DIAG-008 修复），并发下安全。
- 错误路径契约：未训练需预测器的端点返回 **409**（含 `GET /eval/5d`，DIAG-001 已修）；
  非法输入返回 **400**（`/icl/predict`、残缺 PCE scene，DIAG-003/004 已修）；未知路由 404。
- 进程韧性：任意请求触发 5xx 后进程不退出，`/health` 仍 200（ThreadingHTTPServer + 每请求兜底）。


## v3.4 ICM 上下文记忆运维

- **端点**：`POST /icm/demo/register`（注册演示到服务级记忆库）、`POST /icm/predict`（演示条件化预测）。未训练/记忆库空 → 409；非法输入 → 400；未知路由 → 404；异常 → 500 不崩进程。
- **零梯度**：ICM 不改主权重，主件仍 `predictor_v3.4.0.pt`；ICM 延迟仅比 0-shot 高 ~0.4ms（3.34→3.72ms，2 线程 CPU），满足实时预算。
- **日志**：沿用 `udos/logging_config.py`，默认 stderr，`UDOS_LOG_LEVEL` 可配，不污染 HTTP 响应体。
- **backcompat**：v2.1.0..v3.4.0 共 19 件 checkpoint 全部可加载。

## v3.5 SFM 空间基础模型运维

- **端点**：`POST /spatial/query`（`op`=range/box/raycast/los）、`POST /spatial/collision`。请求体带 `objects` 列表（`object_id`/`position[3]`/`velocity[3]`/`radius`）临时构造场景；缺省走服务注册场景。非法输入 → 400；无可用场景 → 409；未知路由 → 404；异常 → 500 不崩进程。
- **零外挂**：SFM 为纯解析几何（numpy），不依赖主 predictor 是否训练；不改 52191 参数。
- **延迟**：50 物体合成场景单次查询亚毫秒~数毫秒（见 `benchmarks/results/feature_latency_v3.5.0.json`）。
- **日志**：沿用 `udos/logging_config.py`，默认 stderr，不污染 HTTP 响应体。
- **backcompat**：v2.1.0..v3.5.0 共 21 件 checkpoint 全部可加载；正式件 `predictor_v3.5.0.pt`（eval_mse=0.045556，与 v3.4.5 逐位一致）。

## v3.8 全域调度 / 数字孪生 / 多体协同运维

- **端点**：`POST /twin/scene`（`n_agents`/`n_obstacles`/`seed`/`bounds` 创建或查询合成孪生场景）、`POST /twin/step`（多体冲突消解+积分一步；可选 `window` 在已挂 predictor 时附跑一步 `ClosedLoopOrchestrator`）。未建场景 → 409；非法参数 → 400；未知路由 → 404；异常 → 500 不崩进程。
- **零外挂**：多体/调度/闭环/孪生均为纯推理外挂（确定性算法或复用既有无学习/外挂模块），不进主 state_dict、不改 52191 参数；默认输出逐位一致，新能力 opt-in。
- **analogy, not reproduction**：多体为合成参数化代理，数字孪生为合成场景，非真机多机器人。
- **延迟**：见 `benchmarks/results/feature_latency_v3.8.0.json`；多体 A/B 见 `multi_agent_ab_v3.8.0.json`；全特性组合见 `comprehensive_eval_v3.8.0.json`。
- **backcompat**：v2.1.0..v3.8.6 共 25 件 checkpoint 全部可加载。



## v4.3 完全自进化线运维（终点 v4.3.9）

- **端点**：`POST /self-evolution/search`（配置自优化搜索）、`POST /self-evolution/ab`（搜索后 vs 默认 A/B）、`POST /self-evolution/long-horizon`（长程闭环）。未挂 predictor → 409；非法参数（horizon/n_sub 越界）→ 400；未知路由 → 404；异常 → 500 不崩进程。
- **零外挂**：配置搜索/A/B/长程闭环均为纯前向编排、零梯度、opt-in，不进主 state_dict、不改 52191 参数；日志沿用 logging_config（默认 stderr），不污染 HTTP 体与 /metrics。
- **基准**：`make self-evolution-bench` → `benchmarks/results/self_evolution_v43.json`；正式件 `predictor_v4.3.0.pt`（第 32 代）、`predictor_v4.3.9.pt`（第 33 代）。
- **backcompat**：v2.1.0..v4.3.9 共 33 件 checkpoint 全部可加载；主参恒 52191、eval_mse=0.045556。

## v4.4 多智能体协作线运维（终点 v4.4.1）

- **端点**：`POST /collab/select`（决策树选型，返回 topology+可解释理由）、`POST /collab/run`（按 topology=star/chain 执行；mesh 需 allow_mesh=true 否则 400）、`POST /collab/handoff`（TransferBundle 五要素校验交接，残缺 400）、`GET /collab/trace/{id}`（责任链查询；不存在 404、路径穿越 400）。
- **错误语义**：非法输入 → 400；未知路由 → 404；异常 → 500 不崩进程。日志沿用 logging_config（默认 stderr），不污染 HTTP 体与 /metrics；JSON 端点纯 JSON，/metrics 纯 Prometheus 文本。
- **安全默认**：swarm/mesh 默认关（须显式 opt-in）；无 owner 或无 stop condition 不得启动；同任务认领锁去重。
- **基准**：`make collab-ab` → `benchmarks/results/collab_ab_v44.json`。
- **backcompat**：checkpoint 代数维持 33 件（本线零正式训练）；主参恒 52191、eval_mse=0.045556、旧锚点 md5 逐位未变。
