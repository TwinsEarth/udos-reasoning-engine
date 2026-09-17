# UDOS 引擎 v2.5 版本规划 —— 效率/工程化与可服务性主线

> 起点：v2.4.16（2.4 线全量绿）
> 终点：**v2.5.2**（对外发布，正式件 `predictor_v2.5.2.pt`）
> 训练节点：**2.5.0** 与 **2.5.2** 各重建一次正式件；2.5.1 不重训。
> 铁律：v2.3.1→v2.4.x 的全部测试持续全绿；新增能力默认不改变旧默认输出；版本号同步且有断言。

## 2.5 线 3 节点清单

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 18 | **2.5.0** | 批量推理引擎 + 推理缓存 + 正式训练重建 | `udos/batch.py`：`BatchPredictor`（批量序列推理，自动 padding + mask，支持大 batch 分片）；`udos/cache.py`：`InferenceCache`（LRU 基于输入哈希的结果缓存，默认关闭 opt-in）；`PhysicsPredictor` 新增 `predict_batch`；正式训练重建 `predictor_v2.5.0.pt` + `training_v2.5.0.json` | `tests/test_v25_batch.py`：批量与逐笔一致、padding mask、空 batch 守卫；`tests/test_v25_cache.py`：命中返回相同、miss 计算、LRU 淘汰、缓存关闭透传 |
| 19 | 2.5.1 | 服务监控指标 + 轻量导出/无状态快照 + 回滚 | `udos/server.py` 新增 `GET /metrics`（请求计数/延迟分位/缓存命中率/OOD 触发率，Prometheus 文本格式）；`udos/persistence.py` 新增 `export_snapshot`（无状态 JSON 快照：配置+校准器+残差分位，不含权重）/ `import_snapshot`；`POST /rollback`（回滚到上一已加载 checkpoint，白名单）；`/evaluate` 含 metrics 段 | `tests/test_v25_metrics.py`：/metrics 200、字段存在、计数递增；`tests/test_v25_snapshot.py`：export/import 配置一致、无权重；`tests/test_v25_rollback.py`：回滚后预测一致、无历史 409 |
| 20 | **2.5.2** | 最终训练重建 + 全量验证 + 打包发布 | 正式训练重建 `predictor_v2.5.2.pt`（front 默认，52191 参数量级）+ `training_v2.5.2.json`（训练/校准/区间指标）；全量 pytest 全绿报总数/覆盖率；旧三版 checkpoint（v2.1.0/v2.2.1/v2.3.1）向后兼容加载测试；真实起 HTTP 服务逐接口验证（含新增接口 200/400/409）；`docs/VERIFICATION_v2.5.2.md`；打包 `udos-engine-v2.5.2.zip`；从 zip 独立解压到 /tmp 复跑验证 | 全量测试通过；zip 内 checkpoint md5 与工程一致；/health=2.5.2；/evaluate 含 calibration/interval 段 |

## 设计约束

1. **批量推理**：结果必须与逐笔 `predict` 逐位一致（浮点容差 1e-5）；padding 位置不参与计算。
2. **推理缓存**：默认关闭；开启后命中必须返回与未命中完全相同的输出；缓存 key 包含输入张量哈希+模型参数哈希。
3. **监控指标**：纯标准库实现，不引入 prometheus_client；线程安全计数。
4. **无状态快照**：仅导出配置/校准器/分位等非权重状态，用于快速恢复推理后处理；权重仍走 .pt。
5. **回滚**：维护已加载 checkpoint 栈；回滚到上一个；无历史时 409。
6. **零重依赖**：同 2.4 线约束。
7. **证据诚实**：批量/缓存若在小模型上收益不明显，照实写文档（CPU 小模型推理本身快，缓存主要收益在重复请求场景）。

## 最终验收清单（v2.5.2）—— ✅ 全部完成（2026-09-13）

- [x] `udos/__init__.py` / `pyproject.toml` / `Makefile` / `docker-compose.yml` / `Dockerfile` 版本号均为 2.5.2
- [x] 全量 pytest 全绿，报总数与覆盖率 —— **218 passed / 92%**
- [x] `checkpoints/predictor_v2.5.2.pt` 存在，参数量 ~52191（可训练 52191；state_dict 含 48 buffer 共 52239）
- [x] `benchmarks/results/training_v2.5.2.json` 含训练/校准/区间指标
- [x] 旧 checkpoint（v2.1.0/v2.2.1/v2.3.1/v2.4.0/v2.5.0）可加载且预测正常
- [x] HTTP 服务真实启动，/health 返回 2.5.2，/evaluate 含 calibration/interval 段
- [x] 新增接口 /detect-ood /metrics /rollback 均验证 200/400/409
- [x] `docs/VERIFICATION_v2.5.2.md` 诚实呈现正反证据与已知限制
- [x] `udos-engine-v2.5.2.zip` 打包，排除 __pycache__/.pytest_cache/.git/.pyc
- [x] 从 zip 独立解压到 /tmp 复跑：版本==2.5.2、pytest 全绿、build --quick 可复现、服务起得来
- [x] zip 内 checkpoint md5 与工程内正式件一致

> 详见 `docs/VERIFICATION_v2.5.2.md`。v2.5.2 为 2.5 线（20 节点）终点正式发布件。
