# UDOS 推演引擎 v3.3.4 hardening patch 验收与独立复跑验证报告

> 日期：2026-09-14
> 性质：hardening 补丁（不重训 / 不改权重 / 不删旧测试）。
> 环境：Python 3.12.11 / torch 2.14.0+cpu / 2 CPU 线程 / 无 GPU / 无 docker daemon。
> 正式件 checkpoint：`checkpoints/predictor_v3.3.3.pt`，md5 `f993bcbdd476473c28dd4604bbbe11d6`（全程未变）。

---

## 0. 验收结论（TL;DR）

| 验证项 | 结果 |
|---|---|
| 版本号 | `udos.__version__ == "3.3.4"` ✓ |
| 全量测试 | **766 passed / 0 failed / 0 skipped**，exit 0 ✓ |
| 覆盖率 | **93%**（TOTAL 5888 stmts / 405 miss）✓ |
| 正式件 md5 | `f993bcbdd476473c28dd4604bbbe11d6`（未变）✓ |
| 18 代 checkpoint | 全部可加载 ✓ |
| `/health` 版本 | 返回 `version=3.3.4` ✓ |
| `/metrics` | `Content-Type: text/plain; version=0.0.4`，纯 Prometheus 文本 ✓ |
| 错误路径 | 400 / 404 / 409 各至少一个用例通过 ✓ |
| 日志红线 | 全 stderr、stdout 0 字节、默认 WARNING ✓ |
| 进程韧性 | 多次异常请求后 `/health` 仍 200 ✓ |
| 全新解压复跑 | /tmp 独立目录全部通过 ✓ |

**发布门禁：go。**

---

## 1. 版本验证

```text
$ python3 -c "import udos; print(udos.__version__)"
3.3.4
```

全局版本单一来源已同步（无 `__version__ == "3.3.3"` 残留）：
- `udos/__init__.py`：`__version__ = "3.3.4"`（docstring 标题与「版本:」同步）
- `pyproject.toml`：`version = "3.3.4"`
- `Dockerfile`：`LABEL org.opencontainers.image.version="3.3.4"`
- `docker-compose.yml`：`image: udos-reasoning-engine:3.3.4`
- `Makefile`：`docker-build`/`docker-run` 镜像 tag `udos-reasoning-engine:3.3.4`
- `tests/`：71 处 `__version__ == "3.3.4"`（仅版本断言，未触碰 `predictor_v3.3.3.pt` checkpoint 文件名引用）

---

## 2. 全量测试结果（版本升级后工程根复跑）

命令：`python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing`

| 指标 | 值 |
|---|---|
| 通过 | **766** |
| 失败 | **0** |
| 跳过 | **0** |
| exit code | **0** |
| 总语句 | 5888 |
| miss 语句 | 405 |
| 总覆盖率 | **93%** |
| `udos/server.py` | 88% |
| `udos/debug.py` | 0%（刻意交互式模块，豁免） |

唯一警告：`udos/lite.py` `torch.ao.quantization` DeprecationWarning（DIAG-007，torch 2.10 前瞻迁移项，非失败）。

---

## 3. checkpoint 验证

| 项 | 结果 |
|---|---|
| 正式件 md5 `predictor_v3.3.3.pt` | `f993bcbdd476473c28dd4604bbbe11d6`（与 v3.3.3 基线一致，**未改权重**） |
| checkpoint 总数 | 18（v2.1.0..v3.3.3） |
| 18 代可加载性 | 18/18 全部加载成功（is_calibrated/has_ood 正常，主参数 52191） |

---

## 4. HTTP 服务逐接口验证（真实起停）

> 在**全新解压目录**起两个实例：S1 预加载 v3.3.3（端口 18334，trained=true）；S2 不带 checkpoint（端口 18335，trained=false）。

| 用例 | 期望 | 实测 |
|---|---|---|
| `GET /health`（S1） | 200，`version=3.3.4`，`predictor_trained=true` | ✓ |
| `GET /health`（S2） | 200，`version=3.3.4`，`predictor_trained=false` | ✓ |
| `GET /metrics` | 200，`Content-Type: text/plain; version=0.0.4`，纯 Prometheus 文本（`# HELP/# TYPE/udos_*` 行），无 JSON/无日志串入 | ✓ |
| `POST /predict {"window":"bad"}` | 400 | **400** ✓ |
| `GET /nonexistent` | 404 | **404** ✓ |
| `POST /icl/predict {"window":"not_a_tensor"}`（DIAG-003） | 400 | **400** ✓ |
| `POST /internalize` 残缺 scene（DIAG-004） | 400 | **400** ✓ |
| `GET /eval/5d`（S2 未训练，DIAG-001） | 409 | **409** ✓ |
| `POST /evaluate`（S2 未训练） | 409 | **409** ✓ |
| 错误后韧性：连发 ~8 个非法请求后再 `GET /health` | 进程存活、200 | ✓（version 仍 3.3.4） |

`/health` 实测响应体（S1）：
```json
{ "status": "ok", "service": "udos-reasoning-engine", "version": "3.3.4",
  "predictor_trained": true, "internalized_scenes": [] }
```

---

## 5. 错误路径覆盖矩阵

| 类别 | 用例 | 实测码 |
|---|---|---|
| 400 | `/predict` 非法 window；`/icl/predict` 非法类型；`/internalize` 残缺 scene；`/detect-ood` 坏 sequence；`/future/predict` 错维度 | 400 |
| 404 | `GET /nonexistent` | 404 |
| 409 | S2 `GET /eval/5d`、`POST /evaluate`（未训练） | 409 |
| 5xx 韧性 | catch-all 兜底后进程不退出，`/health` 仍 200 | 存活 |

> 说明：v3.3.4 已把 DIAG-001/003/004 的历史 500 路径分别修正为 409/400；从客户端非法入参已无法稳定触发 500（这正是补丁目的），catch-all 5xx 仍作安全网，进程韧性由 §4 连发异常请求后 `/health` 200 佐证。

---

## 6. 日志验证

| 项 | 实测 |
|---|---|
| 日志流 | 全部 `sys.stderr`，服务 stdout **0 字节**（不污染 stdout / 不串入 HTTP） |
| 默认级别 | WARNING（启动 INFO、400 记 WARNING、完成记 INFO 耗时） |
| 格式 | ISO 时间 `2026-09-14T11:27:29 WARNING udos.server POST /xxx client error (400): ...` |
| `UDOS_LOG_LEVEL` | 环境变量可覆盖（DEBUG/INFO/WARNING/ERROR） |
| `/metrics` 分离 | 纯 Prometheus 文本，日志不进响应体 |

stderr 实测片段：
```
2026-09-14T11:24:45 INFO udos.server UDOS 推理服务启动: http://127.0.0.1:18334 (preset=small, pretrained=True)
2026-09-14T11:27:50 WARNING udos.server POST /future/predict client error (400): window 需为 [W,RAW] 或 [N,W,RAW]
2026-09-14T11:27:50 INFO udos.server POST /future/predict handled in 0.0004s
```

---

## 7. 性能验证（pre vs post，引用 A/B 报告）

| 指标 | pre | post | 变化 |
|---|---|---|---|
| 单次推理 p50 | 3.12 ms | 3.12 ms | 不变 |
| `physical_loop` 单步 p50 | 52.70 ms | **46.90 ms** | **−11%（C1 ACCEPT）** |
| 批量 bs=64 p50 | 14.58 ms | **8.64 ms** | **−41%（C2 ACCEPT）** |
| 批量 bs=32 p50 | 7.08 ms | 6.89 ms | ≈噪声（控制组零 delta） |
| RSS 峰值 | 299.1 MB | 296.7 MB | 持平 |

两项 ACCEPT 均有同合同 A/B（40 对交替）+ `torch.equal` 逐位等价（max_abs_diff=0）+ mutant 门变红证据；被反证候选 C3a/C3b 未改代码。详见 `scratch/perf_ab_report_v334.md` 与 `benchmarks/results/perf_ab_v334.json`。

---

## 8. 全新解压独立复跑（关键，真实执行）

> 目录：`/tmp/udos-engine-v334-verify/udos-engine`（从交付 zip 全新解压，不使用工程根任何缓存）。

| 步骤 | 命令 | 结果 |
|---|---|---|
| 解压 | `unzip -q udos-engine-v3.3.4.zip` 到 /tmp | ✓ |
| 版本 | `python3 -c "import udos; print(udos.__version__)"` | **3.3.4** ✓ |
| 全量测试 | `pytest tests/ -q --cov=udos --cov-report=term` | **766 passed** / 0 failed，覆盖率 **93%**（5888/405）✓ |
| 起服务 S1 | `python -m udos.server --port 18334 --preset small --checkpoint checkpoints/predictor_v3.3.3.pt` | 启动 INFO 走 stderr ✓ |
| `/health` | `curl /health` | 200，`version=3.3.4` ✓ |
| `/metrics` | `curl -i /metrics \| head -5` | `text/plain; version=0.0.4`，纯 Prometheus 文本 ✓ |
| 400 | `POST /predict -d '{"window":"bad"}'` | **400** ✓ |
| 404 | `GET /nonexistent` | **404** ✓ |
| 409 | S2（无 checkpoint）`GET /eval/5d` / `POST /evaluate` | **409** ✓ |
| 韧性 | 多次异常请求后 `GET /health` | 200，进程存活 ✓ |
| 权重 md5 | `md5sum checkpoints/predictor_v3.3.3.pt` | `f993bcbdd476473c28dd4604bbbe11d6` ✓ |
| 文件一致性 | diff（工程根 vs 解压，排除 scratch/缓存） | 仅运行期 `.pyc` 差异，源文件逐字一致 ✓ |
| 清理 | `pkill` 两个服务实例 | 无残留进程 ✓ |

---

## 9. 未验证范围

- Docker 容器实际 `build/run`（本环境无 docker daemon，仅静态核查 + 裸进程等价验证）。
- GPU 性能与 CUDA 路径（无 GPU）。
- 大规模高并发压测（仅 2 线程，ThreadingHTTPServer 并发仅小规模验证）。
- torch 2.10 升级后 `torch.ao.quantization` 兼容性（DIAG-007 前瞻）。

---

## 10. 复现命令

```bash
# 版本确认
python3 -c "import udos; print(udos.__version__)"     # 3.3.4

# 全量回归 + 覆盖率
python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing

# 权重 md5
md5sum checkpoints/predictor_v3.3.3.pt                 # f993bcbdd476473c28dd4604bbbe11d6

# 起服务 + 健康/指标/错误路径
UDOS_LOG_LEVEL=INFO python3 -m udos.server --host 127.0.0.1 --port 18334 \
    --preset small --checkpoint checkpoints/predictor_v3.3.3.pt
curl -s http://127.0.0.1:18334/health | python3 -m json.tool
curl -s -i http://127.0.0.1:18334/metrics | head -5
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:18334/predict \
    -H 'Content-Type: application/json' -d '{"window":"bad"}'   # 400

# 性能 A/B（C1/C2）
make perf-ab
python3 benchmarks/perf_baseline_v334.py
```

---

## 11. 最终门禁结果

- 版本号全局同步 3.3.4：✓
- 766 全绿 / 0 failed / 0 skipped：✓
- 覆盖率 93%：✓
- 权重 md5 未变、18 代兼容：✓
- HTTP 400/404/409 与 /metrics 纯文本：✓
- 日志红线（stderr / stdout 0 / 默认 WARNING）：✓
- 全新解压独立复跑全部通过：✓

**门禁结论：go。**
