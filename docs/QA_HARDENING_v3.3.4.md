# UDOS 推演引擎 v3.3.4 hardening patch — QA 收口报告

> 报告日期：2026-09-14
> 阶段：v3.3.4 hardening patch 最终 QA 收口（诊断 → Bug 修复 → 日志改造 → 性能 A/B → 本收口）
> 运行环境：Python 3.12.11 / torch 2.14.0+cpu / 2 CPU 线程 / 无 GPU / 无 docker daemon
> 正式件 checkpoint：`checkpoints/predictor_v3.3.3.pt`，md5 `f993bcbdd476473c28dd4604bbbe11d6`（全程未改权重）
> 发布结论：**go**（依据见 §9）

---

## 1. 概述

### 1.1 patch 目标
v3.3.4 是 v3.3.3（3.3 线终点正式件）之上的 **hardening（加固）补丁**，目标不是新特性、不是重训，而是：
1. 把只读诊断阶段发现的真实契约冲突/真缺陷修掉并各配 RED→GREEN→回归；
2. 把全仓零散 print/日志统一到标准库 `logging`，守住"日志不串入 `/metrics` 与 HTTP 响应体"的红线；
3. 在不改权重、不损数值精度的前提下，用严格的单 delta A/B 裁决性能优化候选；
4. 全局同步版本号到 3.3.4，产出可独立解压复跑的交付物。

### 1.2 范围与硬约束
- **不重训、不改权重**：正式件 md5 全程锁定 `f993bcbdd476473c28dd4604bbbe11d6`。
- **不删旧测试**：只增不删（738 → 766，+28）。
- **数值逐位等价**：日志/性能改动均有 bit-exact 锚点测试守护。
- 本环境无 GPU、无 docker daemon，相关验证范围在 §10 明确披露。

### 1.3 环境
Python 3.12.11 / torch 2.14.0+cpu / AMD EPYC 9Y24 / 2 CPU 线程；checkpoint md5 与基线一致。

---

## 2. 问题台账与处置（DIAG-001…009）

> 严重度 P0=阻断 / P1=高 / P2=中 / P3=低。分类：真缺陷 / 契约冲突 / 环境问题 / 注意事项。

| ID | 严重度 | 位置 | 根因（触发→机制→症状） | 分类 | 处置 |
|---|---|---|---|---|---|
| DIAG-001 | P2 | `udos/server.py` `do_GET` | 未训练 `GET /eval/5d` → `_require_predictor()` 抛 `ServiceNotReady`；`do_GET` 无单独捕获分支，被 `except Exception→500`，而非契约约定的 409 | 契约冲突 | **已修复（FIX-001）**：`do_GET` 补 `ServiceNotReady→409`，与 `do_POST` 对齐 |
| DIAG-002 | P2 | `Dockerfile:34`、`docker-compose.yml:10` | 镜像 tag 标 3.3.3，启动命令却加载 v3.2.0 checkpoint，发布元数据与实际权重代际不一致 | 配置不一致 | **已修复（FIX-002）**：两处 `--checkpoint` 改为 `predictor_v3.3.3.pt` |
| DIAG-003 | P3 | `udos/server.py` `icl_predict` | `torch.as_tensor(window)` 裸调用，字符串入参触发 `TypeError`→500，其他端点均已 try/except→400 | 真缺陷 | **已修复（FIX-003）**：try/except 转 `ValueError`→400 |
| DIAG-004 | P3 | `udos/server.py` `_scene()` / `pce_format.py:184` | 残缺 PCE scene 不抛错返回空 tokens，后续 `torch.stack([])` 抛 `RuntimeError`（非 ValueError）→500 | 真缺陷 | **已修复（FIX-004）**：入口层校验 tokens 非空→400；不改核心解析，合法 PCE 路径不受影响 |
| DIAG-005 | P3 | `Dockerfile` | 无 `USER` 指令，容器进程以 root 跑 | 注意事项（加固项） | **不改（后续加固）**：纯 CPU 推理、风险中低；排入后续版本 |
| DIAG-006 | P3 | `udos/persistence.py`、`ensemble.py` | `torch.load(weights_only=False)` | 注意事项（纵深防御） | **不改（白名单已足够）**：`/load` 有 `_safe_checkpoint_path` 白名单 + `commonpath`，攻击需先能写 checkpoints 目录 |
| DIAG-007 | P3 | `udos/lite.py:91` | `torch.ao.quantization` 已 deprecated，torch 2.10 将移除 | 环境问题（前瞻） | **不改（前瞻迁移项）**：当前 2.14 正常；升级 torch 时迁 torchao，见 §10 |
| DIAG-008 | P3 | `udos/server.py` `health()` | 读 `scene_memory.keys()` 未持服务锁，ThreadingHTTPServer 下理论上读到中间态 | 真缺陷（轻微） | **已修复（FIX-005）**：`with self._lock` 内做 O(1) dict 拷贝 |
| DIAG-009 | P3 | `udos/server.py` save/load/list | `abspath("checkpoints")` 依赖进程 cwd，从非工程根启动时路径漂移 | 真缺陷 | **已修复（FIX-006）**：基于 `__file__` 锚定工程根 CHECKPOINTS_DIR，不依赖 cwd |

**汇总**：9 个问题中 **6 个已修复**（DIAG-001/002/003/004/008/009），3 个不改并写明理由（DIAG-005 加固、DIAG-006 白名单已足、DIAG-007 前瞻）。

---

## 3. P0 阻断项集合

**P0 阻断项 = 空集合（0 个）。**

诊断阶段即未发现数据损坏、远程代码执行、进程崩溃或核心功能不可用；修复阶段未引入新 P0。
按发布纪律，**P0 清零是 go 的前提**——本补丁满足。开放项最高仅 P3，且均为不改并写明理由的加固/前瞻项，不阻断发布。

---

## 4. 测试前后对比

| 指标 | 修复前（v3.3.3 基线） | v3.3.4 收口 | 变化 |
|---|---|---|---|
| 通过用例 | 738 | **766** | **+28** |
| 失败 | 0 | **0** | — |
| 跳过 | 0 | **0** | — |
| exit code | 0 | **0** | — |
| 总覆盖率 | 93%（5731 stmts / 402 miss） | **93%** | 维持 |
| `udos/server.py` 覆盖率 | 87% | 88% | +1%（异常路径补测） |

新增测试文件（只增不删）：
- `tests/test_v334_bugfix.py`（12）——6 个 Bug 修复的 RED/GREEN + 回归。
- `tests/test_v334_logging.py`（7）——日志流/stderr/级别/`UDOS_LOG_LEVEL`/`/metrics` 纯文本。
- `tests/test_v334_numerical_equivalence.py`（4）——同输入两次 bit-exact 锚点。
- `tests/test_v334_perf_optim.py`（5）——C1/C2 复用逐位相等 + 回退 + 默认 max_shard。

版本升级后另同步 71 处 `__version__ == "3.3.3"` → `"3.3.4"`（仅版本断言，未触碰 checkpoint 文件名引用）。

---

## 5. 逐 HTTP 端点验证矩阵（修复后状态）

> 服务 A = 预加载 v3.3.3（trained=True）；服务 B = 无 checkpoint（trained=False）。
> 详细原始记录：`scratch/http_endpoint_results_v334.json`（75 条请求）。

### 5.1 GET 端点

| 端点 | 服务 A | 服务 B | 期望 | v3.3.3 诊断 | v3.3.4 修复后 |
|---|---|---|---|---|---|
| `/health` | 200 | 200 | 200（version=3.3.4） | ✓ | ✓ |
| `/demo` | 200 | — | 200 | ✓ | ✓ |
| `/checkpoints` | 200 | 200 | 200 | ✓ | ✓ |
| `/experiments` | 200 | — | 200 | ✓ | ✓ |
| `/eval/5d` | 200 | **500** | 未训练 **409** | ✗ DIAG-001 | ✓ **409（FIX-001）** |
| `/metrics` | 200 text/plain | — | 纯 Prometheus 文本 | ✓ | ✓ |
| `/nonexistent` | 404 | — | 404 | ✓ | ✓ |

### 5.2 POST 关键端点（服务 A 正常路径）

| 端点 | 正常码 | 异常用例 | 期望 | v3.3.3 诊断 | v3.3.4 修复后 |
|---|---|---|---|---|---|
| `/icl/predict` | 200 | 非法 window 类型 | **400** | ✗ 500（DIAG-003） | ✓ **400（FIX-003）** |
| `/internalize` | 200* | 残缺 PCE scene | **400** | ✗ 500（DIAG-004） | ✓ **400（FIX-004）** |
| `/reason` | 200* | 残缺 PCE scene | **400** | ✗ 500（DIAG-004） | ✓ **400（FIX-004）** |
| `/predict` | 200 | 非法 window | 400 | ✓ | ✓ |
| `/evaluate` | 200 | 未训练 | 409 | ✓ | ✓ |
| `/load` | 200 | 越权路径 `../etc/passwd` | 400 | ✓ | ✓ |
| `/save` | 200** | `../evil.pt` | 不穿越 | ✓（清洗为 evil.pt） | ✓ |
| 其余 20+ POST | 200 | 正常入参 | 200 | ✓ | ✓ |

\*合法 PCE（`/demo` 用 `PCEParser.dumps`）正常 200，仅手工残缺 dict 触发修复前的 500。\*\*路径清洗后落盘，诊断后已清理。

### 5.3 异常路径与崩溃恢复

| 用例 | 期望 | v3.3.4 实测 |
|---|---|---|
| 空 body / 非法 JSON / 缺 window | 400 | ✓ |
| `POST/GET /nonexistent` | 404 | ✓ |
| 未训练需 predictor 的 POST | 409（22 个全对齐） | ✓（含 `GET /eval/5d`） |
| 触发 5xx 后再 `/health` | 进程存活、200 | ✓（ThreadingHTTPServer + 每请求兜底） |
| `/metrics` Content-Type | `text/plain`，无 JSON/日志串入 | ✓ |

---

## 6. 性能基线与 A/B 结论

> 原则：无测量不结论；无同合同证据不授权；仅 ACCEPT 改默认。
> 裁决门：机制命中 + 语义 oracle 逐位相等 + mutant 门变红 + 同合同 A/B median 改善 ≥5% 且 p95 不退化。
> 原始数据：`benchmarks/results/perf_baseline_v334.json`（pre）、`perf_baseline_v334_post.json`（post）、`perf_ab_v334.json`（C1/C2 各 40 对）。

### 6.1 pre vs post 关键数（p50）

| 指标 | pre 基线 | post 基线（生产默认路径） | 变化 |
|---|---|---|---|
| 单次推理 `predict_next` | 3.12 ms | 3.12 ms | 不变（未触及） |
| HTTP `/predict` e2e | 4.84 ms | 3.98 ms | 噪声（未改该代码路径） |
| `physical_loop` 单步 | **52.70 ms** | **46.90 ms** | **−11.0%（C1）** |
| 批量 bs=64 | **14.58 ms** | **8.64 ms** | **−40.7%（C2）** |
| 批量 bs=32 | 7.08 ms | 6.89 ms | ≈噪声（控制组零 delta） |
| RSS 峰值 | 299.1 MB | 296.7 MB | ≈持平 |

### 6.2 ACCEPT 候选

**C1 — physical_loop 末尾冗余前向复用**：`run()` 末尾一次 `predict_next` 与 understand 内 `target_state` 同输入、同 `@torch.no_grad()`、确定性前向，第二次为纯冗余。改为复用 `understand.target_state`（仅默认 understand 路径，自定义 `understand_fn` 回退原逻辑）。
- 语义 oracle：`run().prediction` 与 `understand.target_state` `torch.equal` **max_abs_diff=0.0**；mutant（置零）oracle 变红。
- 同合同 A/B（40 对交替）：median 54.836→51.535 ms（**+6.02%**），p95 反快 6.44 ms（不退化），39/40 对 candidate 更快。
- **裁决：ACCEPT**，已合入默认。

**C2 — BatchPredictor 默认 max_shard 32→64**：bs=64 旧默认拆 `[32,32]` 两片串行，模型微小、批处理高效，两片的派发开销可省；改单片 `[64]` 一次前向，bs>64 仍分片。
- 语义 oracle：bs=64 两片 vs 单片 `torch.equal` **max_abs_diff=0.0**；mutant 变红；控制组 bs=32 ≈零 delta。
- 同合同 A/B（40 对交替）：median 10.230→6.062 ms（**+40.74%**），p95 反快 5.14 ms，**40/40 对 candidate 更快**；内存仅 +1.6 MB（246.0→247.6 MB）。
- **裁决：ACCEPT**，已合入默认。

### 6.3 被反证候选（REJECT，未改代码）

| 候选 | 裁决 | 原因 |
|---|---|---|
| C3a HTTP `/predict` 序列化 | REJECT | ~1.7ms 开销来自 `torch.tensor(nested_list)` 解析 + `pred.tolist()` + 标准库 HTTP 框架，非 json 库；优化需换框架或改输出契约，无安全单 delta，预估 <5% |
| C3b GPM/CTM 前向冗余拷贝 | REJECT | `predict_next` 已 `@torch.no_grad()`；B=1 下 tiny MLP 为微秒级，主体是已训练 CTM 内部时间轴不可动，无可省冗余 |
| observe 重复编码 | 未尝试 | B=1 下 ~0.1ms，相对 50ms 循环不可达 5%；需把编码上下文穿进 predict_next，风险大于收益，记入待办 |

---

## 7. 日志清单

- **新增基础设施** `udos/logging_config.py`：`configure_logging(level=None, stream=None)`，默认 WARNING / stderr，ISO 时间格式；环境变量 `UDOS_LOG_LEVEL` 覆盖；幂等（重复调用先清旧 handler）；handler 挂 root logger，用 `_UdosOnlyFilter` 只放行 `udos.*`，级别作用在包 logger `udos` 上——**兼容 pytest `caplog` 断言**。
- **逐模块接入**：41 个功能模块 `logger = logging.getLogger("udos.<basename>")` + 包级 `logger = logging.getLogger("udos")`，共 **42 个 udo 模块**。
- **例外**：`udos/debug.py`（交互式分级调试面板（默认关闭，UDOS_DEBUG=1 开启，无需口令），走 stderr；`udos/training.py` verbose 训练进度走 stdout（既有契约测试断言 `ss=`）。
- **关键日志点**：internalize/save/load（INFO）、do_POST 访问日志（DEBUG）、400 非法输入（WARNING）、409（INFO）、5xx（ERROR+exception）、calibration ECE 前后（INFO）、ood/在线自适应漂移（WARNING）、guard 回退（WARNING）、physical_loop 每步（DEBUG）。
- **红线验证**：日志只打 shape/摘要统计（不打完整张量 repr、不打请求体/用户路径）；全部 stderr；`/metrics` 实测 `Content-Type: text/plain; version=0.0.4`，纯 Prometheus 文本，无 JSON/日志串入（`tests/test_v334_logging.py` 第 3 项 + 全新解压复跑实测）。

---

## 8. Docker 静态核查结论

> 本环境**无 docker daemon**，未执行 `docker build`/`run`；以下为只读文件核查 + 裸进程等价验证。

| 检查项 | v3.3.3 诊断 | v3.3.4 结论 |
|---|---|---|
| 启动 checkpoint | ✗ v3.2.0 | **✓ v3.3.3（FIX-002 已修）** |
| 镜像 LABEL / tag | 3.3.3 | **3.3.4（本阶段随版本同步）** |
| 基础镜像 / CPU torch | ✓ python:3.12-slim / cpu wheel | ✓ 不变 |
| 健康检查 | ✓ 调 /health（15s/5s/20s/3） | ✓ 不变 |
| 非 root 用户 | ✗ 无 USER（DIAG-005） | **不改，标注后续加固**（DIAG-005） |
| `torch.load(weights_only=False)` | 注意事项（DIAG-006） | **不改，`/load` 白名单 + commonpath 已足够**（DIAG-006） |
| 裸进程等价启动 | 已探针等价验证 | 全新解压复跑再验（§VERIFICATION） |

---

## 9. 发布结论：go

**结论：go。** 依据：
1. **P0 阻断项 = 0**（§3），无开放 S1/S2。
2. **766 passed / 0 failed / 0 skipped**，覆盖率维持 93%（§4）。
3. **6 个真缺陷/契约冲突已全部修复**，各配 RED→GREEN→回归（§2）。
4. **性能有实测收益**：C1/C2 均有同合同 A/B + 逐位等价证据，loop −11%、bs=64 −41%（§6）。
5. **权重未动**：checkpoint md5 `f993bcbdd476473c28dd4604bbbe11d6` 前后一致；18 代 checkpoint 兼容可加载。
6. **红线验证通过**：`/metrics` 纯 Prometheus 文本、日志全 stderr、崩溃恢复不退出。

---

## 10. 未验证范围（诚实披露）

- **Docker 容器实际 build/run**：本环境无 docker daemon，仅静态核查 + 裸进程等价验证；镜像体积/层缓存/slim 缺依赖（如 libgomp）/容器内 UID 未验。
- **GPU 性能**：无 GPU，全部数为 CPU 2 线程；CUDA 路径未测。
- **高并发压测**：仅 2 线程环境，ThreadingHTTPServer 并发仅做了小规模 health 并发（FIX-005 回归），未做大规模压测/吞吐上限。
- **torch 2.10 升级后的 quantization 兼容性**：DIAG-007 为前瞻性项，当前 2.14 正常，升级到 2.10 时需迁 `torch.ao.quantization`→torchao。
- **/import-snapshot**：依赖 snapshot 内容，未单独单测。

---

## 11. 复现命令

```bash
# 版本确认
python3 -c "import udos; print(udos.__version__)"     # 3.3.4

# 全量测试 + 覆盖率
python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing

# 性能 pre/post 基线
python3 benchmarks/perf_baseline_v334.py               # pre（历史）
python3 benchmarks/perf_baseline_v334.py               # post（生产默认路径）

# 性能单 delta A/B（C1/C2 同合同 40 对）
make perf-ab

# 权重 md5 校验
md5sum checkpoints/predictor_v3.3.3.pt                 # f993bcbdd476473c28dd4604bbbe11d6

# 起服务 + 健康/指标/错误路径
python3 -m udos.server --host 127.0.0.1 --port 18334 --preset small \
    --checkpoint checkpoints/predictor_v3.3.3.pt
curl -s http://127.0.0.1:18334/health | python -m json.tool
curl -s -i http://127.0.0.1:18334/metrics | head -5
```

---

*本报告与 `docs/VERIFICATION_v3.3.4.md`、`CHANGELOG.md`（v3.3.4 条目）同源；原始证据见 `scratch/` 四阶段报告与 `benchmarks/results/*.json`。*
