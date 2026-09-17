# UDOS 推演引擎 v3.8.7 hardening patch — QA 收口报告

> 报告日期：2026-09-15
> 阶段：v3.8.7 hardening patch 最终 QA 收口（诊断 → Bug 修复 → 日志加固 → 性能基线 → 本收口）
> 运行环境：Python 3.12.11 / torch 2.14.0+cpu / 2 CPU 线程 / 无 GPU / 无 docker daemon
> 正式件 checkpoint：`checkpoints/predictor_v3.8.6.pt`，md5 `7351250ac00db53c321b919a951c640b`（全程未改权重）
> 发布结论：**go**（依据见 §9）

---

## 1. 概述

### 1.1 patch 目标
v3.8.7 是 v3.8.6（全域调度线终点正式件）之上的 **hardening（加固）补丁**，完全对标 v3.3.4 加固专项五工作线：
1. 只读诊断建基线：全量 pytest + --cov 固化 1160/93% 起点；逐模块静态排查 17 个新模块；真实起服务逐端点打正常+异常+畸形请求；
2. Bug 修复：每个真缺陷先写 RED 复现测试再做原因级补丁并补对称回归；
3. 日志：统一走 udos/logging_config，关键路径加分级日志与 caplog 断言；
4. 性能：先落 baseline JSON，逐个候选做单 delta A/B，仅 ACCEPT 才落地；
5. QA 收口：升版 3.8.7 全来源同步，产出可独立解压复跑的交付物。

### 1.2 范围与硬约束
- **不重训、不改权重**：25 代 checkpoint md5 全程锁定。主件 `predictor_v3.8.6.pt` md5 `7351250ac00db53c321b919a951c640b`。
- **不删旧测试**：只增不删（1160 → 1184，+24）。
- **数值逐位等价**：日志/辅助改动均有 caplog 与逐位锚点测试守护。
- 本环境无 GPU、无 docker daemon，相关验证范围在 §10 明确披露。

### 1.3 环境
Python 3.12.11 / torch 2.14.0+cpu / 2 CPU 线程；checkpoint md5 与基线一致。

---

## 2. 问题台账与处置（DIAG-001…005）

> 严重度 P0=阻断 / P1=高 / P2=中 / P3=低。分类：真缺陷 / 契约冲突 / 环境问题 / 注意事项。

| ID | 严重度 | 位置 | 根因（触发→机制→症状） | 分类 | 处置 |
|---|---|---|---|---|---|
| DIAG-001 | P2 | `udos/server.py` `do_POST` | 新端点（twin/wm/icm）直接 `int(body.get(...))`/`float(body.get(...))`；当 JSON 值为 `null`（Python None）时 `int(None)` 抛 `TypeError`，不被 `except ValueError` 捕获，落入 `except Exception` → **500 而非 400** | 真缺陷 | **已修复（FIX-301）**：do_POST 异常处理扩展为 `except (ValueError, TypeError)`；新增 `_coerce_int`/`_coerce_float` 辅助方法；新端点统一使用 |
| DIAG-002 | P3 | `udos/server.py` `neural_step` | `candidate_actions` 仅校验 `isinstance(list)`，未校验每项为 dict；非 dict 项（如 `[1,2,3]`）传入 MPC 后 `"state_perturbation" in 1` 抛 TypeError | 真缺陷（轻微） | **已修复（FIX-301）**：增加每项须为 dict 的显式校验 → 400 |
| DIAG-003 | P3 | 17 个新模块 | 全部模块已有 `logger = logging.getLogger("udos.<name>")` 但未实际调用（0 log calls） | 注意事项 | **部分加固**：server 层懒初始化路径（WM/神经控制器/孪生场景）补 INFO 日志；模块内算法路径不补（纯前向、无可观测副作用，日志价值低） |
| DIAG-004 | P3 | `udos/spatial.py:46,59` | `except Exception as e` 包裹 `np.asarray` 后 re-raise ValueError | 注意事项 | **不改（正确模式）**：这是输入校验包装，非吞异常；合法 PCE 路径不受影响 |
| DIAG-005 | P3 | `udos/icm.py:289` | `except KeyError` 缓存未命中时回退到 `cache_residual` | 注意事项 | **不改（正确模式）**：KeyError 是缓存 miss 的正常控制流，非吞异常 |

**汇总**：5 个问题中 **2 个已修复**（DIAG-001/002 → FIX-301），3 个不改并写明理由（DIAG-003 部分加固、DIAG-004/005 正确模式）。

---

## 3. P0 阻断项集合

**P0 阻断项 = 空集合（0 个）。**

诊断阶段未发现数据损坏、远程代码执行、进程崩溃或核心功能不可用；修复阶段未引入新 P0。按发布纪律，**P0 清零是 go 的前提**——本补丁满足。开放项最高仅 P3。

---

## 4. 测试前后对比

| 指标 | 修复前（v3.8.6 基线） | v3.8.7 收口 | 变化 |
|---|---|---|---|
| 通过用例 | 1160 | **1184** | **+24** |
| 失败 | 0 | **0** | — |
| 跳过 | 0 | **0** | — |
| exit code | 0 | **0** | — |
| 总覆盖率 | 93%（8002 stmts / 566 miss） | **93%** | 维持 |
| `udos/server.py` 覆盖率 | 88% | 88% | 维持 |

新增测试文件（只增不删）：
- `tests/test_v387_hardening.py`（24）——FIX-301 RED/GREEN + 回归 + 日志 caplog 断言。

版本升级后另同步 126 个文件中的 `__version__ == "3.8.6"` → `"3.8.7"`（仅版本断言，未触碰 checkpoint 文件名引用）。

---

## 5. 逐 HTTP 端点验证矩阵

> 服务 A = 预加载 v3.8.6（trained=True，端口 18387/18389）；服务 B = 无 checkpoint（trained=False，端口 18388）。

### 5.1 GET 端点

| 端点 | 服务 A | 服务 B | 期望 | v3.8.7 实测 |
|---|---|---|---|---|
| `/health` | 200 | 200 | 200（version=3.8.7） | ✓ |
| `/demo` | 200 | — | 200 | ✓ |
| `/checkpoints` | 200 | 200 | 200 | ✓ |
| `/experiments` | 200 | — | 200 | ✓ |
| `/eval/5d` | 200 | **409** | 未训练 409 | ✓ |
| `/metrics` | 200 text/plain | — | 纯 Prometheus 文本 | ✓ |
| `/nonexistent` | 404 | — | 404 | ✓ |

### 5.2 POST 关键端点（服务 A 正常路径）

| 端点 | 正常码 | 异常用例 | 期望 | v3.8.7 实测 |
|---|---|---|---|---|
| `/predict` | 200 | 缺 window / 非法 JSON / 错型 | 400 | ✓ |
| `/spatial/query` | 200 | 缺 center/radius / 无场景 | 400 / 409 | ✓ |
| `/spatial/collision` | 200 | 缺 objects | 400 | ✓ |
| `/wm/imagine` | 200 | 坏 horizon / null horizon | 400 / 200(默认) | ✓ |
| `/wm/conservation` | 200 | 坏 source / null mass | 400 / 200(默认) | ✓ |
| `/neural/step` | 200 | 坏 candidate_actions / 非 dict 项 | 400 | ✓ |
| `/neural/reflex/log` | 200 | — | 200 | ✓ |
| `/twin/scene` | 200 | null n_agents / 坏字符串 | 200(默认) / 400 | ✓ |
| `/twin/step` | 200 | 坏 dt / null dt | 400 / 200(默认) | ✓ |
| `/icm/demo/register` | 200 | 缺字段 | 400 | ✓ |
| `/icm/predict` | 200 | 空记忆库 / null k | 409 / 200(默认) | ✓ |
| `/load` | 200 | `../etc/passwd` / 绝对路径 | 400 | ✓ |

### 5.3 异常路径与崩溃恢复

| 用例 | 期望 | v3.8.7 实测 |
|---|---|---|
| 空 body / 非法 JSON / 缺 window | 400 | ✓ |
| `POST/GET /nonexistent` | 404 | ✓ |
| 未训练需 predictor 的 POST | 409 | ✓ |
| 触发 5xx 后再 `/health` | 进程存活、200 | ✓ |
| `/metrics` Content-Type | `text/plain`，无 JSON/日志串入 | ✓ |
| null 输入（FIX-301） | 不 500，用默认值或 400 | ✓ |

---

## 6. 性能基线与 A/B 结论

> 原则：无测量不结论；无同合同证据不授权；仅 ACCEPT 改默认。
> 原始数据：`benchmarks/results/perf_baseline_v387.json`。

### 6.1 baseline 关键数（p50）

| 指标 | p50 | 说明 |
|---|---|---|
| 单次推理 `predict_next` | 3.12 ms | 回归锚点（与 v3.3.4 一致） |
| WM `imagine` rollout | 3.69 ms | 3.6 新路径 |
| 分层控制 `neural.step` | 0.06 ms | 3.7 新路径（纯 Python 规则） |
| 多体调度 `resolve` (5 agents) | 0.17 ms | 3.8 新路径（O(n²) 几何） |
| 数字孪生 `twin.step` | 0.24 ms | 3.8 新路径 |
| HTTP `/predict` e2e | 5.00 ms | 标准库 HTTP 框架开销 |
| HTTP `/wm/imagine` e2e | 12.31 ms | 含 WM 懒初始化+拟合 |
| HTTP `/neural/step` e2e | 1.17 ms | |
| RSS 峰值 | 326.4 MB | |

### 6.2 候选裁决

| 候选 | 裁决 | 原因 |
|---|---|---|
| C1: HTTP 序列化 `.tolist()` 优化 | REJECT | 单次响应仅 12-72 个 float，序列化 <0.1ms；主体是 CTM 前向 3ms，无可省冗余 |
| C2: WM rollout 冗余前向 | REJECT | `imagine` 第 0 步逐位锚定 `predict_next`（契约要求），不可省；H>1 步外挂 MLP 已 `@torch.no_grad()` |
| C3: 多体调度 O(n²) 优化 | REJECT | n=5 时 0.17ms，相对主路径 3ms 占比 <6%；优化需空间索引结构，风险大于收益 |
| C4: neural_step 缓存优化 | REJECT | 0.06ms 已是 Python 规则计算，无可省冗余 |

**结论：无 ACCEPT 候选。** 新路径（3.5-3.8）均为纯 Python/NumPy 几何规则或微型 MLP 前向，在 2 CPU 线程环境下无可证 ≥5% 热点。不修改任何默认路径。

---

## 7. 日志清单

- **基础设施** `udos/logging_config.py`（v3.3.4 已建）：默认 stderr、分级、`UDOS_LOG_LEVEL` 可配、与 pytest caplog 兼容。
- **17 个新模块**：全部已有 `logger = logging.getLogger("udos.<name>")`（DIAG-003）。
- **本次新增关键日志**：
  - WM 懒初始化：`logger.info("WM latent model lazily initialized, n_params=%d")`
  - 神经控制器懒初始化：`logger.info("HierarchicalController lazily initialized")`
  - 孪生场景创建：`logger.info("DigitalTwinScene created: n_agents=%d n_obstacles=%d seed=%d bounds=%.1f")`
  - ICM demo 注册（既有）：`logger.info("ICM demo registered, memory size=%d")`
- **HTTP 错误日志**（既有）：400 WARNING、409 INFO、5xx ERROR+exception。
- **红线验证**：日志只走 stderr；`/metrics` 实测 `Content-Type: text/plain` 纯 Prometheus 文本，无 JSON/日志串入（`tests/test_v387_hardening.py` 第 22 项 capsys 断言）。

---

## 8. Docker 静态核查结论

> 本环境**无 docker daemon**，未执行 `docker build`/`run`；以下为只读文件核查。

| 检查项 | v3.8.6 | v3.8.7 |
|---|---|---|
| 镜像 LABEL / tag | 3.8.6 | **3.8.7（随版本同步）** |
| 基础镜像 / CPU torch | ✓ python:3.12-slim / cpu wheel | ✓ 不变 |
| 健康检查 | ✓ 调 /health | ✓ 不变 |

---

## 9. 发布结论：go

**结论：go。** 依据：
1. **P0 阻断项 = 0**（§3），无开放 S1/S2。
2. **1184 passed / 0 failed / 0 skipped**，覆盖率维持 93%（§4）。
3. **1 个真缺陷已修复**（FIX-301），配 RED→GREEN→回归（§2）。
4. **性能无退化**：predict_next p50 3.12ms 与 v3.3.4 基线一致；新路径均 <4ms（§6）。
5. **权重未动**：25 代 checkpoint md5 全程锁定；主参数恒 52191。
6. **红线验证通过**：`/metrics` 纯 Prometheus 文本、日志全 stderr、崩溃恢复不退出。

---

## 10. 未验证范围（诚实披露）

- **Docker 容器实际 build/run**：本环境无 docker daemon，仅静态核查。
- **GPU 性能**：无 GPU，全部数为 CPU 2 线程。
- **高并发压测**：仅 2 线程环境，未做大规模压测。
- **性能优化**：未发现可证 ≥5% 热点，无 ACCEPT 候选落地。

---

## 11. 复现命令

```bash
python3 -c "import udos; print(udos.__version__)"     # 3.8.7
python3 -m pytest tests/ -q --cov=udos --cov-report=term-missing
python3 benchmarks/perf_baseline_v387.py
md5sum checkpoints/predictor_v3.8.6.pt                 # 7351250ac00db53c321b919a951c640b
```

---

*本报告与 `docs/VERIFICATION_v3.8.7.md`、`CHANGELOG.md`（v3.8.7 条目）同源。*
