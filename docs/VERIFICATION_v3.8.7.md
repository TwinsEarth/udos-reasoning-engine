# UDOS 推演引擎 v3.8.7 — 验证报告

> 日期：2026-09-15
> 版本：v3.8.7（fix/hardening，基于 v3.8.6）
> 环境：Python 3.12.11 / torch 2.14.0+cpu / 2 CPU 线程

---

## 1. 测试与覆盖率

| 指标 | 值 |
|---|---|
| 总用例数 | 1184 |
| 通过 | 1184 |
| 失败 | 0 |
| 跳过 | 0 |
| exit code | 0 |
| 总覆盖率 | 93%（8002 stmts / 566 miss） |
| 新增测试文件 | `tests/test_v387_hardening.py`（24） |

## 2. Checkpoint 兼容性

| checkpoint | md5 | 状态 |
|---|---|---|
| predictor_v3.8.6.pt（主件） | `7351250ac00db53c321b919a951c640b` | 逐位不变 ✓ |
| predictor_v3.4.5.pt | `52993ca743416e6d822cdad78743c397` | 逐位不变 ✓ |
| predictor_v3.3.3.pt | `f993bcbdd476473c28dd4604bbbe11d6` | 逐位不变 ✓ |
| 其余 22 代 | — | 全部可加载 ✓ |

- 主参数恒 **52191**；eval_mse 恒 **0.045556**。
- 未产出新 checkpoint；未重训；未改主权重。

## 3. HTTP 端点矩阵

### 3.1 正常路径（200）

| 端点 | 实测 |
|---|---|
| GET /health | 200, version=3.8.7 |
| GET /demo | 200 |
| GET /checkpoints | 200 |
| GET /experiments | 200 |
| GET /eval/5d | 200（已训练） |
| GET /metrics | 200, text/plain |
| POST /predict | 200 |
| POST /spatial/query | 200 |
| POST /spatial/collision | 200 |
| POST /wm/imagine | 200 |
| POST /wm/conservation | 200 |
| POST /neural/step | 200 |
| POST /neural/reflex/log | 200 |
| POST /twin/scene | 200 |
| POST /twin/step | 200 |
| POST /icm/demo/register | 200 |
| POST /icm/predict | 200 |

### 3.2 错误路径

| 用例 | 期望 | 实测 |
|---|---|---|
| 缺 window / 非法 JSON | 400 | ✓ |
| null 数值字段（FIX-301） | 不 500 | ✓（用默认值） |
| 字符串数值字段 | 400 | ✓ |
| 未训练需 predictor | 409 | ✓ |
| ICM 空记忆库 | 409 | ✓ |
| 无空间场景 | 409 | ✓ |
| 未知路由 | 404 | ✓ |
| 路径穿越 ../ | 400 | ✓ |
| 绝对路径 | 400 | ✓ |
| 进程在 5xx 后存活 | 200 /health | ✓ |

## 4. 已知限制

- 无 docker daemon：未执行容器 build/run。
- 无 GPU：全部性能数据为 CPU 2 线程。
- 性能优化：未发现可证 ≥5% 热点，无 ACCEPT 候选落地。
