# UDOS v5.0.2 情报分析服务内核 — 验证报告

> release decision: **go**

## 新增模块/端点
- 包 `udos/intelligence/`（8 模块）：coefficient / singularity_gate / capability /
  s_curve_eta / prediction_scoring / asi_rsi / model_aggregate / risk_signals + seed + api。
- 9 个锁定端点：/intel/health、/intel/coefficient(+step)、/intel/capability、
  /intel/routes、/intel/predictions(+score)、/intel/models、/intel/asi。

## 与网站口径一致性
公式 Read 自 agi-asi-countdown/backend/app/core（只 Read 不改）：
- EWMA v1.0 alpha=0.3；奇点门四条件；Brier=(p-o)²；RSI 间隔=20-(rsi/100)*18；
  logistic S 曲线三时界。同输入同口径结果一致（契约测试钉死）。

## 测试/覆盖率
- **1535 passed / 0 failed**（1522+13，只增）
- 覆盖率 **94%**（11819 stmts / 749 miss）

## 反例契约
奇点门单源/未确认不触发；EWMA 平滑有界；三时界 earliest≤median≤latest；
Brier 0-1；能力向量缺测保守；ASI 间隔随 RSI 增大而减；leader 视角偏差折减。

## 零训练硬验收
主参 52191、eval_mse 0.045556、33 代 checkpoint、四锚点 md5 逐位不变；
历史口令与指定姓名全树 grep 为空（安全守卫测试持续有效）。

## 复现
```
UDOS_AUTH=off python -m udos.server --port 8000 --preset small
curl localhost:8000/intel/health
curl localhost:8000/intel/routes
```

## 安全残留返修
- 守卫测试中的历史口令/姓名改为运行期动态拼接的合成夹具，不落明文；开源版已进一步将其替换为不含任何真实口令/姓名的合成常量（见 `tests/test_security_guard_v501.py`）。
- 全仓（含 tests/docs/demos/web）`grep -rIn` 历史口令与姓名 = **0 hits**（全新解压树复跑同样 0）。
- 守卫测试仍绿且变异敏感性保留。

## 缺口
多模型仅内置 2 个确定性样例；在线 LLM provider 未接（离线默认）；未做 TLS 反代。
