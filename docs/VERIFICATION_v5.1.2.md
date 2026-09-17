# UDOS v5.1.2 — 验证报告

> release decision: **go**

## 新增（v5.1.1→v5.1.2）
- fuse/cascade/infinity 固定 seed 可复算函数（融合省重算/级联缩主模型输入/infinity 命中率+HBM 常驻下降）。
- POST /kvcache/cost（盈亏平衡 retain-vs-recompute，手算例 breakeven=0.4）。
- GET /kvcache/metrics（纯文本 Prometheus）。
- GET /intel/infrastructure（始终 200，gpu=false/qat=ENV_BLOCKED/disclaimer）。

## curl 取证（on）
- sim/run 200；cost `{tier:ddr,hit:0.2,recall:10,recompute:4}` → retain, breakeven=0.4；cost 未知介质→400。
- /kvcache/metrics Content-Type=text/plain。
- /intel/infrastructure → gpu_available=false, qat=ENV_BLOCKED。
- predictions/score 回归 0.09。
- UDOS_KVCACHE=off → /kvcache/* 全 503。

## 测试/覆盖率
- **1558 passed / 0 failed**（1552+6 只增）。
- TOTAL = **12211 stmts / 868 miss = 93%**（≤870 余量）。

## 零训练硬验收
主参 52191、eval_mse 0.045556、33 代 checkpoint、四锚点逐位不变；
全树 grep 历史口令/姓名=0。

## 缺口
GPU/QAT 未在无硬件环境验证；所有数字为 CPU 合成 seed 机制原型。
