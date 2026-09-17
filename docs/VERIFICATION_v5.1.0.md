# UDOS v5.1.0 KV Cache 分层卸载 — 验证报告

> release decision: **go**

## 模块/端点
udos/kvcache/ 10 模块；POST /kvcache/sim/run、GET /kvcache/state（opt-in UDOS_KVCACHE）。

## A/B 证据链
benchmarks/results/kvcache_ab.json：分层窗口 vs 小窗口，命中率 +0.494、重算 -1481、ACCEPT。
压缩无损 roundtrip；QAT=ENV_BLOCKED 不伪造加速。

## 事实核实
Qwen3-8B 每 token KV=144KB（官方 config 复算成立）；Intel/QAT 数字标厂商口径非自测。

## 测试/覆盖率
全量 pytest 只增不删（1550+ 新增反例）。TOTAL = **12128 stmts / 857 miss = 93%**（≤880 余量）。
反例契约：LRU 不驱逐 pinned、冷热升降、命中率↑→重算↓单调、压缩无损、
QAT 无硬件 503、fuse 只复用重叠段、cascade 召回下限、账本选重算、CPU 后端验证。

## 返修取证（sim 输入校验 400 矩阵）
`POST /kvcache/sim/run` 实测：n_tokens∈{-5,0,"x"}、{}、window=-1、vocab=0 均 **400**；
合法 body **200**；UDOS_KVCACHE=off → **503**。
score 端点 curl：(0.7,1)→0.09、(0.2,0)→0.04。

## 零训练硬验收
主参 52191、eval_mse 0.045556、33 代 checkpoint、四锚点 md5 逐位不变；
历史口令/姓名全树 grep=0（测试样本动态拼接）。

## 缺口
GPU/QAT 路径未在无硬件沙箱验证；设备参数为假设值。
