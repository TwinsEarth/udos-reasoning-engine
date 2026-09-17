# UDOS v5.1.0 KV Cache 分层卸载 — 机制原型

> analogy, not reproduction。CPU-only 纯逻辑/零梯度/零训练/零新权重。
> 用固定种子合成 trace + 显式参数化设备模型, 不下载权重、不伪造 GPU/QAT 加速。

## 模块（udos/kvcache/）
block_store（多级后端 HBM→DDR→SSD→remote）/ paged_cache（PagedAttention 块表+前缀共享）/
policy（LRU/LFU/TTL 冷热）/ compression（zlib/zstd 无损, QAT 仅契约 ENV_BLOCKED）/
cost_ledger（保留vs重算账本）/ fuse（重叠段融合）/ cascade（lite 预筛+召回下限）/
infinity（长任务滑窗）/ device（GPU/CPU 抽象, 仅 CPU 验证）/ sim（合成 trace 基准）。

## 端点
- `POST /kvcache/sim/run`（配置→模拟结果）、`GET /kvcache/state`
- opt-in：`UDOS_KVCACHE=on`；off 返回 503，不影响主预测循环逐位不变。

## A/B 证据链（benchmarks/results/kvcache_ab.json 回算）
- baseline（小窗口8） vs candidate（分层窗口128），同 seed=7/3000 token。
- 命中率 +0.494、重算单位 -1481、**ACCEPT**。
- 压缩 roundtrip 无损；QAT=ENV_BLOCKED 不返回伪造加速。

## 事实核实（厂商口径与 UDOS 自测严格分开）
- **Qwen3-8B（官方 config）**：36 层 / 8 KV 头 / head_dim 128 / BF16，原生 32768、YaRN 131072。
  每 token KV = 2(K+V)×36×8×128×2B = **147456 B = 144 KiB ≈ 147 KB（十进制）**（复算成立，两口径同源）。
- Intel KV Shrink/Fuse/Cascade/Infinity、QAT（至强）、重排压缩收益、
  80%命中 TTFT≈5×、QAT≈软件压缩2× 等数字——均为 **Intel/联合实验室特定条件口径**，
  标注为厂商口径，**非 UDOS 自测结果**。UDOS 仅做机制原型类比。

## 边界与替换
- 有真 GPU/QAT 时替换 device.py 后端与 compression 的 QAT 路径；
  本沙箱 GPU 路径显式"未在无GPU环境验证"。
- 带宽/延迟均为显式假设参数，可替换实测。

## 复现
```bash
UDOS_KVCACHE=on python -m udos.server --port 8000 --preset small
curl -X POST localhost:8000/kvcache/sim/run -d '{"n_tokens":2000}'
curl localhost:8000/kvcache/state
```

## v5.1.2 增补
- fuse/cascade/infinity 补固定 seed 可复算函数：融合省 recompute_units、级联主模型输入缩小比例、infinity 命中率+HBM 常驻下降，均 analogy 标注。
- POST /kvcache/cost：retain-vs-recompute 盈亏平衡（手算例 recall=10/recompute=4→breakeven=0.4）。
- GET /kvcache/metrics：纯文本 Prometheus 风格（text/plain）。
- GET /intel/infrastructure：始终 200，gpu_available=false、qat=ENV_BLOCKED、disclaimer 非厂商数字。
