# UDOS v7.3.0 AI 可观测性体系

在 v7.2.2 协同/军团之上新增**引擎内零依赖可观测层 `udos7/observability/`**。
预测内核与协同层行为不变。证据分级：**verified**（CPU 固定 seed 测试/实测）、
**cpu-proto**（CPU 原型口径）、**unverified-on-infra**（需 Docker/GPU/key，配置已给）。

## 1. 五层架构落点

| 层 | 需求 | 本版实现 | 证据 |
|---|---|---|---|
| 模型内部/行为 | 置信度、决策路径、CoT 可追踪 | `Tracer` 嵌套 Span、协调器 `observe_goal`、事件血缘 | verified |
| 推理执行 | TTFT/ITL/Token/工具/KV/GPU | `MetricsRegistry` AI 指标、资源快照（GPU 留空门禁） | verified / gpu unverified |
| 运营智能 | 四黄金信号→AI 七 SLI、SLO 燃尽率 | `SLIS`、`burn_rate/alert_burn_rate`、Prometheus 暴露 | verified |
| 存储查询 | 指标/追踪/日志 | JSONL 结构化日志、Prom 文本；Tempo/Loki 容器配置 | verified 导出 / 后端 infra |
| 语义治理 | 质量评估、合规、成本、隐私 | `redact` 脱敏、尾采样、`OnlineEvaluator`、LLM-judge 门禁 | verified / LLM unverified |

## 2. 七个小版本功能映射（v7.2.2 → v7.3.0）

1. **Prometheus+Grafana 看板**：`metrics.render_prometheus` + `start_prometheus_exporter`
   （stdlib `/metrics`，verified）；`deploy/observability/` 提供 compose、数据源、
   预置看板、TTFT/ITL/错误率告警（部署 unverified-on-infra）。
2. **思维链/事件实时展示与折叠、六层功能**：Tracing/Event Capture/Metrics/Evaluation/
   Guardrails/Session Correlation——Span 树 + 事件 + trace 会话关联（JSONL 可驱动折叠 UI）。
3. **Agent 名字/用途/状态/进程 + 流水线**：`AgentRegistry`；`run_pipeline` 按
   意图分类→查询重写→检索→重排序→上下文压缩→LLM 生成→事实校验建嵌套 Span。
4. **资源消耗列表**：CPU 核数、RSS 内存（/proc）可测；GPU 利用率/显存、网络、
   API/Token 成本：Token/API 走指标计数，GPU/网络在无权限环境诚实留 null。
5. **全链路 Tracing**：trace→span 父子树，prompt/工具/检索/响应映射为嵌套 span，
   强制 trace_id/span_id/parent_id/latency_ms/token_usage。
6. **在线质量评估**：`OnlineEvaluator` 按采样率在生产流量上跑确定性 judge
   （格式/接地/工具正确性/毒性），质量滚动回归先于投诉告警；LLM-as-judge 需 key（GateError）。
7. **零代码侵入自动埋点**：`UDOS_OTEL_AUTO=1` + `enable_autoinstrument()` 自动包裹
   协调器 run（等价 genai-otel-instrument 的环境变量开关，范围限本引擎协调器）。

## 3. 关键契约
- 语义约定版本化：`SEMCONV_VERSION=gen-ai-semconv-v1-udos1`；标准 `gen_ai.*` 与
  UDOS 扩展 `udos.*` 前缀分离。
- 尾采样：错误 trace、超延迟 trace、命中 PII trace 100% 保留；成功 trace 按 base_rate
  （默认 10%）用 trace_id 稳定哈希采样，同 trace 同留同弃。
- 脱敏：邮箱/手机号/身份证号在 Span 属性与结构化日志导出时确定性替换。
- SLO：`burn_rate=错误率/预算率`，多窗口燃尽率给 page/ticket 两级（14.4/14.4、6/1）。

## 4. 复现
```bash
PYTHONPATH=. python3 -m pytest tests7/test_v73_observability.py -q
PYTHONPATH=. python3 scripts7/observability_demo.py
# 产物 reports7/observability_demo.{json,prom,jsonl}
cd deploy/observability && docker compose up -d   # 需 Docker（infra 闸门）
```

## 5. 不冒充清单
OTel Collector / Jaeger-Tempo / Loki / Grafana 真实容器、eBPF 无侵入、GPU 内核与
KV 命中率、跨节点通信、LLM-as-judge 与商业内容审核：需 Docker 主机/GPU 节点/供应商
key，本 CPU 沙箱不运行，相关代码路径要么留空（gpu=null）要么抛 `GateError`。
