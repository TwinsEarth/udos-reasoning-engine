# UDOS v7.3 AI 可观测性部署（Prometheus + Grafana，开源自建）

证据级：引擎内的追踪/指标/脱敏/尾采样/确定性评估为 **verified**（见
`tests7/test_v73_observability.py`）；本目录的容器栈配置有效，但**真实容器部署、
抓取、GPU/eBPF 采集、LLM-as-judge 在 CPU 沙箱未运行**，标 **unverified-on-infra**，
需你提供 Docker 主机 / GPU 节点 / LLM key。

## 1. 引擎暴露 /metrics（内置 stdlib exporter，无需第三方库）
```bash
PYTHONPATH=. python3 - <<'PY'
from udos7.observability.metrics import MetricsRegistry, start_prometheus_exporter
import time
r = MetricsRegistry()                 # 生产中应传入服务进程持有的同一 registry
start_prometheus_exporter(r, "0.0.0.0", 8778)
time.sleep(10**9)
PY
curl http://127.0.0.1:8778/metrics
```

## 2. 起监控栈
```bash
cd deploy/observability
docker compose up -d      # Prometheus :9090, Grafana :3000(admin/admin)
```
Linux 若用 `host.docker.internal`，在 prometheus 服务加
`extra_hosts: ["host.docker.internal:host-gateway"]`，或把 `prometheus.yml`
target 改为宿主 IP。

## 3. 六步落地路径（与需求对应）
1. 统一语义：`udos7/observability/semconv.py`（OTel GenAI 约定，版本化契约）。
2. 双层采集：SDK 精细埋点（Tracer/run_pipeline/observe_goal）+ 代理/Collector
   （OTel Collector 部署属 infra 闸门）；零代码：`UDOS_OTEL_AUTO=1`
   `enable_autoinstrument()` 自动埋点协调器 run。
3. 冷热存储：引擎内 ring/JSONL（热）；Prometheus/Loki/Tempo（冷，容器栈）。
4. 持续评估：`OnlineEvaluator`（确定性 judge verified；LLM-as-judge 需 key，门禁）。
5. 智能告警：SLO 燃尽率 `alert_burn_rate` + `alerts.yml`（TTFT/ITL/错误率）。
6. 隐私治理：`redact()` 邮箱/手机/身份证脱敏（等价 Redaction Processor）。

## 4. 双轨模式
- OpenTelemetry 轨（基础设施）：本目录 Prometheus/Grafana；追踪可接 Tempo/Jaeger
  （引擎 JSONL 已含 trace_id/span_id/parent_id，可做 OTLP 导出适配）。
- LLM 专用轨（模型层）：Langfuse/Phoenix 负责 prompt/评估管理；通过 OTLP 粘合。
  真实接入与 ClickHouse/Postgres 后端属 infra 部署，未在本沙箱运行。

## 5. 明确不冒充
GPU 利用率/显存、KV 命中率、eBPF 无侵入采集、跨节点通信剖析需 GPU 节点 / 内核权限；
LLM-as-judge、在线毒性商业模型需 API key。`resource_usage()` 在无 GPU 时
`gpu_*=null` 且 evidence=cpu-proto。
