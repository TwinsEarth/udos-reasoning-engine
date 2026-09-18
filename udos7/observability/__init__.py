"""UDOS v7.3 AI 可观测性层（零第三方依赖，CPU 可真跑）。

五大层映射（证据分级见 docs7/OBSERVABILITY_v7.3.md）：
- 模型内部/行为：tracing 记录置信度、决策路径、Agent 状态（verified，引擎内）。
- 推理执行/运营：metrics 记录 TTFT/ITL/Token/工具/错误/资源，Prometheus 暴露（verified）。
- 语义与治理：redaction(PII 脱敏)、尾采样、在线质量评估（确定性 judge verified；
  LLM-as-judge 需 API key，门禁 GateError，unverified）。
Prometheus/Grafana/OTel Collector/eBPF/GPU 的真实部署见 deploy/observability（unverified-on-infra）。
"""
from .semconv import (SEMCONV_VERSION, PIPELINE_STAGES,
                      GEN_AI_SYSTEM, GEN_AI_REQUEST_MODEL, GEN_AI_OPERATION_NAME,
                      GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS,
                      GEN_AI_PROMPT, GEN_AI_COMPLETION,
                      EXT_AGENT_NAME, EXT_AGENT_PURPOSE, EXT_AGENT_STATUS,
                      EXT_STAGE, EXT_EVIDENCE)
from .tracing import Span, Tracer, redact, tail_sample_keep
from .metrics import (Counter, Gauge, Histogram, MetricsRegistry,
                      resource_usage, render_prometheus,
                      burn_rate, alert_burn_rate, SLIS,
                      start_prometheus_exporter)
from .evaluation import (FormatJudge, GroundingJudge, ToolCorrectnessJudge,
                         ToxicityJudge, LLMJudge, OnlineEvaluator, QualityScore)
from .instrument import (run_pipeline, AgentRegistry, observe_goal,
                         enable_autoinstrument, get_tracer, disable_autoinstrument)
from .intelligence import (session_index, Guardrails, GuardrailResult,
                           CostAccounting, AnomalyDetector,
                           trace_topology, flame_profile,
                           to_otlp_json, export_otlp_json)
from .metrics import network_io_counters

__all__ = [
    "SEMCONV_VERSION", "PIPELINE_STAGES",
    "Span", "Tracer", "redact", "tail_sample_keep",
    "Counter", "Gauge", "Histogram", "MetricsRegistry", "resource_usage",
    "render_prometheus", "burn_rate", "alert_burn_rate", "SLIS",
    "start_prometheus_exporter",
    "FormatJudge", "GroundingJudge", "ToolCorrectnessJudge", "ToxicityJudge",
    "LLMJudge", "OnlineEvaluator", "QualityScore",
    "run_pipeline", "AgentRegistry", "observe_goal",
    "enable_autoinstrument", "disable_autoinstrument", "get_tracer",
    "GEN_AI_SYSTEM", "GEN_AI_REQUEST_MODEL", "GEN_AI_OPERATION_NAME",
    "GEN_AI_USAGE_INPUT_TOKENS", "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_PROMPT", "GEN_AI_COMPLETION",
    "EXT_AGENT_NAME", "EXT_AGENT_PURPOSE", "EXT_AGENT_STATUS",
    "EXT_STAGE", "EXT_EVIDENCE",
    "network_io_counters",
    "session_index", "Guardrails", "GuardrailResult", "CostAccounting",
    "AnomalyDetector", "trace_topology", "flame_profile",
    "to_otlp_json", "export_otlp_json",
]
