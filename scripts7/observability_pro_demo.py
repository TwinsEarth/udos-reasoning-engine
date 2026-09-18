"""v7.3.3 可观测加深层端到端演示（零第三方依赖，CPU 可跑）。

覆盖六个生产功能层中 v7.3.0 尚缺的部分：
Session 关联 / Guardrails 实时拦截 / 成本归因 / 流式异常+根因 /
拓扑+火焰图 / OTLP-JSON 导出。产物写入 reports7/observability_pro_demo.*。
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.observability import (  # noqa: E402
    AnomalyDetector, CostAccounting, Guardrails, MetricsRegistry, Tracer,
    export_otlp_json, flame_profile, resource_usage, session_index, trace_topology)
from udos7.observability.semconv import GEN_AI_REQUEST_MODEL  # noqa: E403

OUT = os.path.join(os.path.dirname(__file__), "..", "reports7")


def main() -> None:
    tr = Tracer()
    reg = MetricsRegistry()
    guard = Guardrails(registry=reg)
    cost = CostAccounting(registry=reg)
    anom = AnomalyDetector(warmup=5, k=3.0)

    sessions = ["s1", "s1", "s2"]
    blocked = 0
    for i, sid in enumerate(sessions):
        with tr.span(f"request-{i}", session_id=sid,
                     **{GEN_AI_REQUEST_MODEL: "default", "udos.user": f"u{i%2}",
                        "udos.feature": "rag"}) as root:
            root.record_tokens(120 + i * 10, 80)
            with tr.span("llm.generate"):
                pass
            out = "正常回答" if i != 2 else "联系 a@b.com"  # 第 3 条触发 PII 护栏
            res = guard.inspect({"output": out}, root)
            if not res.allowed:
                blocked += 1
            cost.record("default", 120 + i * 10, 80, user=f"u{i%2}", feature="rag")

    # 流式延迟异常检测 + 根因
    alerts = []
    for v in [120, 130, 125, 128, 122, 130, 126, 124, 900]:
        a = anom.observe("ttft_ms", v)
        if a:
            alerts.append(a)
    tid = "rc1"
    from udos7.observability.tracing import Span
    rootc = Span("root", tid, None)
    rootc.end_ns = rootc.start_ns + int(100 * 1e6)
    slow = Span("retrieval", tid, rootc.span_id)
    slow.end_ns = slow.start_ns + int(80 * 1e6)
    rc = AnomalyDetector.root_cause(rootc, [slow])

    otlp_path = os.path.abspath(os.path.join(OUT, "observability_pro_demo.otlp.json"))
    n_otlp = export_otlp_json(tr, otlp_path)

    report = {
        "version": "7.3.3",
        "evidence": "verified (cpu, zero-3rd-party)",
        "sessions": list(session_index(tr).values()),
        "guardrail_blocks": blocked,
        "guardrail_metric_series": {
            str(k): v for k, v in
            reg.counter("gen_ai_guardrail_blocks_total").series().items()},
        "cost_report": cost.report(),
        "anomaly_alerts": alerts,
        "root_cause": rc,
        "topology_nodes": list(trace_topology(tr.spans).keys()),
        "flame_rows": flame_profile(tr.spans),
        "otlp_spans_exported": n_otlp,
        "otlp_path": otlp_path,
        "resource": resource_usage(),
    }
    os.makedirs(OUT, exist_ok=True)
    p = os.path.abspath(os.path.join(OUT, "observability_pro_demo.json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"sessions": len(report["sessions"]),
                      "guardrail_blocks": blocked,
                      "anomaly_alerts": len(alerts),
                      "root_cause_stage": rc["stage"],
                      "otlp_spans": n_otlp,
                      "platform": report["resource"]["platform"],
                      "rss_mb": report["resource"]["rss_mb"],
                      "report": p}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
