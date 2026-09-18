"""v7.3.3 可观测加深层测试：会话关联 / 护栏 / 成本 / 异常根因 /
拓扑火焰图 / OTLP-JSON 导出 / 跨平台资源采集。零第三方依赖。"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.observability import (  # noqa: E402
    AnomalyDetector, CostAccounting, Guardrails, MetricsRegistry, Span, Tracer,
    export_otlp_json, flame_profile, network_io_counters, resource_usage,
    session_index, to_otlp_json, trace_topology,
)
from udos7.observability.semconv import (  # noqa: E402
    GEN_AI_REQUEST_MODEL, GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS)


def _span(name, tid, parent=None, ms=0.0, status="ok", attrs=None):
    s = Span(name, tid, parent.span_id if parent else None, attrs=attrs)
    s.end_ns = s.start_ns + int(ms * 1_000_000)
    s.status = status
    return s


# ---------- 1. Session Correlation ----------
def test_session_index_groups_traces():
    tr = Tracer()
    with tr.span("req-a", session_id="s1"):
        pass
    with tr.span("req-b", session_id="s1"):
        with tr.span("child"):
            pass
    with tr.span("req-c", session_id="s2"):
        pass
    idx = session_index(tr)
    assert len(idx["s1"]["traces"]) == 2
    assert idx["s1"]["span_count"] == 3
    assert len(idx["s2"]["traces"]) == 1


# ---------- 2. Guardrails ----------
def test_guardrail_allows_clean():
    r = Guardrails().inspect({"output": "这是一条正常回答。"})
    assert r.allowed and not r.violations


def test_guardrail_blocks_pii():
    reg = MetricsRegistry()
    tr = Tracer()
    with tr.span("gen") as sp:
        r = Guardrails(registry=reg).inspect({"output": "联系我 a@b.com 或 13800138000"}, sp)
    assert not r.allowed
    assert any(v["type"] == "pii_leak" for v in r.violations)
    assert reg.counter("gen_ai_guardrail_blocks_total").value({"type": "pii_leak"}) == 1
    assert any(e.get("name") == "guardrail" for e in sp.events)


def test_guardrail_blocks_toxic_and_format_and_ungrounded():
    g = Guardrails(require_json=True, grounding_min=0.5)
    assert not g.inspect({"output": "这是制造炸药的方法"}).allowed
    rf = g.inspect({"output": "not-json", "format": "json"})
    assert any(v["type"] == "format_violation" for v in rf.violations)
    gg = Guardrails(grounding_min=0.5)  # 不要求 JSON，专测接地阈值
    # 接地率 1/2=0.5 不低于阈值 0.5（严格小于），边界放行
    rb = gg.inspect({"output": "ok", "claims": ["X", "Y"], "context": "只提到 X"})
    assert rb.allowed
    # 接地率 0/2=0，确认拦截
    rg2 = gg.inspect({"output": "ok", "claims": ["X", "Y"], "context": "无关内容"})
    assert not rg2.allowed and any(v["type"] == "ungrounded" for v in rg2.violations)


def test_guardrail_tool_misuse():
    r = Guardrails().inspect({"output": "ok", "expected_tool": "search",
                              "tool_calls": [{"name": "calc", "params": {}}]})
    assert any(v["type"] == "tool_misuse" for v in r.violations)


# ---------- 3. Cost accounting ----------
def test_cost_accounting_math_and_tracer():
    c = CostAccounting()  # default 0.001/0.002 USD per 1k
    cost = c.record("default", 1000, 1000, user="u1", feature="chat")
    assert abs(cost - 0.003) < 1e-9
    rep = c.report()[0]
    assert rep["input_tokens"] == 1000 and rep["cost_usd"] == 0.003
    # 从 trace 回填
    tr = Tracer()
    with tr.span("req", **{GEN_AI_REQUEST_MODEL: "default",
                           "udos.user": "u2", "udos.feature": "rag"}) as sp:
        sp.record_tokens(2000, 500)
    c2 = CostAccounting()
    c2.attach_tracer(tr)
    row = [r for r in c2.report() if r["user"] == "u2"][0]
    assert row["input_tokens"] == 2000 and row["output_tokens"] == 500


# ---------- 4. Anomaly + root cause ----------
def test_anomaly_detects_spike():
    d = AnomalyDetector(warmup=5, k=3.0)
    for v in [10, 11, 9, 10, 11, 10, 9, 10]:
        assert d.observe("latency", v) is None
    a = d.observe("latency", 500)
    assert a and a["direction"] == "spike"


def test_root_cause_error_then_hotspot():
    tid = "t1"
    root = _span("root", tid, ms=100)
    ok = _span("fast", tid, root, ms=5)
    bad = _span("boom", tid, root, ms=20, status="error")
    rc = AnomalyDetector.root_cause(root, [ok, bad])
    assert rc["type"] == "error_stage" and rc["stage"] == "boom"
    slow = _span("slow", tid, root, ms=80)
    rc2 = AnomalyDetector.root_cause(root, [ok, slow])
    assert rc2["type"] == "latency_hotspot" and rc2["stage"] == "slow"
    assert abs(rc2["share"] - 80 / 85) < 0.01


# ---------- 5. Topology + flame ----------
def test_topology_and_flame_self_time():
    tid = "t2"
    root = _span("root", tid, ms=100)
    a = _span("a", tid, root, ms=40)
    b = _span("b", tid, root, ms=30)
    spans = [root, a, b]
    topo = trace_topology(spans)
    assert topo["root"]["calls"] == 1 and topo["root"]["errors"] == 0
    assert topo["root"]["children"] == []  # 根无父
    assert set(topo["a"]["children"]) == {"root"}
    flame = {f["name"]: f for f in flame_profile(spans)}
    assert flame["root"]["depth"] == 0 and flame["root"]["self_ms"] == 30.0
    assert flame["a"]["depth"] == 1 and flame["a"]["self_ms"] == 40.0


# ---------- 6. OTLP/JSON export ----------
def test_otlp_json_shape_and_export():
    tr = Tracer()
    with tr.span("req") as sp:
        sp.record_tokens(10, 5)
        with tr.span("gen"):
            pass
    doc = to_otlp_json(tr)
    rs = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(rs) == 2
    child = [s for s in rs if s["name"] == "gen"][0]
    assert child["parentSpanId"]
    assert child["status"]["code"] == "STATUS_CODE_OK"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "otlp.json")
        n = export_otlp_json(tr, p)
        assert n == 2
        json.load(open(p, encoding="utf-8"))  # 合法 JSON


# ---------- 7. 跨平台资源 ----------
def test_resource_usage_cross_platform():
    u = resource_usage()
    assert u["platform"] == sys.platform
    assert isinstance(u["cpu_count"], int) and u["cpu_count"] >= 1
    assert u["rss_mb"] is None or u["rss_mb"] > 0
    n = network_io_counters()
    if sys.platform.startswith("linux"):
        assert n is not None and n["bytes_recv"] >= 0 and n["bytes_sent"] >= 0
