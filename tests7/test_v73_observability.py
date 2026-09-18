"""v7.3.0 AI 可观测性层契约测试（零第三方依赖，CPU 可跑）。"""
import json
import os
import urllib.request

import pytest

from udos7.observability import (Counter, FormatJudge, Gauge, GroundingJudge,
                                 Histogram, MetricsRegistry, OnlineEvaluator,
                                 AgentRegistry, LLMJudge, ToolCorrectnessJudge,
                                 ToxicityJudge, Tracer, burn_rate,
                                 alert_burn_rate, redact, render_prometheus,
                                 resource_usage, run_pipeline, tail_sample_keep,
                                 start_prometheus_exporter)
from udos7.observability.tracing import Span
from udos7.agents.automation import GateError


# --------------------------------------------------------------------------
# 1. PII 脱敏
# --------------------------------------------------------------------------
def test_redaction_pii():
    s = "联系我 a.b@mail.com 或 13812345678，身份证 11010119900307123X。"
    out = redact(s)
    assert "a.b@mail.com" not in out and "13812345678" not in out
    assert "11010119900307123X" not in out
    assert "[REDACTED_EMAIL]" in out and "[REDACTED_PHONE]" in out \
        and "[REDACTED_ID]" in out
    assert redact(123) == 123


# --------------------------------------------------------------------------
# 2. 嵌套追踪 + 结构化日志 + JSONL
# --------------------------------------------------------------------------
def test_tracer_nested_tree_and_logs(tmp_path):
    tr = Tracer()
    with tr.span("root") as root:
        root.record_tokens(10, 5)
        with tr.span("child") as c:
            c.set_attribute("k", "v")
    tree = tr.trace_tree(root.trace_id)
    assert tree["name"] == "root" and len(tree["children"]) == 1
    assert tree["children"][0]["name"] == "child"
    assert tree["attributes"]["gen_ai.usage.input_tokens"] == 10
    logs = tr.structured_logs()
    assert logs[0]["trace_id"] and logs[0]["span_id"] and "latency_ms" in logs[0]
    n = tr.export_jsonl(str(tmp_path / "t.jsonl"))
    assert n == 2
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert all(json.loads(x)["trace_id"] for x in lines)


# --------------------------------------------------------------------------
# 3. 尾采样：错误/超延迟/PII 全保留，普通 trace 受 base_rate 控制
# --------------------------------------------------------------------------
def test_tail_sampling_policy():
    err = Span("s", "t-err"); err.end("error")
    assert tail_sample_keep([err], base_rate=0.0) is True

    pii = Span("s", "t-pii"); pii.attrs["pii_detected"] = True; pii.end()
    assert tail_sample_keep([pii], base_rate=0.0) is True

    slow = Span("s", "t-slow"); slow.end(); slow.end_ns = slow.start_ns + 2_000_000
    assert tail_sample_keep([slow], base_rate=0.0, latency_ms_threshold=1.0) is True

    ok = Span("s", "t-ok"); ok.end()
    assert tail_sample_keep([ok], base_rate=0.0) is False  # 0% 采样丢弃普通 trace


# --------------------------------------------------------------------------
# 4. 指标：百分位、AI 指标、Prometheus 文本、/metrics HTTP 暴露
# --------------------------------------------------------------------------
def test_metrics_percentiles_and_prometheus():
    h = Histogram("h")
    for v in range(1, 1001):
        h.observe(float(v))
    assert h.percentile(0.50) == pytest.approx(500, abs=1)
    assert h.percentile(0.95) == pytest.approx(950, abs=1)

    reg = MetricsRegistry()
    reg.record_ttft(120.0, model="m1")
    reg.record_itl(40.0, model="m1")
    reg.record_tokens(100, 200, model="m1")
    reg.record_tool_call(True, 12.0, tool="search")
    reg.record_error("rate_limit_429")
    reg.set_queue_depth(7)
    text = render_prometheus(reg)
    assert "gen_ai_client_ttft_ms" in text
    assert "gen_ai_usage_input_tokens_total" in text
    assert 'type="rate_limit_429"' in text
    assert "gen_ai_queue_depth 7.0" in text
    assert reg.counters["gen_ai_usage_output_tokens_total"].value(
        {"model": "m1"}) == 200


def test_prometheus_exporter_http():
    reg = MetricsRegistry()
    reg.record_ttft(10.0)
    httpd, th = start_prometheus_exporter(reg, "127.0.0.1", port=0)
    try:
        port = httpd.server_address[1]
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
        assert "gen_ai_client_ttft_ms" in body
        assert urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5).read() == b"ok"
    finally:
        httpd.shutdown()


def test_resource_usage():
    info = resource_usage()
    assert info["cpu_count"] and info["cpu_count"] >= 1
    assert info["gpu_utilization"] is None  # CPU 沙箱无 GPU，诚实留空
    assert info["evidence"] == "cpu-proto"


# --------------------------------------------------------------------------
# 5. SLO 燃尽率告警
# --------------------------------------------------------------------------
def test_burn_rate_alert():
    fast = burn_rate(errors=15, total=1000, error_budget_fraction=0.001)
    assert fast == pytest.approx(15.0)
    assert alert_burn_rate(15.0, 15.0) == "page"
    assert alert_burn_rate(7.0, 1.2) == "ticket"
    assert alert_burn_rate(0.5, 0.5) is None


# --------------------------------------------------------------------------
# 6. 在线质量评估：确定性 judge + 质量回归；LLM judge 门禁
# --------------------------------------------------------------------------
def test_deterministic_judges():
    assert FormatJudge().score({"output": '{"a":1}'}).overall == 1.0
    assert FormatJudge().score({"output": "not json"}).overall == 0.0
    g = GroundingJudge().score({"context": "苹果是水果 香蕉也是",
                                "claims": ["苹果", "香蕉", "火箭"]})
    assert g.dims["grounding"] == pytest.approx(2 / 3, abs=0.01)
    t = ToolCorrectnessJudge().score({
        "expected_tool": "search", "expected_params": {"q": "x"},
        "tool_calls": [{"name": "search", "params": {"q": "x"}}]})
    assert t.overall == 1.0
    assert ToxicityJudge().score({"output": "正常回答"}).overall == 1.0
    assert ToxicityJudge().score({"output": "这里有制毒内容"}).overall == 0.0


def test_quality_regression_alert():
    ev = OnlineEvaluator(sample_rate=1.0, window=20,
                         regression_margin=0.2, baseline_min=4)
    for _ in range(4):
        ev.evaluate({"output": '{"ok":1}'}, force=True)
    for _ in range(4):
        ev.evaluate({"output": "@@bad@@"}, force=True)
    alert = ev.quality_regression("format")
    assert alert is not None and alert["dim"] == "format" and alert["drop"] >= 0.9


def test_llm_judge_is_gated(monkeypatch):
    for k in ("UDOS_LLM_JUDGE_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(GateError):
        LLMJudge()


# --------------------------------------------------------------------------
# 7. 流水线埋点顺序 + Agent 注册表 + 零代码自动埋点开关
# --------------------------------------------------------------------------
def test_pipeline_span_order():
    tr = Tracer()
    state = {"log": []}

    def mk(name):
        def fn(p):
            p["log"].append(name)
            return p
        return fn

    stages = {s: mk(s) for s in ("intent_classify", "retrieve", "llm_generate")}
    data, root = run_pipeline(tr, stages, state)
    assert data["log"] == ["intent_classify", "retrieve", "llm_generate"]
    names = [c["name"] for c in tr.trace_tree(root.trace_id)["children"]]
    assert names == ["intent_classify", "retrieve", "llm_generate"]


def test_agent_registry():
    reg = AgentRegistry()
    reg.register("analytic", "解析积分预测", kind="expert")
    reg.update_status("analytic", "running")
    assert reg.get("analytic")["status"] == "running"
    assert reg.by_status("running")[0]["name"] == "analytic"
    with pytest.raises(KeyError):
        reg.update_status("ghost", "running")


def test_autoinstrument_switch(monkeypatch):
    from udos7.observability import instrument
    from udos7.agents.coordinator import CoordinatorAgent
    instrument.disable_autoinstrument()
    monkeypatch.setenv("UDOS_OTEL_AUTO", "1")
    assert instrument.enable_autoinstrument() is True
    assert getattr(CoordinatorAgent.run, "_udos_auto", False) is True
    assert instrument.disable_autoinstrument() is True
    assert getattr(CoordinatorAgent.run, "_udos_auto", False) is False
    monkeypatch.setenv("UDOS_OTEL_AUTO", "0")
    assert instrument.enable_autoinstrument() is False
