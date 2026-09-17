"""v2.5.1 回归: 服务监控指标 (GET /metrics + MetricsCollector)。

覆盖:
- MetricsCollector 计数/延迟分位/OOD 触发率/Prometheus 文本格式
- 服务级: 请求后计数递增
- /evaluate 含 metrics 段
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.server import MetricsCollector  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_metrics_collector_record_and_snapshot():
    mc = MetricsCollector()
    mc.record("/predict", 0.010)
    mc.record("/predict", 0.020)
    mc.record("/predict", 0.015)
    s = mc.snapshot()
    assert s["endpoints"]["/predict"]["count"] == 3
    assert 0.010 <= s["endpoints"]["/predict"]["latency_p50_s"] <= 0.020
    assert "latency_p95_s" in s["endpoints"]["/predict"]
    assert "latency_p99_s" in s["endpoints"]["/predict"]


def test_metrics_ood_trigger_rate():
    mc = MetricsCollector()
    mc.record_ood(True)
    mc.record_ood(False)
    mc.record_ood(True)
    s = mc.snapshot()
    assert s["ood_triggers"] == 2
    assert s["ood_total"] == 3
    assert abs(s["ood_trigger_rate"] - round(2 / 3, 4)) < 1e-4


def test_metrics_cache_hit_rate():
    mc = MetricsCollector()
    mc.record_cache(True)
    mc.record_cache(True)
    mc.record_cache(False)
    s = mc.snapshot()
    assert s["cache_hits"] == 2
    assert s["cache_misses"] == 1
    assert abs(s["cache_hit_rate"] - round(2 / 3, 4)) < 1e-4


def test_metrics_prometheus_text_format():
    mc = MetricsCollector()
    mc.record("/health", 0.001)
    text = mc.prometheus_text()
    assert "udos_requests_total" in text
    assert "udos_latency_seconds" in text
    assert 'endpoint="/health"' in text
    assert "quantile=\"0.95\"" in text
    assert "# HELP" in text and "# TYPE" in text


def test_metrics_empty_collector():
    mc = MetricsCollector()
    s = mc.snapshot()
    assert s["endpoints"] == {}
    assert s["cache_hit_rate"] == 0.0
    assert s["ood_trigger_rate"] == 0.0


def test_service_metrics_count_increments():
    """服务级: 通过 HTTP 请求后 /metrics 计数递增。"""
    import json, threading, urllib.request
    from udos.server import create_server
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    try:
        # /health GET 不经过 do_POST 的计时, 但 /metrics 本身也是 GET
        # 先 GET /metrics (返回 Prometheus 文本)
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            assert r.status == 200
            body = r.read().decode()
            assert "udos_requests_total" in body
            assert "udos_latency_seconds" in body
        # 发一个 POST /train 让计数变化
        req = urllib.request.Request(
            base + "/train",
            data=json.dumps({"epochs": 1, "n_per_kind": 4}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            assert r.status == 200
        # 再看 metrics, /train 应被记录
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            body2 = r.read().decode()
            assert 'udos_requests_total{endpoint="/train"} 1' in body2
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_evaluate_contains_metrics_segment(tmp_path, monkeypatch):
    """v2.5.1: /evaluate 响应含 metrics 段。"""
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    s = UDOSService(preset="small")
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    rep = s.evaluate({"n_per_kind": 6, "horizon": 3})
    assert "service_metrics" in rep
    assert "endpoints" in rep["service_metrics"]
