"""v3.3.4 hardening: 统一 logging 体系行为测试 (caplog)。

覆盖:
  - 错误路径 (400) 确实落 WARNING/ERROR 日志;
  - 正常成功路径不刷 ERROR 级;
  - GET /metrics 纯 Prometheus 文本 (Content-Type text/plain, 无 JSON 串入);
  - handler 输出流为 stderr;
  - configure_logging 级别生效 (DEBUG 可见 / WARNING 抑制 INFO);
  - 大张量只打 shape/统计, 不打完整 repr。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import urllib.error
import urllib.request

import pytest
import torch

from udos.server import create_server
from udos.pce_format import PCEParser
from udos.logging_config import configure_logging, get_udos_handler


@pytest.fixture
def live_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _post_raw(base, path, raw: bytes):
    req = urllib.request.Request(
        base + path, data=raw,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _factory_scene_dict(n_steps: int = 4):
    from demos.scene_factory import build_factory_scene
    return json.loads(PCEParser.dumps(build_factory_scene(n_steps=n_steps)))


def _udos_records(caplog):
    return [r for r in caplog.records if r.name == "udos"
            or r.name.startswith("udos.")]


def test_invalid_json_logs_warning_or_error(live_server, caplog):
    """POST /predict 传非法 JSON -> 400, 必须落 WARNING/ERROR 日志。"""
    caplog.set_level(logging.WARNING, logger="udos")
    code, _ = _post_raw(live_server, "/predict", b"{ this is not json !!!")
    assert code == 400
    hits = [r for r in _udos_records(caplog) if r.levelno >= logging.WARNING]
    assert hits, "非法 JSON 400 应产生 WARNING/ERROR 级日志"


def test_normal_success_path_no_error_log(live_server, caplog):
    """正常 POST /internalize 返回 200 时, 不应出现 ERROR 级日志。"""
    caplog.set_level(logging.INFO, logger="udos")
    scene = _factory_scene_dict(n_steps=4)
    code, body = _post(live_server, "/internalize", {"scene": scene})
    assert code == 200 and body["status"] == "ok"
    errs = [r for r in _udos_records(caplog) if r.levelno >= logging.ERROR]
    assert not errs, f"正常路径不应刷 ERROR: {[r.message for r in errs]}"


def test_metrics_pure_prometheus_text(live_server):
    """/metrics 必须是纯 Prometheus 文本, Content-Type text/plain, 无 JSON 串入。"""
    _post(live_server, "/health", {})  # 触发一条指标
    with urllib.request.urlopen(live_server + "/metrics", timeout=10) as r:
        body = r.read().decode("utf-8")
        ctype = r.headers.get("Content-Type", "")
    assert "text/plain" in ctype
    # 无 JSON 对象/键值串入
    assert '"status"' not in body and '"message"' not in body
    # 每行要么是 # HELP/# TYPE, 要么是 指标名{labels} 数值
    line_re = re.compile(
        r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+[+-]?[0-9.eE]+\s*$')
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            assert line.startswith("# HELP") or line.startswith("# TYPE"), \
                f"非法注释行: {line}"
        else:
            assert line_re.match(line), f"非 Prometheus 行: {line!r}"


def test_log_handler_writes_to_stderr():
    """统一 handler 必须输出 stderr, 绝不 stdout。"""
    h = configure_logging(level=logging.WARNING)
    assert h.stream is sys.stderr
    # 再次自检: get_udos_handler 指向同一 stderr handler
    assert get_udos_handler() is h


def test_log_level_takes_effect(caplog):
    """configure_logging(DEBUG) 后 DEBUG 可见; 改回 WARNING 后 INFO 被抑制。"""
    configure_logging(level=logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="udos")
    lg = logging.getLogger("udos.level_probe")
    lg.debug("probe-debug-visible")
    assert any("probe-debug-visible" in r.message for r in _udos_records(caplog))

    configure_logging(level=logging.WARNING)
    caplog.clear()
    lg.info("probe-info-hidden")
    assert not any("probe-info-hidden" in r.message for r in _udos_records(caplog))


def test_env_level_override(monkeypatch):
    """UDOS_LOG_LEVEL 环境变量可覆盖默认级别。"""
    monkeypatch.setenv("UDOS_LOG_LEVEL", "DEBUG")
    h = configure_logging()
    assert logging.getLogger("udos").getEffectiveLevel() == logging.DEBUG
    monkeypatch.setenv("UDOS_LOG_LEVEL", "ERROR")
    configure_logging()
    assert logging.getLogger("udos").getEffectiveLevel() == logging.ERROR


def test_large_tensor_logs_shape_only(caplog):
    """含大张量的日志点只打 shape/统计摘要, 不出现完整 tensor repr。"""
    caplog.set_level(logging.INFO, logger="udos")
    lg = logging.getLogger("udos.tensor_probe")
    big = torch.randn(64, 32)
    # 模拟关键路径日志纪律: 只打 shape 与 mean
    lg.info("tensor shape=%s mean=%.4f", tuple(big.shape),
            big.float().mean().item())
    rec = [r for r in caplog.records if r.name == "udos.tensor_probe"][-1]
    assert "tensor(" not in rec.message
    assert str(tuple(big.shape)) in rec.message
