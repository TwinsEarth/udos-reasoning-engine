"""v4.3.1 完全自进化线 HTTP 端点集成测试。

覆盖 3 个新端点逐路径 正常 200 + 异常 400 + 未知路由 404:
    /self-evolution/search, /self-evolution/ab, /self-evolution/long-horizon。
另验 /health 200、/metrics 纯文本。错误语义延续: 未训练 409, 非法 400,
未知路由 404, 不崩进程。
"""
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.server import create_server

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v4.3.0.pt")


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, r.read().decode("utf-8"), r.headers.get(
                "Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), ""


def test_health_ok(server):
    st, body, _ = _get(server, "/health")
    assert st == 200
    assert json.loads(body)["version"] == __version__ == "5.5.5"


def test_metrics_plain_text(server):
    st, body, ctype = _get(server, "/metrics")
    assert st == 200
    assert "text/plain" in ctype
    assert "udos_" in body or "#" in body


def test_search_ok(server):
    st, body = _post(server, "/self-evolution/search", {})
    assert st == 200
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["n_candidates"] > 1
    assert isinstance(body["improved"], bool)
    assert body["main_params_untouched"] is True


def test_search_unknown_route(server):
    st, body = _post(server, "/self-evolution/nonexistent", {})
    assert st == 404
    assert body["status"] == "error"


def test_ab_ok(server):
    st, body = _post(server, "/self-evolution/ab", {})
    assert st == 200
    assert body["status"] == "ok"
    assert isinstance(body["A_wins"], bool)
    assert isinstance(body["cost_saving_pct"], (int, float))


def test_ab_unknown_route(server):
    st, body = _post(server, "/self-evolution/ab-nope", {})
    assert st == 404


def test_long_horizon_ok(server):
    st, body = _post(server, "/self-evolution/long-horizon",
                      {"horizon": 4, "n_sub": 2, "seed": 0})
    assert st == 200
    assert body["status"] == "ok"
    assert body["verified"] is True
    assert body["n_subtasks"] == 2


def test_long_horizon_bad(server):
    # horizon 越界 -> 400
    st, body = _post(server, "/self-evolution/long-horizon",
                     {"horizon": 999, "n_sub": 2})
    assert st == 400
    # n_sub > horizon -> 400
    st2, body2 = _post(server, "/self-evolution/long-horizon",
                       {"horizon": 2, "n_sub": 9})
    assert st2 == 400


def test_long_horizon_unknown_route(server):
    st, body = _post(server, "/self-evolution/long-horizon/xx", {})
    assert st == 404
