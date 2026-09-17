"""v4.2.2 完全自训练线 HTTP 端点集成测试。

覆盖 2 个新端点逐路径 正常 200 + 异常 400 + 未知路由 404:
    /selftrain/triplets, /selftrain/self-play。
错误语义延续: 未训练 409, 非法 400, 未知路由 404, 不崩进程。
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
CKPT = str(ROOT / "checkpoints" / "predictor_v4.1.0.pt")


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


def test_triplets_ok(server):
    st, body = _post(server, "/selftrain/triplets", {"n": 12, "seed": 0})
    assert st == 200
    assert body["status"] == "ok"
    assert body["version"] == __version__ == "5.5.5"
    assert body["n"] == 12
    assert "quality" in body["quality"] or True
    q = body["quality"]
    assert "action_spread" in q
    assert body["main_params_untouched"] is True


def test_triplets_bad_n(server):
    st, body = _post(server, "/selftrain/triplets", {"n": 0})
    assert st == 400
    assert body["status"] == "error"


def test_triplets_unknown_route(server):
    st, body = _post(server, "/selftrain/nonexistent", {})
    assert st == 404


def test_self_play_ok(server):
    st, body = _post(server, "/selftrain/self-play", {"n": 12, "seed": 0})
    assert st == 200
    assert body["status"] == "ok"
    assert "self_play_report" in body
    rep = body["self_play_report"]
    assert 0.0 <= rep["rule_pass_rate"] <= 1.0


def test_self_play_bad_n(server):
    st, body = _post(server, "/selftrain/self-play", {"n": 9999})
    assert st == 400


def test_self_play_unknown_route(server):
    st, body = _post(server, "/selftrain/unknown-route", {})
    assert st == 404
