"""v4.5.1 隐式思考线 HTTP 端点集成测试。

覆盖 POST /reason/latent 与 /reason/route 逐路径 正常200 + 非法effort 400 +
未知路由 404; /metrics 纯 Prometheus 文本, JSON 端点纯 JSON, 不崩进程。
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
from udos.pce_format import PCEParser
from demos.scene_factory import build_factory_scene

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v4.3.0.pt")


def test_version_pinned():
    assert isinstance(__version__, str) and __version__.startswith("5.")


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


@pytest.fixture
def scene_body():
    sc = build_factory_scene(n_steps=8)
    return json.loads(PCEParser.dumps(sc))


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
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


# ---- /reason/latent ---- #
@pytest.mark.parametrize("eff", ["none", "low", "high", "max"])
def test_latent_each_effort_200(server, scene_body, eff):
    st, b = _post(server, "/reason/latent", {"scene": scene_body, "effort": eff})
    assert st == 200, b
    assert b["effort"] == eff
    assert b["n_paths"] >= 1
    assert b["internal_ticks"] > 0
    assert "latent_summary" in b
    assert b["main_params_untouched"] is True
    # high/max 显式化
    if eff in ("high", "max"):
        assert b["externalized"] is True
        assert len(b["explicit_chain"]) > 0


def test_latent_invalid_effort_400(server, scene_body):
    st, b = _post(server, "/reason/latent", {"scene": scene_body, "effort": "medium"})
    assert st == 400
    assert "effort" in b["message"]


def test_latent_missing_effort_400(server, scene_body):
    st, b = _post(server, "/reason/latent", {"scene": scene_body})
    assert st == 400


def test_latent_malformed_scene_400(server):
    # 可解析但无有效 tokens -> 既有 _scene 守卫 -> 400
    st, b = _post(server, "/reason/latent",
                  {"effort": "none", "scene": {"scene_id": "x", "tokens": []}})
    assert st == 400


def test_latent_unknown_route_404(server, scene_body):
    st, _ = _post(server, "/reason/latentxx", {"scene": scene_body, "effort": "none"})
    assert st == 404


# ---- /reason/route ---- #
def test_route_200(server, scene_body):
    st, b = _post(server, "/reason/route", {"scene": scene_body})
    assert st == 200
    assert b["recommended_effort"] in ("none", "low", "high", "max")
    assert 0.0 <= b["difficulty"] <= 1.0
    assert isinstance(b["rationale"], list) and len(b["rationale"]) >= 1


def test_route_with_hint_200(server, scene_body):
    st, b = _post(server, "/reason/route",
                  {"scene": scene_body, "task_complexity": 0.9,
                   "needs_deeper": 0.9})
    assert st == 200
    # 强 hint -> 至少 high
    assert b["recommended_effort"] in ("high", "max")


def test_route_malformed_scene_400(server):
    st, _ = _post(server, "/reason/route", {"scene": {"scene_id": "x", "tokens": []}})
    assert st == 400


# ---- /metrics 纯文本, 不被污染 ---- #
def test_metrics_text_plain(server):
    st, body, ctype = _get(server, "/metrics")
    assert st == 200
    assert "text/plain" in ctype
    assert "udos" in body.lower() or "# " in body
