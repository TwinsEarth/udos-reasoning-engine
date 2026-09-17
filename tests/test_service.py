"""服务层契约: 真实起 HTTP 端口打请求, 覆盖成功/400/404/全链路。"""

import json
import threading
import urllib.request
import urllib.error

import pytest
import torch

from udos.server import create_server
from udos.pce_format import PCEParser


@pytest.fixture
def live_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    yield base
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _factory_scene_dict():
    from demos.scene_factory import build_factory_scene
    return json.loads(PCEParser.dumps(build_factory_scene(n_steps=8)))


def test_health(live_server):
    code, body = _get(live_server, "/health")
    assert code == 200 and body["status"] == "ok"
    from udos import __version__
    assert body["version"] == __version__ == "5.5.5"
    assert body["internalized_scenes"] == []
    assert body["predictor_trained"] is False


def test_train_then_reason_predicts_state(live_server):
    c, b = _post(live_server, "/train", {"epochs": 2, "n_per_kind": 4})
    assert c == 200 and b["trained"] is True
    assert len(b["train_loss_curve"]) == 2 and b["predictor_params"] > 0
    assert _get(live_server, "/health")[1]["predictor_trained"] is True
    # 训练挂载后 reason 输出可解释下一时刻物理量
    scene = _factory_scene_dict()
    _post(live_server, "/internalize", {"scene": scene})
    c, b = _post(live_server, "/reason", {"scene": scene})
    st = b["predicted_next_state"]
    assert len(st["position"]) == 3 and len(st["velocity"]) == 3


def test_train_bad_params_400(live_server):
    c, _ = _post(live_server, "/train", {"epochs": 99999})
    assert c == 400
    c, _ = _post(live_server, "/train", {"n_per_kind": 99999})
    assert c == 400


def test_internalize_reason_reset_flow(live_server):
    scene = _factory_scene_dict()
    c, b = _post(live_server, "/internalize", {"scene": scene})
    assert c == 200 and b["lora_params"] > 0
    assert b["scene_id"] in _get(live_server, "/health")[1]["internalized_scenes"]

    c, b = _post(live_server, "/reason", {"scene": scene, "query": "抓取时机?"})
    assert c == 200
    assert b["ticks_used"] > 0
    assert len(b["certainty_trajectory"]) == b["ticks_used"]
    assert isinstance(b["causal_chain"], list) and b["internalized"] is True

    c, b = _post(live_server, "/reset", {"scene_id": scene["scene_id"]})
    assert c == 200
    assert _get(live_server, "/health")[1]["internalized_scenes"] == []


def test_bad_json_returns_400(live_server):
    # 缺 scene 字段
    c, b = _post(live_server, "/reason", {})
    assert c == 400 and b["status"] == "error"
    # 缺 scene_id
    c, _ = _post(live_server, "/reset", {})
    assert c == 400


def test_unknown_route_404(live_server):
    c, _ = _post(live_server, "/nope", {})
    assert c == 404
    c, _ = _get(live_server, "/nope")
    assert c == 404


def test_demo_endpoint(live_server):
    c, b = _get(live_server, "/demo")
    assert c == 200
    assert set(b) == {"internalize", "reason", "reset"}
    assert b["reason"]["ticks_used"] > 0
