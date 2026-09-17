"""v3.5.1 集成: HTTP /spatial/query、/spatial/collision 端点 + 默认路径逐位一致。"""
import json
import threading
import urllib.request
import urllib.error

import pytest
import torch

from udos import __version__
from udos.server import create_server


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


def test_version():
    assert __version__ == "5.5.5"


_OBJS = [
    {"object_id": "a", "position": [0.0, 0.0, 0.0], "radius": 0.5},
    {"object_id": "b", "position": [5.0, 0.0, 0.0], "radius": 0.5},
    {"object_id": "c", "position": [5.3, 0.0, 0.0], "radius": 0.5},  # 与 b 接触
]


def test_spatial_query_range_200(live_server):
    code, body = _post(live_server, "/spatial/query", {
        "op": "range", "objects": _OBJS, "center": [0, 0, 0], "radius": 1.0})
    assert code == 200, body
    assert body["status"] == "ok"
    ids = {r["object_id"] for r in body["results"]}
    assert ids == {"a"}


def test_spatial_query_raycast(live_server):
    code, body = _post(live_server, "/spatial/query", {
        "op": "raycast", "objects": _OBJS,
        "origin": [2.0, 0, 0], "direction": [1, 0, 0]})
    assert code == 200
    assert body["hit"] is True
    # a 在原点(身后), b(5.0) 与 c(5.3) 中 b 更近
    assert body["object_id"] == "b"


def test_spatial_query_bad_op_400(live_server):
    code, body = _post(live_server, "/spatial/query",
                       {"op": "nope", "objects": _OBJS})
    assert code == 400
    assert body["status"] == "error"


def test_spatial_query_bad_objects_400(live_server):
    code, body = _post(live_server, "/spatial/query",
                       {"op": "range", "objects": [], "center": [0, 0, 0]})
    assert code == 400


def test_spatial_query_no_scene_409(live_server):
    # 不带 objects 且未注册场景 -> 409
    code, body = _post(live_server, "/spatial/query",
                       {"op": "range", "center": [0, 0, 0], "radius": 1.0})
    assert code == 409
    assert body["status"] == "error"


def test_spatial_collision_200(live_server):
    code, body = _post(live_server, "/spatial/collision",
                       {"objects": _OBJS})
    assert code == 200
    assert body["n_objects"] == 3
    # b(5.0) 与 c(5.3) 中心距 0.3 < 1.0 => 接触
    assert body["n_contacts"] == 1
    pair = {body["contacts"][0]["a"], body["contacts"][0]["b"]}
    assert pair == {"b", "c"}


def test_spatial_collision_409(live_server):
    code, body = _post(live_server, "/spatial/collision", {})
    assert code == 409


def test_unknown_route_404(live_server):
    code, body = _post(live_server, "/spatial/nope", {"objects": _OBJS})
    assert code == 404


def test_default_path_untouched(live_server):
    """默认推理路径不受 SFM 影响 (未训练 -> health predictor_trained=False)。"""
    with urllib.request.urlopen(live_server + "/health", timeout=30) as resp:
        hb = json.loads(resp.read().decode("utf-8"))
    assert hb["predictor_trained"] is False
