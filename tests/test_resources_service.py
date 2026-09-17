"""v4.5.4 资源端点 HTTP 契约: /resources, /{id}/probe, /{id}/invoke 全状态码。"""
import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from udos.server import create_server


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


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _post(base, path, obj):
    req = urllib.request.Request(
        base + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_get_resources_200(live_server):
    code, body = _get(live_server, "/resources")
    assert code == 200 and body["status"] == "ok"
    assert body["summary"]["total_registered"] == 79
    # performance 默认视图不暴露 L3
    assert body["summary"]["profile"] == "performance"


def test_get_resources_full(live_server):
    code, body = _get(live_server, "/resources?profile=full")
    assert code == 200
    assert body["summary"]["visible_in_profile"] == 79


def test_get_resources_filters(live_server):
    code, body = _get(live_server, "/resources?profile=full&kind=model")
    assert code == 200
    assert body["summary"]["by_kind"]["model"] == len(body["resources"])


def test_probe_unknown_id_404(live_server):
    code, body = _post(live_server, "/resources/nope/probe", {})
    assert code == 404


def test_probe_known_200(live_server):
    code, body = _post(live_server, "/resources/yourdfpy/probe", {})
    assert code == 200 and body["id"] == "yourdfpy"
    assert body["capability"]["status"] in ("available", "degraded", "absent",
                                             "env_blocked")


def test_invoke_absent_l3_503(live_server):
    code, body = _post(live_server, "/resources/openvla/invoke",
                       {"action": "convert", "params": {"payload": {}}})
    assert code == 503
    assert body["resource_status"] == "absent"
    assert body["requires"]["weights"] is True


def test_invoke_missing_action_400(live_server):
    code, body = _post(live_server, "/resources/yourdfpy/invoke", {})
    assert code == 400


def test_invoke_unknown_id_404(live_server):
    code, body = _post(live_server, "/resources/ghost/invoke",
                       {"action": "convert"})
    assert code == 404


def test_invoke_convert_l1_200(live_server):
    code, body = _post(live_server, "/resources/diffusion_policy/invoke",
                       {"action": "convert",
                        "params": {"payload": {"chunk": [[0.1, 0.2], [0.3, 0.4]]}}})
    assert code == 200
    assert body["udos"]["n_steps"] == 2
