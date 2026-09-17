"""v3.7.1 集成测试: HierarchicalController + HTTP /neural/step、/neural/reflex/log。

覆盖:
    * loop+neural 组合, 默认逐位一致 (/loop/step 不受影响);
    * /neural/step 200 (command[6]/priority_winner/safety_state);
    * /neural/step 400 (缺 window);
    * /neural/step 409 (未训练);
    * /neural/reflex/log 200 (count/events);
    * 未知路由 404。
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
CKPT = str(ROOT / "checkpoints" / "predictor_v3.7.0.pt")


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def trained_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown(); httpd.server_close(); t.join(timeout=5)


@pytest.fixture(scope="module")
def bare_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")  # 无 checkpoint
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown(); httpd.server_close(); t.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_neural_step_200(trained_server):
    w = torch.randn(1, 6, 6).tolist()
    sp = torch.zeros(1, 4).tolist()
    code, body = _post(trained_server, "/neural/step",
                       {"window": w, "scene_params": sp})
    assert code == 200, body
    assert body["status"] == "ok"
    assert len(body["command"]) == 6
    assert body["priority_winner"] in ("spinal", "cerebellum", "cortex")
    assert body["safety_state"] in ("nominal", "caution", "reflex")
    assert body["priority_matrix"]["spinal"] < body["priority_matrix"]["cortex"]


def test_neural_step_400(trained_server):
    code, body = _post(trained_server, "/neural/step", {})
    assert code == 400
    assert body["status"] == "error"


def test_neural_step_409(bare_server):
    w = torch.randn(1, 6, 6).tolist()
    code, body = _post(bare_server, "/neural/step", {"window": w})
    assert code == 409


def test_reflex_log_200(trained_server):
    # 先触发一次 /neural/step (良性)
    w = torch.randn(1, 6, 6).tolist()
    _post(trained_server, "/neural/step", {"window": w})
    code, body = _post(trained_server, "/neural/reflex/log", {})
    assert code == 200, body
    assert body["status"] == "ok"
    assert body["count"] >= 0
    assert "events" in body and "scheduler" in body


def test_reflex_log_reset(trained_server):
    code, body = _post(trained_server, "/neural/reflex/log", {"reset": True})
    assert code == 200
    assert body["count"] == 0


def test_unknown_route_404(trained_server):
    code, body = _post(trained_server, "/neural/nope", {"window": []})
    assert code == 404


def test_loop_step_unchanged(trained_server):
    """默认 opt-in: /loop/step 仍可用, 不受神经控制端点影响。"""
    w = torch.randn(1, 6, 6).tolist()
    sp = torch.zeros(1, 4).tolist()
    code, body = _post(trained_server, "/loop/step",
                       {"window": w, "scene_params": sp})
    assert code == 200, body
    assert body["steps"][0] == "observe"
