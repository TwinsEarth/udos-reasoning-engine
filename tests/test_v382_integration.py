"""v3.8.2 系统集成: HTTP /twin/step、/twin/scene。

覆盖:
    * POST /twin/scene 创建/查询 200;
    * POST /twin/step 推进一步 200 (含冲突计数);
    * /twin/step 未建场景 409; /twin/scene 查询未建 409;
    * 非法参数 400 (n_agents=0 / dt<=0);
    * 未知路由 404;
    * 带 window 时附跑 ClosedLoopOrchestrator (已挂 predictor);
    * /loop/step 默认逐位路径不受影响 (全特性兼容)。
"""
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

import torch
import pytest

from udos import __version__
from udos.server import create_server

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def server():
    import pytest
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
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
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_step_before_scene_409(server):
    code, body = _post(server, "/twin/step", {})
    assert code == 409, body


def test_scene_query_before_create_409(server):
    code, body = _post(server, "/twin/scene", {})
    assert code == 409, body


def test_create_scene_200(server):
    code, body = _post(server, "/twin/scene",
                       {"n_agents": 4, "n_obstacles": 3, "seed": 1, "bounds": 8})
    assert code == 200, body
    assert body["action"] == "create"
    assert body["summary"]["n_agents"] == 4
    assert body["summary"]["n_obstacles"] == 3


def test_query_scene_200(server):
    code, body = _post(server, "/twin/scene", {})
    assert code == 200, body
    assert body["action"] == "query"


def test_step_200_and_advances(server):
    code, body = _post(server, "/twin/step", {})
    assert code == 200, body
    assert body["step"] == 0
    assert "scene" in body
    code, body = _post(server, "/twin/step", {})
    assert code == 200
    assert body["step"] == 1


def test_step_with_closed_loop_200(server):
    w = torch.randn(1, 6, 6).tolist()
    code, body = _post(server, "/twin/step", {"window": w})
    assert code == 200, body
    assert "closed_loop" in body
    assert len(body["closed_loop"]["command"]) == 6


def test_bad_params_400(server):
    code, body = _post(server, "/twin/scene", {"n_agents": 0})
    assert code == 400, body
    code, body = _post(server, "/twin/step", {"dt": -1})
    assert code == 400, body


def test_unknown_route_404(server):
    code, body = _post(server, "/twin/nope", {})
    assert code == 404, body


def test_default_loop_unchanged(server):
    w = torch.randn(1, 6, 6).tolist()
    code, body = _post(server, "/loop/step", {"window": w})
    assert code == 200, body
