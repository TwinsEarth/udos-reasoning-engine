"""v3.9.2 多任务统一头 + /wla/er HTTP 端点集成测试。

覆盖:
    * POST /wla/er 正常 200 (结构化代理输出形状/有限);
    * 非法 window -> 400; 未知路由 -> 404;
    * 多任务评测矩阵 JSON 存在且 n_task_kinds>0;
    * 版本断言。
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
CKPT = str(ROOT / "checkpoints" / "predictor_v3.9.0.pt")


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
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_wla_er_endpoint_ok(server):
    w = torch.randn(2, 6, 6).tolist()
    code, body = _post(server, "/wla/er", {"window": w})
    assert code == 200, body
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert len(body["spatial_relation"]) == 2
    assert len(body["target_point"][0]) == 2
    assert body["main_params_untouched"] is True


def test_wla_er_bad_window_400(server):
    code, body = _post(server, "/wla/er", {"window": "not-an-array"})
    assert code == 400, body


def test_wla_er_unknown_route_404(server):
    code, body = _post(server, "/wla/does-not-exist", {"window": []})
    assert code == 404, body


def test_multitask_matrix_json():
    p = ROOT / "benchmarks" / "results" / "wla_multitask_matrix.json"
    assert p.exists()
    d = json.load(open(p, encoding="utf-8"))
    assert d["n_task_kinds"] >= 1
    assert d["analogy_not_reproduction"] is True
