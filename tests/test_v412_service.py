"""v4.1.2 自规划自监督线 HTTP 端点集成测试。

覆盖 5 个新端点逐路径 正常 200 + 异常 400 + 未知路由 404:
    /curriculum/generate, /curriculum/solvable,
    /selfsup/pseudo-label, /selfplan/decompose, /selfplan/decide。
未训练语义由未挂载 checkpoint 的服务单独验证 409。
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


def _win():
    return torch.randn(1, 6, 6).tolist()


def test_curriculum_generate_ok(server):
    code, body = _post(server, "/curriculum/generate", {"stage": 1})
    assert code == 200, body
    assert body["n_samples"] > 0
    assert set(body["kinds"].keys()) == {"uniform", "accel", "spring", "collision"}


def test_curriculum_generate_bad_stage_400(server):
    code, body = _post(server, "/curriculum/generate", {"stage": -1})
    assert code == 400, body


def test_curriculum_solvable_ok(server):
    code, body = _post(server, "/curriculum/solvable",
                       {"window": _win(), "horizon": 2})
    assert code == 200, body
    assert "solvable" in body


def test_curriculum_solvable_bad_window_400(server):
    code, body = _post(server, "/curriculum/solvable", {"window": "x"})
    assert code == 400, body


def test_selfsup_pseudo_label_ok(server):
    code, body = _post(server, "/selfsup/pseudo-label",
                       {"window": _win(), "horizon": 3})
    assert code == 200, body
    assert len(body["step_mse"]) == 3
    assert 0.0 <= body["mean_consistency"] <= 1.0


def test_selfplan_decompose_ok(server):
    goal = torch.randn(6).tolist()
    code, body = _post(server, "/selfplan/decompose",
                       {"window": _win(), "goal": goal, "n_subgoals": 3})
    assert code == 200, body
    assert len(body["subgoal_chain"]) == 3


def test_selfplan_decompose_missing_goal_400(server):
    code, body = _post(server, "/selfplan/decompose", {"window": _win()})
    assert code == 400, body


def test_selfplan_decide_ok(server):
    goal = torch.randn(6).tolist()
    code, body = _post(server, "/selfplan/decide",
                       {"window": _win(), "goal": goal, "confidence": 0.9})
    assert code == 200, body
    assert body["decision"] in ("accept", "stop", "self_correct", "verify")


def test_unknown_route_404(server):
    code, body = _post(server, "/curriculum/nope", {})
    assert code == 404, body


def test_not_ready_409():
    """未挂载 checkpoint 的服务: 依赖预测器的端点须 409 而非 500。"""
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address
    try:
        code, body = _post(f"http://{host}:{port}", "/curriculum/solvable",
                           {"window": torch.randn(1, 6, 6).tolist()})
        assert code == 409, body
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)
