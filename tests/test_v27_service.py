"""
v2.7.0.dev6 服务接口扩展 (MPC / 在线 / 主动 / 实验) 契约测试
============================================================
真实起 HTTP 端口打请求。200 用例先 POST /load 或预加载 v2.7.0 checkpoint 后再调;
409 用例用全新未训练 UDOSService 直接调方法验证 ServiceNotReady 语义;
400 用例打非法输入。
"""
import json
import threading
import urllib.request
import urllib.error

import pytest
import torch

from udos import __version__
from udos.server import create_server, UDOSService, ServiceNotReady
from udos.dynamics import build_parametric_dataset
from udos.policy import MPCActionSelector
from udos.active_learning import UncertaintySampler

CKPT = "checkpoints/predictor_v2.7.0.pt"


@pytest.fixture(scope="module")
def loaded_server():
    httpd = create_server("127.0.0.1", 0, preset="small",
                          checkpoint=CKPT)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


@pytest.fixture(scope="module")
def sample():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=401)
    return ds.X[:1].tolist(), ds.P[:1].tolist()


def test_version():
    assert __version__ == "5.5.5"


# ---------------- /policy/select ---------------- #
def test_policy_select_200(loaded_server, sample):
    w, p = sample
    actions = [{}, {"state_perturbation": [0.0] * 6},
               {"scene_param": p[0]}]
    c, b = _post(loaded_server, "/policy/select",
                 {"window": w, "scene_params": p, "actions": actions,
                  "horizon": 2, "lambda_risk": 1.0})
    assert c == 200, b
    for k in ("best_action", "best_score", "best_index", "ranked_actions",
              "no_valid_action"):
        assert k in b
    assert b["no_valid_action"] is False
    assert len(b["ranked_actions"]) == 3
    scores = [r["score"] for r in b["ranked_actions"]]
    assert scores == sorted(scores, reverse=True)


def test_policy_select_empty_actions_200(loaded_server, sample):
    """空动作集 => no_valid_action, 仍 200 (不是错误)。"""
    w, _ = sample
    c, b = _post(loaded_server, "/policy/select",
                 {"window": w, "actions": [], "horizon": 1})
    assert c == 200, b
    assert b["no_valid_action"] is True
    assert b["best_action"] is None


def test_policy_select_400(loaded_server, sample):
    w, _ = sample
    c, b = _post(loaded_server, "/policy/select",
                 {"window": w})            # 缺 actions
    assert c == 400
    c, b = _post(loaded_server, "/policy/select",
                 {"actions": [{}]})          # 缺 window
    assert c == 400


def test_policy_select_409():
    svc = UDOSService(preset="small")
    with pytest.raises(ServiceNotReady):
        svc.policy_select({"window": [[0.0] * 6] * 6, "actions": [{}]})


def test_policy_select_matches_offline(loaded_server, sample):
    """HTTP 结果与离线 MPCActionSelector 一致 (best_index)。"""
    w, p = sample
    actions = [{}, {"state_perturbation": [0.0] * 6}]
    c, b = _post(loaded_server, "/policy/select",
                 {"window": w, "scene_params": p, "actions": actions,
                  "horizon": 2})
    assert c == 200
    # 离线复算 (服务内 predictor 与 HTTP 同一 checkpoint)
    from udos.persistence import load_predictor
    model, _ = load_predictor(CKPT)
    off = MPCActionSelector(model, horizon=2).select(
        torch.tensor(w), scene_params=torch.tensor(p),
        candidate_actions=actions)
    assert b["best_index"] == off["best_index"]


# ---------------- /online/adapt ---------------- #
def test_online_adapt_200(loaded_server, sample):
    w, p = sample
    c, b = _post(loaded_server, "/online/adapt",
                 {"window": w, "enable_finetune": False})
    assert c == 200, b
    for k in ("adapted", "reason", "drift_score", "weights_modified",
              "log_length"):
        assert k in b
    # 分布内单窗口 => 不应触发微调权重
    assert b["weights_modified"] is False


def test_online_adapt_400(loaded_server):
    c, b = _post(loaded_server, "/online/adapt", {})
    assert c == 400


def test_online_adapt_409():
    svc = UDOSService(preset="small")
    with pytest.raises(ServiceNotReady):
        svc.online_adapt({"window": [[0.0] * 6] * 6})


# ---------------- /active/sample ---------------- #
def test_active_sample_200(loaded_server, sample):
    _, p = sample
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=777)
    pool = ds.X[:5].tolist()
    sp = ds.P[:5].tolist()
    c, b = _post(loaded_server, "/active/sample",
                 {"sample_pool": pool, "k": 3, "scene_params": sp})
    assert c == 200, b
    assert b["pool_size"] == 5
    assert len(b["indices"]) == 3
    assert len(b["scores"]) == 3
    # top-K 分数降序
    assert b["scores"] == sorted(b["scores"], reverse=True)


def test_active_sample_k_gt_pool_clamps(loaded_server, sample):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=778)
    pool = ds.X[:2].tolist()
    c, b = _post(loaded_server, "/active/sample",
                 {"sample_pool": pool, "k": 10})
    assert c == 200
    assert len(b["indices"]) == 2          # k>池 => 返回全部


def test_active_sample_400(loaded_server):
    c, b = _post(loaded_server, "/active/sample",
                 {"sample_pool": [], "k": 3})     # 空池
    assert c == 400
    c, b = _post(loaded_server, "/active/sample",
                 {"sample_pool": [[[0.0] * 6] * 6], "k": 0})   # k<=0
    assert c == 400


def test_active_sample_409():
    svc = UDOSService(preset="small")
    with pytest.raises(ServiceNotReady):
        svc.active_sample({"sample_pool": [[[0.0] * 6] * 6], "k": 1})


# ---------------- /experiments ---------------- #
def test_experiments_get(loaded_server):
    c, b = _get(loaded_server, "/experiments")
    assert c == 200, b
    assert "experiments" in b and "count" in b
    assert isinstance(b["experiments"], list)
