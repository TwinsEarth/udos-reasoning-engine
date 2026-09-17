"""
v2.6.0+dev6 服务接口扩展 (反事实/辨识/风险/快照差分) 契约测试
==============================================================
真实起 HTTP 端口打请求。200 用例先 POST /load 加载 v2.6.0 checkpoint 后再调;
409 用例用全新未训练 UDOSService 直接调方法验证 ServiceNotReady 语义。
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

CKPT = "predictor_v2.6.0.pt"


@pytest.fixture(scope="module")
def loaded_server():
    """启动服务并预加载 v2.6.0 checkpoint。"""
    httpd = create_server("127.0.0.1", 0, preset="small",
                          checkpoint="checkpoints/predictor_v2.6.0.pt")
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


def _window():
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=401)
    return ds.X[:1].tolist(), ds.P[:1].tolist()


def test_version():
    assert __version__ == "5.5.5"


def test_counterfactual_endpoint(loaded_server):
    w, p = _window()
    c, b = _post(loaded_server, "/counterfactual",
                 {"window": w, "horizon": 3, "scene_params": p})
    assert c == 200, b
    for k in ("baseline", "counterfactual", "ate_by_step", "ate_mean",
              "final_state_diff"):
        assert k in b
    # 零干预 => 基线与反事实逐位一致, ATE≈0
    assert b["ate_mean"] == pytest.approx(0.0, abs=1e-9)
    assert b["baseline"] == b["counterfactual"]
    # 非零干预 => 轨迹不同
    c2, b2 = _post(loaded_server, "/counterfactual",
                   {"window": w, "horizon": 3, "scene_params": p,
                    "intervention": {"scene_params": {"2": 1.2}}})
    assert c2 == 200
    assert b2["ate_mean"] > 0.0


def test_counterfactual_409():
    """全新未训练 UDOSService 直接调 -> ServiceNotReady (HTTP 层映射为 409)。"""
    svc = UDOSService(preset="small")
    with pytest.raises(ServiceNotReady):
        svc.counterfactual({"window": [[0.0] * 6] * 6, "horizon": 2})


def test_identify_endpoint(loaded_server):
    w, _ = _window()
    c, b = _post(loaded_server, "/identify",
                 {"window": w, "horizon": 2, "grid_size": 4})
    assert c == 200, b
    assert len(b["identified_params"]) == 4
    assert b["param_names"] == ["v0", "accel_a", "spring_omega", "other_v2"]
    assert b["loss_min"] >= 0.0


def test_risk_endpoint(loaded_server):
    w, p = _window()
    c, b = _post(loaded_server, "/risk",
                 {"window": w, "horizon": 1, "scene_params": p})
    assert c == 200, b
    assert b["risk_level"] in ("low", "medium", "high")
    assert 0.0 <= b["risk_score"] <= 1.0
    assert set(("interval_width", "ood_score", "confidence")).issubset(
        b["components"].keys())


def test_diff_checkpoints_endpoint(loaded_server):
    c, b = _post(loaded_server, "/diff-checkpoints",
                 {"name_a": "predictor_v2.5.2.pt",
                  "name_b": "predictor_v2.6.0.pt",
                  "n_per_kind": 16})
    assert c == 200, b
    for k in ("a_metrics", "b_metrics", "pred_diff_max", "pred_diff_mean",
              "mse_diff", "params_a", "params_b"):
        assert k in b
    assert b["params_a"] == b["params_b"] == 52191


def test_diff_checkpoints_400(loaded_server):
    c, b = _post(loaded_server, "/diff-checkpoints",
                 {"name_a": "predictor_v2.5.2.pt",
                  "name_b": "does_not_exist.pt"})
    assert c == 400
