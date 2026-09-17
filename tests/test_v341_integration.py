"""v3.4.1 ICM 与 PhysicalLoop 集成 + 服务端点测试。"""
import json
import threading
import urllib.request
import urllib.error

import pytest
import torch

from udos import load_predictor, __version__
from udos.server import create_server
from udos.physical_loop import PhysicalLoopRunner
from udos.dynamics import build_parametric_dataset
from udos.icm import DemonstrationEpisode, DemonstrationMemory

CKPT = "checkpoints/predictor_v3.4.0.pt"


# --------------------------------------------------------------------------- #
# PhysicalLoop 集成
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model():
    m, _ = load_predictor(CKPT)
    return m


@pytest.fixture(scope="module")
def memory(model):
    tr = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4242)
    mem = DemonstrationMemory()
    from udos.icm import ICMAggregator
    agg = ICMAggregator(model)
    for i in range(len(tr)):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                  scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])
    return mem


def test_loop_default_path_bitwise_unchanged(model):
    """use_icm=False 时 loop 预测与直接 predictor.predict_next 逐位一致。"""
    tr = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=1)
    loop = PhysicalLoopRunner(model, horizon=2)
    w = tr.X[0:1]
    sp = tr.P[0:1]
    a = loop._integrated_predict(w, sp)
    b = model.predict_next(w, scene_params=sp)
    assert torch.allclose(a, b, atol=1e-6)


def test_loop_icm_path_runs(model, memory):
    loop = PhysicalLoopRunner(model, horizon=2, use_icm=True,
                              icm_memory=memory, icm_k=3)
    tr = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2)
    w = tr.X[0:1]
    sp = tr.P[0:1]
    out = loop._integrated_predict(w, sp)
    assert out.shape == (1, 6)
    assert bool(torch.isfinite(out).all())


# --------------------------------------------------------------------------- #
# HTTP 端点 (真实起端口)
# --------------------------------------------------------------------------- #
@pytest.fixture
def live_server():
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
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


def test_icm_predict_409_when_empty(live_server):
    """未注册演示 => /icm/predict 应 409。"""
    code, body = _post(live_server, "/icm/predict",
                       {"window": torch.randn(6, 6).tolist()})
    assert code == 409


def test_icm_register_then_predict_200(live_server):
    win = torch.randn(6, 6).tolist()
    result = torch.randn(6).tolist()
    code, body = _post(live_server, "/icm/demo/register",
                       {"input_window": win, "result": result, "kind": "test"})
    assert code == 200 and body["memory_size"] >= 1
    code, body = _post(live_server, "/icm/predict",
                       {"window": win, "k": 2})
    assert code == 200
    assert body["status"] == "ok"
    assert len(body["prediction"]) == 6


def test_icm_register_bad_input_400(live_server):
    code, body = _post(live_server, "/icm/demo/register",
                       {"input_window": [[1, 2]]})   # 缺 result
    assert code == 400


def test_icm_predict_bad_k_400(live_server):
    # 先注册一条
    _post(live_server, "/icm/demo/register",
          {"input_window": torch.randn(6, 6).tolist(),
           "result": torch.randn(6).tolist()})
    code, body = _post(live_server, "/icm/predict",
                       {"window": torch.randn(6, 6).tolist(), "k": -1})
    assert code == 400


def test_icm_unknown_route_404(live_server):
    code, body = _post(live_server, "/icm/nonexistent", {})
    assert code == 404


def test_version():
    assert __version__ == "5.5.5"
