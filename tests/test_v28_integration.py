"""
v2.8.2 集成加固: loop + multitask 组合 + 新服务端点 + 性能基准
=================================================================
锚点纪律:
    * loop + multitask 组合调用互不冲突、互不污染; 默认路径逐位一致;
    * backcompat 11 件 (v2.1.0..v2.8.0) 全部可加载;
    * POST /loop/step 与 POST /multitask/predict: 200 / 400(非法输入) / 409(未训练);
    * benchmarks/results/feature_latency_v2.8.0.json 落盘可复算。
"""
import json
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.server import create_server, UDOSService
from udos.persistence import load_predictor
from udos.physical_loop import PhysicalLoopRunner
from udos.multitask import (MultiTaskHead, SpatialCoordHead, ActionTrajectoryHead,
                            FutureStateHead)
from udos.dynamics import build_parametric_dataset, RAW_DIM

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.8.0.pt")
LAT_JSON = ROOT / "benchmarks" / "results" / "feature_latency_v2.8.0.json"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


@pytest.fixture(scope="module")
def sample():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=818)
    return ds.X[:1].tolist(), ds.P[:1].tolist()


@pytest.fixture(scope="module")
def loaded_server():
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


def test_version():
    assert __version__ == "5.5.5"


def test_loop_multitask_composition_no_conflict(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=818)
    wb, pb = ds.X[:1], ds.P[:1]
    loop = PhysicalLoopRunner(predictor, horizon=2)
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("spatial", SpatialCoordHead(32, n_pts=4))
    mth.register_head("action", ActionTrajectoryHead(32, horizon=4, action_dim=RAW_DIM))
    mth.register_head("future", FutureStateHead(32, horizon=4, state_dim=RAW_DIM))
    # 先跑 loop 再跑 multitask, 再跑 loop
    before = predictor.predict_next(wb, scene_params=pb)
    loop.run(wb, scene_params=pb, candidate_actions=[{}])
    out = mth.forward(wb, scene_params=pb)
    assert set(out.keys()) == {"spatial", "action", "future"}
    loop.run(wb, scene_params=pb, candidate_actions=[{}])
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after), "组合调用后主路径必须逐位一致"


def test_backcompat_11_checkpoints_load():
    names = ["predictor_v2.1.0.pt", "predictor_v2.2.1.pt", "predictor_v2.3.1.pt",
             "predictor_v2.4.0.pt", "predictor_v2.5.0.pt", "predictor_v2.5.2.pt",
             "predictor_v2.6.0.pt", "predictor_v2.6.2.pt", "predictor_v2.7.0.pt",
             "predictor_v2.7.3.pt", "predictor_v2.8.0.pt"]
    for n in names:
        m, meta = load_predictor(str(ROOT / "checkpoints" / n))
        assert meta["udos_version"] in n or n.endswith(meta["udos_version"] + ".pt")
        out = m.predict_next(torch.randn(1, 6, m.raw_dim),
                             scene_params=(torch.randn(1, m.scene_param_dim)
                                           if m.scene_encoder is not None else None))
        assert out.shape == (1, 6) and torch.isfinite(out).all()


def test_loop_step_200(loaded_server, sample):
    w, p = sample
    c, b = _post(loaded_server, "/loop/step",
                 {"window": w, "scene_params": p, "horizon": 2,
                  "actions": [{}, {"state_perturbation": [0.0] * 6}]})
    assert c == 200
    assert b["steps"] == ["observe", "understand", "predict_action",
                          "future_state", "feedback"]
    assert b["prediction_shape"] == [1, 6]


def test_multitask_predict_200(loaded_server, sample):
    w, p = sample
    c, b = _post(loaded_server, "/multitask/predict",
                 {"window": w, "scene_params": p})
    assert c == 200
    assert len(b["spatial"]) == 1 and len(b["spatial"][0]) == 4
    assert len(b["action"][0]) == 4 and len(b["action"][0][0]) == 6
    assert len(b["future"]["trajectory"][0]) == 4
    assert len(b["future"]["uncertainty"][0]) == 4


def test_endpoints_409_when_untrained():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):  # ServiceNotReady -> HTTP 409
        svc.loop_step({"window": [[0.0] * 6] * 6})
    with pytest.raises(Exception):
        svc.multitask_predict({"window": [[0.0] * 6] * 6})


def test_loop_step_400_bad_input(loaded_server, sample):
    w, p = sample
    # 缺 window
    c, b = _post(loaded_server, "/loop/step", {"scene_params": p})
    assert c == 400
    # 非法 horizon
    c, b = _post(loaded_server, "/loop/step",
                 {"window": w, "scene_params": p, "horizon": 99})
    assert c == 400


def test_latency_json_written():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_v28_features.py")],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(LAT_JSON))
    assert d["n_repeats"] == 50
    for k in ("predict_next", "physical_loop_step", "multitask_three_heads"):
        assert d[k]["p50_ms"] >= 0
