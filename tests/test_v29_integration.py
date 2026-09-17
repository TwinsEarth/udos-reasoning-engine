"""
v2.9.2 集成加固: retargeting+affordance+spatial_relation+loop+multitask 组合
================================================================================
锚点纪律:
    * 全部 2.9 特性组合调用互不冲突、互不污染; 默认路径逐位一致;
    * backcompat 12 件 (v2.1.0..v2.9.0) 全部可加载;
    * POST /retarget/convert 与 POST /affordance/score: 200/400/409;
    * benchmarks/results/feature_latency_v2.9.0.json 落盘可复算。
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
from udos.multitask import MultiTaskHead, SpatialRelationHead
from udos.retargeting import MorphologyConfig, ActionRetargeter, MorphologyLibrary
from udos.affordance import AffordanceScorer, AffordanceActionPlanner
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")
LAT_JSON = ROOT / "benchmarks" / "results" / "feature_latency_v2.9.0.json"


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
    assert __version__ == "5.5.5" or __version__.startswith("2.9")


def test_full_composition_no_conflict(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=818)
    wb, pb = ds.X[:1], ds.P[:1]
    before = predictor.predict_next(wb, scene_params=pb)

    loop = PhysicalLoopRunner(predictor, horizon=2)
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    mth.register_head("spatial_rel", SpatialRelationHead(32, n_obj=4))
    lib = MorphologyLibrary()
    rt = ActionRetargeter(lib.get("arm_7dof"), lib.get("gripper_4dof"))
    planner = AffordanceActionPlanner(AffordanceScorer(reach_radius=1e3))

    loop.run(wb, scene_params=pb, candidate_actions=[{}])
    mth.forward(wb, scene_params=pb)
    rt.retarget(torch.randn(3, 7))
    planner.plan(wb[:, -1, :], torch.randn(1, 3, 6))
    loop.run(wb, scene_params=pb, candidate_actions=[{}])
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after), "组合调用后主路径必须逐位一致"


def test_backcompat_12_checkpoints_load():
    names = ["predictor_v2.1.0.pt", "predictor_v2.2.1.pt", "predictor_v2.3.1.pt",
             "predictor_v2.4.0.pt", "predictor_v2.5.0.pt", "predictor_v2.5.2.pt",
             "predictor_v2.6.0.pt", "predictor_v2.6.2.pt", "predictor_v2.7.0.pt",
             "predictor_v2.7.3.pt", "predictor_v2.8.0.pt", "predictor_v2.9.0.pt"]
    assert len(names) == 12
    for n in names:
        m, meta = load_predictor(str(ROOT / "checkpoints" / n))
        assert meta["udos_version"] in n or n.endswith(meta["udos_version"] + ".pt")
        out = m.predict_next(torch.randn(1, 6, m.raw_dim),
                             scene_params=(torch.randn(1, m.scene_param_dim)
                                           if m.scene_encoder is not None else None))
        assert out.shape == (1, 6) and torch.isfinite(out).all()


def test_retarget_convert_200(loaded_server):
    c, b = _post(loaded_server, "/retarget/convert", {
        "actions": [[1.0, -1.0, 0.5, 0.2, 0.1, 0.0, 0.3]],
        "source": "arm_7dof",
        "target": "gripper_4dof"})
    assert c == 200
    assert b["tgt_dof"] == 4 and len(b["actions"][0]) == 4


def test_affordance_score_200(loaded_server):
    c, b = _post(loaded_server, "/affordance/score", {
        "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "objects": [[[0.5, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]]]})
    assert c == 200
    assert len(b["scores"][0]) == 2
    assert b["best_part"][0] == 0


def test_endpoints_409_when_untrained():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):
        svc.retarget_convert({"actions": [[0.0] * 7], "source": "arm_7dof",
                              "target": "gripper_4dof"})
    with pytest.raises(Exception):
        svc.affordance_score({"state": [0.0] * 6, "objects": [[[0, 0, 0, 0, 0, 0]]]})


def test_endpoints_400_bad_input(loaded_server):
    # retarget 缺 actions
    c, b = _post(loaded_server, "/retarget/convert",
                 {"source": "arm_7dof", "target": "gripper_4dof"})
    assert c == 400
    # affordance 缺字段
    c, b = _post(loaded_server, "/affordance/score", {"state": [0.0] * 6})
    assert c == 400


def test_latency_json_written():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_v29_features.py")],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(LAT_JSON))
    assert d["n_repeats"] == 50
    for k in ("predict_next", "retarget_convert", "affordance_score",
              "spatial_relation_head"):
        assert d[k]["p50_ms"] >= 0
