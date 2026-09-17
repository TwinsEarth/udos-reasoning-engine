"""
v3.2.2 节点49: 集成加固 + backcompat 16 件 + 新服务端点 + 性能基准
====================================================================
锚点纪律:
    * checkpoints v2.1.0..v3.2.0 共 16 件全部可加载并 predict_next;
    * POST /augment/generate (无状态, 不需模型) 输出形状与配置;
    * POST /icl/predict 已训练可用 / 未训练 409;
    * benchmarks/results/feature_latency_v3.2.0.json 落盘。
"""
import json
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.server import UDOSService
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
LAT_JSON = ROOT / "benchmarks" / "results" / "feature_latency_v3.2.0.json"

# v2.1.0..v3.2.0 共 16 件
ALL_CKPTS = [
    "predictor_v2.1.0.pt", "predictor_v2.2.1.pt", "predictor_v2.3.1.pt",
    "predictor_v2.4.0.pt", "predictor_v2.5.0.pt", "predictor_v2.5.2.pt",
    "predictor_v2.6.0.pt", "predictor_v2.6.2.pt", "predictor_v2.7.0.pt",
    "predictor_v2.7.3.pt", "predictor_v2.8.0.pt", "predictor_v2.9.0.pt",
    "predictor_v3.0.0.pt", "predictor_v3.0.3.pt", "predictor_v3.1.0.pt",
    "predictor_v3.2.0.pt",
]


def test_backcompat_16_checkpoints():
    ds = build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                  horizon=2, dt=0.5, seed=99)
    assert len(ALL_CKPTS) == 16
    for name in ALL_CKPTS:
        path = CKPT_DIR / name
        assert path.exists(), f"{name} 缺失"
        m, _ = load_predictor(str(path))
        m.eval()
        out = m.predict_next(ds.X[:1], scene_params=ds.P[:1])
        assert out.shape == (1, 6) and torch.isfinite(out).all()


@pytest.fixture(scope="module")
def svc():
    return UDOSService(preset="small",
                       checkpoint=str(CKPT_DIR / "predictor_v3.2.0.pt"))


@pytest.fixture(scope="module")
def ds():
    return build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                    horizon=2, dt=0.5, seed=7)


def test_augment_generate(svc, ds):
    body = {"window": ds.X[0].tolist(), "view_rotate_deg": 20.0,
            "traj_perturb": 0.01, "noise_sigma": 0.005, "time_scale": 1.0,
            "seed": 1}
    out = svc.augment_generate(body)
    assert out["status"] == "ok"
    assert out["shape"] == [6, 6]
    assert out["config"]["view_rotate_deg"] == 20.0


def test_augment_generate_no_model_needed(ds):
    svc = UDOSService(preset="small")   # 未加载 checkpoint
    out = svc.augment_generate({"window": ds.X[0].tolist()})
    assert out["status"] == "ok"


def test_augment_bad_input():
    svc = UDOSService(preset="small")
    try:
        svc.augment_generate({})
        assert False
    except ValueError:
        pass


def test_icl_predict(svc, ds):
    out = svc.icl_predict({
        "window": ds.X[0].tolist(),
        "examples": [ds.X[1].tolist(), ds.X[2].tolist()],
        "scene_params": ds.P[0].tolist(),
    })
    assert out["status"] == "ok"
    assert out["shape"] == [6]
    assert all(v == v for v in out["prediction"])   # finite


def test_icl_predict_untrained_409(ds):
    from udos.server import ServiceNotReady
    svc = UDOSService(preset="small")
    try:
        svc.icl_predict({"window": ds.X[0].tolist()})
        assert False
    except ServiceNotReady:
        pass


def test_latency_json():
    d = json.loads(LAT_JSON.read_text(encoding="utf-8"))
    for k in ("predict_ms", "rollout_ms", "augment_ms", "icl_ms"):
        assert k in d
        assert d[k] > 0


def test_version():
    assert __version__ == "5.5.5"
