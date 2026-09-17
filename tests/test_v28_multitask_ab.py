"""
v2.8.1 FutureStateHead + 多任务 A/B 单元测试
===============================================
锚点纪律:
    * FutureStateHead 输出 trajectory[B,H,6] + uncertainty[B,H,1] 有限;
    * A/B 脚本落盘 benchmarks/results/multitask_ab_v2.8.0.json 且可复算;
    * opt-in 默认关 (MultiTaskHead.enable=False) => 旧 predict_next 逐位一致;
    * enable=True 后三头 (spatial/action/future) 联合推理;
    * 被否决候选 (三头默认开启) 保留在 JSON。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.multitask import (MultiTaskHead, SpatialCoordHead, ActionTrajectoryHead,
                            FutureStateHead)
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset, RAW_DIM

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.8.0.pt")
AB_JSON = ROOT / "benchmarks" / "results" / "multitask_ab_v2.8.0.json"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=5050)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_future_head_shape_finite(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("future", FutureStateHead(32, horizon=4, state_dim=RAW_DIM))
    out = mth.forward(wb, scene_params=pb)
    fut = out["future"]
    assert fut["trajectory"].shape == (1, 4, RAW_DIM)
    assert fut["uncertainty"].shape == (1, 4, 1)
    assert torch.isfinite(fut["trajectory"]).all()
    assert (fut["uncertainty"] >= 0).all()


def test_ab_json_written_and_recomputable():
    # 真实运行 A/B 脚本落盘
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "multitask_ab_v28.py")],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert AB_JSON.exists()
    d = json.load(open(AB_JSON))
    # 可复算字段
    assert d["n_heads"] == 3
    assert d["shared_backbone"]["encode_calls"] == 1
    assert d["independent_heads"]["encode_calls"] == 3
    assert d["latency_speedup_x"] >= 1.0
    assert "rejected_candidate" in d and d["rejected_candidate"]


def test_optin_default_off_bit_identical(predictor, window_batch):
    wb, pb = window_batch
    # 默认 enable=False
    mth = MultiTaskHead(predictor, latent_dim=32)
    assert mth.enable is False
    assert mth.forward(wb, scene_params=pb) == {}
    before = predictor.predict_next(wb, scene_params=pb)
    z = mth.encode(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_three_heads_joint_inference(predictor, window_batch):
    wb, pb = window_batch
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("spatial", SpatialCoordHead(32, n_pts=4))
    mth.register_head("action", ActionTrajectoryHead(32, horizon=4, action_dim=RAW_DIM))
    mth.register_head("future", FutureStateHead(32, horizon=4, state_dim=RAW_DIM))
    out = mth.forward(wb, scene_params=pb)
    assert set(out.keys()) == {"spatial", "action", "future"}
    assert torch.isfinite(out["spatial"]).all()
    assert torch.isfinite(out["action"]).all()
    assert torch.isfinite(out["future"]["trajectory"]).all()
