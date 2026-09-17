"""v2.4.10 回归: per_step_confidence 置信矩阵形状/值域/与标量置信一致 + /evaluate 段。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset, RAW_DIM  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _tiny_model():
    return PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)


def test_matrix_shape_and_range():
    model = _tiny_model(); model.eval()
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=41)
    out = model.per_step_confidence(ds.X, horizon=3, scene_params=ds.P)
    assert out.shape == (len(ds), 3, RAW_DIM)
    assert bool((out >= 0.0).all() and (out <= 1.0).all())


def test_consistent_with_scalar_confidence():
    """未挂 conformal 半宽时, 逐维广播 -> mean(dim=-1) 应等于逐步标量 certainty。"""
    model = _tiny_model(); model.eval()
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=42)
    out = model.per_step_confidence(ds.X, horizon=3, scene_params=ds.P)  # [N,3,R]
    # 第一步标量 certainty = evaluate 用的 certs[:,1,-1]
    _, certs, _, _ = model(ds.X, scene_params=ds.P)
    scalar0 = certs[:, 1, -1]
    # 广播下 mean over dims == step conf
    assert torch.allclose(out[:, 0, :].mean(dim=-1), scalar0, atol=1e-5)


def test_evaluate_contains_confidence_breakdown():
    model = _tiny_model(); model.eval()
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=43)
    rep = evaluate_predictor(model, ds)
    bd = rep["confidence_breakdown"]
    assert bd["shape"] == [3, RAW_DIM]
    assert len(bd["matrix"]) == 3 and len(bd["matrix"][0]) == RAW_DIM
    assert len(bd["mean_by_step"]) == 3
    assert 0.0 <= bd["min"] <= bd["max"] <= 1.0
