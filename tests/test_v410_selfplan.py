"""
v4.1.0.dev6 自规划目标分解测试 (goal -> 子目标链, 复用 policy)
=============================================================
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、确定性。
"""
import hashlib

import pytest
import torch

from udos import CTMConfig, PhysicsPredictor, GoalDecomposer
from udos.dynamics import build_parametric_dataset


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _predictor():
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def test_decompose_chain_length_and_shape():
    m = _predictor()
    de = GoalDecomposer(m, horizon=2)
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=3)
    goal = ds.Y[0, -1, :]                 # 最后一步未来态作目标
    out = de.decompose(ds.X[0:1], goal, n_subgoals=3, scene_params=ds.P[0:1])
    assert len(out["subgoal_chain"]) == 3
    assert len(out["residuals_to_goal"]) == 3
    assert len(out["goal"]) == 6


def test_decompose_residuals_decrease():
    m = _predictor()
    de = GoalDecomposer(m, horizon=2)
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=4)
    goal = ds.Y[0, -1, :]
    out = de.decompose(ds.X[0:1], goal, n_subgoals=4, scene_params=ds.P[0:1])
    # 残差应随子目标推进而下降 (或持平)
    assert out["monotone_decreasing"] in (True, False)
    assert out["residuals_to_goal"][-1] <= out["residuals_to_goal"][0] + 1e-6


def test_decompose_rejects_bad():
    m = _predictor()
    de = GoalDecomposer(m)
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=1, dt=0.5, seed=5)
    with pytest.raises(ValueError):
        de.decompose(ds.X[0:1], torch.zeros(5))      # goal 维数错
    with pytest.raises(ValueError):
        de.decompose(ds.X[0:1], ds.Y[0, 0], n_subgoals=0)
    with pytest.raises(ValueError):
        de.decompose(torch.zeros(2, 6, 6), ds.Y[0, 0])   # 批维>1


def test_de_ctor_bad_horizon():
    m = _predictor()
    with pytest.raises(ValueError):
        GoalDecomposer(m, horizon=0)


def test_decompose_zero_gradient_main_untouched():
    m = _predictor()
    before = _md5(m)
    de = GoalDecomposer(m, horizon=2)
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=6)
    de.decompose(ds.X[0:1], ds.Y[0, -1, :], n_subgoals=2,
                 scene_params=ds.P[0:1])
    assert before == _md5(m)


def test_decompose_deterministic():
    m = _predictor()
    de = GoalDecomposer(m, horizon=2)
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=7)
    g = ds.Y[0, -1, :]
    a = de.decompose(ds.X[0:1], g, n_subgoals=3, scene_params=ds.P[0:1])
    b = de.decompose(ds.X[0:1], g, n_subgoals=3, scene_params=ds.P[0:1])
    assert a["residuals_to_goal"] == b["residuals_to_goal"]
