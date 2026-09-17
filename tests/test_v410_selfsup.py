"""
v4.1.0.dev3 自监督伪信号测试 (PWM rollout 一致性伪标签)
=========================================================
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、确定性。
"""
import hashlib

import pytest
import torch

from udos import CTMConfig, PhysicsPredictor, PWMConsistencyPseudoLabeler
from udos.world_model import LatentWorldModel
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


def _wm_fitted(m):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=7)
    wm = LatentWorldModel(m)
    wm.fit(ds, epochs=10)
    return wm, ds


def test_pseudo_label_shape_and_consistency():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    pl = PWMConsistencyPseudoLabeler(m, wm=wm)
    out = pl.pseudo_label(ds.X[0:1], horizon=3, scene_params=ds.P[0:1])
    assert out["pseudo_label_states"].shape == (1, 3, 6)
    assert out["step_mse"].shape == (3,)
    c = out["consistency"]
    assert bool((c >= 0.0).all()) and bool((c <= 1.0).all())
    assert 0.0 <= out["mean_consistency"] <= 1.0


def test_lazy_fit_from_dataset():
    m = _predictor()
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=8)
    pl = PWMConsistencyPseudoLabeler(m, fit_dataset=ds, fit_epochs=8)
    out = pl.pseudo_label(ds.X[0:1], horizon=2, scene_params=ds.P[0:1])
    assert out["wm_params"] > 0


def test_no_wm_no_dataset_raises():
    m = _predictor()
    pl = PWMConsistencyPseudoLabeler(m)
    with pytest.raises(RuntimeError):
        pl.pseudo_label(torch.zeros(1, 6, 6), horizon=2)


def test_pseudo_label_rejects_bad():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    pl = PWMConsistencyPseudoLabeler(m, wm=wm)
    with pytest.raises(ValueError):
        pl.pseudo_label(torch.zeros(2, 6, 6))     # 批维>1
    with pytest.raises(ValueError):
        pl.pseudo_label(torch.zeros(1, 6, 6), horizon=0)


def test_zero_gradient_main_untouched():
    m = _predictor()
    before = _md5(m)
    wm, ds = _wm_fitted(m)
    pl = PWMConsistencyPseudoLabeler(m, wm=wm)
    for i in range(4):
        pl.pseudo_label(ds.X[i:i + 1], horizon=2, scene_params=ds.P[i:i + 1])
    assert before == _md5(m)


# ---------------- v4.1.0.dev4 物理守恒 + 多视角一致伪信号门 ---------------- #

from udos import PhysicsMultiviewGate  # noqa: E402


def _traj(scale=1.0, n=4):
    # 匀速直线: pos = v*t, vel 恒定 => 动量/动能严格守恒
    t = torch.arange(n, dtype=torch.float32)
    pos = torch.stack([1.0 * t, 0.5 * t, 0.0 * t], dim=-1) * scale
    vel = torch.stack([torch.full((n,), scale), torch.full((n,), 0.5 * scale),
                       torch.zeros(n)], dim=-1)
    return torch.cat([pos, vel], dim=-1).unsqueeze(0)   # [1,n,6]


def test_conservation_score_uniform_motion():
    g = PhysicsMultiviewGate()
    s = g.conservation_score(_traj())
    assert s["conserved"] is True
    assert s["conservation_score"] > 0.99


def test_multiview_score_finite():
    g = PhysicsMultiviewGate()
    m = g.multiview_score(_traj())
    assert 0.0 <= m["multiview_score"] <= 1.0
    assert m["multiview_error"] >= 0.0


def test_gate_combines_and_bounds():
    g = PhysicsMultiviewGate()
    out = g.gate(_traj(), pwm_consistency=0.9)
    assert 0.0 <= out["final_pseudo_weight"] <= 0.9 + 1e-6
    # 匀速运动守恒门接近 1, 多视角门接近 1
    assert out["conservation_score"] > 0.99


def test_gate_rejects_bad():
    g = PhysicsMultiviewGate()
    with pytest.raises(ValueError):
        g.gate(torch.zeros(1, 1, 6))              # 不足 2 步
    with pytest.raises(ValueError):
        g.gate(torch.zeros(1, 4, 5))              # 维数错


def test_gate_ctor_bad_scale():
    with pytest.raises(ValueError):
        PhysicsMultiviewGate(conservation_scale=0.0)


def test_gate_zero_gradient_untouched():
    m = _predictor()
    before = _md5(m)
    g = PhysicsMultiviewGate()
    g.gate(_traj())
    assert before == _md5(m)
