"""v5.5.2 四类运动识别 + 估计路由 + 类型相关不确定性 (TDD)。"""
from __future__ import annotations

import math

import pytest
import torch

from udos.dynamics import (
    build_parametric_dataset, traj_uniform, traj_accel, traj_spring,
    traj_collision, SCENE_PARAM_NAMES,
)
from udos.dynamics_router import (
    classify_dynamics, routed_scene_params, DynamicsRoute,
    KindSpecificParamErrorModel, apply_kind_inflation,
    fit_kind_conformal_inflation,
    CLASS_NAMES, JUMP_K, JUMP_FLOOR,
)


def _win(traj, w: int = 6, dt: float = 0.5) -> torch.Tensor:
    raw = torch.tensor([[list(p) + list(v)] for p, v in traj],
                       dtype=torch.float32).squeeze(1)
    return raw[:w]


def test_class_names_align_dynamics():
    assert tuple(CLASS_NAMES) == ("uniform", "accel", "spring", "collision")


def test_uniform_window_classified_and_routed():
    w = _win(traj_uniform(8, 0.5, v0=1.3, x0=0.2)).unsqueeze(0)
    r = classify_dynamics(w, 0.5)
    assert isinstance(r, DynamicsRoute)
    assert r.labels.tolist() == [CLASS_NAMES.index("uniform")]
    p = routed_scene_params(w, 0.5)
    assert p.shape == (1, 4)
    assert p[0, 0].item() == pytest.approx(1.3, abs=1e-4)
    assert torch.count_nonzero(p[0, 1:]) == 0          # a/omega/v2 全 0


def test_accel_window_classified_and_routed():
    w = _win(traj_accel(8, 0.5, v0=0.4, a=1.1)).unsqueeze(0)
    r = classify_dynamics(w, 0.5)
    assert r.labels.tolist() == [CLASS_NAMES.index("accel")]
    p = routed_scene_params(w, 0.5)
    assert p[0, 1].item() == pytest.approx(1.1, abs=6e-2)
    assert p[0, 2].item() == 0.0 and p[0, 3].item() == 0.0


def test_spring_window_classified_and_routed_no_v0_contamination():
    omega = 1.1
    w = _win(traj_spring(8, 0.5, amp=1.6, omega=omega, phi=0.3)).unsqueeze(0)
    r = classify_dynamics(w, 0.5)
    assert r.labels.tolist() == [CLASS_NAMES.index("spring")]
    p = routed_scene_params(w, 0.5)
    assert p[0, 0].item() == 0.0 and p[0, 1].item() == 0.0   # 不再污染 v0/a
    assert p[0, 2].item() == pytest.approx(omega, abs=4e-2)
    assert p[0, 3].item() == 0.0


def test_collision_window_with_jump_classified():
    # 让碰撞发生在窗口内: p1=-2 v1=2.4, p2 靠近
    traj = traj_collision(8, 0.5, x1=-2.0, v1=2.4, x2=1.2, v2=0.1)
    w = _win(traj).unsqueeze(0)
    r = classify_dynamics(w, 0.5)
    # 含跳变的窗口必须识别为 collision, 且路由参数不含 omega
    assert r.labels.tolist() == [CLASS_NAMES.index("collision")]
    p = routed_scene_params(w, 0.5)
    assert p[0, 2].item() == 0.0


def test_router_validates_input():
    with pytest.raises(ValueError):
        classify_dynamics(torch.randn(6, 5), 0.5)          # 末维不是 6
    bad = _win(traj_uniform(8, 0.5, 1.0)).unsqueeze(0).clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        classify_dynamics(bad, 0.5)
    with pytest.raises(ValueError):
        classify_dynamics(_win(traj_uniform(8, 0.5, 1.0)).unsqueeze(0), 0.0)


def test_router_batch_deterministic_and_shapes():
    ds = build_parametric_dataset(n_per_kind=8, seed=1)
    r = classify_dynamics(ds.X, ds.dt)
    assert r.labels.shape == (ds.X.size(0),)
    assert set(r.labels.tolist()).issubset({0, 1, 2, 3})
    p1 = routed_scene_params(ds.X, ds.dt)
    p2 = routed_scene_params(ds.X, ds.dt)
    assert p1.shape == (ds.X.size(0), 4) and torch.equal(p1, p2)
    assert torch.isfinite(p1).all()


def test_heldout_spring_recall_high_and_zero_uniform_false_positive():
    """决策树在独立 seed 上: 弹簧高召回, 匀速零误判为弹簧。"""
    ds = build_parametric_dataset(n_per_kind=128, seed=2026)
    r = classify_dynamics(ds.X, ds.dt)
    spr = CLASS_NAMES.index("spring"); uni = CLASS_NAMES.index("uniform")
    true = torch.tensor([CLASS_NAMES.index(k) for k in ds.kinds])
    spring_mask = true == spr
    recall = float((r.labels[spring_mask] == spr).float().mean())
    fp_uniform = int(((r.labels == spr) & (true == uni)).sum())
    assert recall >= 0.98
    assert fp_uniform == 0


# ---- 类型相关误差模型 ----
def test_kind_specific_error_model_fit_and_bound_sample():
    torch.manual_seed(0)
    B, K = 20, 4
    p_true = torch.zeros(B, 4)
    p_hat = p_true + 0.1
    true_labels = torch.arange(K).repeat_interleave(B // K)
    model = KindSpecificParamErrorModel.fit(p_hat, p_true, true_labels, n_kinds=K)
    assert model.bias.shape == (K, 4) and model.std.shape == (K, 4)
    assert (model.std > 0).all()
    pred_labels = true_labels.clone()
    bound = model.bind(pred_labels)
    s = bound.sample(p_hat, n=300, generator=torch.Generator().manual_seed(1))
    assert s.shape == (300, B, 4)
    # 去偏后样本中心在 p_true 附近 (按采样维)
    assert torch.allclose(s.mean(dim=0), p_true, atol=3e-2)


def test_apply_kind_inflation_geometry():
    labels = torch.tensor([0, 1])
    fan_med = torch.zeros(2, 3, 6)
    fan_low = fan_med - 1.0
    fan_high = fan_med + 1.0
    infl = torch.tensor([1.0, 2.0])
    low, high = apply_kind_inflation(fan_low, fan_med, fan_high, infl, labels)
    assert torch.allclose(low[0], fan_med[0] - 1.0)
    assert torch.allclose(high[0], fan_med[0] + 1.0)
    assert torch.allclose(low[1], fan_med[1] - 2.0)
    assert torch.allclose(high[1], fan_med[1] + 2.0)


def test_apply_kind_inflation_validates():
    with pytest.raises(ValueError):
        apply_kind_inflation(torch.zeros(2, 1, 6), torch.zeros(2, 1, 6),
                             torch.zeros(2, 1, 6), torch.tensor([1.0, -0.2]),
                             torch.tensor([0, 1]))
