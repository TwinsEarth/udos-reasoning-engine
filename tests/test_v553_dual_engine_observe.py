"""v5.5.3 双引擎可观测性 (TDD): 来源 / 置信 / 场景门贡献 / 盲-感知轨迹差。"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from udos.dynamics import build_parametric_dataset, traj_spring
from udos.dynamics_router import CLASS_NAMES
from udos.persistence import load_predictor
from udos.dual_engine_observe import observe_conditioning, EngineObservation

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "predictor_v4.3.9.pt"
HEAD_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "scene_head_v5.5.0.pt"
W, H, DT = 6, 4, 0.5


@pytest.fixture(scope="module")
def predictor():
    p, _ = load_predictor(CKPT)
    p.eval()
    return p


def _kind_window(ds, kind: str, seed_idx: int = 0):
    m = ds.kind_mask(kind)
    idx = torch.nonzero(m, as_tuple=False).flatten()
    i = int(idx[seed_idx])
    return ds.X[i:i + 1], ds.P[i:i + 1], ds.Y[i:i + 1]


def test_blind_observation_has_zero_gate_contribution(predictor):
    ds = build_parametric_dataset(n_per_kind=8, seed=2026)
    win, _, _ = _kind_window(ds, "uniform")
    obs = observe_conditioning(predictor, win, horizon=H, dt=DT)
    assert isinstance(obs, EngineObservation)
    assert obs.conditioned is False
    assert obs.source == "blind"
    assert obs.gate_position_delta.shape == (H,)
    assert torch.allclose(obs.blind_trajectory, obs.conditioned_trajectory)
    assert obs.gate_contribution == pytest.approx(0.0, abs=1e-7)
    assert float(obs.gate_position_delta.abs().sum()) == pytest.approx(0.0, abs=1e-7)


def test_explicit_spring_gate_contribution_large_and_improves(predictor):
    ds = build_parametric_dataset(n_per_kind=64, seed=2026)
    # 取若干弹簧窗, 场景门 (显式 omega) 应显著改变并改善轨迹
    m = ds.kind_mask("spring")
    idx = torch.nonzero(m, as_tuple=False).flatten()[:48]
    deltas, blind_err, cond_err = [], [], []
    for i in idx.tolist():
        win, P, Y = ds.X[i:i + 1], ds.P[i:i + 1], ds.Y[i:i + 1]
        obs = observe_conditioning(predictor, win, conditioned_params=P,
                                   source="metadata", horizon=H, dt=DT)
        deltas.append(obs.gate_position_delta)
        blind_err.append(float(((obs.blind_trajectory - Y) ** 2).mean()))
        cond_err.append(float(((obs.conditioned_trajectory - Y) ** 2).mean()))
    mean_final_delta = float(torch.stack(deltas)[:, -1].mean())
    assert obs.conditioned is True and obs.source == "metadata"
    assert obs.route_name == "spring"
    assert mean_final_delta > 0.1                  # 盲 vs 感知末步明显分叉
    assert sum(cond_err) < sum(blind_err) / 3      # 条件化显著更准


def test_gate_deltas_are_nonneg_and_shapes(predictor):
    ds = build_parametric_dataset(n_per_kind=16, seed=2026)
    win, P, _ = _kind_window(ds, "accel")
    obs = observe_conditioning(predictor, win, conditioned_params=P,
                               source="metadata", horizon=H, dt=DT)
    assert obs.blind_trajectory.shape == (H, 6)
    assert obs.conditioned_trajectory.shape == (H, 6)
    assert obs.gate_velocity_delta.shape == (H,)
    assert bool((obs.gate_position_delta >= 0).all())
    assert bool((obs.gate_velocity_delta >= 0).all())
    assert obs.gate_contribution >= 0.0 and obs.gate_final_delta >= 0.0


def test_route_confidence_unit_interval(predictor):
    ds = build_parametric_dataset(n_per_kind=32, seed=2026)
    for kind in CLASS_NAMES:
        m = ds.kind_mask(kind)
        i = int(torch.nonzero(m, as_tuple=False).flatten()[0])
        obs = observe_conditioning(predictor, ds.X[i:i + 1], horizon=H, dt=DT)
        assert 0.0 <= obs.route_confidence <= 1.0
        assert obs.route_name in CLASS_NAMES


def test_precomputed_conditioned_matches_recomputed(predictor):
    ds = build_parametric_dataset(n_per_kind=16, seed=2026)
    win, P, _ = _kind_window(ds, "spring", seed_idx=2)
    a = observe_conditioning(predictor, win, conditioned_params=P,
                             source="learned_head", horizon=H, dt=DT)
    with torch.no_grad():
        pre = predictor.rollout(win, H, scene_params=P)[0]
    b = observe_conditioning(predictor, win, conditioned_params=P,
                             source="learned_head", horizon=H, dt=DT,
                             precomputed_conditioned=pre)
    assert torch.allclose(a.conditioned_trajectory, b.conditioned_trajectory)
    assert torch.allclose(a.gate_position_delta, b.gate_position_delta)


def test_optional_fan_halfwidth(predictor):
    from udos.scene_fan import TrajectoryFan
    ds = build_parametric_dataset(n_per_kind=16, seed=2026)
    win, P, _ = _kind_window(ds, "spring")
    Hh = H
    low = torch.zeros(1, Hh, 6)
    high = torch.ones(1, Hh, 6)
    fan = TrajectoryFan(low=low, median=(low + high) / 2, high=high,
                        quantiles=(0.1, 0.5, 0.9))
    obs = observe_conditioning(predictor, win, conditioned_params=P,
                               source="metadata", horizon=H, dt=DT, fan=fan)
    assert obs.fan_position_halfwidth is not None
    # 位置三维半宽 = 0.5 (high-low=1), 向量范数 = sqrt(3)*0.5
    assert obs.fan_position_halfwidth[0].item() == pytest.approx(
        (3 * 0.5 ** 2) ** 0.5, abs=1e-5)


def test_observe_validates_input(predictor):
    bad = torch.randn(H, 5)
    with pytest.raises(ValueError):
        observe_conditioning(predictor, bad, horizon=H, dt=DT)
    ds = build_parametric_dataset(n_per_kind=4, seed=1)
    win, _, _ = _kind_window(ds, "uniform")
    with pytest.raises(ValueError):
        observe_conditioning(predictor, win, horizon=0, dt=DT)


def test_as_dict_is_json_friendly(predictor):
    ds = build_parametric_dataset(n_per_kind=8, seed=2026)
    win, P, _ = _kind_window(ds, "spring")
    obs = observe_conditioning(predictor, win, conditioned_params=P,
                               source="metadata", horizon=H, dt=DT)
    d = obs.as_dict()
    for key in ("source", "conditioned", "route_name", "route_confidence",
                "gate_contribution", "gate_final_delta",
                "gate_position_delta_per_step",
                "gate_velocity_delta_per_step"):
        assert key in d
    assert isinstance(d["gate_position_delta_per_step"], list)
    assert all(isinstance(v, float) for v in d["gate_position_delta_per_step"])


# ---- reason() opt-in 接线 ----
from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.pce_format import PhysicsScene, PhysicalToken
from udos.reasoning import UDOSReasoningEngine
from udos.scene_head import load_scene_head

OMEGA, AMP = 1.2, 1.5


def _engine(predictor, head=None):
    return UDOSReasoningEngine(
        CTMConfig(iterations=2, d_model=16, d_input=16, heads=2,
                  n_synch_out=8, n_synch_action=8, memory_length=4,
                  nlm_hidden=8, out_dims=16, certainty_threshold=0.0,
                  n_random_pairing_self=2),
        GPMConfig(feature_dim=16, latent_size=8, n_latents=4, lora_rank=4,
                  layer_indices=(0, 1), num_pre_head_layers=1, heads=2),
        base_model=TinyBaseModel(hidden=16, n_layers=2),
        predictor=predictor, scene_head=head)


def _spring_scene():
    traj = traj_spring(W + H, DT, amp=AMP, omega=OMEGA, phi=0.0)
    scene = PhysicsScene(scene_id="spring", duration=W + H)
    for i, (pos, vel) in enumerate(traj[:W]):
        scene.add(PhysicalToken(object_id="obj-a", timestamp=i,
                                position=pos, velocity=vel))
    return scene


def test_reason_observe_default_off(predictor):
    eng = _engine(predictor)
    r = eng.reason(_spring_scene(), horizon=H)
    assert r.gate_observation is None


@pytest.mark.skipif(not HEAD_CKPT.exists(),
                    reason="需先运行 scripts/train_scene_head.py 生成产物")
def test_reason_observe_optin_reports_gate(predictor):
    head, _ = load_scene_head(HEAD_CKPT)
    eng = _engine(predictor, head=head)
    r = eng.reason(_spring_scene(), horizon=H, observe=True)
    obs = r.gate_observation
    assert obs is not None
    assert obs["source"] == "learned_head"
    assert obs["conditioned"] is True
    assert obs["route_name"] == "spring"
    assert len(obs["gate_position_delta_per_step"]) == H
    assert obs["gate_final_delta"] > 0.1
    assert 0.0 <= obs["route_confidence"] <= 1.0
