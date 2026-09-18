"""v7.0.3 解析运动学积分 + 学习门控混合头的契约测试。

覆盖：
- 未训练严格恒等（前向与 rollout，门零初始化 g≡0）；
- 恒定加速度解析积分步的数学正确性；
- ca_conf 干净分离 uniform/accel（≈1）与 spring/collision（≈0）；
- 门控正向饱和时把 accel 路由到解析解、spring（ca=0）仍走学习路径；
- use_kinematics=False 不构建门控；
- v7.0.2 checkpoint 非严格加载、门控零初始化；
- 端到端：小训练后门控被启用且 accel 盲路径误差实质下降（突变敏感性）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from udos7.contracts import DT, STATE_DIM
from udos7.dynamics import build_split, three_way_splits
from udos7.kinematics import KIN_DIM, kinematic_features
from udos7.model import WorldModelCore
from udos7.train import TrainConfig, fit

torch.set_num_threads(2)


def _window_from_constant_accel(p0, v0, a, w=6, dt=DT):
    """构造恒定加速度解析窗（t 居中），返回 [1,W,6]。"""
    t = (torch.arange(w) - (w - 1) / 2.0) * dt
    pos = p0[None, :] + v0[None, :] * t[:, None] + 0.5 * a[None, :] * t[:, None] ** 2
    vel = v0[None, :] + a[None, :] * t[:, None]
    return torch.cat([pos, vel], dim=1).unsqueeze(0)


def test_untrained_gate_is_identity_forward_and_rollout():
    m = WorldModelCore(window=6, hidden=32)
    m.eval()
    ds = build_split(seed=11, n_traj_per_kind=4)
    with torch.no_grad():
        assert torch.allclose(m(ds.X), ds.X[:, -1, :], atol=1e-6)
        P = m.rollout(ds.X, 4)
        # 未训练：解析门 g≡0 => 纯恒等，rollout 每帧都等于最后一帧
        assert torch.allclose(P, ds.X[:, -1:, :].repeat(1, 4, 1), atol=1e-6)


def test_analytic_step_constant_acceleration_exact():
    m = WorldModelCore(window=6, hidden=32)
    p0 = torch.tensor([0.0, 1.0, -2.0])
    v0 = torch.tensor([0.5, -0.3, 0.2])
    a = torch.tensor([0.4, 0.1, -0.2])
    w = _window_from_constant_accel(p0, v0, a)
    feats = kinematic_features(w, DT)
    nxt = m.analytic_kinematic_step(w[:, -1, :], feats)
    last = w[0, -1, :]
    p_last, v_last = last[:3], last[3:]
    expect_p = p_last + v_last * DT + 0.5 * a * DT ** 2
    expect_v = v_last + a * DT
    assert torch.allclose(nxt[0, :3], expect_p, atol=1e-4)
    assert torch.allclose(nxt[0, 3:], expect_v, atol=1e-4)


def test_ca_conf_separates_constant_from_oscillation_and_impact():
    ds = build_split(seed=2026, n_traj_per_kind=256)
    kin = kinematic_features(ds.X, DT)
    ca = kin[:, 10]
    # uniform/accel：恒定加速度模型成立 => ca_conf≈1（p05 也接近 1）
    for kind in ("uniform", "accel"):
        v = ca[ds.kind_mask(kind)]
        assert torch.quantile(v, 0.05) > 0.99
    # spring：ω 全部检出 => ca_conf 严格为 0
    assert float((ca[ds.kind_mask("spring")]).abs().max()) == 0.0
    # collision：跨跳变窗 ca=0（无跳变窗允许为 1，但整体不应全部误开）
    assert float(ca[ds.kind_mask("collision")].median()) == 0.0


def test_saturated_gate_routes_accel_to_analytic_but_spring_stays_learned():
    m = WorldModelCore(window=6, hidden=32)
    # 门控末层偏置置大正 => tanh 饱和≈1，g≈ca_conf
    with torch.no_grad():
        m.kin_gate[-1].bias.fill_(10.0)
    ds = build_split(seed=2026, n_traj_per_kind=128)
    with torch.no_grad():
        nxt = m(ds.X)
        kin = kinematic_features(ds.X, DT)
        analytic = m.analytic_kinematic_step(ds.X[:, -1, :], kin)
        acc = ds.kind_mask("accel")
        spr = ds.kind_mask("spring")
        # accel（ca=1）几乎完全采用解析解
        assert torch.allclose(nxt[acc], analytic[acc], atol=2e-3)
        # spring（ca=0）门严格关闭 => 未训练学习路径=恒等（最后一帧）
        assert torch.allclose(nxt[spr], ds.X[spr, -1, :], atol=1e-6)


def test_no_gate_when_kinematics_disabled():
    m = WorldModelCore(window=6, hidden=32, use_kinematics=False)
    assert m.kin_gate is None
    ds = build_split(seed=5, n_traj_per_kind=2)
    with torch.no_grad():
        assert torch.allclose(m(ds.X), ds.X[:, -1, :], atol=1e-6)


def test_v702_checkpoint_loads_with_zero_init_gate():
    from udos7.persistence import load_worldmodel
    ck = Path(__file__).resolve().parents[1] / "checkpoints7" / "worldmodel_v7.0.2.pt"
    if not ck.exists():
        pytest.skip("v7.0.2 checkpoint 不存在")
    m, meta = load_worldmodel(ck)
    missing = meta["meta"]["load_missing_zero_init"]
    assert any(k.startswith("kin_gate") for k in missing)
    # 门控末层缺失并零初始化 => 无论首层如何，门输出恒为 0（等价 v7.0.2）
    assert torch.allclose(m.kin_gate[2].weight,
                          torch.zeros_like(m.kin_gate[2].weight), atol=1e-8)
    assert torch.allclose(m.kin_gate[2].bias,
                          torch.zeros_like(m.kin_gate[2].bias), atol=1e-8)
    ds = build_split(seed=2026, n_traj_per_kind=8)
    with torch.no_grad():
        kin = kinematic_features(ds.X, DT)
        g = kin[:, 10:11] * torch.tanh(m.kin_gate(kin) / 2.0)
        assert torch.allclose(g, torch.zeros_like(g), atol=1e-8)


def test_small_training_gate_activates_and_cuts_accel_error():
    """突变敏感性：门可学 vs 门冻结，前者 accel 盲路径误差必须实质更低、门被启用。"""
    sp = three_way_splits(n_traj_per_kind=48)
    H = 4

    def train(freeze_gate):
        torch.manual_seed(0)
        m = WorldModelCore(window=6, hidden=64, n_layers=2)
        if freeze_gate:
            for p_ in m.kin_gate.parameters():
                p_.requires_grad_(False)
        fit(m, sp["train"], sp["val"],
            TrainConfig(epochs=40, batch=128, lr=2e-3, patience=40,
                        scene_dropout=0.5, seed=0))
        return m

    from udos7.metrics import evaluate
    m_off = train(True)
    m_on = train(False)
    b_off = evaluate(m_off, sp["test"], H, use_explicit=False)
    b_on = evaluate(m_on, sp["test"], H, use_explicit=False)
    assert b_on["blind_accel"] < 0.6 * b_off["blind_accel"]
    # spring 不得被门控拖累（不超过冻结对照的 1.3 倍）
    assert b_on["blind_spring"] < 1.3 * b_off["blind_spring"]
    # accel 门确实被启用
    ds = sp["test"]
    with torch.no_grad():
        kin = kinematic_features(ds.X, DT)
        g = kin[:, 10:11] * torch.tanh(m_on.kin_gate(kin) / 2.0)
        assert float(g[ds.kind_mask("accel")].mean()) > 0.3
