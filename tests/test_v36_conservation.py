"""v3.6.0.dev3 物理守恒一致性检验测试: ConservationChecker。

覆盖:
    * 匀速直线段 m*v 与 0.5*m*v^2 严格守恒 (违反≈0, conserved=True);
    * 加速/发散轨迹 => 动量/能量违反量 >0 (conserved=False);
    * 弹簧势下 K+U 守恒 (合成 spring 轨迹可验);
    * 空序列 (<2 步) / 非法质量 / 非有限输入守卫;
    * 纯解析: 不依赖主 predictor。
"""
import math
import pytest
import torch

from udos import __version__
from udos.wm_conservation import ConservationChecker
from udos.dynamics import traj_spring


def test_version():
    assert __version__ == "5.5.5"


def _states_from_traj(traj):
    return torch.tensor([[*p, *v] for p, v in traj],
                        dtype=torch.float32).unsqueeze(0)


def test_uniform_motion_conserved():
    # 匀速 v0=1.5: pos 线性增长, vel 恒定 => m*v 与动能严格守恒
    dt = 0.5
    traj = [([t * dt * 1.5, 0, 0], [1.5, 0, 0]) for t in range(8)]
    s = torch.tensor([[*p, *v] for p, v in traj], dtype=torch.float32).unsqueeze(0)
    out = ConservationChecker(mass=2.0).check(s)
    r = out["batch"][0]
    assert r["conserved"] is True
    assert r["momentum_violation"] < 1e-5
    assert r["energy_violation"] < 1e-5
    # 动量大小 = m*|v| = 2*1.5 = 3.0 每步
    assert all(abs(mp - 3.0) < 1e-4 for mp in r["momentum_per_step"])


def test_accelerating_detected_as_violation():
    # 匀加速: 速度逐段变化 => 动量大小漂移 > tol
    dt = 0.5
    rows = []
    for t in range(8):
        v = 0.5 * t
        rows.append(([0.5 * t, 0, 0], [v, 0, 0]))
    s = torch.tensor([[*p, *v] for p, v in rows], dtype=torch.float32).unsqueeze(0)
    out = ConservationChecker(mass=1.0).check(s)
    r = out["batch"][0]
    assert r["momentum_violation"] > 1e-3
    assert r["conserved"] is False


def test_spring_total_energy_conserved():
    # 简谐振动: K + 0.5*k*x^2 近似守恒 (omega 与 k 一致: k = m*omega^2)
    omega = 1.2
    mass = 1.0
    k = mass * omega * omega
    traj = traj_spring(n_steps=24, dt=0.2, amp=1.5, omega=omega, phi=0.3)
    s = _states_from_traj(traj)
    out = ConservationChecker(mass=mass, spring_k=k).check(
        s, momentum_tol=1e-2, energy_tol=5e-2)
    r = out["batch"][0]
    # 弹簧总能量 K+U 应近似守恒 (时间步有限的一阶误差内)
    assert r["energy_violation"] < 0.1
    # 但动量大小 |m*v| 在振动中变化 => 不要求动量守恒
    assert "momentum_violation" in r


def test_empty_sequence_guard():
    with pytest.raises(ValueError):
        ConservationChecker().check(torch.zeros(1, 1, 6))   # 仅 1 步
    with pytest.raises(ValueError):
        ConservationChecker().check(torch.zeros(1, 0, 6))


def test_bad_mass_guard():
    with pytest.raises(ValueError):
        ConservationChecker(mass=0.0)
    with pytest.raises(ValueError):
        ConservationChecker(mass=-1.0)


def test_non_finite_guard():
    bad = torch.tensor([[[0, 0, 0, 1, 0, 0],
                         [1, 0, 0, float("inf"), 0, 0]]])
    with pytest.raises(ValueError):
        ConservationChecker().check(bad)


def test_is_conserved_helper():
    dt = 0.5
    rows = [([t * dt, 0, 0], [1.0, 0, 0]) for t in range(6)]
    s = torch.tensor([[*p, *v] for p, v in rows], dtype=torch.float32).unsqueeze(0)
    assert ConservationChecker().is_conserved(s) is True
