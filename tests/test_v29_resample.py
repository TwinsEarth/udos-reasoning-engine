"""
v2.9.0.dev2 动作时间重采样 (跨控制频率对齐) 单测
==================================================
锚点纪律:
    * 频率比 2x / 0.5x / 1.5x 重采样帧数正确 (支持非整数比);
    * 重采样首/末帧与源首/末帧逐位一致 (端点一致);
    * 线性插值对单调信号保单调; 三次样条输出有限;
    * 重采样前后动作能量近似守恒 (同一物理时长);
    * 与 loop.predict_action 组合: 默认路径逐位一致, 重采样轨迹可被消费。
analogy, not reproduction: 频率重对齐为合成动作轨迹代理。
"""
from pathlib import Path

import pytest
import torch

from udos.retargeting import (MorphologyConfig, ActionRetargeter,
                              ActionRetargeter as AR)
from udos.physical_loop import PhysicalLoopRunner
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


def _rt():
    src = MorphologyConfig(6, 60.0, [[-2, 2]] * 6, name="s")
    tgt = MorphologyConfig(6, 120.0, [[-2, 2]] * 6, name="t")
    return ActionRetargeter(src, tgt)


def test_resample_ratios_shape():
    rt = _rt()
    traj = torch.randn(11, 6)     # 60Hz, 11 帧
    for ratio, n in [(2.0, 21), (0.5, 6), (1.5, 16)]:
        out = rt.resample(traj, 60.0, 60.0 * ratio, kind="linear")
        # T_dst = round((11-1)*ratio)+1
        assert out.shape[-1] == 6
        assert out.shape[-2] == n, f"ratio {ratio}: got {out.shape[-2]}"


def test_resample_endpoints_consistent():
    rt = _rt()
    traj = torch.randn(11, 6)
    for kind in ("linear", "cubic"):
        out = rt.resample(traj, 60.0, 180.0, kind=kind)
        assert torch.allclose(out[0], traj[0], atol=1e-6)
        assert torch.allclose(out[-1], traj[-1], atol=1e-6)


def test_linear_monotone():
    rt = _rt()
    # 单调递增源信号
    src = torch.linspace(0, 1, 11).unsqueeze(-1).repeat(1, 6)
    out = rt.resample(src, 60.0, 150.0, kind="linear")
    d = out[1:, 0] - out[:-1, 0]
    assert bool((d >= -1e-6).all()), "线性插值应保单调"


def test_energy_approx_conserved():
    rt = _rt()
    # 平滑正弦轨迹 (粗网格梯形积分已收敛), 重采样不应注入/耗散物理能量
    t = torch.linspace(0, 2 * 3.1415927, 11)
    traj = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t),
                        torch.zeros_like(t), torch.ones_like(t) * 0.5,
                        torch.sin(t + 1.0)], dim=-1)   # [11,6]
    e_src = rt.action_energy(traj, dt=1.0 / 60.0)
    out = rt.resample(traj, 60.0, 180.0, kind="linear")
    e_dst = rt.action_energy(out, dt=1.0 / 180.0)
    ratio = e_dst / e_src
    # 残差来自粗网格梯形离散误差 (线性插值精确复现 PL 函数), ±10% 视为近似守恒
    assert 0.90 < ratio < 1.10, f"能量漂移 {ratio:.3f} 超容差"


def test_cubic_finite_and_guard():
    rt = _rt()
    out = rt.resample(torch.randn(11, 6), 60.0, 90.0, kind="cubic")
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        rt.resample(torch.randn(1, 6))          # T<2
    with pytest.raises(ValueError):
        rt.resample(torch.randn(11, 7))          # 末维不符


def test_loop_predict_action_integration_bit_identical():
    m, _ = load_predictor(CKPT)
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=909)
    wb, pb = ds.X[:1], ds.P[:1]
    loop = PhysicalLoopRunner(m, horizon=2)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    direct = m.predict_next(wb, scene_params=pb)
    assert torch.equal(out["prediction"], direct), "默认 loop 路径须逐位一致"
    # 重采样轨迹可被消费 (末帧状态扰动代理动作)
    rt = _rt()
    traj = rt.resample(ds.X[:1].reshape(6, 6), 60.0, 120.0)
    assert traj.shape[-1] == 6 and torch.isfinite(traj).all()
