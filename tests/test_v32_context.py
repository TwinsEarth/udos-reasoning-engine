"""
v3.2.0.dev2 节点43: ExtendedContextWindow 长历史上下文
========================================================
锚点纪律:
    * 长窗口 (W=12/24) 输入形状正确、可推理;
    * 位置编码表形状/对称性;
    * W=6 默认路径逐位等价锚点 (use_pe=False, pad_to=None);
    * W=6/12/24 精度不崩 (均为有限值, 与 W=6 同量级);
    * 空窗口 / W=0 守卫。
"""
from pathlib import Path

import pytest
import torch

from udos.persistence import load_predictor
from udos.extended_context import ExtendedContextWindow, sinusoidal_pe
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.1.0.pt"


@pytest.fixture(scope="module")
def pred():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


def _ds(seed):
    return build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=seed)


def test_pe_table():
    pe = sinusoidal_pe(16, 32)
    assert pe.shape == (16, 32)
    # 位置 0 与位置 1 不同
    assert not torch.allclose(pe[0], pe[1])


def test_w6_bit_identical_anchor(pred):
    ecw = ExtendedContextWindow(pred, max_len=24)
    ds = _ds(3)
    X = ds.X[:8]
    ref = pred.predict_next(X, scene_params=ds.P[:8])
    got = ecw.predict_next(X, scene_params=ds.P[:8], use_pe=False)
    assert torch.equal(ref, got)   # 逐位一致


def test_long_window_shape(pred):
    ecw = ExtendedContextWindow(pred, max_len=24)
    # 用更长的历史: 取 n_steps=30, window=24 构造
    ds = build_parametric_dataset(n_per_kind=3, n_steps=30, window=24,
                                  horizon=2, dt=0.5, seed=5)
    X = ds.X[:4]   # [4,24,6]
    out = ecw.predict_next(X, scene_params=ds.P[:4], use_pe=True)
    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_w12_w24_precision_finite(pred):
    ecw = ExtendedContextWindow(pred, max_len=24)
    base = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=9)
    ref6 = ecw.predict_next(base.X[:6], scene_params=base.P[:6],
                            use_pe=False)
    # 构造 W=12 / W=24 的窗口: 用同一批轨迹拉长 (n_steps 足够)
    for W in (12, 24):
        ds = build_parametric_dataset(n_per_kind=3, n_steps=W + 4,
                                      window=W, horizon=2, dt=0.5, seed=W)
        out = ecw.predict_next(ds.X[:4], scene_params=ds.P[:4],
                               use_pe=True)
        assert out.shape == (4, 6)
        assert torch.isfinite(out).all()
        # 与 W=6 输出量级一致 (不会爆炸)
        assert out.abs().max() < ref6.abs().max() * 20 + 5.0


def test_truncate_and_pad(pred):
    ecw = ExtendedContextWindow(pred, max_len=10)
    big = torch.randn(2, 30, 6)
    trunc = ecw.truncate(big)
    assert trunc.shape == (2, 10, 6)
    # 截断保留最近帧
    assert torch.equal(trunc, big[:, -10:, :])
    small = torch.randn(2, 4, 6)
    padded = ecw.pad(small, 10)
    assert padded.shape == (2, 10, 6)
    # 尾部 (后 4 帧) 与原窗口一致 (首帧重复前插)
    assert torch.equal(padded[:, 6:, :], small)


def test_empty_window_guard(pred):
    ecw = ExtendedContextWindow(pred, max_len=24)
    try:
        ecw.predict_next(torch.zeros(2, 0, 6))
        assert False, "应抛 ValueError"
    except ValueError:
        pass
