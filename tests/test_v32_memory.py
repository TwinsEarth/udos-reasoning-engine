"""
v3.2.0.dev3 节点44: TemporalMemory 滑动窗口记忆 + 摘要
========================================================
锚点纪律:
    * 记忆更新正确 (增长/超容量淘汰最旧);
    * EMA 摘要向量正确 (首帧=该帧, 后续指数平滑);
    * 长时域信息保留 (缓冲可查询超出窗口的旧帧);
    * reset 清空;
    * 与 predictor 集成 (扩展窗口预测有限);
    * 空记忆 / 维度守卫。
"""
from pathlib import Path

import pytest
import torch

from udos.temporal_memory import TemporalMemory
from udos.extended_context import ExtendedContextWindow
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.1.0.pt"


@pytest.fixture(scope="module")
def pred():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


def test_update_and_evict():
    mem = TemporalMemory(capacity=3)
    for i in range(5):
        mem.update(torch.full((6,), float(i)))
    assert mem.buffered == 3
    # 保留最近 3 帧 (i=2,3,4)
    hist = mem.history(3)
    assert torch.equal(hist[:, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_ema_summary():
    mem = TemporalMemory(capacity=8, ema_alpha=0.5)
    mem.update(torch.zeros(6))
    s0 = mem.summary()
    assert torch.equal(s0, torch.zeros(6))
    mem.update(torch.full((6,), 2.0))
    s1 = mem.summary()
    # 0.5*2 + 0.5*0 = 1.0
    assert torch.allclose(s1, torch.ones(6), atol=1e-6)


def test_long_horizon_retention():
    mem = TemporalMemory(capacity=100)
    for i in range(50):
        mem.update(torch.tensor([float(i), 0, 0, 0, 0, 0]))
    # 能查到超出近期窗口的旧帧 (第 0 帧仍在缓冲)
    old = mem.history(50)
    assert float(old[0, 0]) == 0.0
    assert float(old[-1, 0]) == 49.0


def test_reset():
    mem = TemporalMemory(capacity=4)
    mem.update(torch.ones(6))
    mem.reset()
    assert mem.buffered == 0
    try:
        mem.summary()
        assert False, "应报 RuntimeError"
    except RuntimeError:
        pass


def test_empty_memory_guard():
    mem = TemporalMemory(capacity=4)
    try:
        mem.summary()
        assert False
    except RuntimeError:
        pass
    try:
        mem.update(torch.zeros(3))   # 错误维度
        assert False
    except ValueError:
        pass


def test_build_extended_window(pred):
    ds = build_parametric_dataset(n_per_kind=3, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=21)
    mem = TemporalMemory(capacity=16)
    # 推入历史帧
    for t in range(ds.X[0].size(0)):
        mem.update(ds.X[0, t])
    w = ds.X[0]                      # [6,6]
    ext = mem.build_extended_window(w, k=4)
    assert ext.shape == (10, 6)
    # 尾部仍是原窗口
    assert torch.equal(ext[4:], w)
    # 集成 predictor (单条 -> batch=1)
    ecw = ExtendedContextWindow(pred, max_len=24)
    out = ecw.predict_next(ext.unsqueeze(0), scene_params=ds.P[0:1],
                           use_pe=True)
    assert out.shape == (1, 6)
    assert torch.isfinite(out).all()


def test_k_zero_returns_original(pred):
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=22)
    mem = TemporalMemory(capacity=8)
    mem.update(ds.X[0, 0])
    w = ds.X[0]
    ext = mem.build_extended_window(w, k=0)
    assert torch.equal(ext, w)
