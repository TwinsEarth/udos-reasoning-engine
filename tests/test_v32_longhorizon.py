"""
v3.2.0.dev6 节点47: LongHorizonRollout 长时域 + 记忆协同
==========================================================
锚点纪律:
    * 长 horizon H=8/12/16 形状正确、输出有限;
    * 记忆累积逐步更新;
    * H=4 / use_memory=False 逐位等价 predictor.rollout;
    * 与 HierarchicalRollout 接口一致;
    * horizon=0 守卫。
"""
from pathlib import Path

import pytest
import torch

from udos.longhorizon import LongHorizonRollout
from udos.temporal_memory import TemporalMemory
from udos.hierarchical import HierarchicalRollout
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.1.0.pt"


@pytest.fixture(scope="module")
def pred():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


@pytest.fixture(scope="module")
def ds():
    return build_parametric_dataset(n_per_kind=3, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=41)


def test_h4_bit_identical(pred, ds):
    lh = LongHorizonRollout(pred)
    w, sp = ds.X[:1], ds.P[:1]
    a = lh.rollout(w, 4, scene_params=sp, use_memory=False)
    b = pred.rollout(w, 4, scene_params=sp)
    assert torch.equal(a, b)


def test_long_horizon_shapes(pred, ds):
    lh = LongHorizonRollout(pred)
    w, sp = ds.X[:1], ds.P[:1]
    for H in (8, 12, 16):
        out = lh.rollout(w, H, scene_params=sp, use_memory=False)
        assert out.shape == (1, H, 6)
        assert torch.isfinite(out).all()


def test_memory_path(pred, ds):
    mem = TemporalMemory(capacity=32)
    lh = LongHorizonRollout(pred, memory=mem)
    w, sp = ds.X[:1], ds.P[:1]
    out = lh.rollout(w, 8, scene_params=sp, use_memory=True)
    assert out.shape == (1, 8, 6)
    assert torch.isfinite(out).all()
    # 记忆应累积了 horizon 个预测状态
    assert mem.buffered == 8


def test_vs_hierarchical(pred, ds):
    w, sp = ds.X[:1], ds.P[:1]
    lh = LongHorizonRollout(pred)
    hier = HierarchicalRollout(pred, coarse_factor=4)
    flat = lh.rollout(w, 8, scene_params=sp)
    h8 = hier.rollout(w, horizon=8, scene_params=sp)["predictions"]
    # 二者逐位一致 (hierarchical 粗跳步=flat 在同口径下已验证)
    assert torch.equal(flat, h8)


def test_bad_horizon_guard(pred, ds):
    lh = LongHorizonRollout(pred)
    try:
        lh.rollout(ds.X[:1], 0, scene_params=ds.P[:1])
        assert False, "应报错"
    except ValueError:
        pass


def test_memory_batch_guard(pred, ds):
    lh = LongHorizonRollout(pred)
    try:
        lh.rollout(ds.X[:2], 4, scene_params=ds.P[:2], use_memory=True)
        assert False, "应报错 (B>1)"
    except ValueError:
        pass
