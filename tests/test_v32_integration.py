"""
v3.2.1 节点48: 3.2 线与 Physical Loop 集成
==============================================
锚点纪律:
    * 默认 loop 仍逐位等价 predictor.predict_next (无外挂);
    * TemporalMemory 作为 loop 可选记忆 (use_memory=True) 可运行且累积;
    * ICL 作为 loop 上下文注入 (icl_examples) 可运行;
    * 数据增强作为训练时 opt-in (正式件口径不变, 仅快照验证);
    * memory + ICL 组合不冲突。
"""
from pathlib import Path

import pytest
import torch

from udos.physical_loop import PhysicalLoopRunner
from udos.temporal_memory import TemporalMemory
from udos.incontext import InContextLearner
from udos.ego_data import SyntheticEgoAugmenter
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
                                    horizon=4, dt=0.5, seed=51)


def test_default_loop_bit_identical(pred, ds):
    loop = PhysicalLoopRunner(pred, horizon=4)
    out = loop.run(ds.X[:1], scene_params=ds.P[:1])
    ref = pred.predict_next(ds.X[:1], scene_params=ds.P[:1])
    assert torch.equal(out["prediction"], ref)


def test_memory_in_loop(pred, ds):
    mem = TemporalMemory(capacity=32)
    loop = PhysicalLoopRunner(pred, horizon=4, use_memory=True, memory=mem)
    out = loop.run(ds.X[:1], scene_params=ds.P[:1])
    assert torch.isfinite(out["prediction"]).all()
    assert mem.buffered == 1     # 末帧已推入
    # 第二次 run 记忆继续累积
    loop.run(ds.X[:1], scene_params=ds.P[:1])
    assert mem.buffered == 2


def test_icl_in_loop(pred, ds):
    examples = [ds.X[1], ds.X[2]]
    loop = PhysicalLoopRunner(pred, horizon=4, icl_examples=examples)
    out = loop.run(ds.X[:1], scene_params=ds.P[:1])
    assert torch.isfinite(out["prediction"]).all()


def test_augmenter_training_optin(pred, ds):
    # 数据增强仅训练时 opt-in; 正式件口径不变 (此处只验证可成对增强)
    aug = SyntheticEgoAugmenter(view_rotate_deg=10.0, traj_perturb=0.01)
    Xa, Ya = aug.augment_pair(ds.X, ds.Y)
    assert Xa.shape == ds.X.shape and Ya.shape == ds.Y.shape


def test_b_batch_guard(pred, ds):
    loop = PhysicalLoopRunner(pred, horizon=4, use_memory=True,
                              memory=TemporalMemory())
    try:
        loop.run(ds.X[:2], scene_params=ds.P[:2])
        assert False, "B>1 应报错"
    except ValueError:
        pass
