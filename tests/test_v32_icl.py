"""
v3.2.0.dev5 节点46: InContextLearner
==========================================
锚点纪律:
    * 上下文拼接形状正确 (k 示例 + 查询);
    * 任务描述向量注入;
    * 0/1/3-shot 可运行且 MSE 有限;
    * 空示例守卫、示例形状不一致守卫;
    * 与 extended_context 接口一致。
"""
from pathlib import Path

import pytest
import torch

from udos.incontext import InContextLearner
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
    return build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=31)


def test_context_shape(pred, ds):
    icl = InContextLearner(pred, max_len=64)
    q = ds.X[0]                       # [6,6]
    exs = [ds.X[1], ds.X[2], ds.X[3]]
    ctx = icl.build_context(q, exs)
    assert ctx.shape == (1, 24, 6)    # 3*6 + 6
    ctx0 = icl.build_context(q, None)
    assert ctx0.shape == (1, 6, 6)


def test_predict_0shot(pred, ds):
    icl = InContextLearner(pred)
    out = icl.predict(ds.X[0], examples=None, scene_params=ds.P[0:1])
    assert out.shape == (6,)
    assert torch.isfinite(out).all()


def test_task_desc_injected(pred, ds):
    icl = InContextLearner(pred)
    td = torch.tensor([0.1, -0.2, 0.3, 0.4, 9.9])  # 取前 4 维
    out = icl.predict(ds.X[0], examples=[ds.X[1]], task_desc=td)
    assert out.shape == (6,) and torch.isfinite(out).all()


def test_shot_mse_0_1_3(pred, ds):
    icl = InContextLearner(pred)
    pool = [ds.X[i] for i in range(4, 12)]
    qw = ds.X[:6]
    qt = ds.Y[:6, 0, :]
    sp = ds.P[:6]
    m0 = icl.shot_mse(qw, qt, pool, 0, scene_params=sp)
    m1 = icl.shot_mse(qw, qt, pool, 1, scene_params=sp)
    m3 = icl.shot_mse(qw, qt, pool, 3, scene_params=sp)
    for m in (m0, m1, m3):
        assert m == m and m < 1e6    # finite, 不爆炸
    # 如实记录: 不强制 shot 越多越好 (冻结小模型 ICL 不保证)


def test_empty_examples_guard(pred, ds):
    icl = InContextLearner(pred)
    try:
        icl.shot_mse(ds.X[:2], ds.Y[:2, 0], [], 1)
        assert False, "应报错"
    except ValueError:
        pass


def test_shape_mismatch_guard(pred, ds):
    icl = InContextLearner(pred)
    bad = torch.randn(5, 6)   # W=5 != query W=6
    try:
        icl.build_context(ds.X[0], [bad])
        assert False, "应报错"
    except ValueError:
        pass


def test_task_desc_too_short(pred, ds):
    icl = InContextLearner(pred)
    try:
        icl.predict(ds.X[0], task_desc=torch.tensor([0.1, 0.2]))
        assert False, "应报错"
    except ValueError:
        pass
