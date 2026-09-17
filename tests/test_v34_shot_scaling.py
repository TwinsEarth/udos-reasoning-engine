"""v3.4.0.dev5 k-shot scaling 曲线 + 权重逐位不变锚点测试。"""
import hashlib

import pytest
import torch

from udos import load_predictor, __version__
from udos.dynamics import build_parametric_dataset
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator
from udos.incontext import InContextLearner

CKPT = "checkpoints/predictor_v3.4.0.pt"


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def setup():
    m, _ = load_predictor(CKPT)
    tr = build_parametric_dataset(n_per_kind=32, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4242)
    te = build_parametric_dataset(n_per_kind=24, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=7777)
    mem = DemonstrationMemory()
    agg = ICMAggregator(m, temperature=1.0, lamb=1.0)
    for i in range(len(tr)):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                 scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])
    return m, tr, te, mem, agg


def test_kshot_scaling_curve_finite(setup):
    """0/1/3/5/10-shot MSE 曲线有限且不退化。"""
    m, tr, te, mem, agg = setup
    idx = torch.arange(min(80, len(te)))
    curve = {}
    for k in (0, 1, 3, 5, 10):
        curve[k] = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=k,
                               scene_params=te.P[idx])
    print("ICM curve:", curve)
    for v in curve.values():
        assert v == v and v != float("inf")       # finite
        assert v < 0.2                              # 不爆炸
    # 10-shot 不应比 0-shot 退化 (检索聚合路径)
    assert curve[10] <= curve[0] * 1.5


def test_weight_md5_bitwise_anchor(setup):
    """权重不变锚点: 多轮 ICM 推理后 state_dict md5 逐位一致。"""
    m, tr, te, mem, agg = setup
    before = _md5(m)
    for k in (1, 3, 5, 10):
        for i in range(5):
            agg.predict(te.X[i], memory=mem, k=k, scene_params=te.P[i:i+1])
    after = _md5(m)
    assert before == after


def test_vs_naive_icl_compare(setup):
    """ICM 检索聚合 vs naive 朴素拼接: ICM k-shot 远不爆炸。"""
    m, tr, te, mem, agg = setup
    idx = torch.arange(min(40, len(te)))
    icm3 = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=3,
                        scene_params=te.P[idx])
    pool = [tr.X[j] for j in range(min(16, len(tr)))]
    icl = InContextLearner(m)
    naive3 = icl.shot_mse(te.X[idx], te.Y[idx, 0], pool, 3,
                          scene_params=te.P[idx])
    print(f"ICM k3={icm3:.5f}  naive k3={naive3:.5f}")
    assert icm3 < 0.2
    assert naive3 > icm3 * 3.0     # naive 显著更差 (对照)


def test_empty_example_guard(setup):
    m, tr, te, mem, agg = setup
    empty_mem = DemonstrationMemory()
    out = agg.predict(te.X[0], memory=empty_mem, k=3, scene_params=te.P[0:1])
    assert out.shape == (6,)
    assert bool(torch.isfinite(out).all())


def test_version():
    assert __version__ == "5.5.5"
