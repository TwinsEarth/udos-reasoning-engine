"""v3.4.0.dev4 上下文预算/检索压缩 A/B 测试。"""
import json
import os

import pytest
import torch

from udos import load_predictor, __version__
from udos.dynamics import build_parametric_dataset
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator
from udos.icm_budget import ContextBudgetManager

CKPT = "checkpoints/predictor_v3.4.0.pt"


@pytest.fixture(scope="module")
def setup():
    m, _ = load_predictor(CKPT)
    tr = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4242)
    mem = DemonstrationMemory()
    agg = ICMAggregator(m)
    for i in range(len(tr)):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                  scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])
    return m, mem, agg


def test_cap_k_respects_budget(setup):
    _, _, _ = setup
    bm = ContextBudgetManager(budget=4)
    assert bm.cap_k(16, available=100) == 4
    assert bm.cap_k(2, available=100) == 2       # 请求小于预算, 取请求
    assert bm.cap_k(16, available=2) == 2        # 可用不足, 取可用
    assert bm.cap_k(0, 100) == 0


def test_no_budget_means_unlimited(setup):
    _, _, _ = setup
    bm = ContextBudgetManager()                  # opt-in 默认不限
    assert bm.cap_k(50, available=100) == 50


def test_compress_no_op_when_nproto_ge_k(setup):
    _, _, _ = setup
    bm = ContextBudgetManager(n_proto=None)
    res = [torch.randn(6) for _ in range(3)]
    out, tot = bm.compress(res, [0.5, 0.3, 0.2])
    assert out.shape == (6,)
    assert abs(tot - 1.0) < 1e-5


def test_compress_reduces_buckets(setup):
    _, _, _ = setup
    bm = ContextBudgetManager(n_proto=2)
    res = [torch.randn(6) for _ in range(5)]
    out, tot = bm.compress(res, [0.4, 0.3, 0.15, 0.1, 0.05])
    assert out.shape == (6,)
    assert tot == pytest.approx(1.0, abs=1e-5)


def test_budget_shrinks_k_in_predict(setup):
    m, mem, agg = setup
    te = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=99)
    # 不限预算
    bm_none = ContextBudgetManager()
    k1 = bm_none.cap_k(8, mem.size)
    # 预算=2
    bm_small = ContextBudgetManager(budget=2)
    k2 = bm_small.cap_k(8, mem.size)
    assert k2 == 2 and k1 > k2
    out = agg.predict(te.X[0], memory=mem, k=k2, scene_params=te.P[0:1])
    assert bool(torch.isfinite(out).all())


def test_budget_ab_json_written():
    p = "benchmarks/results/icm_budget_ab_v3.4.0.json"
    assert os.path.exists(p), "A/B JSON 应已由 scripts/icm_budget_ab_v340.py 落盘"
    d = json.load(open(p, encoding="utf-8"))
    assert d["feature"] == "icm_context_budget"
    assert len(d["budgets"]) == 4
    for row in d["budgets"]:
        assert {"budget", "k_eff", "mse", "latency_ms_per_query"} <= row.keys()


def test_bad_budget():
    with pytest.raises(ValueError):
        ContextBudgetManager(budget=0)
    with pytest.raises(ValueError):
        ContextBudgetManager(n_proto=0)


def test_version():
    assert __version__ == "5.5.5"
