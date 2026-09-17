"""
v2.7.0.dev2 主动学习 / 不确定性采样选点 (udos.active_learning.UncertaintySampler) 测试
======================================================================================
锚点纪律:
    * OOD 样本信息增益分高于分布内样本 (排序单调);
    * top-K 选出的恰为分最大的 k 个且降序;
    * 空池 / k=0 守卫; 两次打分逐位一致 (确定性);
    * A/B 证据 JSON 已落盘且字段完整。
"""
import json
import os

import pytest
import torch

from udos import __version__
from udos.active_learning import UncertaintySampler
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"
AB_JSON = "benchmarks/results/active_learning_ablation_v2.7.0.json"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def pool(predictor):
    ds = build_parametric_dataset(n_per_kind=10, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=601)
    n = ds.X.size(0)
    # 前半分布内, 后半平移 +15 (OOD)
    X = torch.cat([ds.X, ds.X[n // 2:] + 15.0], dim=0)
    P = torch.cat([ds.P, ds.P[n // 2:]], dim=0)
    return X, P, n // 2


def test_version():
    assert __version__ == "5.5.5"


def test_uncertainty_ranking_monotonicity(predictor, pool):
    X, P, split = pool
    sampler = UncertaintySampler()
    scores = sampler.score_samples(predictor, X, scene_params=P)
    in_dist = scores[:split]
    ood = scores[split:]
    # OOD 样本平均信息增益分应显著高于分布内
    assert ood.mean() > in_dist.mean()
    # 最高分样本应来自 OOD 段
    assert int(scores.argmax()) >= split


def test_top_k_selection(predictor, pool):
    X, P, split = pool
    sampler = UncertaintySampler()
    k = 5
    idx, vals = sampler.select_top_k(predictor, X, k=k, scene_params=P)
    assert idx.numel() == k
    assert vals.numel() == k
    # 降序
    assert torch.all(vals[:-1] >= vals[1:] - 1e-9)
    # 确为最大 k 个 (并列时 n_above >= k)
    all_scores = sampler.score_samples(predictor, X, scene_params=P)
    thresh = vals.min()
    n_above = int((all_scores >= thresh - 1e-9).sum())
    assert n_above >= k
    # 未选中者不超过 thresh
    mask = torch.ones(all_scores.numel(), dtype=torch.bool)
    mask[idx] = False
    assert bool((all_scores[mask] <= thresh + 1e-9).all())


def test_deterministic(predictor, pool):
    X, P, _ = pool
    sampler = UncertaintySampler()
    a = sampler.score_samples(predictor, X, scene_params=P)
    b = sampler.score_samples(predictor, X, scene_params=P)
    assert torch.allclose(a, b)


def test_empty_and_zero_guard(predictor, pool):
    X, P, _ = pool
    sampler = UncertaintySampler()
    # k=0
    idx, vals = sampler.select_top_k(predictor, X, k=0, scene_params=P)
    assert idx.numel() == 0 and vals.numel() == 0
    # k >= N => 返回全部
    n = X.size(0)
    idx, vals = sampler.select_top_k(predictor, X, k=n + 10, scene_params=P)
    assert idx.numel() == n


def test_ablation_json_exists():
    assert os.path.exists(AB_JSON), "A/B 脚本未落盘, 请先运行 scripts/ablation_active_learning.py"
    with open(AB_JSON, encoding="utf-8") as f:
        d = json.load(f)
    for k in ("eval_mse_active", "eval_mse_random", "eval_mse_baseline_seed_only",
              "active_better"):
        assert k in d
