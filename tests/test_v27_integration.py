"""
v2.7.1 集成加固: 跨特性组合 + 默认路径逐位一致
=================================================
锚点:
    * policy + online + active + lite + hierarchical 组合调用互不冲突、互不污染;
    * 默认路径 (不挂载任何 2.7 新特性) 与直接 predict_next 逐位一致;
    * 各模块 opt-in 独立工作;
    * 任一特性调用前后, predict_next 输出逐位不变 (无副作用)。
"""
import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.policy import MPCActionSelector
from udos.online import OnlineAdapter
from udos.active_learning import UncertaintySampler
from udos.lite import DynamicQuantizer
from udos.hierarchical import HierarchicalRollout
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    m.eval()
    return m


@pytest.fixture(scope="module")
def data():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=401)
    return ds.X[:1], ds.P[:1], ds.X[:5], ds.P[:5]


def test_version():
    assert __version__ == "5.5.5"


def test_default_path_unchanged(predictor, data):
    """不调用任何新特性时, predict_next 两次逐位一致 (默认路径无副作用)。"""
    w, p, _, _ = data
    a = predictor.predict_next(w, scene_params=p).clone()
    b = predictor.predict_next(w, scene_params=p)
    assert torch.equal(a, b)


def test_combined_features_no_conflict(predictor, data):
    """policy + active + hierarchical + online 组合跑一遍, 互不报错。"""
    w, p, pool, pool_p = data
    # policy
    mpc = MPCActionSelector(predictor, horizon=2)
    r1 = mpc.select(w, scene_params=p, candidate_actions=[{}, {}])
    assert r1["no_valid_action"] is False
    # active
    idx, scores = UncertaintySampler().select_top_k(predictor, pool, 3,
                                                    scene_params=pool_p)
    assert idx.numel() == 3
    # hierarchical
    hr = HierarchicalRollout(predictor, coarse_factor=2).rollout(
        w, horizon=8, scene_params=p)
    assert hr["predictions"].shape == (1, 8, predictor.raw_dim)
    # online
    adapter = OnlineAdapter(pool)
    adapter.observe(w)
    res = adapter.check_and_adapt(predictor, build_parametric_dataset(
        n_per_kind=8, n_steps=14, window=6, horizon=4, dt=0.5, seed=999))
    assert "adapted" in res
    # 全部成功 => 无冲突
    assert r1["best_action"] is not None


def test_features_do_not_mutate_predictor(predictor, data):
    """调用各特性前后 predict_next 逐位一致 (外挂只读)。"""
    w, p, pool, pool_p = data
    before = predictor.predict_next(w, scene_params=p).clone()
    MPCActionSelector(predictor, horizon=2).select(
        w, scene_params=p, candidate_actions=[{}, {}])
    UncertaintySampler().score_samples(predictor, pool, pool_p)
    HierarchicalRollout(predictor, coarse_factor=2).rollout(
        w, horizon=4, scene_params=p)
    after = predictor.predict_next(w, scene_params=p)
    assert torch.equal(before, after)


def test_lite_quantize_independent(predictor, data):
    """lite 量化是独立副本, 不改原模型; 量化模型在容差内输出。"""
    w, p, _, _ = data
    full = predictor.predict_next(w, scene_params=p)
    q = DynamicQuantizer.quantize(predictor)
    qout = q.predict_next(w, scene_params=p)
    # INT8 动态量化允许小误差, 但不应发散 (与 test_v27_lite 同纪律)
    assert (qout - full).abs().max().item() < 0.5
    assert torch.isfinite(qout).all()
    # 原模型仍逐位一致
    assert torch.equal(predictor.predict_next(w, scene_params=p), full)


def test_each_feature_independent(predictor, data):
    """单独开启任一特性均工作 (opt-in 独立)。"""
    w, p, pool, pool_p = data
    assert MPCActionSelector(predictor, horizon=1).select(
        w, scene_params=p, candidate_actions=[{}])["best_action"] == {}
    assert UncertaintySampler().score_samples(predictor, pool, pool_p).numel() == 5
    assert HierarchicalRollout(predictor, coarse_factor=4).rollout(
        w, horizon=2, scene_params=p)["predictions"].shape[1] == 2
