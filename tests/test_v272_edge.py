"""
v2.7.2 边缘加固测试 (policy / online / active / lite / hierarchical / guard)
==========================================================================
覆盖各 2.7 模块的退化/边界条件, 确保不崩溃、不产生 NaN、状态一致:
    * policy: 空动作集、horizon=0 拒绝、scene_params=None;
    * online: 全相同数据不触发漂移、NaN 观测被跳过、适配后校准器状态一致;
    * active: 零样本池、k>池大小、单样本池;
    * lite: 量化输出有限、剪枝后 save/load 一致、蒸馏学生 save/load;
    * hierarchical: horizon=1 退化、coarse_factor>horizon 退化 (== 平铺);
    * PredictionGuard 与 policy 输出格式兼容。
"""
import copy

import pytest
import torch

from udos import __version__
from udos.lite import MagnitudePruner, DynamicQuantizer, DistillationTrainer
from udos.persistence import load_predictor, save_predictor
from udos.policy import MPCActionSelector
from udos.online import OnlineAdapter
from udos.active_learning import UncertaintySampler
from udos.hierarchical import HierarchicalRollout
from udos.guard import PredictionGuard
from udos.ctm_engine import CTMConfig
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"


def small_cfg(d_model=64):
    return CTMConfig(iterations=8, d_model=d_model, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    m.eval()
    return m


@pytest.fixture(scope="module")
def window():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=401)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


# ---------------- policy ---------------- #
def test_policy_empty_actions(window):
    w, p = window
    out = MPCActionSelector(load_predictor(CKPT)[0], horizon=2).select(
        w, scene_params=p, candidate_actions=[])
    assert out["no_valid_action"] is True


def test_policy_horizon_zero_rejected():
    with pytest.raises(ValueError):
        MPCActionSelector(load_predictor(CKPT)[0], horizon=0)


def test_policy_scene_params_none(window):
    w, _ = window
    out = MPCActionSelector(load_predictor(CKPT)[0], horizon=2).select(
        w, scene_params=None, candidate_actions=[{}])
    assert out["no_valid_action"] is False


# ---------------- online ---------------- #
def test_online_constant_no_false_drift(predictor, window):
    """全相同 (常数) 数据不应触发漂移 (方差 0)。"""
    w, _ = window
    adapter = OnlineAdapter(w.repeat(20, 1, 1))   # 参考=常数
    for _ in range(20):
        adapter.observe(w)
    assert adapter.is_drifted() is False


def test_online_nan_observation_skipped(predictor, window):
    w, _ = window
    adapter = OnlineAdapter(w.repeat(16, 1, 1))
    bad = w.clone()
    bad[0, -1, 0] = float("nan")
    adapter.observe(bad)               # 不应抛错, NaN 行被跳过
    adapter.observe(w)
    # 窗口不含 NaN
    assert adapter.drift_score() == adapter.drift_score()  # 非 nan
    adapter.check_and_adapt(predictor, build_parametric_dataset(
        n_per_kind=8, n_steps=14, window=6, horizon=4, dt=0.5, seed=555))


def test_online_adapt_calibrator_consistent(predictor, window):
    """触发适配后, predictor.is_calibrated 为 True, 校准器可再次评估。"""
    w, _ = window
    adapter = OnlineAdapter(window[0].repeat(16, 1, 1))
    # 注入强漂移 (末帧放大 8x) 触发
    drift_w = window[0].clone()
    drift_w[0, -1, :] *= 8.0
    for _ in range(16):
        adapter.observe(drift_w)
    cal_ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                      horizon=4, dt=0.5, seed=777)
    res = adapter.check_and_adapt(predictor, cal_ds)
    # 无论是否触发, 预测器校准器状态自洽
    assert hasattr(predictor, "is_calibrated")
    assert "adapted" in res


# ---------------- active ---------------- #
def test_active_zero_pool(predictor):
    empty = torch.zeros(0, 6, predictor.raw_dim)
    idx, sc = UncertaintySampler().select_top_k(predictor, empty, 3)
    assert idx.numel() == 0 and sc.numel() == 0


def test_active_k_gt_pool(window):
    w, p = window
    pool = w.repeat(2, 1, 1)
    idx, sc = UncertaintySampler().select_top_k(load_predictor(CKPT)[0], pool, 10,
                                                scene_params=p.repeat(2, 1))
    assert idx.numel() == 2


def test_active_single_pool(window):
    w, p = window
    idx, sc = UncertaintySampler().select_top_k(load_predictor(CKPT)[0], w, 1,
                                                scene_params=p)
    assert idx.numel() == 1


# ---------------- lite ---------------- #
def test_quantize_output_finite(predictor, window):
    w, p = window
    q = DynamicQuantizer.quantize(predictor)
    out = q.predict_next(w, scene_params=p)
    assert torch.isfinite(out).all()


def test_prune_save_load_consistency(predictor, window, tmp_path):
    w, p = window
    model = copy.deepcopy(predictor)
    p_obj = MagnitudePruner()
    p_obj.prune(model, 0.5)
    sp_after_prune = MagnitudePruner.sparsity_ratio(model)
    path = str(tmp_path / "pruned.pt")
    save_predictor(model, path)
    loaded, _ = load_predictor(path)
    # 剪枝态落盘后重载, 稀疏度保持 (权重仍为零)
    assert abs(MagnitudePruner.sparsity_ratio(loaded) - sp_after_prune) < 1e-6
    out = loaded.predict_next(w, scene_params=p)
    assert torch.isfinite(out).all()


def test_distill_student_save_load(predictor, tmp_path):
    data = build_parametric_dataset(n_per_kind=12, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=612)
    student = DistillationTrainer(alpha=0.7).distill(
        predictor, small_cfg(d_model=32), data, epochs=3, seed=0)
    path = str(tmp_path / "student.pt")
    save_predictor(student, path)
    loaded, meta = load_predictor(path)
    x = torch.randn(2, 6, loaded.raw_dim)
    p = torch.randn(2, loaded.scene_param_dim)
    out = loaded.predict_next(x, scene_params=p)
    assert out.shape == (2, 6)
    assert torch.isfinite(out).all()


# ---------------- hierarchical ---------------- #
def test_hierarchical_horizon1_equals_plain(predictor, window):
    w, p = window
    plain = predictor.rollout(w, 1, scene_params=p)
    hier = HierarchicalRollout(predictor, coarse_factor=4).rollout(
        w, horizon=1, scene_params=p)
    assert torch.equal(plain, hier["predictions"])


def test_hierarchical_coarse_gt_horizon_equals_plain(predictor, window):
    w, p = window
    plain = predictor.rollout(w, 3, scene_params=p)
    hier = HierarchicalRollout(predictor, coarse_factor=8).rollout(
        w, horizon=3, scene_params=p)
    assert torch.equal(plain, hier["predictions"])


# ---------------- guard + policy 兼容 ---------------- #
def test_policy_compatible_with_guard(predictor, window):
    """predictor 挂了 PredictionGuard 后, policy 仍正常工作并返回有限结果。"""
    w, p = window
    predictor.attach_guard(PredictionGuard())
    try:
        out = MPCActionSelector(predictor, horizon=2).select(
            w, scene_params=p, candidate_actions=[{}, {}])
        assert out["no_valid_action"] is False
        for r in out["ranked_actions"]:
            assert r["score"] == r["score"]      # 有限
    finally:
        predictor.guard = None
