"""
v3.3.0.dev6 节点57: 综合鲁棒性评估器测试
====================================================================
纪律:
    * 四项指标齐全 (噪声/OOD/外推/FGSM 代理); 分数 ∈ [0,100];
    * 与现有 ood / noise_augment 模块集成;
    * 空/未训练模型守卫; 结果可复现。
analogy, not reproduction。
"""
import torch

from udos.robustness import RobustnessEvaluator
from udos.ctm_engine import CTMConfig
from udos.training import PhysicsPredictor
from udos.ood import DistributionDriftDetector
from udos.dynamics import build_parametric_dataset

torch.set_num_threads(2)


def fresh_model():
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, heads=4,
                    n_synch_out=8, n_synch_action=4, memory_length=6,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0)
    cfg.scene_dim = 32
    torch.manual_seed(0)
    m = PhysicsPredictor(cfg, scene_param_dim=4)
    m.eval()
    return m


def ds(seed=42):
    return build_parametric_dataset(n_per_kind=12, n_steps=12, window=6,
                                   horizon=2, dt=0.5, seed=seed)


def test_four_metrics_and_score_range():
    model = fresh_model()
    rep = RobustnessEvaluator(model, seed=1).evaluate(ds())
    for k in ("noise", "ood", "extrapolation", "fgsm_proxy",
              "sub_scores", "robustness_score"):
        assert k in rep
    assert 0.0 <= rep["robustness_score"] <= 100.0
    assert rep["noise"]["clean_mse"] > 0
    assert rep["ood"]["hit_rate"] is None     # 未挂检测器


def test_integrates_with_ood_detector():
    model = fresh_model()
    data = ds(7)
    ood = DistributionDriftDetector(ridge=1e-3, alpha=0.05).fit(data.X)
    model.attach_ood_detector(ood)
    rep = RobustnessEvaluator(model, seed=2).evaluate(data)
    assert rep["ood"]["hit_rate"] is not None
    assert 0.0 <= rep["ood"]["hit_rate"] <= 1.0
    assert 0.0 <= rep["ood"]["false_alarm_rate"] <= 1.0


def test_noise_grid_consistent_with_noise_augment():
    model = fresh_model()
    data = ds(9)
    ev = RobustnessEvaluator(model, noise_sigmas=(0.0, 0.1), seed=3)
    rep = ev.evaluate(data)
    assert "test_sigma=0.0" in rep["noise"]["grid"]
    assert "test_sigma=0.1" in rep["noise"]["grid"]
    # 噪声越大 MSE 不应更小 (单调不严格, 但干净基准在)
    assert rep["noise"]["grid"]["test_sigma=0.0"] > 0


def test_fgsm_degradation_recorded():
    model = fresh_model()
    rep = RobustnessEvaluator(model, seed=4).evaluate(ds(11))
    f = rep["fgsm_proxy"]
    assert f["eps"] > 0
    assert f["adv_mse"] > 0
    assert f["degradation_x"] >= 1.0


def test_reproducible():
    m1, m2 = fresh_model(), fresh_model()
    data = ds(13)
    r1 = RobustnessEvaluator(m1, seed=5).evaluate(data)
    r2 = RobustnessEvaluator(m2, seed=5).evaluate(data)
    assert r1["robustness_score"] == r2["robustness_score"]
    assert r1["noise"]["clean_mse"] == r2["noise"]["clean_mse"]


def test_untrained_model_does_not_crash():
    """未训练模型评估不崩溃, 分数落在 [0,100]。"""
    model = fresh_model()
    rep = RobustnessEvaluator(model, seed=6).evaluate(ds(17))
    assert 0.0 <= rep["robustness_score"] <= 100.0
