"""
v5.5.1 参数不确定性 -> 蒙特卡洛 p10/p50/p90 轨迹扇形 + 覆盖率
=============================================================
契约:
  1. 参数误差模型: 逐槽位 bias/std 有限、std>0、形状 [4]; 采样去偏且形状正确;
  2. 扇形: low<=median<=high, 形状 [B,H,6], 固定 seed 可复现;
  3. 零方差误差模型 -> 所有样本相同 -> 扇形退化为点 rollout;
  4. 非法分位数三元组显式拒绝;
  5. split-conformal 膨胀在独立 held-out 上把 pooled 覆盖率拉到名义 80% 附近
     (校准集拟合, held 验证, 不泄漏), 且不低于原始 MC 覆盖的合理下界。
分类型欠/过覆盖是全局高斯的已知性质 (交 v5.5.2 路由收紧), 本版只守卫 pooled。
"""

from pathlib import Path

import pytest
import torch

from udos.dynamics import build_parametric_dataset
from udos.persistence import load_predictor
from udos.scene_fan import (ParamErrorModel, TrajectoryFan,
                            calibrated_band, coverage_fraction,
                            fit_conformal_inflation, monte_carlo_rollout,
                            per_step_coverage)
from udos.scene_head import load_scene_head

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "predictor_v4.3.9.pt"
HEAD = Path(__file__).resolve().parents[1] / "checkpoints" / "scene_head_v5.5.0.pt"
H = 4


@pytest.fixture(scope="module")
def predictor():
    p, _ = load_predictor(CKPT)
    p.eval()
    return p


@pytest.fixture(scope="module")
def head():
    h, _ = load_scene_head(HEAD)
    return h


def test_error_model_fit_shapes(head):
    ds = build_parametric_dataset(n_per_kind=32, seed=5)
    em = ParamErrorModel.fit(head, ds)
    assert em.bias.shape == (4,)
    assert em.std.shape == (4,)
    assert torch.isfinite(em.bias).all() and torch.isfinite(em.std).all()
    assert (em.std > 0).all()


def test_sample_shape_and_debias(head):
    ds = build_parametric_dataset(n_per_kind=16, seed=5)
    em = ParamErrorModel.fit(head, ds)
    with torch.no_grad():
        phat = head(ds.X)
    g = torch.Generator().manual_seed(0)
    samples = em.sample(phat, 4000, g)
    assert samples.shape == (4000, ds.X.size(0), 4)
    # 大样本逐槽位总均值应在去偏中心 (phat-bias) 的总体均值附近
    grand = samples.mean(dim=(0, 1))
    center = (phat - em.bias).mean(dim=0)
    assert torch.allclose(grand, center, atol=0.03)
    # 固定样本、仅在采样维上的 std 应接近拟合 std (避免混入中心的样本间方差)
    assert torch.allclose(samples.std(dim=0).mean(dim=0), em.std, rtol=0.12)


def test_fan_ordering_shape_and_determinism(predictor, head):
    ds = build_parametric_dataset(n_per_kind=16, seed=9)
    em = ParamErrorModel.fit(head, ds)
    with torch.no_grad():
        phat = head(ds.X)
    g1 = torch.Generator().manual_seed(42)
    fan = monte_carlo_rollout(predictor, ds.X, phat, em, 32, H, generator=g1)
    B = ds.X.size(0)
    assert fan.low.shape == (B, H, 6)
    assert torch.all(fan.low <= fan.median + 1e-6)
    assert torch.all(fan.median <= fan.high + 1e-6)
    g2 = torch.Generator().manual_seed(42)
    fan2 = monte_carlo_rollout(predictor, ds.X, phat, em, 32, H, generator=g2)
    assert torch.allclose(fan.low, fan2.low, atol=1e-6)


def test_zero_variance_degenerates_to_point_rollout(predictor, head):
    ds = build_parametric_dataset(n_per_kind=8, seed=2)
    em = ParamErrorModel(bias=torch.zeros(4), std=torch.zeros(4))
    with torch.no_grad():
        phat = head(ds.X)
        point = predictor.rollout(ds.X, H, scene_params=phat)
    fan = monte_carlo_rollout(predictor, ds.X, phat, em, 5, H,
                              generator=torch.Generator().manual_seed(0))
    assert torch.allclose(fan.low, point, atol=1e-6)
    assert torch.allclose(fan.high, point, atol=1e-6)


def test_invalid_quantiles_rejected(predictor, head):
    ds = build_parametric_dataset(n_per_kind=8, seed=2)
    em = ParamErrorModel.fit(head, ds)
    with torch.no_grad():
        phat = head(ds.X)
    with pytest.raises(ValueError):
        monte_carlo_rollout(predictor, ds.X, phat, em, 5, H,
                            quantiles=(0.1, 0.5))
    with pytest.raises(ValueError):
        monte_carlo_rollout(predictor, ds.X, phat, em, 5, H,
                            quantiles=(0.9, 0.5, 0.1))


def test_coverage_helpers():
    low = torch.zeros(2, H, 6)
    high = torch.ones(2, H, 6)
    truth = torch.full((2, H, 6), 0.5)
    assert coverage_fraction(low, high, truth) == 1.0
    truth_out = torch.full((2, H, 6), 2.0)
    assert coverage_fraction(low, high, truth_out) == 0.0
    pc = per_step_coverage(low, high, truth)
    assert len(pc) == H and all(c == 1.0 for c in pc)


@pytest.mark.skipif(not HEAD.exists(), reason="需 v5.5.0 场景头产物")
def test_conformal_calibrated_pooled_coverage(predictor, head):
    cal = build_parametric_dataset(n_per_kind=64, seed=314)
    held = build_parametric_dataset(n_per_kind=64, seed=2026)
    em = ParamErrorModel.fit(head, cal)
    g = torch.Generator().manual_seed(7)
    inflation = fit_conformal_inflation(
        predictor, head, cal, em, n_samples=40, horizon=H,
        nominal=0.80, generator=g)

    def bands(ds, gen):
        with torch.no_grad():
            phat = head(ds.X)
        fan = monte_carlo_rollout(predictor, ds.X, phat, em, 40, H,
                                  generator=gen)
        lo, hi = calibrated_band(fan, inflation)
        raw = coverage_fraction(fan.low, fan.high, ds.Y)
        cal_cov = coverage_fraction(lo, hi, ds.Y)
        return raw, cal_cov

    raw, cal_cov = bands(held, torch.Generator().manual_seed(8))
    assert 0.77 <= cal_cov <= 0.88            # 名义 80%, 有限样本容差
    assert cal_cov >= raw - 0.02              # 膨胀不应显著降低覆盖
