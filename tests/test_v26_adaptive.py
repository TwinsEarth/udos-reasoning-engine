"""
v2.6.0+dev3 自适应计算 (自适应 iterations + horizon) 单元测试
===============================================================
锚点纪律:
    * stopper=None 时 predict_next_adaptive 与旧 predict_next 逐位一致。
    * AdaptiveStopper 纯逻辑: 收敛停 / 未收敛不停 / 点数不足不停。
    * 未挂 conformal 半宽时 adaptive_rollout 跑满 max_horizon。
    * 构造快速增长的半宽 => 提前截断。
"""
import pytest
import torch

from udos import __version__
from udos.adaptive import AdaptiveStopper, adaptive_rollout
from udos.ctm_engine import CTMConfig
from udos.dynamics import build_parametric_dataset
from udos.training import PhysicsPredictor
from udos.persistence import load_predictor

CKPT = "checkpoints/predictor_v2.6.0.pt"


def _cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


@pytest.fixture(scope="module")
def model():
    m, _ = load_predictor(CKPT)
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_adaptive_disabled_equivalence(model):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=7)
    xb, pb = ds.X[:4], ds.P[:4]
    ref = model.predict_next(xb, scene_params=pb)
    ad = model.predict_next_adaptive(xb, scene_params=pb, stopper=None)
    assert torch.allclose(ref, ad, atol=1e-6)


def test_stopper_logic():
    s = AdaptiveStopper(threshold=1e-3, patience=3)
    # 收敛: 末 3 步变化都 < 阈值
    assert s.should_stop([0.1, 0.10001, 0.10002, 0.10001]) is True
    # 未收敛: 仍在大幅波动
    assert s.should_stop([0.1, 0.5, 0.1, 0.5]) is False
    # 点数不足
    assert s.should_stop([0.1, 0.10001]) is False


def test_adaptive_rollout_no_quantile():
    torch.manual_seed(1)
    m = PhysicsPredictor(_cfg(), scene_param_dim=4)
    assert m.residual_quantiles is None
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=9)
    out = adaptive_rollout(m, ds.X[:2], max_horizon=5, scene_params=ds.P[:2])
    assert out["horizon_used"] == 5
    assert out["truncated"] is False
    assert out["trajectory"].shape[1] == 5


def test_adaptive_rollout_truncation(model):
    """构造快速增长的 conformal 半宽 => 第 2 步触发宽度增长率阈值, 提前截断。"""
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=10)
    # 临时挂载快速膨胀的半宽 (第 1 步 1.0 -> 第 2 步 100.0, 增长率 100x > 2.0)
    saved = model.residual_quantiles
    model.residual_quantiles = [torch.tensor([1.0]), torch.tensor([100.0])]
    try:
        out = adaptive_rollout(model, ds.X[:2], max_horizon=6,
                               scene_params=ds.P[:2],
                               width_growth_threshold=2.0)
        assert out["truncated"] is True
        assert out["reason"] == "interval_width_growth"
        assert out["horizon_used"] == 2
    finally:
        model.residual_quantiles = saved
