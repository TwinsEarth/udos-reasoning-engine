"""v2.4.5 回归: 退化守卫 (NaN/inf 回退 / 越界截断 / guard=False 透传)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.guard import PredictionGuard  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_nan_inf_fallback_to_zero():
    g = PredictionGuard()
    pred = torch.tensor([[1.0, float("nan"), 3.0],
                         [float("inf"), -1.0, 2.0]])
    out = g.sanitize(pred)
    assert torch.isfinite(out).all()
    assert g.n_fallbacks >= 1


def test_out_of_bounds_clipped():
    g = PredictionGuard(low=-1.0, high=1.0)
    pred = torch.tensor([[0.5, 5.0, -3.0], [0.0, 0.2, -1.5]])
    out = g.sanitize(pred)
    assert bool((out <= 1.0 + 1e-6).all())
    assert bool((out >= -1.0 - 1e-6).all())
    assert g.n_clips > 0


def test_fallback_to_last_valid():
    g = PredictionGuard()
    v = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    g.sanitize(v)                       # 建立 last_valid
    bad = torch.tensor([[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = g.sanitize(bad)
    assert torch.isfinite(out).all()
    # 回退到上一有效行 (全 0.5)
    assert torch.allclose(out, torch.full((1, 6), 0.5))


def test_guard_false_is_transparent():
    model = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)
    model.eval()
    x = torch.randn(3, 6, 6)
    p = torch.randn(3, 4)
    a = model.predict_next(x, scene_params=p)
    b = model.predict_next(x, scene_params=p, guard=False)
    assert torch.equal(a, b)          # 默认不挂守卫 => 逐位一致


def test_guard_true_on_model_produces_finite():
    model = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)
    model.eval()
    x = torch.randn(2, 6, 6)
    p = torch.randn(2, 4)
    out = model.predict_next(x, scene_params=p, guard=True)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 6)
    # 守卫被自动挂载且记录了统计
    assert model.guard is not None
