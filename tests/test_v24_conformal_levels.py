"""v2.4.4 回归: 多名义水平 conformal 区间 (80/90/95, 默认 0.1 不变, 宽度随 alpha 减小增宽)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.calibration import (  # noqa: E402
    fit_predictor_calibration, compute_conformal_halfwidths, ALLOWED_ALPHAS,
)
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"
    assert ALLOWED_ALPHAS == (0.2, 0.1, 0.05)


def _pred():
    return PhysicsPredictor(CTMConfig(
        iterations=4, d_model=32, d_input=20, heads=2,
        n_synch_out=10, n_synch_action=8, memory_length=6,
        nlm_hidden=12, out_dims=20, certainty_threshold=0.0),
        scene_param_dim=4)


def _ds(seed=1, n=8):
    return build_parametric_dataset(n_per_kind=n, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=seed)


def _attach(model, cal_ds):
    cal, rep, rq = fit_predictor_calibration(model, cal_ds)
    # 把 report 里的多水平半宽 (list) 转回张量后挂载
    cba = {float(a): [torch.tensor(q) for q in hw]
           for a, hw in rep["conformal_by_alpha"].items()}
    model.attach_calibration(cal, rq, conformal_by_alpha=cba)
    return model


def test_default_alpha01_unchanged():
    model = _attach(_pred(), _ds(seed=2))
    iv = model.predict_interval(_ds(seed=3).X[:4], 4,
                                scene_params=_ds(seed=3).P[:4])
    iv01 = model.predict_interval(_ds(seed=3).X[:4], 4,
                                  scene_params=_ds(seed=3).P[:4], alpha=0.1)
    assert iv["alpha"] == 0.1
    for k in ("lower", "upper", "median"):
        assert torch.allclose(iv[k], iv01[k])


def test_width_nondecreasing_as_alpha_shrinks():
    model = _attach(_pred(), _ds(seed=4))
    ds = _ds(seed=5)
    widths = {}
    for a in (0.2, 0.1, 0.05):
        iv = model.predict_interval(ds.X[:6], 4, scene_params=ds.P[:6], alpha=a)
        widths[a] = sum(iv["half_width_by_step"]) / len(iv["half_width_by_step"])
    # 名义覆盖越高 (alpha 越小) => 区间越宽
    assert widths[0.05] > widths[0.1] > widths[0.2]


def test_coverage_ordering_sanity():
    model = _attach(_pred(), _ds(seed=6))
    ds = _ds(seed=7)
    covs = {}
    for a in (0.2, 0.1, 0.05):
        iv = model.predict_interval(ds.X, ds.Y.size(1), scene_params=ds.P, alpha=a)
        inside = ((ds.Y >= iv["lower"] - 1e-9) & (ds.Y <= iv["upper"] + 1e-9))
        covs[a] = float(inside.float().mean())
    # 更宽的区间覆盖率不更低
    assert covs[0.05] >= covs[0.2] - 1e-6
    assert 0.0 <= covs[0.1] <= 1.0


def test_invalid_alpha_rejected():
    model = _attach(_pred(), _ds(seed=8))
    try:
        model.predict_interval(_ds(seed=9).X[:2], 4,
                               scene_params=_ds(seed=9).P[:2], alpha=0.01)
        assert False
    except ValueError:
        pass


def test_compute_conformal_halfwidths_deterministic():
    torch.manual_seed(0)
    resid = torch.randn(50, 4, 6)
    out = compute_conformal_halfwidths(resid)
    assert set(out.keys()) == {0.2, 0.1, 0.05}
    # 越宽水平半宽越大 (逐维)
    for h in range(4):
        assert bool((out[0.05][h] >= out[0.1][h] - 1e-6).all())
        assert bool((out[0.1][h] >= out[0.2][h] - 1e-6).all())
