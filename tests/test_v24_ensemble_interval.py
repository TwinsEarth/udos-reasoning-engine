"""v2.4.7 回归: 集成不确定性 + conformal 区间融合 (N=1 退化普通区间, N>1 方差增宽)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
from udos.ensemble import DeepEnsemble  # noqa: E402
from udos.calibration import fit_predictor_calibration  # noqa: E402


def _factory():
    return lambda: CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0)


def test_version_bumped():
    assert __version__ == "5.5.5"


def _window(seed=3, n=4):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=seed)
    return ds.X[:n], ds.P[:n]


def _attached_ens(n):
    ens = DeepEnsemble.create(_factory(), n=n, base_seed=0,
                              scene_param_dim=4)
    ens.eval()
    cal_ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                      horizon=3, dt=0.5, seed=31)
    cal, _, rq = fit_predictor_calibration(ens.members[0], cal_ds)
    # 同一半宽挂到所有成员 (q 来自校准残差, 与成员权重无关)
    for m in ens.members:
        m.attach_calibration(cal, rq)
    return ens


def test_n1_degrades_to_ordinary_interval():
    ens = _attached_ens(1)
    x, p = _window()
    iv = ens.predict_interval(x, 3, scene_params=p)
    single = ens.members[0].predict_interval(x, 3, scene_params=p)
    assert torch.allclose(iv["median"], single["median"], atol=1e-6)
    assert torch.allclose(iv["lower"], single["lower"], atol=1e-5)
    assert torch.allclose(iv["upper"], single["upper"], atol=1e-5)
    assert torch.count_nonzero(iv["variance"]) == 0


def test_n_gt1_interval_wider_and_finite():
    ens1 = _attached_ens(1)
    ens5 = _attached_ens(5)
    x, p = _window()
    iv1 = ens1.predict_interval(x, 3, scene_params=p)
    iv5 = ens5.predict_interval(x, 3, scene_params=p)
    assert torch.isfinite(iv5["lower"]).all()
    assert torch.isfinite(iv5["upper"]).all()
    # 集成方差>0 => 平均半宽不窄于单模型
    w1 = sum(iv1["half_width_by_step"]) / len(iv1["half_width_by_step"])
    w5 = sum(iv5["half_width_by_step"]) / len(iv5["half_width_by_step"])
    assert w5 >= w1 - 1e-6


def test_interval_covers_some_truth():
    ens = _attached_ens(4)
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=41)
    iv = ens.predict_interval(ds.X, 3, scene_params=ds.P)
    inside = ((ds.Y >= iv["lower"] - 1e-9) & (ds.Y <= iv["upper"] + 1e-9))
    cover = float(inside.float().mean())
    assert 0.0 <= cover <= 1.0
