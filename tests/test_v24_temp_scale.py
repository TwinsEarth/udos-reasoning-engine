"""v2.4.2 回归: 温度缩放校准 (T=1 恒等 / T 优化降 ECE / method 分派 / 持久化)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.calibration import (  # noqa: E402
    TemperatureScaling, ConfidenceCalibrator, fit_predictor_calibration,
    reliability,
)
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_temperature_one_is_identity():
    ts = TemperatureScaling(temperature=1.0)
    ts.n_fit = 10  # 标记已拟合, 跳过 fit
    c = torch.linspace(0.05, 0.95, 50)
    out = ts.transform(c)
    assert torch.allclose(out, c, atol=1e-5)


def test_fit_reduces_or_matches_ece_on_fit_set():
    torch.manual_seed(0)
    # 过置信: 原始置信普遍偏高, 但误差中等 => 经验精度约 0.5, T>1 应收拢
    conf = torch.linspace(0.6, 0.99, 200)
    err = torch.full((200,), 0.7)  # 使 exp(-err/scale)~0.5
    ts = TemperatureScaling().fit(conf, err)
    scale = ts.scale
    raw_ece = reliability(conf, err, scale, 5)["ece"]
    cal_ece = reliability(ts.transform(conf), err, scale, 5)["ece"]
    # 网格搜索以 ECE 为目标, 拟合集上不得更差
    assert cal_ece <= raw_ece + 1e-9
    assert ts.temperature > 0 and ts.fitted


def test_temperature_is_monotone():
    ts = TemperatureScaling(temperature=2.5)
    ts.n_fit = 10
    probe = torch.linspace(0.01, 0.99, 100)
    out = ts.transform(probe)
    assert bool((out.diff() >= -1e-7).all())


def test_edge_guards():
    try:
        TemperatureScaling().fit(torch.tensor([0.5]), torch.tensor([0.1]))
        assert False
    except ValueError:
        pass
    try:
        TemperatureScaling().fit(torch.tensor([0.2, 0.5]),
                                 torch.tensor([0.1, 0.2, 0.3])); assert False
    except ValueError:
        pass
    try:
        TemperatureScaling().transform(torch.tensor([0.5])); assert False
    except RuntimeError:
        pass


def _pred():
    return PhysicsPredictor(CTMConfig(
        iterations=4, d_model=32, d_input=20, heads=2,
        n_synch_out=10, n_synch_action=8, memory_length=6,
        nlm_hidden=12, out_dims=20, certainty_threshold=0.0),
        scene_param_dim=4)


def _ds(seed=1):
    return build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=seed)


def test_fit_predictor_calibration_method_dispatch():
    ds = _ds()
    # 默认 pava 不变
    cal_pava, rep_pava, rq_pava = fit_predictor_calibration(_pred(), ds)
    assert isinstance(cal_pava, ConfidenceCalibrator)
    assert rep_pava["method"] == "pava"
    # temperature 方法
    cal_t, rep_t, rq_t = fit_predictor_calibration(_pred(), ds, method="temperature")
    assert isinstance(cal_t, TemperatureScaling)
    assert rep_t["method"] == "temperature"
    assert len(rq_t) == ds.Y.size(1)
    # none 方法: 无校准器但仍给半宽
    cal_none, rep_none, rq_none = fit_predictor_calibration(_pred(), ds, method="none")
    assert cal_none is None and rep_none["method"] == "none"
    assert len(rq_none) == ds.Y.size(1)
    # 非法 method
    try:
        fit_predictor_calibration(_pred(), ds, method="weird"); assert False
    except ValueError:
        pass


def test_temperature_persistence_roundtrip(tmp_path):
    from udos import save_predictor, load_predictor
    model = _pred(); model.eval()
    cal, _, rq = fit_predictor_calibration(model, _ds(seed=5),
                                           method="temperature")
    model.attach_calibration(cal, rq)
    assert isinstance(model.calibrator, TemperatureScaling)
    p = tmp_path / "pt.pt"
    save_predictor(model, p)
    loaded, meta = load_predictor(p)
    assert isinstance(loaded.calibrator, TemperatureScaling)
    assert abs(loaded.calibrator.temperature - cal.temperature) < 1e-9
    probe = torch.linspace(0.1, 0.9, 12)
    assert torch.allclose(loaded.calibrator.transform(probe),
                          cal.transform(probe), atol=1e-6)
