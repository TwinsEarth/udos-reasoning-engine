"""v2.4.12 回归: guard 接入 server — /evaluate guard 段 + POST /predict (含 guard=True 回退)。"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _service(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService, ServiceNotReady
    s = UDOSService(preset="small")
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    return s


def test_predict_409_when_untrained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService, ServiceNotReady
    s = UDOSService(preset="small")
    try:
        s.predict({"window": [[[0.0] * 6] * 6]}); assert False
    except ServiceNotReady:
        pass


def test_predict_basic_and_shape(tmp_path, monkeypatch):
    s = _service(tmp_path, monkeypatch)
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=61)
    out = s.predict({"window": ds.X[:3].tolist(),
                     "scene_params": ds.P[:3].tolist()})
    assert out["status"] == "ok"
    assert out["shape"] == [3, 6]          # [N, RAW_DIM]
    assert len(out["prediction"]) == 3
    # 默认 guard 关闭
    assert out["guard"]["enabled"] is False


def test_predict_bad_input(tmp_path, monkeypatch):
    s = _service(tmp_path, monkeypatch)
    for bad in ({}, {"window": "x"}, {"window": [[0.0] * 6]}):  # 2D 缺 batch
        try:
            s.predict(bad)
        except ValueError:
            pass


def test_predict_guard_triggers_fallback_on_nan(tmp_path, monkeypatch):
    s = _service(tmp_path, monkeypatch)
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=62)
    win = ds.X[:2].clone()
    win[0, 2, 1] = float("nan")            # 注入非有限输入 -> 输出应含 NaN
    out = s.predict({"window": win.tolist(), "scene_params": ds.P[:2].tolist(),
                     "guard": True})
    assert out["guard"]["enabled"] is True
    assert out["guard"]["n_fallbacks"] >= 1     # NaN 行触发回退
    # 回退后输出不再含 NaN/inf
    flat = [v for row in out["prediction"] for v in row]
    assert all(math.isfinite(v) for v in flat)


def test_evaluate_guard_segment(tmp_path, monkeypatch):
    s = _service(tmp_path, monkeypatch)
    rep = s.evaluate({"n_per_kind": 6, "horizon": 3, "guard": True})
    assert "guard" in rep
    assert rep["guard"]["enabled"] is True
    assert "n_fallbacks" in rep["guard"] and "n_clips" in rep["guard"]
    # 不带 guard 时不出现 guard 段 (默认不变)
    rep2 = s.evaluate({"n_per_kind": 6, "horizon": 3})
    assert "guard" not in rep2
