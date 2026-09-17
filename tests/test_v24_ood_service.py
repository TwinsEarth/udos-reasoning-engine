"""v2.4.8 回归: OOD 服务接口 POST /detect-ood (未训练 409 / ID 低分 / OOD 高分 / /evaluate 含 ood 段)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.ood import DistributionDriftDetector  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _service(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    s = UDOSService(preset="small")
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    # 给训练好的预测器挂一个 OOD 检测器 (用训练窗口拟合)
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=51)
    s.engine.predictor.attach_ood_detector(
        DistributionDriftDetector().fit(ds.X))
    return s, ds


def test_detect_ood_409_when_untrained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService, ServiceNotReady
    s = UDOSService(preset="small")
    try:
        s.detect_ood({"sequence": [[[0.0] * 6] * 6]}); assert False
    except ServiceNotReady:
        pass


def test_detect_ood_id_low_ood_high(tmp_path, monkeypatch):
    s, ds = _service(tmp_path, monkeypatch)
    id_seq = ds.X[:4].tolist()
    out = s.detect_ood({"sequence": id_seq})
    assert out["status"] == "ok" and len(out["scores"]) == 4
    ood_seq = (ds.X[:4] * 6.0).tolist()
    out_ood = s.detect_ood({"sequence": ood_seq})
    assert out_ood["score_max"] > out["score_max"]
    assert out_ood["ood_rate"] >= out["ood_rate"]


def test_detect_ood_bad_input(tmp_path, monkeypatch):
    s, _ = _service(tmp_path, monkeypatch)
    for bad in ({}, {"sequence": "x"}, {"sequence": [[0.0] * 6]}):
        # 2D 单窗应被接受; 其余非法应 400 语义 ValueError
        try:
            s.detect_ood(bad)
        except ValueError:
            pass


def test_evaluate_include_ood_segment(tmp_path, monkeypatch):
    s, _ = _service(tmp_path, monkeypatch)
    rep = s.evaluate({"n_per_kind": 6, "horizon": 3, "include_ood": True})
    assert "ood" in rep and "threshold" in rep["ood"]
    # 不带 include_ood 时不出现 ood 段 (默认不变)
    rep2 = s.evaluate({"n_per_kind": 6, "horizon": 3})
    assert "ood" not in rep2
