"""v2.4.6 回归: 校准 A/B 脚本逻辑 (三方法产出结构 / ECE 可比 / JSON 含三方法)。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import ablation_calibration as ab  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _tiny_model():
    return PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)


def test_eval_method_three_methods_structure():
    model = _tiny_model(); model.eval()
    cal = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                   horizon=3, dt=0.5, seed=21)
    te = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=22)
    for m in ("pava", "temperature", "none"):
        row = ab.eval_method(model, cal, te, m)
        assert row["method"] == m
        for k in ("ece", "spearman_conf_err", "coverage_overall"):
            assert k in row
        assert 0.0 <= row["coverage_overall"] <= 1.0


def test_ablation_json_written():
    p = ROOT / "benchmarks" / "results" / "calibration_ablation_v2.4.6.json"
    assert p.exists(), "请先运行 make calib-ablation 生成结果"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data["summary"].keys()) == {"pava", "temperature", "none"}
    for m in ("pava", "temperature", "none"):
        assert "ece" in data["summary"][m]
