"""v2.4.11 回归: 多水平区间覆盖率基准脚本逻辑 (三水平产出结构 / 覆盖单调 / JSON 含三水平)。"""
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
import ablation_conformal_levels as ab  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _tiny():
    return PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)


def test_eval_levels_structure_and_monotonic():
    m = _tiny(); m.eval()
    cal = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                   horizon=3, dt=0.5, seed=51)
    te = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=52)
    rows = ab.eval_levels(m, cal, te)
    assert [r["alpha"] for r in rows] == [0.2, 0.1, 0.05]
    covs = [r["coverage"] for r in rows]
    widths = [r["mean_width"] for r in rows]
    for c in covs:
        assert 0.0 <= c <= 1.0
    # 名义覆盖越高 => 覆盖不更低、宽度更宽
    assert covs[2] >= covs[0] - 1e-6
    assert widths[2] > widths[1] > widths[0]


def test_ablation_json_written():
    p = ROOT / "benchmarks" / "results" / "conformal_levels_v2.4.11.json"
    assert p.exists(), "请先运行 make conformal-levels 生成结果"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data["summary"].keys()) >= {"80%", "90%", "95%"}
    for lv in ("80%", "90%", "95%"):
        assert "coverage" in data["summary"][lv]
        assert "mean_width" in data["summary"][lv]
        assert "nominal_coverage" in data["summary"][lv]
