"""
v3.0.0.dev6 composite_score + 四代五维演化
==============================================
锚点纪律:
    * composite_score 等权/加权平均计算正确, 权重自动归一化;
    * 四代 (v2.7.3/v2.8.0/v2.9.0/v3.0.0) 五维评测 JSON 落盘;
    * 分数单调仅作合理性记录 (不强制提升);
    * eval5d 脚本可运行; 与 PhysBrain 分数严格分开标注。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from udos import __version__
from udos.eval_suite import FiveDimensionEvaluator
from udos.persistence import load_predictor

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.0.0.pt")
EVOL_JSON = ROOT / "benchmarks" / "results" / "five_dim_evolution_v3.0.0.json"
EVOL_SCRIPT = ROOT / "scripts" / "eval5d_evolution.py"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_composite_equal_weight(predictor):
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    s = ev.evaluate()
    comp = ev.composite_score(s)
    expect = sum(s.values()) / len(s)
    assert abs(comp - expect) < 1e-6
    assert 0.0 <= comp <= 100.0


def test_composite_custom_weights_normalized(predictor):
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    s = ev.evaluate()
    # 只给视觉轨迹推理维度 2 倍权, 其余 1 倍 (和=6, 不必为 1)
    w = {d: 1.0 for d in FiveDimensionEvaluator.DIMENSIONS}
    w["visual_trajectory_reasoning"] = 2.0
    comp = ev.composite_score(s, weights=w)
    denom = 6.0
    expect = (sum(s[d] for d in FiveDimensionEvaluator.DIMENSIONS)
              + s["visual_trajectory_reasoning"]) / denom
    assert abs(comp - expect) < 1e-6


def test_composite_bad_weights_rejected(predictor):
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    s = ev.evaluate()
    with pytest.raises(ValueError):
        ev.composite_score(s, weights={d: 0.0 for d in
                                       FiveDimensionEvaluator.DIMENSIONS})
    with pytest.raises(ValueError):
        w = {d: 1.0 for d in FiveDimensionEvaluator.DIMENSIONS}
        del w["visual_trajectory_reasoning"]
        ev.composite_score(s, weights=w)


def test_evolution_json_written(predictor):
    r = subprocess.run([sys.executable, str(EVOL_SCRIPT)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(EVOL_JSON))
    assert d["not_physbrain_leaderboard"] is True
    assert len(d["generations"]) == 4
    gens = [g["version"] for g in d["generations"]]
    assert gens == ["v2.7.3", "v2.8.0", "v2.9.0", "v3.0.0"]
    for g in d["generations"]:
        assert 0.0 <= g["composite"] <= 100.0
        for dim in FiveDimensionEvaluator.DIMENSIONS:
            assert 0.0 <= g[dim] <= 100.0
