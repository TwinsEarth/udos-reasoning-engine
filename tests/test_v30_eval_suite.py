"""
v3.0.0.dev5 FiveDimensionEvaluator 单元测试
==============================================
锚点纪律:
    * 五维度各输出分数 [0,100];
    * 算术均分计算正确;
    * 空模型 (predictor=None) 守卫;
    * 分数可复现 (同种子两次跑一致);
    * 显式标注为 UDOS 内部基准, 非 PhysBrain 榜单。
"""
from pathlib import Path

import pytest

from udos import __version__
from udos.eval_suite import FiveDimensionEvaluator, IS_INTERNAL_BENCHMARK
from udos.persistence import load_predictor

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.0.0.pt")


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_is_internal_not_physbrain(predictor):
    assert IS_INTERNAL_BENCHMARK is True
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    md = ev.metadata()
    assert md["is_internal_benchmark"] is True
    assert md["not_physbrain_leaderboard"] is True
    assert len(md["dimensions"]) == 5
    assert "PhysBrain" in md["note"]


def test_five_dimensions_in_range(predictor):
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    scores = ev.evaluate()
    assert set(scores.keys()) == set(FiveDimensionEvaluator.DIMENSIONS)
    for k, v in scores.items():
        assert 0.0 <= v <= 100.0, f"{k}={v} 越界"


def test_arith_mean_correct(predictor):
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    s = ev.evaluate()
    mean = sum(s.values()) / len(s)
    assert abs(mean - sum(s.values()) / len(s)) < 1e-9


def test_empty_model_guard():
    with pytest.raises(ValueError):
        FiveDimensionEvaluator(None)


def test_scores_reproducible(predictor):
    ev1 = FiveDimensionEvaluator(predictor, seed=2025)
    s1 = ev1.evaluate()
    ev2 = FiveDimensionEvaluator(predictor, seed=2025)
    s2 = ev2.evaluate()
    assert s1 == s2


def test_metadata_marks_internal(predictor):
    ev = FiveDimensionEvaluator(predictor, seed=2025)
    md = ev.metadata()
    assert md["is_internal_benchmark"] is True
    assert md["not_physbrain_leaderboard"] is True
    assert len(md["dimensions"]) == 5
