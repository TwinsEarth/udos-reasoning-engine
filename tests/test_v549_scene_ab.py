"""
v5.4.9 场景条件三方 A/B (blind / explicit / estimated)
======================================================
验证 udos.scene_estimation_ab.three_way_scene_ab 在 held-out 上诚实度量
v5.4.8 经典估计器相对"显式真值场景参数上界"恢复了多少增益。

关键反例守卫:
  * 若实现把 estimated 错接成 blind (估计器被忽略), 匀速 recovery 会为 0,
    test_uniform_recovers_explicit 立即变红;
  * 显式条件必须严格优于场景盲 (发布模型已知的场景增益);
  * 估计不得整体劣于场景盲; 弹簧是已记录短板 (短窗 omega 召回 + v0 槽污染),
    其 recovery 允许很低但不应为负 (不造成净伤害)。
"""

from pathlib import Path

import pytest
import torch

from udos.dynamics import build_parametric_dataset
from udos.persistence import load_predictor
from udos.scene_estimation_ab import three_way_scene_ab

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "predictor_v4.3.9.pt"
KINDS = ("uniform", "accel", "spring", "collision")


@pytest.fixture(scope="module")
def report():
    predictor, _ = load_predictor(CKPT)
    predictor.eval()
    ds = build_parametric_dataset(n_per_kind=64, seed=7)
    return three_way_scene_ab(predictor, ds, dt=0.5, horizon=4)


def test_report_structure(report):
    for block in ("single_step", "rollout_4"):
        t = report["overall"][block]
        for key in ("blind", "explicit", "estimated", "recovery", "est_over_blind"):
            assert key in t
    assert set(report["per_kind"]) == set(KINDS)
    assert report["config"]["n_samples"] > 0


def test_explicit_beats_blind(report):
    # 显式真值场景参数必须严格降低误差 (已知场景增益)
    for block in ("single_step", "rollout_4"):
        t = report["overall"][block]
        assert t["explicit"] < t["blind"]


def test_uniform_recovers_explicit(report):
    # 最高风险: 匀速下估计参数≈真值, 应几乎完全恢复显式增益
    t = report["per_kind"]["uniform"]["rollout_4"]
    assert t["recovery"] is not None and t["recovery"] >= 0.95
    assert t["estimated"] == pytest.approx(t["explicit"], rel=0.05)


def test_overall_estimated_substantially_beats_blind(report):
    # 实测整体恢复率单步/4步约 0.92/0.87; 留保守裕度
    assert report["overall"]["single_step"]["recovery"] >= 0.70
    assert report["overall"]["rollout_4"]["recovery"] >= 0.70
    assert report["overall"]["rollout_4"]["estimated"] < \
        report["overall"]["rollout_4"]["blind"]


def test_spring_is_weak_but_not_harmful(report):
    # 弹簧是已记录短板 (omega 短窗召回~0.78 + v0 槽污染); 不要求高恢复,
    # 但估计条件不应造成净伤害 (recovery>=0)
    t = report["per_kind"]["spring"]["rollout_4"]
    assert t["recovery"] is not None and t["recovery"] >= 0.0


def test_observability_stats_honest(report):
    obs = report["estimator_observability"]
    rates = obs["slot_observable_rates"]
    assert rates["other_v2"] == 0.0          # 碰撞对方速度恒不可观测
    rate = obs["spring_omega_detection_rate"]
    assert rate is not None and 0.65 <= rate <= 0.92


def test_deterministic(report):
    predictor, _ = load_predictor(CKPT)
    predictor.eval()
    ds = build_parametric_dataset(n_per_kind=32, seed=11)
    r1 = three_way_scene_ab(predictor, ds, dt=0.5, horizon=4)
    r2 = three_way_scene_ab(predictor, ds, dt=0.5, horizon=4)
    assert r1["overall"] == r2["overall"]
