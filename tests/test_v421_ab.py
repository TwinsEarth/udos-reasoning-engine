"""
v4.2.1 收益递减判据 + 自训练 A/B 测试
==========================================
纪律: 诚实 A/B 不预设胜负; 主预测器只读; 空/非法 ValueError。
"""
import hashlib

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, TransitionTripletGenerator,
                  DiminishingReturnsCriterion, SelfTrainAB)
from udos.world_model import LatentWorldModel
from udos.dynamics import build_parametric_dataset


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _setup(seed=7):
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg, scene_param_dim=4)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=seed)
    wm = LatentWorldModel(m, action_dim=0)
    wm.fit(ds, epochs=10)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    trip = gen.generate(ds, n=48, seed=0)
    return m, trip, ds


def test_diminishing_returns_triggers():
    cr = DiminishingReturnsCriterion(patience=3, min_delta=1e-3)
    rep = cr.check([0.1000, 0.0999, 0.0998, 0.0997])  # 改善均 < 1e-3
    assert rep["diminishing_returns"] is True
    assert rep["recommendation"] == "stop_self_training"


def test_diminishing_returns_continue():
    cr = DiminishingReturnsCriterion(patience=3, min_delta=1e-3)
    rep = cr.check([0.100, 0.080, 0.060, 0.050])  # 持续大降
    assert rep["diminishing_returns"] is False
    assert rep["recommendation"] == "continue"


def test_diminishing_bad_input():
    cr = DiminishingReturnsCriterion()
    with pytest.raises(ValueError):
        cr.check([0.1])


def test_ab_runs_and_reports():
    m, trip, ds = _setup()
    before = _md5(m)
    ab = SelfTrainAB(m, epochs=30, seed=0)
    rep = ab.run(trip, ds, holdout_dataset=ds)
    after = _md5(m)
    assert before == after, "A/B 不得改动 teacher"
    assert "arm_A_self_generated_mse" in rep
    assert "arm_B_original_mse" in rep
    assert rep["self_generated_wins"] in (True, False)
    # 诚实: 不预设胜负
    assert "verdict" in rep


def test_ab_bad_triplet():
    m, trip, ds = _setup()
    ab = SelfTrainAB(m, epochs=5)
    with pytest.raises(ValueError):
        ab.run(self_trip=object(), orig_dataset=ds, holdout_dataset=ds)


def test_ab_verdict_honest():
    m, trip, ds = _setup()
    ab = SelfTrainAB(m, epochs=30, seed=0)
    rep = ab.run(trip, ds, holdout_dataset=ds)
    if not rep["self_generated_wins"]:
        assert "未优于" in rep["verdict"] or "不宣称" in rep["verdict"]
