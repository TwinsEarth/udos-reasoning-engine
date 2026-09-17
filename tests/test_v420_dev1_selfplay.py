"""
v4.2.0.dev1 自博弈探索 + 规则校验测试
==========================================
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、确定性。
"""
import hashlib

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, TransitionTripletGenerator,
                  SelfPlayExplorer)
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
    return m, gen, ds


def test_explore_returns_triplets_and_report():
    m, gen, ds = _setup()
    expl = SelfPlayExplorer(gen)
    trip, rep = expl.explore(ds, n=12)
    assert len(trip) == 12
    assert trip.actions.shape == (12, 1)
    assert trip.next_states.shape == (12, 6)
    for k in ("n", "n_actions", "chosen_action_counts",
              "mean_momentum_violation", "max_momentum_violation",
              "rule_pass_rate"):
        assert k in rep
    assert rep["n"] == 12
    assert rep["n_actions"] == 3
    # 动作分布三类都在
    assert set(rep["chosen_action_counts"].keys()) == {"a=-1.0", "a=+0.0", "a=+1.0"}


def test_explore_deterministic():
    m, gen, ds = _setup()
    expl = SelfPlayExplorer(gen)
    t1, r1 = expl.explore(ds, n=8)
    t2, r2 = expl.explore(ds, n=8)
    assert torch.allclose(t1.actions, t2.actions)
    assert torch.allclose(t1.next_states, t2.next_states)
    assert r1["chosen_action_counts"] == r2["chosen_action_counts"]


def test_explore_zero_grad():
    m, gen, ds = _setup()
    before = _md5(m)
    expl = SelfPlayExplorer(gen)
    _, _ = expl.explore(ds, n=8)
    after = _md5(m)
    assert before == after, "自博弈不得改动主预测器权重"


def test_explore_report_within_bounds():
    m, gen, ds = _setup()
    expl = SelfPlayExplorer(gen)
    trip, rep = expl.explore(ds, n=12)
    assert 0.0 <= rep["rule_pass_rate"] <= 1.0
    assert rep["mean_momentum_violation"] >= 0.0
    assert bool(torch.isfinite(trip.next_states).all())


def test_bad_generator_rejected():
    with pytest.raises(ValueError):
        SelfPlayExplorer(object())


def test_bad_mass_rejected():
    m, gen, ds = _setup()
    with pytest.raises(ValueError):
        SelfPlayExplorer(gen, mass=0.0)


def test_bad_n_rejected():
    m, gen, ds = _setup()
    expl = SelfPlayExplorer(gen)
    with pytest.raises(ValueError):
        expl.explore(ds, n=0)
