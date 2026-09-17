"""
v4.2.0 完全自训练线测试 —— 世界模型自生成 (s,a,s') 三元组骨架
=================================================================
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、确定性、
      不宣称 RSI 改进 (防退化纪律; 本节点只验骨架可复算)。
"""
import hashlib

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, TransitionTripletGenerator,
                  SelfGeneratedTriplets)
from udos.world_model import LatentWorldModel
from udos.dynamics import build_parametric_dataset


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _predictor():
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def _wm_fitted(m, seed=7):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=seed)
    wm = LatentWorldModel(m, action_dim=0)   # 无动作自治转移 (兼容 wm.fit)
    wm.fit(ds, epochs=10)
    return wm, ds


def test_generate_triplet_shapes():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    tr = gen.generate(ds, n=12, seed=0)
    assert isinstance(tr, SelfGeneratedTriplets)
    assert len(tr) == 12
    assert tr.states.shape == (12, 6)
    assert tr.actions.shape == (12, 1)
    assert tr.next_states.shape == (12, 6)
    assert tr.origin == "self_generated"
    assert bool(torch.isfinite(tr.next_states).all())


def test_deterministic_generation():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    a = gen.generate(ds, n=8, seed=0)
    b = gen.generate(ds, n=8, seed=0)
    assert torch.allclose(a.next_states, b.next_states)
    assert torch.allclose(a.actions, b.actions)


def test_main_weights_untouched_zero_grad():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    before = _md5(m)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    tr = gen.generate(ds, n=8, seed=0)
    _ = gen.triplet_quality(tr)
    after = _md5(m)
    assert before == after, "自生成不得改动主预测器权重 (外挂零梯度)"


def test_quality_metrics_present():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    tr = gen.generate(ds, n=12, seed=0)
    q = gen.triplet_quality(tr)
    for k in ("n", "action_spread", "next_state_finite",
              "next_state_std", "transition_norm", "momentum_residual"):
        assert k in q
    assert q["next_state_finite"] == 1.0
    assert q["n"] == 12.0


def test_unfitted_wm_rejected():
    m = _predictor()
    wm = LatentWorldModel(m, action_dim=0)
    with pytest.raises(ValueError):
        TransitionTripletGenerator(m, wm, action_dim=1)


def test_non_wm_rejected():
    m = _predictor()
    with pytest.raises(ValueError):
        TransitionTripletGenerator(m, object(), action_dim=1)


def test_action_independent_channel():
    # v4.2.0 骨架: action 是自博弈独立通道, 与 wm 自治转移解耦
    m = _predictor()
    wm, ds = _wm_fitted(m)
    gen = TransitionTripletGenerator(m, wm, action_dim=3)
    tr = gen.generate(ds, n=6, seed=0)
    assert tr.actions.shape == (6, 3)
    assert tr.next_states.shape == (6, 6)


def test_bad_n_rejected():
    m = _predictor()
    wm, ds = _wm_fitted(m)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    with pytest.raises(ValueError):
        gen.generate(ds, n=0)
    with pytest.raises(ValueError):
        gen.generate(ds, n=-3)


def test_triplet_container_rejects_bad_shape():
    with pytest.raises(ValueError):
        SelfGeneratedTriplets(
            states=torch.zeros(4, 5),   # 非 RAW_DIM=6
            actions=torch.zeros(4, 1),
            next_states=torch.zeros(4, 5),
            scene_params=None, kinds=["k"] * 4)


def test_triplet_container_rejects_nonfinite():
    bad = torch.zeros(4, 6)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        SelfGeneratedTriplets(
            states=torch.zeros(4, 6), actions=torch.zeros(4, 1),
            next_states=bad, scene_params=None, kinds=["k"] * 4)


def test_line_stages_ten():
    from udos import LINE_STAGES
    assert len(LINE_STAGES) == 10
    assert LINE_STAGES[0] == "4.2.0"
    assert LINE_STAGES[-1] == "4.2.9"
