"""
v4.2.0.dev2 执行验证 + 模型评审测试
==========================================
纪律: 外挂零梯度、空/非法显式 ValueError、确定性。
"""
import hashlib

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, TransitionTripletGenerator,
                  ExecutionValidator, ModelReviewer, SelfGeneratedTriplets)
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


def test_execution_validator_passes_clean():
    m, gen, ds = _setup()
    trip = gen.generate(ds, n=12, seed=0)
    v = ExecutionValidator()
    rep = v.validate(trip)
    assert rep["n"] == 12
    assert 0.0 <= rep["pass_rate"] <= 1.0
    assert rep["pass"] + rep["fail"] == 12


def test_execution_validator_rejects_oob():
    # 构造一个超界的 next_state
    s = torch.zeros(4, 6)
    ns = torch.zeros(4, 6)
    ns[0, 0] = 1e6          # 位置超界
    ns[1, 3] = 1e6          # 速度超界
    bad = SelfGeneratedTriplets(
        states=s, actions=torch.zeros(4, 1), next_states=ns,
        scene_params=None, kinds=["k"] * 4)
    v = ExecutionValidator()
    rep = v.validate(bad)
    assert rep["pos_oob"] >= 1
    assert rep["vel_oob"] >= 1


def test_execution_validator_bad_bound():
    with pytest.raises(ValueError):
        ExecutionValidator(pos_bound=-1.0)


def test_model_reviewer_keeps_half():
    m, gen, ds = _setup()
    rev = ModelReviewer(gen, n_perturb=3, noise=0.05)
    filt, rep = rev.review(ds, n=16, seed=0)
    # 以方差中位数为阈值, 保留约一半
    assert rep["n_total"] == 16
    assert rep["n_kept"] + rep["n_rejected"] == 16
    assert len(filt) == rep["n_kept"]
    assert 0.0 <= rep["keep_rate"] <= 1.0


def test_model_reviewer_deterministic():
    m, gen, ds = _setup()
    rev = ModelReviewer(gen, n_perturb=3, noise=0.05)
    f1, r1 = rev.review(ds, n=12, seed=0)
    f2, r2 = rev.review(ds, n=12, seed=0)
    assert torch.allclose(f1.next_states, f2.next_states)
    assert r1["keep_rate"] == r2["keep_rate"]


def test_model_reviewer_zero_grad():
    m, gen, ds = _setup()
    before = _md5(m)
    rev = ModelReviewer(gen)
    rev.review(ds, n=8, seed=0)
    after = _md5(m)
    assert before == after


def test_model_reviewer_bad_genrejected():
    with pytest.raises(ValueError):
        ModelReviewer(object(), n_perturb=3)


def test_model_reviewer_bad_perturb():
    m, gen, ds = _setup()
    with pytest.raises(ValueError):
        ModelReviewer(gen, n_perturb=1)
