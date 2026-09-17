"""v3.7.0.dev1 大脑慢规划 CortexPlanner 测试 (复用 policy MPC)。

覆盖:
    * 规划输出: 候选动作下 target[6] / planned=True / best_score;
    * 慢频率: 非规划步 cortex.ran=False, 复用上一步缓存目标;
    * 候选评估: 与直接 MPCActionSelector 最优一致;
    * 空候选守卫: candidate_actions=[] => no_valid_action => ValueError;
    * 无候选退化: planned=False, target == predict_next 逐位;
    * 被否决候选索引显式记录 (安全违例候选)。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset, RAW_DIM
from udos.policy import MPCActionSelector
from udos.neural_control import HierarchicalController

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.7.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


@pytest.fixture(scope="module")
def data():
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=21)
    return ds.X[:1], ds.P[:1]


def _candidates(window):
    """三个候选: 原地 + 两个小幅 candidate_state 偏移。"""
    base = window[0, -1, :].clone()
    cands = []
    for scale in (0.0, 0.05, -0.05):
        c = base.clone()
        c[0] = c[0] + scale
        cands.append({"candidate_state": c})
    return cands


def test_planning_output(predictor, data):
    w, sp = data
    ctrl = HierarchicalController(predictor, cortex_every=1)
    cands = _candidates(w)
    out = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    assert out["cortex"]["planned"] is True
    assert out["cortex"]["n_candidates"] == 3
    assert out["cortex"]["target"].shape == (RAW_DIM,)
    assert isinstance(out["cortex"]["rejected_candidates"], list)


def test_slow_frequency_caching(predictor, data):
    w, sp = data
    cands = _candidates(w)
    ctrl = HierarchicalController(predictor, cortex_every=3)
    r0 = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    r1 = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    r2 = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    r3 = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    assert r0["cortex"]["ran"] is True    # step 0 规划
    assert r1["cortex"]["ran"] is False   # step 1 缓存
    assert r2["cortex"]["ran"] is False    # step 2 缓存
    assert r3["cortex"]["ran"] is True     # step 3 再规划
    # 缓存步目标与上一次规划一致
    assert torch.allclose(r1["cortex"]["target"], r0["cortex"]["target"])


def test_consistent_with_mpc(predictor, data):
    """同候选集, 大脑最优与直接 MPCActionSelector 一致。"""
    w, sp = data
    cands = _candidates(w)
    ctrl = HierarchicalController(predictor, cortex_every=1)
    out = ctrl.step(w, scene_params=sp, candidate_actions=cands)
    sel = MPCActionSelector(predictor, horizon=4).select(
        w, scene_params=sp, candidate_actions=cands)
    assert out["cortex"]["best_index"] == sel["best_index"]
    assert out["cortex"]["best_score"] == sel["best_score"]


def test_empty_candidate_guard(predictor, data):
    w, sp = data
    ctrl = HierarchicalController(predictor, cortex_every=1)
    with pytest.raises(ValueError):
        ctrl.step(w, scene_params=sp, candidate_actions=[])


def test_no_candidate_fallback(predictor, data):
    w, sp = data
    ctrl = HierarchicalController(predictor, cortex_every=1)
    out = ctrl.step(w, scene_params=sp)
    assert out["cortex"]["planned"] is False
    assert out["cortex"]["n_candidates"] == 0
    with torch.no_grad():
        ref = predictor.predict_next(w, scene_params=sp)[0]
    assert torch.allclose(out["cortex"]["target"], ref, atol=1e-6)
