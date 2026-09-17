"""
v2.8.0.dev2 predict_action 阶段单元测试
==========================================
锚点纪律:
    * 空候选集 => no_valid_action=True, best_action=None;
    * 已知最优动作 (自定义 objective_reward) 被正确选为 best_index;
    * 风险惩罚单调性: 大扰动 (OOD) 动作 risk 更高 => score 更低;
    * 动作历史跨 run 累积进 loop_state["action_history"];
    * 与旧 policy 逐位一致: loop.predict_action 委托同一 MPCActionSelector,
      best_index/best_score 与直接 select 相同。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.physical_loop import PhysicalLoopRunner
from udos.persistence import load_predictor
from udos.policy import MPCActionSelector
from udos.dynamics import build_parametric_dataset, RAW_DIM

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.8.0.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=888)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_empty_candidate_no_valid_action(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    out = loop.run(wb, scene_params=pb, candidate_actions=[])
    pa = out["loop_state"]["outputs"]["predict_action"]
    assert pa["no_valid_action"] is True
    assert pa["best_action"] is None


def test_known_best_selected(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    # 自定义奖励: 第 1 个候选 (索引 1) 给巨大正奖励 => 必被选中
    def reward(pred, action):
        return 1e6 if action.get("tag") == "best" else 0.0
    loop.mpc = MPCActionSelector(predictor, horizon=2, objective_reward=reward)
    cands = [{"tag": "neutral"}, {"tag": "best"}]
    out = loop.run(wb, scene_params=pb, candidate_actions=cands)
    pa = out["loop_state"]["outputs"]["predict_action"]
    assert pa["no_valid_action"] is False
    assert pa["best_index"] == 1
    assert pa["best_action"]["tag"] == "best"


def test_risk_penalty_monotonic(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    # 无操作 vs 大扰动 (推向 OOD): 大扰动 risk 更高 => 排序更靠后
    cands = [{}, {"state_perturbation": [5.0] * RAW_DIM}]
    out = loop.run(wb, scene_params=pb, candidate_actions=cands)
    pa = out["loop_state"]["outputs"]["predict_action"]
    ranked = pa["ranked_actions"]
    assert len(ranked) == 2
    # 扰动动作的 risk 应 >= 无操作动作
    risk_map = {r["index"]: r["risk"] for r in ranked}
    assert risk_map[1] >= risk_map[0] - 1e-9
    # 排序第一的是无操作 (风险更低)
    assert ranked[0]["index"] == 0


def test_action_history_accumulates(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    assert len(loop.loop_state["action_history"]) == 0
    loop.run(wb, scene_params=pb, candidate_actions=[{}, {}])
    loop.run(wb, scene_params=pb, candidate_actions=[{}, {}])
    assert len(loop.loop_state["action_history"]) == 2
    assert loop.loop_state["action_history"][0]["best_index"] is not None


def test_bit_identical_to_policy(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    cands = [{}, {"state_perturbation": [0.1] * RAW_DIM}]
    out = loop.run(wb, scene_params=pb, candidate_actions=cands)
    pa = out["loop_state"]["outputs"]["predict_action"]
    direct = MPCActionSelector(predictor, horizon=2).select(
        wb, scene_params=pb, candidate_actions=cands)
    assert pa["best_index"] == direct["best_index"]
    assert pa["best_score"] == direct["best_score"]
    assert pa["no_valid_action"] == direct["no_valid_action"]


def test_default_candidate_when_none(predictor, window_batch):
    """未提供候选 => 默认单个无操作动作, 不崩、no_valid_action=False。"""
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    out = loop.run(wb, scene_params=pb)
    pa = out["loop_state"]["outputs"]["predict_action"]
    assert pa["n_candidates"] == 1
    assert pa["no_valid_action"] is False
