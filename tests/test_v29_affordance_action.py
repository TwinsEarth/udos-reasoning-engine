"""
v2.9.0.dev6 affordance + action 联合推理 (AffordanceActionPlanner) 单测
======================================================================
锚点纪律:
    * 联合推理输出动作有限、面向 best_part;
    * 低 affordance / 不可达部位不生成动作 (轨迹全零, no_valid_action);
    * 空物体守卫;
    * opt-in: 未挂 planner 时 loop.predict_action 旧 policy 路径逐位一致;
      挂 predict_action_fn hook 后 loop 最终预测仍与 predict_next 逐位一致。
analogy, not reproduction: 合成代理向量上的接近动作规划。
"""
from pathlib import Path

import pytest
import torch

from udos.affordance import AffordanceScorer, AffordanceActionPlanner
from udos.persistence import load_predictor
from udos.physical_loop import PhysicalLoopRunner
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


def test_joint_planning_finite(predictor):
    planner = AffordanceActionPlanner(AffordanceScorer(reach_radius=2.0),
                                       horizon=4)
    state = torch.zeros(1, 6)
    obj = torch.tensor([[[0.5, 0.2, 0.1, 0, 0, 0],
                         [5.0, 0, 0, 0, 0, 0]]])    # part0 近, part1 远
    res = planner.plan(state, obj)
    assert res["action_trajectory"].shape == (1, 4, 6)
    assert torch.isfinite(res["action_trajectory"]).all()
    assert res["best_part"][0] == 0          # 近部位得分最高
    assert bool(res["targeted"][0]) is True


def test_low_affordance_filtered(predictor):
    planner = AffordanceActionPlanner(AffordanceScorer(reach_radius=1.0),
                                       horizon=3)
    state = torch.zeros(1, 6)
    obj = torch.zeros(1, 3, 6) + 10.0       # 全部超距
    res = planner.plan(state, obj)
    assert res["no_valid_action"] is True
    assert bool((res["action_trajectory"] == 0).all())   # 不生成动作
    d = planner.plan_action_dict(state, obj)
    assert d["best_action"] is None and d["no_valid_action"] is True


def test_empty_objects_guard(predictor):
    planner = AffordanceActionPlanner(AffordanceScorer())
    with pytest.raises(ValueError):
        planner.plan(torch.zeros(1, 6), torch.zeros(1, 0, 6))


def test_opt_in_loop_predict_action_bit_identical(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=606)
    wb, pb = ds.X[:1], ds.P[:1]
    # 默认未挂 planner: loop.predict_action 走旧 MPC, 最终预测逐位一致
    loop = PhysicalLoopRunner(predictor, horizon=2)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    direct = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(out["prediction"], direct)

    # opt-in: 挂 planner 作为 predict_action_fn hook (不改旧默认, 仅本次 run)
    planner = AffordanceActionPlanner(AffordanceScorer(reach_radius=1e3))
    obj = torch.randn(1, 3, 6)

    def hook(ctx):
        state = ctx["window"][:, -1, :]                 # [B,6]
        d = planner.plan_action_dict(state, obj)
        return {"best_action": d["best_action"], "best_score": 1.0,
                "best_index": 0, "ranked_actions": [],
                "no_valid_action": d["no_valid_action"], "best_risk": 0.0,
                "n_candidates": 1}

    loop2 = PhysicalLoopRunner(predictor, horizon=2, predict_action_fn=hook)
    out2 = loop2.run(wb, scene_params=pb)
    # loop 最终预测仍 = predict_next (与 planner hook 解耦)
    assert torch.equal(out2["prediction"], direct)
