"""
v2.9.0.dev4 可供性打分 (AffordanceScorer) 单测
==============================================
锚点纪律:
    * 打分形状 [B,N_parts] 有限、对可达部位归一化求和=1;
    * 归一化正确; 空物体 (N_parts=0) 显式 ValueError;
    * 可达性过滤 (超 reach_radius 的部位得分为 0);
    * 与 loop.understand 集成 (用 understand 目标状态代理机器人状态)。
analogy, not reproduction: 用状态向量子集代理"物体部位"。
"""
from pathlib import Path

import pytest
import torch

from udos.affordance import AffordanceScorer
from udos.persistence import load_predictor
from udos.physical_loop import PhysicalLoopRunner
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


@pytest.fixture(scope="module")
def scorer():
    return AffordanceScorer(reach_radius=2.0)


def test_scores_shape_and_normalization(scorer):
    state = torch.zeros(2, 6)
    obj = torch.randn(2, 4, 6) * 0.5       # 近距离 => 均可达
    out = scorer.score(state, obj)
    assert out["scores"].shape == (2, 4)
    assert torch.isfinite(out["scores"]).all()
    # 归一化: 每个 batch 的得分和为 1
    ssum = out["scores"].sum(dim=-1)
    assert torch.allclose(ssum, torch.ones(2), atol=1e-5)
    assert out["best_part"].shape == (2,)


def test_reachability_filter(scorer):
    state = torch.zeros(1, 6)
    # part0 近 (可达), part1 远 (不可达)
    obj = torch.tensor([[[0.5, 0, 0, 0, 0, 0],     # dist 0.5
                         [5.0, 0, 0, 0, 0, 0]]])   # dist 5.0 > reach_radius
    out = scorer.score(state, obj)
    assert out["reachable"][0, 0] == True
    assert out["reachable"][0, 1] == False
    # 远部位得分为 0, 近部位吃满归一化
    assert out["scores"][0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert out["scores"][0, 0].item() == pytest.approx(1.0, abs=1e-5)


def test_empty_objects_guard(scorer):
    with pytest.raises(ValueError):
        scorer.score(torch.zeros(1, 6), torch.zeros(1, 0, 6))
    with pytest.raises(ValueError):
        scorer.score(torch.zeros(1, 6), torch.tensor([[float("nan")] * 6]))  # 形状错
    with pytest.raises(ValueError):
        AffordanceScorer(reach_radius=-1.0)


def test_no_reachable_part_honest_zero(scorer):
    state = torch.zeros(1, 6)
    obj = torch.zeros(1, 3, 6) + 10.0     # 全部超距
    out = scorer.score(state, obj)
    assert bool((out["scores"] == 0).all())
    assert out["suggestion"][0] == "no_reachable_part"


def test_loop_understand_integration():
    m, _ = load_predictor(CKPT)
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=707)
    wb, pb = ds.X[:1], ds.P[:1]
    loop = PhysicalLoopRunner(m, horizon=2)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    target = out["loop_state"]["outputs"]["understand"]["target_state"]  # [1,6]
    # 用 understand 目标状态代理机器人状态, 构造 3 个部位
    scorer = AffordanceScorer(reach_radius=1e3)   # 宽松半径 => 均可达
    obj = torch.randn(1, 3, 6)
    res = scorer.score(target, obj)
    assert torch.allclose(res["scores"].sum(dim=-1), torch.ones(1), atol=1e-5)
