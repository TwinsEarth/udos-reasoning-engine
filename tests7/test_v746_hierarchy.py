"""v7.4.6 分层混合拓扑与扇出基准测试。"""
import pytest

from udos7.topology.hierarchy import (LayeredMatrix, ExplicitSim,
                                      star_routing, scale_benchmark,
                                      fit_depth)


def test_tree_agent_counts():
    m = LayeredMatrix(branch=4, depth=3)
    assert m.n_workers == 64
    assert m.n_internal == 1 + 4 + 16
    assert m.n_agents == 85


def test_level_roles_map_to_topologies():
    m = LayeredMatrix(branch=8, depth=4)
    assert m.level_role(0) == "orchestrator"
    assert m.level_role(1) == "handoff"
    assert m.level_role(3) == "swarm"
    assert m.level_role(4) == "swarm"


def test_explicit_sim_matches_formula_small():
    m = LayeredMatrix(branch=2, depth=3)        # 8 workers, 15 节点
    lay = m.route(8)
    ex = ExplicitSim(m).run(8)
    assert ex["total_messages"] == lay["total_messages"]
    assert ex["max_node_fanin"] == lay["max_node_fanin"]
    assert ex["top_fanin"] == lay["top_fanin"]


def test_layered_fanin_constant_star_grows():
    """关键命题：分层每节点扇入恒为 b+1；星型中心扇入随 N 线性增长。"""
    rows = scale_benchmark([64, 512, 4096], branch=4)
    fanins = [r["layered_max_fanin"] for r in rows]
    assert set(fanins) == {5}                    # b+1，与规模无关
    stars = [r["star_center_fanin"] for r in rows]
    assert stars[0] < stars[1] < stars[2]
    assert stars[2] > 30 * fanins[2]


def test_rounds_logarithmic_vs_star_serial():
    rows = scale_benchmark([64, 512, 4096, 1_000_000], branch=8)
    for r in rows:
        assert r["layered_rounds"] == 2 * r["depth"]
    assert rows[-1]["layered_rounds"] == 14      # log_8(1e6)≈6.65→7
    assert rows[-1]["star_rounds"] > rows[-1]["layered_rounds"] * 10000


def test_grades_explicit_then_analytical():
    rows = scale_benchmark([100, 100_000], branch=8, explicit_limit=4000)
    assert rows[0]["grade"] == "explicit"
    assert rows[0]["explicit_matches_formula"] is True
    assert rows[1]["grade"] == "analytical"
    assert "explicit_matches_formula" not in rows[1]


def test_fit_depth():
    assert fit_depth(64, 4) == 3
    assert fit_depth(65, 4) == 4
    assert fit_depth(1_000_000, 8) == 7


def test_top_fanin_independent_of_scale():
    for d in (2, 4, 7):
        m = LayeredMatrix(branch=8, depth=d)
        assert m.route(m.n_workers)["top_fanin"] == 8


def test_million_agent_extrapolation():
    rows = scale_benchmark([1_000_000], branch=8)
    r = rows[0]
    assert r["actual_agents"] >= 1_000_000
    assert r["layered_max_fanin"] == 9
    # 星型中心要承接百万级消息，分层顶层只见 8 条领域汇总
    assert r["star_center_fanin"] > 1_000_000
    assert r["layered_top_fanin"] == 8
