"""v7.4.5 拓扑决策树测试。"""
import pytest

from udos7.topology.decision import TaskProfile, choose_topology, choose_layered


@pytest.mark.parametrize("prof,expected", [
    (TaskProfile(flow_explicit=True), "orchestrator"),
    (TaskProfile(flow_explicit=False, expert_relay=True), "handoff"),
    (TaskProfile(flow_explicit=False, expert_relay=False,
                 encapsulable=True), "agent_as_tool"),
    (TaskProfile(flow_explicit=False, open_exploration=True,
                 risk="low"), "swarm"),
    (TaskProfile(flow_explicit=False, open_exploration=True,
                 risk="high"), "escalate"),
    (TaskProfile(flow_explicit=False), "escalate"),
])
def test_decision_tree_branches(prof, expected):
    assert choose_topology(prof).choice == expected


def test_default_is_orchestrator_for_clear_flow_regardless_of_flags():
    # 即使同时声明开放探索，流程明确时仍优先星型（强控制优先）
    p = TaskProfile(flow_explicit=True, open_exploration=True, risk="high")
    assert choose_topology(p).choice == "orchestrator"


def test_high_risk_swarm_is_escalated_not_released():
    d = choose_topology(TaskProfile(flow_explicit=False,
                                    open_exploration=True, risk="high"))
    assert d.choice == "escalate" and d.autonomy_level == 0


def test_autonomy_monotone_with_branch():
    levels = [
        choose_topology(TaskProfile(True)).autonomy_level,
        choose_topology(TaskProfile(False, True)).autonomy_level,
        choose_topology(TaskProfile(False, False, True)).autonomy_level,
        choose_topology(TaskProfile(False, False, False, True, "low")
                        ).autonomy_level,
    ]
    assert levels == sorted(levels)


def test_layered_hybrid_only_at_scale():
    p = TaskProfile(False, open_exploration=True, risk="low")
    assert choose_layered(p, 100) is None
    d = choose_layered(p, 1000)
    assert d and d.choice == "layered_hybrid"


def test_layered_not_suggested_for_pure_clear_flow():
    p = TaskProfile(True)
    assert choose_layered(p, 1_000_000) is None


def test_every_decision_has_rationale():
    for kw in (dict(flow_explicit=True),
               dict(flow_explicit=False, expert_relay=True),
               dict(flow_explicit=False, encapsulable=True),
               dict(flow_explicit=False, open_exploration=True),
               dict(flow_explicit=False)):
        d = choose_topology(TaskProfile(**kw))
        assert d.rationale and len(d.rationale) > 10
