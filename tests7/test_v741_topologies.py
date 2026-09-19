"""v7.4.1 三种拓扑内核契约测试。"""
import pytest

from udos7.topology import (Orchestrator, Handoff, Swarm, build_default_fleet,
                            standard_workload)
from udos7.topology.base import AgentSpec, WorkOrder, DOMAINS


@pytest.fixture
def orders():
    return standard_workload(n=12, seed=7)


def test_clean_fleet_all_topologies_succeed(orders):
    fleet = build_default_fleet()
    for cls in (Orchestrator, Handoff, Swarm):
        m = cls(fleet).run(orders)
        assert m.succeeded == 12, cls.name
        assert m.duplicate_work == 0
        # 每阶段都有唯一 owner
        assert len(m.owner_claims) == 12 * 3


def test_trace_is_parented_and_ordered(orders):
    m = Handoff(build_default_fleet()).run(orders[:2])
    seqs = [e.seq for e in m.trace]
    assert seqs == sorted(seqs)
    # 每个工单三阶段的 parent 链：后两阶段有父事件
    for oid in (orders[0].id, orders[1].id):
        evs = [e for e in m.trace if e.order_id == oid]
        assert evs[0].parent_event is None
        assert evs[1].parent_event == evs[0].seq
        assert evs[2].parent_event == evs[1].seq


def test_message_patterns_differ(orders):
    """星型 2 消息/阶段；链式 1 移交/阶段；网状含投标广播，消息最多。"""
    o = Orchestrator(build_default_fleet()).run(orders)
    h = Handoff(build_default_fleet()).run(orders)
    s = Swarm(build_default_fleet()).run(orders)
    assert o.messages == 12 * 3 * 2
    assert h.messages == 12 * 3
    assert s.messages > o.messages


def test_byzantine_specialist_fails_qa_and_is_traced():
    orders = [WorkOrder("W1", "g", "kinematics", (1, 2, 3))]
    fleet = build_default_fleet({"spec-kinematics": "byzantine"})
    m = Orchestrator(fleet).run(orders)
    assert m.succeeded == 0
    f = [f for f in m.failures if f["order"] == "W1"]
    assert f and f[0]["stage"] == "qa" and f[0]["reason"] == "qa_reject"


def test_crash_agent_reports_typed_error():
    orders = [WorkOrder("W1", "g", "spatial", (4, 5))]
    fleet = build_default_fleet({"triage-1": "crash"})
    m = Swarm(fleet).run(orders)
    assert m.succeeded == 0
    assert m.failures[0]["reason"] == "crash"


def test_missing_capability_is_recorded_not_silent():
    orders = [WorkOrder("W1", "g", "dataflywheel", (1,))]
    # 没有 triage 能力的舰队
    fleet = [AgentSpec("s", ("specialist",), domain="dataflywheel"),
             AgentSpec("q", ("qa",))]
    m = Orchestrator(fleet).run(orders)
    assert m.succeeded == 0
    assert m.failures[0]["reason"] == "no_capable_agent"


def test_swarm_load_balances_qa_across_two_agents():
    orders = standard_workload(n=10, seed=3)
    fleet = build_default_fleet()
    Swarm(fleet).run(orders)
    qa1 = next(a for a in fleet if a.id == "qa-1")
    qa2 = next(a for a in fleet if a.id == "qa-2")
    # 最低负载中标 → 两个 QA 都应被使用
    assert qa1.load >= 1 and qa2.load >= 1


def test_handoff_transfers_todo_list():
    orders = [WorkOrder("W1", "g", "kinematics", (2, 2))]
    m = Handoff(build_default_fleet()).run(orders)
    handoffs = [e for e in m.trace if e.kind == "handoff"]
    assert handoffs[0].payload["todo"] == ["triage", "specialist", "qa"]
    assert handoffs[1].payload["todo"] == ["specialist", "qa"]
    assert handoffs[2].payload["todo"] == ["qa"]
