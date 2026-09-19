"""v7.4.4 Owner/Trace/Stop Condition 治理测试。"""
import pytest

from udos7.topology import (Orchestrator, Handoff, build_default_fleet,
                            standard_workload)
from udos7.topology.base import WorkOrder, RunMetrics
from udos7.topology.governance import (TraceLedger, StopCondition, audit_run,
                                       hop_guarded_run)


def succeeded_set(m, orders):
    # 用 owner_claims 完整 + 无失败近似成功集合；测试里直接按真值核验
    return {o.id for o in orders
            if all(f"{o.id}:{s}" in m.owner_claims for s in
                   ("triage", "specialist", "qa"))
            and not [f for f in m.failures if f["order"] == o.id]}


def test_clean_run_passes_audit():
    orders = standard_workload(8, seed=2)
    m = Orchestrator(build_default_fleet()).run(orders)
    rep = audit_run(m, [o.id for o in orders],
                    succeeded_ids=succeeded_set(m, orders))
    assert rep.ok, rep.summary()


def test_state_loss_detected_in_handoff():
    orders = [WorkOrder("W1", "g", "kinematics", (1, 2))]
    fleet = build_default_fleet({"spec-kinematics": "drop_context"})
    m = Handoff(fleet).run(orders)
    rep = audit_run(m, ["W1"])
    assert rep.state_loss and rep.state_loss[0]["todo_left"]
    assert "W1" in rep.no_closer


def test_duplicate_work_flagged():
    m = RunMetrics("t", 1)
    m.claim("W1", "triage", "a")
    second = m.claim("W1", "triage", "b")
    assert second is False and m.duplicate_work == 1
    rep = audit_run(m, ["W1"])
    assert rep.duplicate_work and rep.no_closer  # qa 无人收口


def test_no_closer_when_final_stage_missing():
    m = RunMetrics("t", 1)
    m.claim("W1", "triage", "a")
    m.claim("W1", "specialist", "b")
    rep = audit_run(m, ["W1"])
    assert rep.no_closer == ["W1"]


def test_premature_completion_detected():
    m = RunMetrics("t", 1)
    m.claim("W1", "triage", "a")
    # 谎称成功，但 specialist/qa 无 owner
    rep = audit_run(m, ["W1"], succeeded_ids={"W1"})
    assert rep.premature_completion == ["W1"]


def test_trace_ledger_detects_tamper_and_gap():
    led = TraceLedger()
    led.append({"a": 1})
    led.append({"a": 2})
    assert led.verify() == []
    led.records[0]["a"] = 999                 # 篡改历史
    assert 0 in led.verify()
    led2 = TraceLedger()
    led2.append({"a": 1})
    led2.append({"a": 2})
    led2.records[1]["prev"] = "FAKE"          # 断链
    assert 1 in led2.verify()


def test_stop_condition_done_predicate():
    stop = StopCondition(is_done=lambda s: s.get("n", 0) >= 3, max_hops=100)
    state, reason, hops = hop_guarded_run(
        lambda s: {**s, "n": s.get("n", 0) + 1}, {}, stop)
    assert reason == "done" and state["n"] == 3 and hops == 3


def test_stop_condition_hop_budget_prevents_infinite_loop():
    stop = StopCondition(is_done=lambda s: False, max_hops=5)
    _, reason, hops = hop_guarded_run(lambda s: s, {}, stop)
    assert reason == "hop_budget_exceeded" and hops == 5


def test_byzantine_qa_loop_has_governance_evidence():
    orders = [WorkOrder("W1", "g", "spatial", (3, 4))]
    fleet = build_default_fleet({"spec-spatial": "byzantine"})
    m = Orchestrator(fleet).run(orders)
    rep = audit_run(m, ["W1"], succeeded_ids=set())
    # QA 拒绝导致工单无收口
    assert "W1" in rep.no_closer
    assert any(f["stage"] == "qa" for f in m.failures)
