"""v7.4.3 Transfer Bundle 契约测试。"""
import pytest

from udos7.topology.transfer import (TransferBundle, make_bundle, advance,
                                     handoff, replay, BundleError)


def test_complete_bundle_handoff_succeeds():
    b = make_bundle("g", {"units": (1, 2, 3)},
                    ("triage", "specialist", "qa"), trace0=0)
    b.owner = "triage-1"
    b, probs = handoff(b, "spec-kin", required_context=("units",))
    assert probs == [] and b.owner == "spec-kin"


def test_missing_owner_blocks_transfer():
    b = make_bundle("g", {"units": (1,)}, ("triage",), 0)
    out, probs = handoff(b, "spec-x")
    assert out is None and "owner" in probs


def test_missing_required_context_is_state_loss():
    b = make_bundle("g", {}, ("specialist", "qa"), 0)
    b.owner = "triage-1"
    out, probs = handoff(b, "spec-kin", required_context=("domain", "units"))
    assert out is None
    assert "context.domain" in probs and "context.units" in probs


def test_empty_todo_blocks_handoff_nothing_to_do():
    b = TransferBundle(goal="g", context={"x": 1}, done=["a"], todo=[],
                       trace=[1], owner="a1")
    out, probs = handoff(b, "a2")
    assert out is None and "todo_empty" in probs


def test_done_todo_overlap_rejected():
    b = TransferBundle(goal="g", context={}, done=["triage"],
                       todo=["triage", "qa"], trace=[1], owner="a1")
    assert "done_todo_overlap" in b.validate()


def test_advance_moves_stage_and_records_outputs():
    b = make_bundle("g", {"units": (1, 2)}, ("triage", "specialist", "qa"), 0)
    b.owner = "triage-1"
    b = advance(b, "triage", {"domain": "kinematics"}, 1, "spec-kin")
    assert b.done == ["triage"] and b.todo == ["specialist", "qa"]
    assert b.context["domain"] == "kinematics" and b.trace == [0, 1]
    assert b.owner == "spec-kin"


def test_advance_unknown_stage_is_stale():
    b = make_bundle("g", {}, ("triage",), 0)
    with pytest.raises(BundleError):
        advance(b, "qa", {}, 1, "q")


def test_replay_reconstructs_state_without_sender():
    b = make_bundle("g", {"units": (1,), "domain": "spatial"},
                    ("specialist", "qa"), 0)
    b.owner = "triage-1"
    b = advance(b, "specialist", {"result": 1}, 1, "qa-1")
    state = replay(b)
    assert state["remaining"] == ["qa"]
    assert state["facts"] == {"units": (1,), "domain": "spatial", "result": 1}
    assert state["owner"] == "qa-1"


def test_drop_context_agent_leaves_bundle_invalid():
    """故障注入：drop_context 的 specialist 不写 result，QA 无法接手。"""
    b = make_bundle("g", {"units": (1, 2)}, ("specialist", "qa"), 0)
    b.owner = "triage-1"
    # specialist 崩溃式交接：未产出 result 就想交给 qa
    out, probs = handoff(b, "qa-1", required_context=("domain", "result"))
    assert out is None and "context.result" in probs
