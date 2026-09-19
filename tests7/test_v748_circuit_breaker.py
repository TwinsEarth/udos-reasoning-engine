"""v7.4.8 熔断/隔离/回滚测试。"""
import pytest

from udos7.topology.circuit_breaker import (AgentBreaker, GroupBreaker,
                                            KillSwitch, ResilienceGrid,
                                            snapshot, rollback,
                                            CLOSED, OPEN, HALF_OPEN)
from udos7.topology.governance import TraceLedger


def test_breaker_trips_after_error_rate():
    b = AgentBreaker("a", trip_threshold=0.5, min_requests=4)
    for ok in (False, True, False, False):
        b.record(ok, tick=0)
    assert b.state == OPEN and not b.available
    assert b.error_rate == 0.75


def test_healthy_agent_not_isolated():
    b = AgentBreaker("a")
    for _ in range(10):
        b.record(True, 0)
    assert b.state == CLOSED and b.available


def test_half_open_probe_recovers():
    b = AgentBreaker("a", min_requests=2, cooldown_ticks=2)
    b.record(False, 0); b.record(False, 0)
    assert b.state == OPEN
    assert b.tick(1) == OPEN
    assert b.tick(2) == HALF_OPEN
    assert b.probe(True) == CLOSED and b.available


def test_failed_probe_reopens():
    b = AgentBreaker("a", min_requests=2, cooldown_ticks=1)
    b.record(False, 0); b.record(False, 0)
    b.tick(1)
    assert b.probe(False) == OPEN


def test_group_breaker_trips_on_mass_isolation():
    g = GroupBreaker("g1", ["a", "b", "c", "d"], isolate_ratio_threshold=0.5)
    grid = ResilienceGrid(["a", "b", "c", "d"], [g])
    for a in ("a", "b"):
        grid.breakers[a].state = OPEN
    assert g.evaluate(grid.breakers) is True


def test_kill_switch_blocks_all_dispatch():
    grid = ResilienceGrid(["a", "b"], [])
    assert grid.healthy_agents() == ["a", "b"]
    grid.kill.trip("global emergency")
    assert grid.healthy_agents() == []
    assert grid.reassign("W1", "a", ["b"]) is None


def test_reassign_to_healthy_peer():
    grid = ResilienceGrid(["a", "b", "c"], [])
    grid.breakers["a"].state = OPEN
    to = grid.reassign("W1", "a", ["b", "c"])
    assert to == "b"
    assert grid.reassignments[0] == {"order": "W1", "from": "a", "to": "b"}


def test_reassign_skips_isolated_and_tripped_group():
    g = GroupBreaker("g1", ["b", "c"], isolate_ratio_threshold=0.5)
    grid = ResilienceGrid(["a", "b", "c"], [g])
    grid.breakers["a"].state = OPEN
    grid.breakers["b"].state = OPEN
    grid.breakers["c"].state = OPEN
    g.evaluate(grid.breakers)
    assert grid.reassign("W1", "a", ["b", "c"]) is None


def test_ledger_snapshot_rollback_removes_half_state():
    led = TraceLedger()
    led.append({"stage": "triage"})
    snap = snapshot(led)
    led.append({"stage": "specialist"})
    led.append({"stage": "qa_partial"})
    assert len(led.records) == 3
    removed = rollback(led, snap)
    assert removed == 2 and len(led.records) == 1
    assert led.verify() == []


def test_rollback_then_replay_keeps_chain_valid():
    led = TraceLedger()
    for i in range(4):
        led.append({"i": i})
    snap = snapshot(led)
    led.append({"i": 99})
    rollback(led, snap)
    led.append({"i": 4})           # 重新派发产生新记录
    assert led.verify() == []
    assert len(led.records) == 5


def test_isolated_orders_recorded_for_reassignment():
    grid = ResilienceGrid(["a"], [])
    grid.breakers["a"].min_requests = 1
    grid.record("a", False, 0, in_flight=["W7", "W8"])
    assert grid.breakers["a"].state == OPEN
    assert set(grid.breakers["a"].isolated_orders) == {"W7", "W8"}
