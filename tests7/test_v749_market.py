"""v7.4.9 内部市场与贡献结算测试。"""
import pytest

from udos7.topology.market import Bid, TaskAuction, ContributionLedger


def test_auction_picks_best_quality_cost_ratio():
    a = TaskAuction("T1", reward=10)
    bids = [Bid("a", cost=8, quality=0.8),
            Bid("b", cost=4, quality=0.8),     # 同质低价，性价比高
            Bid("c", cost=3, quality=0.2)]
    assert a.award(bids) == "b"


def test_quality_floor_filters_cheap_low_quality():
    a = TaskAuction("T1", reward=10, quality_floor=0.6)
    bids = [Bid("a", cost=1, quality=0.1), Bid("b", cost=9, quality=0.7)]
    assert a.award(bids) == "b"


def test_no_eligible_bid_no_winner():
    a = TaskAuction("T1", reward=10, quality_floor=0.9)
    assert a.award([Bid("a", 1, 0.1)]) is None
    assert a.winner is None


def test_load_breaks_tie():
    a = TaskAuction("T1", reward=10)
    bids = [Bid("a", cost=4, quality=0.8, load=5),
            Bid("b", cost=4, quality=0.8, load=1)]
    assert a.award(bids) == "b"


def test_only_accepted_unique_work_paid():
    led = ContributionLedger()
    led.deposit("T1", 10.0)
    r1 = led.settle("T1", "a", accepted=True)
    assert r1["paid"] == 10.0 and led.balances["a"] == 10.0
    r2 = led.settle("T1", "b", accepted=True)   # 重复完成
    assert r2["paid"] == 0.0 and r2["reason"] == "already_paid"
    assert "b" not in led.balances


def test_duplicate_work_flag_not_paid():
    led = ContributionLedger()
    led.deposit("T1", 5.0)
    r = led.settle("T1", "a", accepted=True, duplicate=True)
    assert r["reason"] == "duplicate_work" and r["paid"] == 0
    assert led.total_paid == 0


def test_rejected_byzantine_work_unpaid_and_slashable():
    led = ContributionLedger()
    led.deposit("T1", 5.0)
    led.balances["z"] = 3.0
    r = led.settle("T1", "z", accepted=False, slash=2.0)
    assert r["paid"] == 0 and led.balances["z"] == 1.0
    assert led.slashed == 2.0 and led.total_paid == 0


def test_conservation_invariant():
    led = ContributionLedger()
    for t, rw in (("T1", 10), ("T2", 8), ("T3", 6)):
        led.deposit(t, rw)
    led.settle("T1", "a", accepted=True)
    led.settle("T2", "b", accepted=False, slash=1.0)
    led.balances.setdefault("b", 0.0)
    led.settle("T3", "c", accepted=True)
    chk = led.conservation_check()
    assert chk["conserved"]
    assert chk["total_paid"] == 16
    assert chk["holds"] == 8          # T2 未付预算仍在池子
    assert chk["balance_sum"] == 15   # 16 已付 − 1 罚没


def test_payment_never_exceeds_budget():
    led = ContributionLedger()
    led.deposit("T1", 5.0)
    led.settle("T1", "a", accepted=True)
    chk = led.conservation_check()
    assert chk["total_paid"] <= chk["total_budget"]


def test_rejected_work_recorded():
    led = ContributionLedger()
    led.deposit("T1", 5.0)
    led.settle("T1", "a", accepted=False)
    assert led.rejected[0]["reason"] == "rejected"
