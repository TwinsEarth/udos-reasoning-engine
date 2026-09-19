"""v7.4.7 BFT-lite 停止共识测试。"""
import pytest

from udos7.topology.consensus import (StopConsensus, Ballot, sign, STOP,
                                      CONTINUE, NO_QUORUM)


def voters(n):
    return [f"qa-{i}" for i in range(n)]


def test_unsafe_committee_rejected():
    with pytest.raises(ValueError):
        StopConsensus(voters(6), f=2)        # 6 < 3*2+1


def test_honest_supermajority_stops():
    c = StopConsensus(voters(7), f=2)
    for v in voters(7)[:5]:
        c.cast(v, STOP, 0)
    r = c.decide(0)
    assert r.decision == STOP and r.reason == "quorum"
    assert not r.safety_violation


def test_byzantine_cannot_force_stop_against_honest_continue():
    c = StopConsensus(voters(7), f=2)
    honest = voters(7)[:5]
    byz = voters(7)[5:]
    for v in honest:
        c.cast(v, CONTINUE, 0)
    for v in byz:                        # f=2 谎报 stop
        c.cast(v, STOP, 0)
    r = c.decide(0)
    assert r.decision == CONTINUE
    assert tally_for(c, STOP, 0) == 2


def test_equivocation_detected_and_voided():
    c = StopConsensus(voters(7), f=2)
    for v in voters(7)[:4]:
        c.cast(v, STOP, 0)
    b1, b2 = voters(7)[4], voters(7)[5]
    c.cast(b1, STOP, 0); c.cast(b1, CONTINUE, 0)   # 同一轮反复
    c.cast(b2, STOP, 0)
    r = c.decide(0)
    # b1 作废后 stop 只有 5 张？4+b2=5 → 仍达 2f+1=5
    assert b1 in r.equivocators
    assert r.decision == STOP and r.quorum == 5


def test_equivocation_breaks_would_be_quorum():
    c = StopConsensus(voters(7), f=2)
    for v in voters(7)[:4]:
        c.cast(v, STOP, 0)
    b = voters(7)[4]
    c.cast(b, STOP, 0); c.cast(b, CONTINUE, 0)
    r = c.decide(0)
    # 4 票 < 5，b 作废 → 无法定人数
    assert r.decision == NO_QUORUM and b in r.equivocators


def test_conflicting_quorums_impossible_with_honest_votes():
    """n=7,f=2：诚实票只有 5 张，不可能两边各 5。"""
    c = StopConsensus(voters(7), f=2)
    for v in voters(7)[:5]:
        c.cast(v, STOP, 0)
    for v in voters(7)[5:]:
        c.cast(v, CONTINUE, 0)
    r = c.decide(0)
    assert not r.safety_violation and r.decision == STOP


def test_too_many_silent_triggers_view_change():
    c = StopConsensus(voters(7), f=2)
    for v in voters(7)[:3]:                 # 4 个缺席 > f=2
        c.cast(v, CONTINUE, 0)
    r = c.decide(0)
    assert r.decision == NO_QUORUM and r.reason == "view_change"
    assert len(r.silent) == 4


def test_waiting_when_quorum_not_yet_met():
    c = StopConsensus(voters(7), f=2)
    for v in voters(7)[:4]:
        c.cast(v, CONTINUE, 0)
    c.cast(voters(7)[4], STOP, 0)   # 5 票已投但两边都不到 5，沉默 2 ≤ f
    r = c.decide(0)
    assert r.decision == NO_QUORUM and r.reason == "waiting"


def test_unknown_voter_rejected():
    c = StopConsensus(voters(7), f=2)
    with pytest.raises(ValueError):
        c.cast("sybil", STOP, 0)


def test_signature_is_deterministic():
    assert sign("a", STOP, 1) == sign("a", STOP, 1)
    assert sign("a", STOP, 1) != sign("a", CONTINUE, 1)


def tally_for(c: StopConsensus, dec, round_id):
    r = c.decide(round_id)
    # 辅助：直接数未作废票
    n = 0
    for v in c.voters:
        if v in r.equivocators:
            continue
        bs = [b for b in c._ballots.get(v, []) if b.round_id == round_id]
        if bs and bs[0].decision == dec:
            n += 1
    return n
