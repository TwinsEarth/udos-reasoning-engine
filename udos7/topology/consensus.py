"""分布式停止共识 BFT-lite（v7.4.7）。

百万级矩阵不能由单点决定"何时停止"。委员会对 stop/continue 投票，
容忍至多 f 个拜占庭成员（谎报、重复投票/equivocation、缺席）：

- 安全性：n ≥ 3f+1 时，相互冲突的决定不可能各凑齐 2f+1 张
  *诚实*票；拜占庭重复投票被检出并只计一次；
- 活性：超过 f 个成员缺席则凑不齐法定人数，触发 view change，
  而不是提前终止或无限等待。

确定性 CPU 模拟（签名用哈希占位，非密码学安全），证据 cpu-proto。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

STOP = "stop"
CONTINUE = "continue"
NO_QUORUM = "no_quorum"


def sign(voter: str, decision: str, round_id: int) -> str:
    return hashlib.sha256(f"{voter}|{decision}|{round_id}".encode()) \
        .hexdigest()[:16]


@dataclass(frozen=True)
class Ballot:
    voter: str
    decision: str
    round_id: int
    sig: str


@dataclass
class ConsensusResult:
    decision: Optional[str]
    quorum: int
    equivocators: List[str]
    silent: List[str]
    safety_violation: bool
    reason: str


class StopConsensus:
    def __init__(self, voters: List[str], f: int):
        if len(voters) < 3 * f + 1:
            raise ValueError(
                f"unsafe committee: n={len(voters)} < 3f+1={3*f+1}")
        self.voters = list(voters)
        self.f = f
        self.quorum_size = 2 * f + 1
        self._ballots: Dict[str, List[Ballot]] = {}

    def cast(self, voter: str, decision: str, round_id: int) -> None:
        if voter not in self.voters:
            raise ValueError(f"unknown voter {voter}")
        sig = sign(voter, decision, round_id)
        self._ballots.setdefault(voter, []).append(
            Ballot(voter, decision, round_id, sig))

    def equivocators(self, round_id: int) -> List[str]:
        out = []
        for voter, bs in self._ballots.items():
            decisions = {b.decision for b in bs if b.round_id == round_id}
            if len(decisions) > 1:
                out.append(voter)
        return out

    def decide(self, round_id: int) -> ConsensusResult:
        equiv = self.equivocators(round_id)
        # 诚实票：每轮每人按其第一次投票计一票；equivocator 整轮作废
        tally: Dict[str, int] = {STOP: 0, CONTINUE: 0}
        for voter in self.voters:
            if voter in equiv:
                continue
            bs = [b for b in self._ballots.get(voter, [])
                  if b.round_id == round_id]
            if bs:
                tally[bs[0].decision] += 1
        voted = set(v for v in self.voters
                    if any(b.round_id == round_id
                           for b in self._ballots.get(v, [])))
        silent = [v for v in self.voters if v not in voted]
        safety_violation = (tally[STOP] >= self.quorum_size and
                            tally[CONTINUE] >= self.quorum_size)
        if safety_violation:
            return ConsensusResult(None, self.quorum_size, equiv, silent,
                                   True, "conflicting_quorums")
        for dec in (STOP, CONTINUE):
            if tally[dec] >= self.quorum_size:
                return ConsensusResult(dec, self.quorum_size, equiv, silent,
                                       False, "quorum")
        return ConsensusResult(NO_QUORUM, self.quorum_size, equiv, silent,
                               False,
                               "view_change" if len(silent) > self.f
                               else "waiting")
