"""多层熔断、隔离与回滚（v7.4.8）。

三层防御：
- Agent 级断路器：滑动窗口错误率超阈值 → open（隔离），冷却后半开探测，
  探测成功才恢复 closed；
- 子矩阵级：组内隔离比例超阈值 → 整组熔断、任务改派健康组；
- 全局 kill-switch：一旦置位，所有派发立即停止。

回滚：在派发前对 TraceLedger 做快照（记录长度+末哈希）；故障隔离后
把账本截断回快照点、把在途工单改派，保证不留半截状态。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class AgentBreaker:
    name: str
    trip_threshold: float = 0.5
    min_requests: int = 4
    cooldown_ticks: int = 3
    state: str = CLOSED
    successes: int = 0
    failures: int = 0
    tripped_at: Optional[int] = None
    isolated_orders: List[str] = field(default_factory=list)

    @property
    def error_rate(self):
        tot = self.successes + self.failures
        return self.failures / tot if tot else 0.0

    def record(self, ok: bool, tick: int):
        if self.state == OPEN:
            return
        if ok:
            self.successes += 1
        else:
            self.failures += 1
        if (self.successes + self.failures >= self.min_requests
                and self.error_rate >= self.trip_threshold):
            self.state = OPEN
            self.tripped_at = tick

    def tick(self, current_tick: int) -> str:
        """冷却后半开，等待一次探测。"""
        if (self.state == OPEN and self.tripped_at is not None
                and current_tick - self.tripped_at >= self.cooldown_ticks):
            self.state = HALF_OPEN
        return self.state

    def probe(self, ok: bool):
        if self.state != HALF_OPEN:
            return self.state
        self.state = CLOSED if ok else OPEN
        if ok:
            self.successes = self.failures = 0
            self.tripped_at = None
        return self.state

    @property
    def available(self):
        return self.state == CLOSED


@dataclass
class GroupBreaker:
    name: str
    members: List[str]
    isolate_ratio_threshold: float = 0.5
    tripped: bool = False

    def evaluate(self, breakers: Dict[str, AgentBreaker]) -> bool:
        iso = sum(1 for m in self.members
                  if breakers[m].state == OPEN)
        if iso / len(self.members) >= self.isolate_ratio_threshold:
            self.tripped = True
        return self.tripped


class KillSwitch:
    def __init__(self):
        self.engaged = False
        self.reason = None

    def trip(self, reason: str):
        self.engaged = True
        self.reason = reason

    def allow_dispatch(self):
        return not self.engaged


@dataclass
class LedgerSnapshot:
    length: int
    head_hash: Optional[str]


def snapshot(ledger) -> LedgerSnapshot:
    return LedgerSnapshot(len(ledger.records),
                          ledger.records[-1]["h"] if ledger.records else None)


def rollback(ledger, snap: LedgerSnapshot) -> int:
    """截断到快照点；返回被撤销的记录数。"""
    removed = len(ledger.records) - snap.length
    if removed < 0:
        raise ValueError("snapshot ahead of ledger")
    if removed:
        del ledger.records[snap.length:]
    return removed


class ResilienceGrid:
    """聚合三层熔断 + 故障时改派。"""

    def __init__(self, agent_names: List[str], groups: List[GroupBreaker]):
        self.breakers = {n: AgentBreaker(n) for n in agent_names}
        self.groups = {g.name: g for g in groups}
        self.kill = KillSwitch()
        self.reassignments: List[Dict] = []

    def record(self, agent: str, ok: bool, tick: int,
               in_flight: Optional[List[str]] = None):
        b = self.breakers[agent]
        before = b.state
        b.record(ok, tick)
        if before != OPEN and b.state == OPEN and in_flight:
            b.isolated_orders.extend(in_flight)
        for g in self.groups.values():
            g.evaluate(self.breakers)

    def healthy_agents(self) -> List[str]:
        if self.kill.engaged:
            return []
        tripped_groups = {m for g in self.groups.values() if g.tripped
                          for m in g.members}
        return [n for n, b in self.breakers.items()
                if b.available and n not in tripped_groups]

    def reassign(self, order_id: str, failed_agent: str,
                 candidates: List[str]) -> Optional[str]:
        if self.kill.engaged:
            return None
        for c in candidates:
            if c != failed_agent and self.breakers[c].available:
                self.reassignments.append(
                    {"order": order_id, "from": failed_agent, "to": c})
                return c
        return None
