"""Owner / Trace / Stop Condition 治理（v7.4.4）。

三类典型失败的机械检测：
- 状态丢失 state_loss：交接缺上下文（missing_context）或仍有 todo 即中断；
- 重复劳动 duplicate_work：同一 order:stage 被两个不同 owner 认领；
- 责任不清 no_closer：工单走完却没有最终阶段 owner / 无人收口。
另含追加式哈希链 TraceLedger（可发现篡改/断链）与 StopCondition
（完成谓词 + 最大跳数，区分提前终止与无限循环）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .base import RunMetrics, STAGES


@dataclass
class TraceLedger:
    """追加式账本：每条记录含前一条哈希，断链/改写可被 verify 发现。"""
    records: List[Dict] = field(default_factory=list)

    def append(self, rec: Dict) -> str:
        prev = self.records[-1]["h"] if self.records else "GENESIS"
        h = hashlib.sha256(
            (prev + json.dumps(rec, sort_keys=True,
                               default=str)).encode()).hexdigest()[:16]
        rec = dict(rec, h=h, prev=prev)
        self.records.append(rec)
        return h

    def verify(self) -> List[int]:
        """返回断链或内容被改写的记录下标；空列表表示账本完整。"""
        bad, prev = [], "GENESIS"
        for i, r in enumerate(self.records):
            body = {k: v for k, v in r.items() if k not in ("h", "prev")}
            expect = hashlib.sha256(
                (prev + json.dumps(body, sort_keys=True,
                                   default=str)).encode()).hexdigest()[:16]
            if r.get("prev") != prev or r.get("h") != expect:
                bad.append(i)
            prev = r.get("h", expect)
        return bad


@dataclass
class StopCondition:
    is_done: Callable[[Dict], bool]
    max_hops: int = 12

    def check(self, state: Dict, hops: int):
        """返回 (stop, reason)。"""
        if self.is_done(state):
            return True, "done"
        if hops >= self.max_hops:
            return True, "hop_budget_exceeded"     # 防无限循环
        return False, None


@dataclass
class GovernanceReport:
    state_loss: List[Dict] = field(default_factory=list)
    duplicate_work: List[Dict] = field(default_factory=list)
    no_closer: List[str] = field(default_factory=list)
    premature_completion: List[str] = field(default_factory=list)
    unbounded: List[str] = field(default_factory=list)
    trace_gaps: List[int] = field(default_factory=list)

    @property
    def ok(self):
        return not (self.state_loss or self.duplicate_work or self.no_closer
                    or self.premature_completion or self.unbounded
                    or self.trace_gaps)

    def summary(self):
        return {"ok": self.ok, "state_loss": len(self.state_loss),
                "duplicate_work": len(self.duplicate_work),
                "no_closer": len(self.no_closer),
                "premature_completion": len(self.premature_completion),
                "unbounded": len(self.unbounded),
                "trace_gaps": len(self.trace_gaps)}


def audit_run(m: RunMetrics, order_ids: List[str],
              stages=STAGES, succeeded_ids: Optional[set] = None,
              ledger: Optional[TraceLedger] = None) -> GovernanceReport:
    rep = GovernanceReport()
    # 1) 状态丢失：失败记录里带缺失上下文，或中断时仍有 todo
    for f in m.failures:
        if f.get("reason") in ("missing_context",) or f.get("todo_left"):
            rep.state_loss.append(f)
    # 2) 重复劳动
    if m.duplicate_work:
        seen: Dict[str, List[str]] = {}
        for key, owner in m.owner_claims.items():
            seen.setdefault(key, []).append(owner)
        # owner_claims 只保留最后一个 owner，重复计数来自 m.duplicate_work
        rep.duplicate_work.append({"count": m.duplicate_work})
    # 3) 责任不清：未收口的工单（无最终 owner，或最终阶段未通过）
    final = stages[-1]
    closed_ids = succeeded_ids if succeeded_ids is not None else set()
    for oid in order_ids:
        if succeeded_ids is not None:
            if oid not in closed_ids:
                rep.no_closer.append(oid)
        elif f"{oid}:{final}" not in m.owner_claims:
            rep.no_closer.append(oid)
    # 4) 提前终止：声称成功但缺任一阶段 owner
    for oid in closed_ids:
        if any(f"{oid}:{s}" not in m.owner_claims for s in stages):
            rep.premature_completion.append(oid)
    # 5) Trace 完整性
    if ledger is not None:
        rep.trace_gaps = ledger.verify()
    return rep


def hop_guarded_run(step_fn, init_state, stop: StopCondition):
    """通用停止条件执行器：step_fn(state)->state；返回 (state, reason, hops)。"""
    state, hops = init_state, 0
    while True:
        do_stop, reason = stop.check(state, hops)
        if do_stop:
            return state, reason, hops
        state = step_fn(state)
        hops += 1
