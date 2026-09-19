"""三种拓扑内核（v7.4.1）：Orchestrator / Handoff / Swarm。

工作负载：工单（WorkOrder）经 triage（分诊领域）→ specialist（领域处理，
对 work units 求和）→ qa（核验）三阶段。真值在工单内，成功可机械判定。
Agent 可注入故障：drop_context / byzantine / duplicate / crash。

三种拓扑跑**同一套处理器**，差异只在控制权与消息路径：
- Orchestrator：中心规划/派发/合并，worker 不互相通信；
- Handoff：工单沿 triage→specialist→qa 接力，责任随 Transfer Bundle 转移；
- Swarm：每阶段广播任务（contract-net），有能力者投标，按负载/确定性规则中标。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

DOMAINS = ("kinematics", "spatial", "dataflywheel")
STAGES = ("triage", "specialist", "qa")
FAULTS = (None, "drop_context", "byzantine", "duplicate", "crash")


@dataclass
class WorkOrder:
    id: str
    goal: str
    true_domain: str
    units: Tuple[int, ...]
    required_skills: Tuple[str, ...] = ("triage", "specialist", "qa")

    @property
    def true_result(self) -> int:
        return sum(self.units)

    def verify(self, result: Optional[int], domain: Optional[str]) -> bool:
        return result == self.true_result and domain == self.true_domain


@dataclass
class AgentReport:
    agent_id: str
    stage: str
    ok: bool
    domain: Optional[str] = None
    result: Optional[int] = None
    error_type: Optional[str] = None
    note: str = ""


@dataclass
class TraceEvent:
    seq: int
    order_id: str
    stage: str
    actor: str
    kind: str                       # dispatch/handoff/bid/award/return
    parent_event: Optional[int] = None
    payload: Dict = field(default_factory=dict)


@dataclass
class RunMetrics:
    topology: str
    n_orders: int
    succeeded: int = 0
    messages: int = 0
    hops: int = 0
    trace: List[TraceEvent] = field(default_factory=list)
    duplicate_work: int = 0
    owner_claims: Dict[str, str] = field(default_factory=dict)   # order:stage -> agent
    failures: List[Dict] = field(default_factory=list)
    _seq: int = 0

    def event(self, order_id, stage, actor, kind, parent=None, **payload):
        ev = TraceEvent(self._seq, order_id, stage, actor, kind, parent, payload)
        self._seq += 1
        self.trace.append(ev)
        return ev.seq

    def claim(self, order_id, stage, agent_id) -> bool:
        """登记唯一 Owner；返回 False 表示发生重复劳动。"""
        key = f"{order_id}:{stage}"
        if key in self.owner_claims and self.owner_claims[key] != agent_id:
            self.duplicate_work += 1
            return False
        self.owner_claims[key] = agent_id
        return True


# ---- 处理器（同一套，三种拓扑共用） -------------------------------------

def triage_handler(agent: "AgentSpec", order: WorkOrder, ctx: Dict) -> AgentReport:
    if agent.fault == "crash":
        return AgentReport(agent.id, "triage", False, error_type="crash")
    if agent.fault == "byzantine":
        return AgentReport(agent.id, "triage", True,
                           domain=DOMAINS[(DOMAINS.index(order.true_domain) + 1) % 3])
    return AgentReport(agent.id, "triage", True, domain=order.true_domain)


def specialist_handler(agent: "AgentSpec", order: WorkOrder, ctx: Dict) -> AgentReport:
    if agent.fault == "crash":
        return AgentReport(agent.id, "specialist", False, error_type="crash")
    if agent.fault == "drop_context":
        return AgentReport(agent.id, "specialist", False,
                           error_type="missing_context",
                           note="未收到 domain/units")
    if agent.fault == "byzantine":
        return AgentReport(agent.id, "specialist", True,
                           domain=ctx.get("domain"), result=order.true_result + 7)
    return AgentReport(agent.id, "specialist", True,
                       domain=ctx.get("domain"), result=order.true_result)


def qa_handler(agent: "AgentSpec", order: WorkOrder, ctx: Dict) -> AgentReport:
    if agent.fault == "crash":
        return AgentReport(agent.id, "qa", False, error_type="crash")
    ok = order.verify(ctx.get("result"), ctx.get("domain"))
    return AgentReport(agent.id, "qa", ok, domain=ctx.get("domain"),
                       result=ctx.get("result"),
                       note="pass" if ok else "reject")


HANDLERS = {"triage": triage_handler, "specialist": specialist_handler,
            "qa": qa_handler}


@dataclass
class AgentSpec:
    id: str
    capabilities: Tuple[str, ...]
    domain: Optional[str] = None      # specialist 绑定领域
    fault: Optional[str] = None
    load: int = 0

    def can(self, stage, domain=None):
        if stage not in self.capabilities:
            return False
        if stage == "specialist" and self.domain is not None and \
                domain is not None and self.domain != domain:
            return False
        return True

    def run(self, stage, order, ctx):
        return HANDLERS[stage](self, order, ctx)


def build_default_fleet(seed_faults: Optional[Dict[str, str]] = None
                        ) -> List[AgentSpec]:
    """1 triage + 每领域 1 specialist + 2 QA（QA 可投票）。"""
    faults = seed_faults or {}
    fleet = [AgentSpec("triage-1", ("triage",), fault=faults.get("triage-1"))]
    for d in DOMAINS:
        fleet.append(AgentSpec(f"spec-{d}", ("specialist",), domain=d,
                               fault=faults.get(f"spec-{d}")))
    fleet += [AgentSpec("qa-1", ("qa",), fault=faults.get("qa-1")),
              AgentSpec("qa-2", ("qa",), fault=faults.get("qa-2"))]
    return fleet


def standard_workload(n=12, seed=0) -> List[WorkOrder]:
    import torch
    g = torch.Generator().manual_seed(seed)
    orders = []
    for i in range(n):
        d = DOMAINS[int(torch.randint(0, len(DOMAINS), (1,), generator=g))]
        k = int(torch.randint(2, 6, (1,), generator=g))
        units = tuple(int(x) for x in torch.randint(1, 9, (k,), generator=g))
        orders.append(WorkOrder(f"WO-{i:04d}", f"process-{i}", d, units))
    return orders


class _Topology:
    name = "base"

    def __init__(self, fleet: List[AgentSpec]):
        self.fleet = fleet

    def _pick(self, stage, domain=None, rng_index=0):
        cand = [a for a in self.fleet if a.can(stage, domain)]
        if not cand:
            return None
        return cand[rng_index % len(cand)]

    def _metrics(self, orders):
        return RunMetrics(self.name, len(orders))


class Orchestrator(_Topology):
    """星型：中心对每阶段选 agent、派发、收回、合并。"""
    name = "orchestrator"

    def run(self, orders: List[WorkOrder]) -> RunMetrics:
        m = self._metrics(orders)
        for order in orders:
            ctx, parent = {}, None
            for stage in STAGES:
                domain = ctx.get("domain")
                agent = self._pick(stage, domain)
                if agent is None:
                    m.failures.append({"order": order.id, "stage": stage,
                                       "reason": "no_capable_agent"})
                    break
                m.claim(order.id, stage, agent.id)
                d_id = m.event(order.id, stage, agent.id, "dispatch", parent)
                m.messages += 2                       # 派发 + 回传
                rep = agent.run(stage, order, ctx)
                m.event(order.id, stage, agent.id, "return", d_id,
                        ok=rep.ok, error=rep.error_type)
                m.messages += 0
                parent = d_id
                agent.load += 1
                if not rep.ok:
                    m.failures.append({"order": order.id, "stage": stage,
                                       "actor": agent.id,
                                       "reason": rep.error_type or "qa_reject"})
                    break
                if stage == "triage":
                    ctx["domain"] = rep.domain
                elif stage == "specialist":
                    ctx["result"] = rep.result
                m.hops += 1
            else:
                if order.verify(ctx.get("result"), ctx.get("domain")):
                    m.succeeded += 1
        return m


class Handoff(_Topology):
    """链式：工单与上下文在 triage→specialist→qa 间接力转移。"""
    name = "handoff"

    def run(self, orders: List[WorkOrder]) -> RunMetrics:
        m = self._metrics(orders)
        for order in orders:
            bundle = {"goal": order.goal, "context": {"units": order.units},
                      "done": [], "todo": list(STAGES), "trace": [],
                      "owner": None}
            ctx, parent = {}, None
            for stage in STAGES:
                agent = self._pick(stage, ctx.get("domain"))
                if agent is None:
                    m.failures.append({"order": order.id, "stage": stage,
                                       "reason": "no_capable_agent"})
                    break
                m.claim(order.id, stage, agent.id)
                h_id = m.event(order.id, stage, agent.id, "handoff", parent,
                               from_=bundle["owner"], todo=list(bundle["todo"]))
                m.messages += 1                        # 责任移交
                bundle["owner"] = agent.id
                rep = agent.run(stage, order, ctx)
                bundle["trace"].append(h_id)
                bundle["done"].append(stage)
                bundle["todo"].remove(stage)
                parent = h_id
                agent.load += 1
                if not rep.ok:
                    m.failures.append({"order": order.id, "stage": stage,
                                       "actor": agent.id,
                                       "reason": rep.error_type or "qa_reject",
                                       "todo_left": list(bundle["todo"])})
                    break
                if stage == "triage":
                    ctx["domain"] = rep.domain
                    bundle["context"]["domain"] = rep.domain
                elif stage == "specialist":
                    ctx["result"] = rep.result
                    bundle["context"]["result"] = rep.result
                m.hops += 1
            else:
                if order.verify(ctx.get("result"), ctx.get("domain")):
                    m.succeeded += 1
        return m


class Swarm(_Topology):
    """网状：每阶段 contract-net 广播→投标→授标，无中心。"""
    name = "swarm"

    def run(self, orders: List[WorkOrder]) -> RunMetrics:
        m = self._metrics(orders)
        for order in orders:
            ctx, parent = {}, None
            for stage in STAGES:
                bidders = [a for a in self.fleet
                           if a.can(stage, ctx.get("domain"))]
                if not bidders:
                    m.failures.append({"order": order.id, "stage": stage,
                                       "reason": "no_bidder"})
                    break
                m.messages += len(bidders)            # 广播 + 投标
                # 确定性中标：当前负载最低，id 次序打破平手
                agent = sorted(bidders, key=lambda a: (a.load, a.id))[0]
                m.claim(order.id, stage, agent.id)
                a_id = m.event(order.id, stage, agent.id, "award", parent,
                               bids=len(bidders))
                m.messages += 1                        # 授标
                rep = agent.run(stage, order, ctx)
                m.event(order.id, stage, agent.id, "return", a_id, ok=rep.ok)
                m.messages += 1                        # 结果回公告板
                parent = a_id
                agent.load += 1
                if not rep.ok:
                    m.failures.append({"order": order.id, "stage": stage,
                                       "actor": agent.id,
                                       "reason": rep.error_type or "qa_reject"})
                    break
                if stage == "triage":
                    ctx["domain"] = rep.domain
                elif stage == "specialist":
                    ctx["result"] = rep.result
                m.hops += 1
            else:
                if order.verify(ctx.get("result"), ctx.get("domain")):
                    m.succeeded += 1
        return m


TOPOLOGIES = {"orchestrator": Orchestrator, "handoff": Handoff, "swarm": Swarm}
