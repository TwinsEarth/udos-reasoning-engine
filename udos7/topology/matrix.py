"""百万级 Agent 矩阵集成（v7.5.0）。

把 v7.4.1–v7.4.10 的机制串成一条可运行的任务闭环（小规模显式模拟，
大规模消息复杂度走 hierarchy 闭式外推）：

  Stigmergy 黑板认领 → 分层 specialist 执行 → 故障熔断/隔离/改派
  → BFT-lite QA 委员会验收（2f+1）→ 内部市场按验收结算
  → TraceLedger 哈希链 + 治理审计 → 委员会共识决定批次停止。

所有 Agent 均为确定性 CPU 模拟（非真实 LLM 进程），证据等级 cpu-proto。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import (DOMAINS, STAGES, WorkOrder, AgentSpec, standard_workload,
                   specialist_handler, triage_handler)
from .stigmergy import Blackboard
from .circuit_breaker import ResilienceGrid, GroupBreaker, snapshot, rollback
from .consensus import StopConsensus, STOP, CONTINUE
from .market import ContributionLedger
from .governance import TraceLedger, audit_run, RunMetrics
from .hierarchy import scale_benchmark


@dataclass
class MatrixConfig:
    specialists_per_domain: int = 3
    qa_voters: int = 7
    qa_f: int = 2
    reward_per_order: float = 10.0
    max_rounds: int = 20
    breaker_min_requests: int = 2
    breaker_threshold: float = 0.5
    global_retry_budget: int = 999


def build_matrix(cfg: MatrixConfig, faulty_workers: Optional[Dict[str, str]]
                 = None, seed: int = 0):
    faulty = faulty_workers or {}
    specialists = []
    for d in DOMAINS:
        for k in range(cfg.specialists_per_domain):
            aid = f"spec-{d}-{k}"
            specialists.append(AgentSpec(
                aid, ("specialist",), domain=d, fault=faulty.get(aid)))
    qa = [f"qa-{i}" for i in range(cfg.qa_voters)]
    byz_qa = set(list(qa)[:cfg.qa_f]) if cfg.qa_f else set()
    groups = [GroupBreaker(f"grp-{d}",
                           [a.id for a in specialists if a.domain == d])
              for d in DOMAINS]
    grid = ResilienceGrid([a.id for a in specialists], groups)
    for b in grid.breakers.values():
        b.min_requests = cfg.breaker_min_requests
        b.trip_threshold = cfg.breaker_threshold
    return specialists, qa, byz_qa, grid


def run_mission(orders: List[WorkOrder], cfg: MatrixConfig = MatrixConfig(),
                faulty_workers: Optional[Dict[str, str]] = None,
                seed: int = 0) -> Dict:
    rng = random.Random(seed)
    specialists, qa, byz_qa, grid = build_matrix(cfg, faulty_workers, seed)
    board = Blackboard()
    market = ContributionLedger()
    trace = TraceLedger()
    metrics = RunMetrics("layered_matrix", len(orders))
    for o in orders:
        board.post(o.id, value=1.0)
        market.deposit(o.id, cfg.reward_per_order)

    by_id = {o.id: o for o in orders}
    origin = {o.id: o.id for o in orders}   # rework 项 → 原始工单
    accepted, qa_rounds, retries = set(), 0, 0
    in_flight_failures = 0

    for round_id in range(cfg.max_rounds):
        if not board.open_items():
            break
        # 1) 健康 specialist 经黑板原子认领并执行（熔断节点被排除）
        for a in specialists:
            if not grid.breakers[a.id].available:
                continue
            opens = board.open_items()
            cand = [k for k in opens
                    if by_id[k].true_domain == a.domain]
            if not cand:
                continue
            key = rng.choice(cand)
            if not board.claim(a.id, key):
                continue
            order = by_id[key]
            snap = snapshot(trace)
            trace.append({"order": key, "stage": "specialist",
                          "actor": a.id, "round": round_id})
            rep = specialist_handler(a, order, {"domain": order.true_domain})
            if not rep.ok:
                in_flight_failures += 1
                rollback(trace, snap)               # 撤销半截 trace
                grid.record(a.id, False, round_id, in_flight=[key])
                board.items[key].claimed_by = None  # 释放改派
                board.messages -= 0
                retries += 1
                if retries > cfg.global_retry_budget:
                    grid.kill.trip("retry_budget_exceeded")
                continue
            grid.record(a.id, True, round_id)
            board.complete(key)

            # 2) BFT-lite QA 委员会就"验收通过(stop)/返工(continue)"投票
            cons = StopConsensus(qa, f=cfg.qa_f)
            for v in qa:
                honest_accept = order.verify(rep.result, rep.domain)
                vote = honest_accept if v not in byz_qa else not honest_accept
                cons.cast(v, STOP if vote else CONTINUE, round_id)
            qa_rounds += 1
            res = cons.decide(round_id)
            trace.append({"order": key, "stage": "qa",
                          "decision": res.decision,
                          "equivocators": res.equivocators})
            metrics.claim(key, "triage", "triage-1")
            metrics.claim(key, "specialist", a.id)
            metrics.claim(key, "qa", "qa-committee")
            orig = origin[key]
            if res.decision == STOP:
                accepted.add(orig)
                market.settle(orig, a.id, accepted=True)
            else:
                # 验收未过：记一次质量失败（累积可隔离拜占庭 worker），
                # 以 rework 项重新挂回黑板，仍对应原始工单
                grid.record(a.id, False, round_id, in_flight=[orig])
                rk = key + f":rework{round_id}"
                board.post(rk, 1.0)
                by_id[rk] = order
                origin[rk] = orig
                retries += 1
        if grid.kill.engaged:
            break

    # 3) 治理审计
    closed = set(accepted)
    metrics.succeeded = len(closed)
    gov = audit_run(metrics, [o.id for o in orders], succeeded_ids=closed)
    conservation = market.conservation_check()
    return {
        "evidence_grade": "cpu-proto",
        "n_orders": len(orders),
        "accepted": len(accepted),
        "qa_rounds": qa_rounds,
        "retries": retries,
        "isolated_agents": [n for n, b in grid.breakers.items()
                            if not b.available],
        "tripped_groups": [n for n, g in grid.groups.items() if g.tripped],
        "kill_switch": grid.kill.engaged,
        "board_messages": board.messages,
        "market": conservation,
        "governance": gov.summary(),
        "trace_verified": trace.verify() == [],
        "trace_len": len(trace.records),
        "scale": scale_benchmark([10, 100, 1000, 10000, 100000, 1000000],
                                 branch=8),
    }
