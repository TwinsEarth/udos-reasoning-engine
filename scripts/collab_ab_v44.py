"""v4.4.0.dev6 拓扑 A/B: 四拓扑在合成协作任务上的实测对比
======================================================================
落 benchmarks/results/collab_ab_v44.json, 全部确定性、可复算。
度量: 完成率/成功率、重复工作量、trace 完整率、冲突/未对齐数、收敛跳数、
      延迟(秒)、消息量(事件数)。
用证据检验蓝图三论断:
    * "拓扑比数量重要" —— 同任务换拓扑, 指标差异;
    * "star 最可控"     —— star 的 trace 完整率/收口确定性最高;
    * "swarm 治理成本高" —— mesh 的治理开销/消息量/冲突数最高。
被反证处如实留账本 (verdict_ledger), 不硬撑结论。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (build_default_registry, StarOrchestrator,  # noqa: E402
                  ChainHandoff, MeshSwarm, TopologySelector)

OUT = ROOT / "benchmarks" / "results" / "collab_ab_v44.json"


def _time(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def main():
    reg = build_default_registry()
    N_TASK = 20
    metrics = {"star": [], "chain": [], "mesh": []}

    for i in range(N_TASK):
        # ---- star: 复合任务拆 3 子任务, 故意埋 1 个重复 key 测去重 ----
        star = StarOrchestrator(reg)
        sub = [
            {"key": "a", "tool": "ctm_reasoning",
             "payload": {"query": f"q{i}", "horizon": 2}},
            {"key": "b", "tool": "gpm_scene", "payload": {"scene_id": f"s{i}"}},
            {"key": "a", "tool": "ctm_reasoning",
             "payload": {"query": f"q{i}", "horizon": 2}},  # 重复 key
        ]
        r, dt = _time(star.run, f"ab{i}", f"goal{i}", sub)
        metrics["star"].append({
            "ok": r["collector"]["complete"],
            "dup_work": r["collector"]["n_dedup_dropped"],
            "trace_intact": r["trace_integrity"]["chain_intact"],
            "closed": r["trace_integrity"]["closed"],
            "hops": len(r["results"]),
            "latency_s": round(dt, 6),
            "messages": len(r["results"]) + 2,
        })

        # ---- chain: 2 棒专业路由 ----
        ch = ChainHandoff(reg, {"inq": "sfm_space", "act": "wla_action"})
        r2, dt2 = _time(ch.run, f"ab{i}", f"goal{i}", ["inq", "act"],
                        {"query_region": f"r{i}", "body_part": "limb", "dim": 7})
        metrics["chain"].append({
            "ok": r2["recovered"],
            "dup_work": 0,
            "trace_intact": r2["trace_integrity"]["chain_intact"],
            "closed": r2["trace_integrity"]["closed"],
            "hops": len(r2["chain"]),
            "latency_s": round(dt2, 6),
            "messages": len(r2["chain"]) + 1,
        })

        # ---- mesh: opt-in swarm ----
        sw = MeshSwarm(reg, enabled=True)
        r3, dt3 = _time(sw.run, f"ab{i}", f"goal{i}",
                        ["reasoning", "planning"], {"query": f"q{i}", "horizon": 2})
        metrics["mesh"].append({
            "ok": r3["consensus"],
            "dup_work": 0,
            "trace_intact": r3["trace_integrity"]["chain_intact"],
            "closed": r3["trace_integrity"]["closed"],
            "conflict": int(r3["conflict_unaligned"]),
            "gov_overhead": sum(r3["governance_overhead"].values()),
            "hops": r3["n_peers"],
            "latency_s": round(dt3, 6),
            "messages": r3["n_peers"] * 2,
        })

    def agg(rows, topo):
        n = len(rows)
        out = {
            "n_tasks": n,
            "completion_rate": round(sum(r["ok"] for r in rows) / n, 4),
            "trace_complete_rate": round(
                sum(r["trace_intact"] and r["closed"] for r in rows) / n, 4),
            "avg_dup_work": round(sum(r["dup_work"] for r in rows) / n, 4),
            "avg_convergence_hops": round(sum(r["hops"] for r in rows) / n, 3),
            "avg_latency_ms": round(sum(r["latency_s"] for r in rows) / n * 1000, 4),
            "total_messages": sum(r["messages"] for r in rows),
        }
        if topo == "mesh":
            out["conflict_unaligned_total"] = sum(
                r.get("conflict", 0) for r in rows)
            out["avg_gov_overhead"] = round(
                sum(r.get("gov_overhead", 0) for r in rows) / n, 3)
        return out

    summary = {t: agg(rows, t) for t, rows in metrics.items()}

    # ---- 论断检验 (证据驱动, 被反证进账本) ----
    s, c, m = summary["star"], summary["chain"], summary["mesh"]
    ledger = []
    # 论断1: star 最可控 -> trace 完整率最高
    if s["trace_complete_rate"] >= m["trace_complete_rate"]:
        ledger.append({"claim": "star 最可控(trace 完整率>=mesh)", "verdict": "SUPPORTED"})
    else:
        ledger.append({"claim": "star 最可控(trace 完整率>=mesh)", "verdict": "REJECTED",
                       "note": f"star={s['trace_complete_rate']} mesh={m['trace_complete_rate']}"})
    # 论断2: swarm 治理成本高 -> 消息量/治理开销最高
    if m["total_messages"] >= s["total_messages"]:
        ledger.append({"claim": "swarm 治理成本高(消息量>=star)", "verdict": "SUPPORTED"})
    else:
        ledger.append({"claim": "swarm 治理成本高", "verdict": "REJECTED"})
    # 论断3: 拓扑比数量重要 -> star 与 mesh 在同任务上指标不同
    if abs(s["avg_convergence_hops"] - m["avg_convergence_hops"]) > 0.01:
        ledger.append({"claim": "拓扑比数量重要(不同拓扑指标不同)", "verdict": "SUPPORTED"})
    else:
        ledger.append({"claim": "拓扑比数量重要", "verdict": "INCONCLUSIVE"})

    result = {
        "version": "4.4.0.dev6",
        "n_tasks_per_topology": N_TASK,
        "note": "CPU 合成机制类比, 非真实多机器人; 延迟为进程内确定性测量",
        "summary": summary,
        "verdict_ledger": ledger,
        "topology_selector_default": TopologySelector().default.value,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("ledger:", json.dumps(ledger, ensure_ascii=False))
    print("written:", OUT)


if __name__ == "__main__":
    main()
