"""Direct / Motion-only / Hybrid 三模式闭环对比基准（CPU 合成动力学）。

外部参照（匿名第三方仿真报告，GPT-6 Astra + π0.5）**不是本仓实测**，仅作口径
对照，证据等级 UNVERIFIED；本函数产出的所有数字为本机固定 seed 可复现，等级
CPU_PROTO。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..contracts import EvidenceGrade
from .env import standard_suite
from .hybrid import run_episode

MODES = ("direct", "motion", "hybrid")

# 外部报告口径（RoboDojo，50 次/任务，双臂仿真）；仅作对照，勿当成本仓结论。
EXTERNAL_REFERENCE = {
    "source": "匿名 GitHub 仿真测评（银河通用团队作者，媒体转述），GPT-6 Astra × π0.5",
    "grade": EvidenceGrade.UNVERIFIED.value,
    "note": "非本仓复现；任务、仿真器、初始条件与本原型不同，数字不可直接比较。",
    "roboDojo": {
        "direct": {"success_rate": 0.26, "mean_score": 37.81,
                   "tokens": "1.13e9（累计，代理原文口径）"},
        "hybrid": {"success_rate": 0.48, "mean_score": 62.60,
                   "intervention_share": 0.144,
                   "tokens": "6.25e8（累计，代理原文口径）"},
        "score_gap_vs_runner_up": 0.64,
    },
    "roboLab": {"direct_success": 0.98, "hybrid_success": 0.92,
                "note": "10 任务×5 槽位，非完整榜单复现，含授权重试"},
}


def run_suite(episodes_per_task: int = 10, K: int = 32, tasks=None,
              tracer=None) -> Dict:
    tasks = tasks or standard_suite()
    rows: List[Dict] = []
    for task in tasks:
        for mode in MODES:
            for ep in range(episodes_per_task):
                # 同任务各模式用同一组 seed，保证配对可比
                rows.append(run_episode(task, mode, seed=1000 + ep, K=K,
                                        tracer=tracer))

    def agg(mode: str) -> Dict:
        rs = [r for r in rows if r["mode"] == mode]
        n = len(rs)
        return {
            "episodes": n,
            "success_rate": round(sum(r["success"] for r in rs) / n, 3),
            "mean_score": round(sum(r["score"] for r in rs) / n, 2),
            "mean_collisions": round(sum(r["collisions"] for r in rs) / n, 2),
            "intervention_rate": round(
                sum(r["intervention_rate"] for r in rs) / n, 3),
            "tokens_in_proxy_total": sum(r["tokens_in_proxy"] for r in rs),
            "tokens_out_proxy_total": sum(r["tokens_out_proxy"] for r in rs),
        }

    by_task: Dict[str, Dict] = defaultdict(dict)
    for t in tasks:
        for mode in MODES:
            rs = [r for r in rows if r["task"] == t.name and r["mode"] == mode]
            by_task[t.name][mode] = {
                "success_rate": round(sum(r["success"] for r in rs) / len(rs), 3),
                "mean_score": round(sum(r["score"] for r in rs) / len(rs), 2),
                "intervention_rate": round(
                    sum(r["intervention_rate"] for r in rs) / len(rs), 3),
            }

    return {
        "module": "udos7.embodied hybrid control",
        "version": "v7.3.7",
        "evidence_grade": EvidenceGrade.CPU_PROTO.value,
        "config": {"episodes_per_task": episodes_per_task, "K": K,
                   "seg_len": 6, "override_len": 3},
        "aggregate": {m: agg(m) for m in MODES},
        "by_task": by_task,
        "external_reference": EXTERNAL_REFERENCE,
        "gates": ["LLM 语义裁判需多供应商 API key（token proxy→真实 usage）",
                  "真机/MuJoCo(-MJX) 接触动力学需 GPU/HPC"],
        "raw_rows": rows,
    }
