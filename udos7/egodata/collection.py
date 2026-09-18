"""采集策略：被动（偏态、按序）vs Coverage-aware 主动（子模贪心补缺口）。

主动采集对候选池每个片段预算其状态单元，逐轮选择边际新单元最多的片段
（覆盖函数单调子模，贪心有 1-1/e 近似保证）；被动采集模拟“固定环境简单
动作”的偏态上量。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .episodes import EpisodeRecord, TASK_NAMES, generate_pool
from .coverage import CoverageMap, state_cells
from .processing import process_batch

# 被动采集的任务偏态：大量简单 reach + 少量其他（“满屏固定搬运/抓取”）
PASSIVE_SKEW = {"reach_free": 0.60, "contact_gate": 0.25,
                "ordered_sort": 0.10, "disturb_recover": 0.05}


def passive_select(pool: List[EpisodeRecord], budget: int
                   ) -> List[EpisodeRecord]:
    return pool[:budget]


def active_select(pool: List[EpisodeRecord], budget: int
                  ) -> List[EpisodeRecord]:
    chosen: List[EpisodeRecord] = []
    covered = set()
    used = [False] * len(pool)
    cell_sets = [state_cells(ep) for ep in pool]
    for _ in range(budget):
        best_i, best_gain = -1, -1
        for i, cells in enumerate(cell_sets):
            if used[i]:
                continue
            gain = len(cells - covered)
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0:
            break
        used[best_i] = True
        chosen.append(pool[best_i])
        covered |= cell_sets[best_i]
    return chosen


def flywheel_curve(eps: List[EpisodeRecord]) -> Dict:
    """按给定顺序计算每加入一个合格片段的累计新状态单元与边际增量。"""
    covered = set()
    cumulative, marginal = [], []
    for ep in eps:
        if not ep.accepted:
            cumulative.append(len(covered)); marginal.append(0); continue
        cells = state_cells(ep)
        marginal.append(len(cells - covered))
        covered |= cells
        cumulative.append(len(covered))
    return {"cumulative_cells": cumulative, "marginal_new_cells": marginal}


def build_pools(pool_size: int, budget: int, seed: int = 0, K: int = 16
                ) -> Tuple[List[EpisodeRecord], List[EpisodeRecord]]:
    """同一世界、同一缺陷率下生成两个候选池：被动偏态池与主动均匀池。"""
    passive_pool = generate_pool(pool_size, seed=seed, K=K,
                                 task_skew=PASSIVE_SKEW)
    active_pool = generate_pool(pool_size, seed=seed + 1, K=K, task_skew=None)
    return passive_pool, active_pool


def run_strategy(pool: List[EpisodeRecord], budget: int, active: bool
                 ) -> Dict:
    raw = active_select(pool, budget) if active else passive_select(pool, budget)
    accepted, ledger, rejects = process_batch(raw)
    curve = flywheel_curve(raw)
    densities = [e.density["density_score"] for e in accepted]
    return {
        "raw_selected": len(raw),
        "accepted": accepted,
        "ledger": ledger.as_dict(),
        "reject_reasons": rejects,
        "coverage_curve": curve,
        "final_state_cells": curve["cumulative_cells"][-1]
        if curve["cumulative_cells"] else 0,
        "mean_density": round(sum(densities) / len(densities), 2)
        if densities else 0.0,
        "redundancy_rate": round(
            1 - sum(curve["marginal_new_cells"]) /
            max(1, sum(len(state_cells(e)) for e in accepted)), 3),
    }
