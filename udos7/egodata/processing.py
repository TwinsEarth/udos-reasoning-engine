"""自动质检（QC）、经验密度（Experience Density）与 Yield 良率台账。

质检规则全部基于可观测信号盲检，不使用注入的缺陷真值标签：
- movement：手部轨迹总长过短 → 无效/静止片段；
- drift：相机系轨迹二阶差分（抖动代理）过大 → SLAM/镜头漂移；
- duplicate：量化签名重复 → 重复摆拍/复刻。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .episodes import EpisodeRecord
from .coverage import state_cells

MIN_PATH_M = 0.6
MAX_JERK = 0.115


def _path_length(track) -> float:
    total = 0.0
    for a, b in zip(track[:-1], track[1:]):
        total += math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    return total


def _jerk_score(track) -> float:
    """相机系手部轨迹的二阶差分中位数（漂移/抖动代理）。"""
    if len(track) < 4:
        return 0.0
    d2 = []
    for i in range(2, len(track)):
        d2.append(math.sqrt(sum(
            (track[i][j] - 2 * track[i - 1][j] + track[i - 2][j]) ** 2
            for j in range(3))))
    d2.sort()
    return d2[len(d2) // 2]


def signature(ep: EpisodeRecord) -> Tuple:
    start = [round(v, 1) for v in ep.states[0][:2]]
    end = [round(v, 1) for v in ep.states[-1][:2]]
    return (ep.params_key, len(ep.states) // 10, tuple(start), tuple(end))


def quality_control(ep: EpisodeRecord, seen_signatures: set) -> Dict:
    path = _path_length(ep.hand_track)
    jerk = _jerk_score(ep.hand_track)
    sig = signature(ep)
    reasons = []
    if path < MIN_PATH_M:
        reasons.append("static_or_empty")
    if jerk > MAX_JERK:
        reasons.append("camera_drift")
    if sig in seen_signatures:
        reasons.append("duplicate")
    return {"path_m": round(path, 3), "jerk": round(jerk, 4),
            "signature": sig, "reasons": reasons,
            "accepted": not reasons}


def experience_density(ep: EpisodeRecord) -> Dict:
    """每片段的经验密度：状态单元数、接触、子任务切换、单位时间手部路程。"""
    cells = len(state_cells(ep))
    transitions = len(ep.subtask_events)
    path = _path_length(ep.hand_track)
    minutes = max(ep.raw_minutes, 1e-9)
    return {
        "unique_state_cells": cells,
        "contact_events": ep.contacts,
        "subtask_transitions": transitions,
        "hand_path_m": round(path, 3),
        "cells_per_minute": round(cells / minutes, 2),
        "density_score": round(
            (cells + 8 * ep.contacts + 6 * transitions + 0.5 * path) / minutes, 2),
    }


@dataclass
class YieldLedger:
    raw_episodes: int = 0
    accepted_episodes: int = 0
    raw_minutes: float = 0.0
    accepted_minutes: float = 0.0

    @property
    def yield_rate(self) -> float:
        return round(self.accepted_minutes / self.raw_minutes, 4) \
            if self.raw_minutes else 0.0

    def as_dict(self) -> Dict:
        return {"raw_episodes": self.raw_episodes,
                "accepted_episodes": self.accepted_episodes,
                "raw_minutes": round(self.raw_minutes, 1),
                "accepted_minutes": round(self.accepted_minutes, 1),
                "yield_rate": self.yield_rate}


def process_batch(eps: List[EpisodeRecord]) -> Tuple[List[EpisodeRecord],
                                                     YieldLedger, Dict]:
    """对一批原始片段跑 QC + 密度，返回合格片段、良率台账、拒绝原因统计。"""
    ledger = YieldLedger()
    seen: set = set()
    accepted: List[EpisodeRecord] = []
    reject_counts: Dict[str, int] = {}
    for ep in eps:
        ledger.raw_episodes += 1
        ledger.raw_minutes += ep.raw_minutes
        ep.qc = quality_control(ep, seen)
        if ep.qc["accepted"]:
            seen.add(ep.qc["signature"])
            ep.accepted = True
            ep.density = experience_density(ep)
            accepted.append(ep)
            ledger.accepted_episodes += 1
            ledger.accepted_minutes += ep.raw_minutes
        else:
            for r in ep.qc["reasons"]:
                reject_counts[r] = reject_counts.get(r, 0) + 1
    return accepted, ledger, reject_counts
