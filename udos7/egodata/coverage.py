"""状态覆盖（State Coverage）：把片段映射为离散状态单元集合。

Task Coverage 只问“见过哪几类任务”；State Coverage 问“在任务内部见过哪些
（位置×速度×接触×阶段）状态”。后者是 Experience Density 的度量基础。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Set, Tuple

from .episodes import EpisodeRecord, TASK_NAMES

X_BINS = 12
Y_BINS = 12
SPEED_BINS = 3
BOUND = 2.5

Cell = Tuple[str, int, int, int, int]          # task,x,y,speed,contact
TASK_INDEX = {t: i for i, t in enumerate(TASK_NAMES)}


def _bin(v: float, n: int, bound: float) -> int:
    b = int((v + bound) / (2 * bound) * n)
    return max(0, min(n - 1, b))


def state_cells(ep: EpisodeRecord) -> Set[Cell]:
    cells: Set[Cell] = set()
    contact = ep.contacts > 0
    for s in ep.states:
        x, y, _z, vx, vy, _vz = s
        speed = math.sqrt(vx * vx + vy * vy)
        cb = 0 if speed < 0.4 else (1 if speed < 1.2 else 2)
        # 接触信息是片段级的：碰撞后的状态归入接触单元
        cells.add((ep.task_name, _bin(x, X_BINS, BOUND),
                   _bin(y, Y_BINS, BOUND), cb, int(contact)))
    return cells


class CoverageMap:
    def __init__(self) -> None:
        self.cells: Set[Cell] = set()
        self.task_cells: Dict[str, Set[Cell]] = {t: set() for t in TASK_NAMES}

    def add(self, ep: EpisodeRecord) -> int:
        new = state_cells(ep) - self.cells
        self.cells |= new
        self.task_cells[ep.task_name] |= state_cells(ep)
        return len(new)

    def task_coverage(self) -> float:
        seen = sum(1 for t in TASK_NAMES if len(self.task_cells[t]) >= 5)
        return seen / len(TASK_NAMES)

    def state_count(self) -> int:
        return len(self.cells)

    def sparsest_tasks(self) -> List[str]:
        return sorted(TASK_NAMES, key=lambda t: len(self.task_cells[t]))

    def fraction_of(self, reference: "CoverageMap") -> float:
        if not reference.cells:
            return 0.0
        return len(self.cells & reference.cells) / len(reference.cells)
