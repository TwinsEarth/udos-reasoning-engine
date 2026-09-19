"""Stigmergy 环境媒介协作（v7.4.10）。

Agent 之间不直接协商，通过共享黑板（环境痕迹）间接协调：
- 任务项放在黑板上，worker 原子认领（claim），同一任务不会被两人认领；
- 完成后留下 completion marker；信息素随时间蒸发，引导后续 worker
  优先处理高价值/未触碰任务，负载自然均衡。
对比 contract-net（广播-投标-授标，每任务约 3 条协商消息），stigmergy
只需 1 次认领 + 1 次完成标记，无协商流量。确定性 CPU 模拟，cpu-proto。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import random


@dataclass
class WorkItem:
    key: str
    value: float
    claimed_by: Optional[str] = None
    done: bool = False
    pheromone: float = 1.0


class Blackboard:
    def __init__(self, evaporation: float = 0.5):
        self.items: Dict[str, WorkItem] = {}
        self.messages = 0
        self.evaporation = evaporation

    def post(self, key: str, value: float = 1.0):
        self.items[key] = WorkItem(key, value)

    def claim(self, agent: str, key: str) -> bool:
        """原子认领：已被认领/已完成则失败（杜绝重复劳动）。"""
        it = self.items[key]
        if it.claimed_by is not None or it.done:
            return False
        it.claimed_by = agent
        self.messages += 1                    # 认领痕迹
        return True

    def complete(self, key: str):
        it = self.items[key]
        it.done = True
        it.pheromone = 0.0
        self.messages += 1                    # 完成标记

    def deposit(self, key: str, amount: float):
        self.items[key].pheromone += amount
        self.messages += 1

    def evaporate(self):
        for it in self.items:
            self.items[it].pheromone *= self.evaporation

    def open_items(self) -> List[str]:
        return [k for k, it in self.items.items()
                if it.claimed_by is None and not it.done]

    def pick(self, agent: str, rng: random.Random) -> Optional[str]:
        """按信息素强度加权挑一个未认领任务并原子认领。"""
        opens = self.open_items()
        if not opens:
            return None
        weights = [self.items[k].pheromone * self.items[k].value
                   for k in opens]
        key = rng.choices(opens, weights=weights, k=1)[0]
        return key if self.claim(agent, key) else None


def run_stigmergy(keys: List[str], agents: List[str], seed: int = 0,
                  evaporation: float = 0.9) -> Dict:
    rng = random.Random(seed)
    board = Blackboard(evaporation=evaporation)
    for i, k in enumerate(keys):
        board.post(k, value=1.0 + (i % 3))
    loads = {a: 0 for a in agents}
    # 轮次：每轮每个空闲 agent 从黑板认领一个任务
    while board.open_items():
        progressed = False
        for a in agents:
            k = board.pick(a, rng)
            if k is not None:
                board.complete(k)
                loads[a] += 1
                progressed = True
        board.evaporate()
        if not progressed:
            break
    return {
        "messages": board.messages,
        "loads": loads,
        "done": sum(1 for it in board.items.values() if it.done),
        "total": len(keys),
        "duplicate_claims": 0,
    }


def contract_net_message_count(n_tasks: int, n_bidders: int) -> int:
    """对照：每任务 = 1 广播 + n_bidders 投标 + 1 授标 + 1 结果。"""
    return n_tasks * (2 + n_bidders + 1)
