"""分层混合拓扑与扇出基准（v7.4.6）。

一棵 b 叉聚合树，层级角色映射：
- 顶层（level 0）：Orchestrator，战略调度；
- 中间层：Handoff，按领域/部门接力聚合；
- 倒数两层：Swarm team + worker，大规模并行探索。

关键可测量命题：星型中心扇入 O(N)、串行轮次 O(units)；b 叉分层树
每个节点扇入恒为 b（与 N 无关），并行轮次 O(log_b N)。
小规模逐边显式模拟；超过 explicit_limit 用已验证的闭式计数外推
（grade = explicit / analytical，均为 cpu-proto 消息级模拟，非真实进程）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LayeredMatrix:
    branch: int = 8
    depth: int = 3                    # 根到 worker 的层数（worker 在 level depth）

    @property
    def n_workers(self) -> int:
        return self.branch ** self.depth

    @property
    def n_internal(self) -> int:
        # 等比级数 1 + b + ... + b^(d-1)
        return (self.n_workers - 1) // (self.branch - 1)

    @property
    def n_agents(self) -> int:
        return self.n_workers + self.n_internal

    def level_role(self, level: int) -> str:
        if level == 0:
            return "orchestrator"
        if level >= self.depth - 1:
            return "swarm"
        return "handoff"

    def route(self, n_units: int) -> Dict:
        """满载时每个内部节点收 b 条子级汇总 + 1 条父级派发。"""
        b, d = self.branch, self.depth
        active = min(self.n_workers, n_units)
        down_edges = self.n_agents - 1                # 每个非根节点收 1 条派发
        up_edges = self.n_internal - 1                # 内部节点（除根）逐级聚合
        down = down_edges + n_units
        up = active + up_edges                        # worker 汇报 + 逐级聚合
        return {
            "total_messages": down + up,
            "max_node_fanin": b + 1,                  # b 条汇报 + 1 条派发
            "top_fanin": b,
            "parallel_rounds": 2 * d,
            "active_workers": active,
        }


def star_routing(n_workers: int, n_units: int) -> Dict:
    return {
        "total_messages": 2 * n_units + 2 * n_workers,
        "center_fanin": 2 * n_units + 2 * n_workers,
        "parallel_rounds": n_units,                   # 中心串行处理
    }


@dataclass
class ExplicitSim:
    matrix: LayeredMatrix
    inbox: Dict[str, int] = field(default_factory=dict)
    messages: int = 0

    def _send(self, node):
        self.inbox[node] = self.inbox.get(node, 0) + 1
        self.messages += 1

    def run(self, n_units: int) -> Dict:
        b, d = self.matrix.branch, self.matrix.depth
        # 节点 id：(level, index)；父 (l,i) 的孩子为 (l+1, i*b + k)
        def parent(l, i):
            return (l - 1, i // b)

        active_workers = min(self.matrix.n_workers, n_units)
        # 下行：每条树边一次派发
        for l in range(1, d + 1):
            for i in range(b ** l):
                self._send(("n", l, i))
        # 任务到活跃 worker（每单元一条）
        for u in range(n_units):
            self._send(("w", d, u % self.matrix.n_workers))
        # 上行：活跃 worker 汇报
        for w in range(active_workers):
            self._send(("n", d - 1, w // b) if d > 1 else ("n", 0, 0))
        # 内部逐级聚合：每个非根内部节点向父节点发一条
        for l in range(d - 1, 0, -1):
            for i in range(b ** l):
                self._send(("n", l - 1, i // b))
        return {"total_messages": self.messages,
                "max_node_fanin": max(self.inbox.values()),
                "top_fanin": self.inbox.get(("n", 0, 0), 0)}


def fit_depth(target_agents: int, branch: int) -> int:
    return max(1, math.ceil(math.log(max(target_agents, branch), branch)))


def scale_benchmark(sizes: List[int], branch: int = 8,
                    units_per_worker: int = 4,
                    explicit_limit: int = 4000) -> List[Dict]:
    rows = []
    for n in sizes:
        d = fit_depth(n, branch)
        m = LayeredMatrix(branch, d)
        units = m.n_workers * units_per_worker
        grade = "explicit" if m.n_agents <= explicit_limit else "analytical"
        lay = m.route(units)
        star = star_routing(m.n_workers, units)
        row = {"target_agents": n, "actual_agents": m.n_agents,
               "depth": d, "units": units, "grade": grade,
               "star_center_fanin": star["center_fanin"],
               "layered_max_fanin": lay["max_node_fanin"],
               "layered_top_fanin": lay["top_fanin"],
               "star_rounds": star["parallel_rounds"],
               "layered_rounds": lay["parallel_rounds"]}
        if grade == "explicit":
            ex = ExplicitSim(m).run(units)
            row["explicit_total_messages"] = ex["total_messages"]
            row["formula_total_messages"] = lay["total_messages"]
            row["explicit_matches_formula"] = (
                ex["total_messages"] == lay["total_messages"]
                and ex["max_node_fanin"] == lay["max_node_fanin"]
                and ex["top_fanin"] == lay["top_fanin"])
        rows.append(row)
    return rows
