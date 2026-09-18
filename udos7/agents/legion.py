"""Agent 军团（v7.2）—— 组织层级、分层聚合与 Agent Scaling Law。

诚实边界（务必读）：
- 这里能 *真实运行* 的是 CPU 上的轻量确定性 Agent（接世界模型做预测/验证）。
  用它们实测两条规模曲线：① 处理独立任务的吞吐随并发 Agent 数的变化（含协调
  开销与线程饱和）；② 多专家集成质量随 Agent 数的变化（边际递减/饱和）。
- “几千几万几亿岗位/部门/国家军团”的 *组织结构* 可以用元数据廉价表示
  （build_org 只建树、不跑模型）；但让每个成员都成为在线 LLM 子 Agent 需要
  LLM key + 云预算（AL4 门禁，见 automation.py），本文件不假装已运行。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..dynamics import build_split
from .coordinator import CoordinatorAgent
from .workers import AgentContext, PredictionExpert, _to_tensor
from .types import Task, TaskKind

# 自底向上的层级命名：0=agent，1=team，2=department，3=domain，…
LEVEL_NAMES = ["agent", "team", "department", "domain", "federation",
               "planet", "cluster", "civilization"]


def level_name(depth: int) -> str:
    if depth < len(LEVEL_NAMES):
        return LEVEL_NAMES[depth]
    return f"higher{depth}"


@dataclass
class OrgNode:
    name: str
    level: str
    headcount: int = 0                 # 该节点下 Agent 总数（含虚拟计数）
    children: List["OrgNode"] = field(default_factory=list)
    level_sizes: Dict[str, int] = field(default_factory=dict)  # 精确编制（全军团）
    materialized: bool = True          # False=该子树仅以 headcount 计数表示

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)


def org_level_sizes(headcount: int, span: int = 8):
    """精确编制表：自底向上每层节点数（数学计算，O(层数)，可到亿级）。"""
    sizes = [int(headcount)]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + span - 1) // span)
    return sizes


def build_org(headcount: int, span: int = 8, name: str = "UDOS Legion",
              max_nodes: int = 2000) -> OrgNode:
    """按管理幅度 span 建组织树：编制精确、对象有界。

    - 每层 *精确人数/节点数* 由 org_level_sizes 公式给出（root.level_sizes），
      可到亿级且 O(层数)，瞬间完成；
    - 真正实例化的树对象受 max_nodes 约束，仅作结构预览/遍历；预算耗尽处的
      子树以 headcount 整数“虚拟”表示（materialized=False）。
    这与算力完全解耦：组织图描述岗位/部门结构，实际运行的 Agent 由协调器按
    任务与并发创建，亿级在线 LLM 子 Agent 需 LLM key+云预算（AL4 门禁）。
    """
    import math
    headcount = int(headcount)
    sizes = org_level_sizes(headcount, span)
    L = len(sizes) - 1                      # 顶层 depth
    level_sizes = {level_name(d): sizes[d] for d in range(len(sizes))}
    counter = {"n": 1}

    def make(depth: int, units: int, budget: int, path: str) -> OrgNode:
        node = OrgNode(name=f"{level_name(depth)}-{path}", level=level_name(depth),
                       headcount=units, level_sizes=level_sizes)
        if depth == 0:
            # 单个 agent 节点（units 必为 1）
            return node
        if budget <= 0:
            node.materialized = False
            return node
        if depth == 1:
            # team 的直接下级就是一个个 agent；预算够才展开，否则整体虚拟
            if budget >= units:
                node.children = [OrgNode(f"agent-{path}-{j}", "agent", 1)
                                 for j in range(units)]
                counter["n"] += units
            else:
                node.materialized = False
            return node
        child_cap = span ** depth          # 每个下层节点最多管辖的 agent 数
        n_child = min(span, max(1, math.ceil(units / child_cap)))
        base, rem = divmod(units, n_child)
        children = []
        for i in range(n_child):
            cu = base + (1 if i < rem else 0)
            if cu <= 0:
                continue
            if counter["n"] + 1 > max_nodes:
                # 预算不足：剩余子树整体虚拟
                v = OrgNode(f"{level_name(depth-1)}-{path}-{i}",
                            level_name(depth - 1), headcount=cu,
                            level_sizes=level_sizes, materialized=False)
                children.append(v)
                counter["n"] += 1
                continue
            counter["n"] += 1
            children.append(make(depth - 1, cu,
                                 max_nodes - counter["n"], f"{path}-{i}"))
        node.children = children
        return node

    root = make(L, headcount, max_nodes, "0")
    root.name = name
    return root


def level_headcounts(root: OrgNode) -> Dict[str, int]:
    """返回全军团 *精确* 编制（每层节点数），与物化与否无关。"""
    if root.level_sizes:
        return dict(root.level_sizes)
    counts: Dict[str, int] = {}

    def walk(n: OrgNode):
        counts[n.level] = counts.get(n.level, 0) + 1
        for c in n.children:
            walk(c)
    walk(root)
    return counts


def hierarchical_select(teams: Dict[str, List[Dict[str, Any]]]
                        ) -> Dict[str, Any]:
    """分层投票：team 内选最高分 → department 从各 team 胜者中再选。

    入参 teams: {team_name: [candidate dict with score ...]}；
    演示两级，可按 OrgNode 层级递归扩展。
    """
    team_winners = {}
    for tname, cands in teams.items():
        if cands:
            team_winners[tname] = max(cands, key=lambda c: c.get("score", 0))
    if not team_winners:
        return {"winner": None}
    dept_winner = max(team_winners.values(), key=lambda c: c.get("score", 0))
    return {"winner": dept_winner,
            "team_winners": {k: v.get("id") for k, v in team_winners.items()}}


# ---------------------------------------------------------------------------
# Scaling Law —— 实测，而非宣称
# ---------------------------------------------------------------------------
async def _run_m_independent_tasks(coord: CoordinatorAgent, windows,
                                   truths, n_workers: int):
    coord.sem = asyncio.Semaphore(n_workers)
    tasks = []
    for i in range(len(windows)):
        tasks.append(Task(goal="预测", kind=TaskKind.PREDICT,
                          payload={"window": windows[i], "truth": truths[i],
                                   "expert_modes": ["blind"]},
                          branch=f"job-{i}"))
    t0 = time.perf_counter()
    products = await coord._execute_dag(tasks)
    dt = time.perf_counter() - t0
    return dt, len(products)


def _batch_data(n_tasks: int, seed: int = 2026):
    ds = build_split(seed, n_traj_per_kind=max(2, n_tasks // 4 + 1))
    idx = np.linspace(0, ds.X.size(0) - 1, n_tasks).astype(int)
    windows = ds.X[idx].tolist()
    truths = ds.Y[idx].tolist()
    return windows, truths


def throughput_curve(model, n_tasks: int = 64,
                     workers_list=(1, 2, 4, 8, 16)) -> List[Dict[str, Any]]:
    """吞吐扩展：固定 M 个独立任务，改变并发 Agent 数，实测墙钟与 tasks/s。"""
    windows, truths = _batch_data(n_tasks)
    ctx = AgentContext(model=model)
    records = []

    async def main():
        for n in workers_list:
            # 让协调器自持执行器，aclose(wait=True) 确定性回收线程
            coord = CoordinatorAgent(project=f"scale-{n}", model=model,
                                     concurrency=n)
            dt, done = await _run_m_independent_tasks(
                coord, windows, truths, n)
            await coord.aclose()
            records.append({"agents": n, "tasks": done,
                            "wall_s": round(dt, 4),
                            "tasks_per_s": round(done / dt, 3)})
    asyncio.run(main())
    base = records[0]["wall_s"]
    for r in records:
        r["speedup_vs_1"] = round(base / r["wall_s"], 3)
    return records


def _expert_pred(model, mode: str, windows, H):
    expert = PredictionExpert(f"pred-{mode}", AgentContext(model=model), mode)
    task = Task(goal="预测", kind=TaskKind.PREDICT,
                payload={"window": windows, "horizon": H}, branch=f"pred-{mode}")

    class _Mem:
        def append(self, e):
            return e

    return np.asarray(expert.run_sync(task, _Mem()).output, dtype=np.float32)


def quality_curve(model, n_samples: int = 64,
                  k_list=(1, 2, 3, 5, 9, 17),
                  test_seed: int = 2026, calib_seed: int = 314,
                  tau: float = 0.05) -> List[Dict[str, Any]]:
    """质量扩展（复刻协调器真实协议：校准集加权，而非朴素等权平均）。

    实测会暴露一个关键事实：**等权平均差专家会拖差结果**，所以生产路径按各专家
    在 *独立校准集* 上的 MSE 做 softmax 加权（不偷看 test）。
    - 前 D 个为真正独立信息源（blind 学习模型 / analytic 解析基线），verified；
    - k>D 后追加 *最强专家的近相关副本*（模拟同质化扩招），权重随之摊薄，
      质量饱和而非无限提升——这才是诚实的 Agent Scaling Law：增益来自多样性与
      验证加权，而非单纯堆人头。标 cpu-proto，不冒充 k 个独立 LLM。
    同时给出 best_single / naive_equal 对照，不隐藏等权平均可能变差。
    """
    windows, truths = _batch_data(n_samples, test_seed)
    cw, ct = _batch_data(max(16, n_samples // 2), calib_seed)
    truth = np.asarray(truths, dtype=np.float32)
    calib_truth = np.asarray(ct, dtype=np.float32)
    H = truth.shape[1]

    modes = ("blind", "analytic")
    test_pred = [_expert_pred(model, m, windows, H) for m in modes]
    calib_pred = [_expert_pred(model, m, cw, H) for m in modes]
    calib_mse = np.array([np.mean((p - calib_truth) ** 2) for p in calib_pred])
    w = np.exp(-calib_mse / tau)
    w = w / w.sum()
    order = np.argsort(-w)                       # 最强专家在前
    best_i = int(order[0])

    def weighted(preds, weights):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        return float(np.mean((np.sum([weights[i] * preds[i]
                                      for i in range(len(preds))], axis=0)
                              - truth) ** 2))

    best_single = float(np.mean((test_pred[best_i] - truth) ** 2))
    naive_two = float(np.mean((np.mean(test_pred, axis=0) - truth) ** 2))
    D = len(modes)

    rng = np.random.default_rng(0)
    sigma = float(np.std(test_pred[best_i]) * 0.01)
    records = []
    for k in k_list:
        preds, weights = [], []
        for i in range(k):
            if i < D:
                preds.append(test_pred[i]); weights.append(w[i])
            else:
                # 同质化扩招：复制最强专家 + 零均值近相关噪声
                preds.append(test_pred[best_i]
                             + rng.normal(0, sigma, test_pred[best_i].shape)
                             .astype(np.float32))
                weights.append(w[best_i])
        mse = weighted(preds, weights)
        records.append({"agents": k,
                        "weighted_ensemble_mse": round(mse, 6),
                        "best_single_mse": round(best_single, 6),
                        "naive_equal_2_mse": round(naive_two, 6),
                        "independent": min(k, D),
                        "evidence": "verified" if k <= D else "cpu-proto"})
    return records


def io_bound_curve(n_tasks: int = 64, latency_s: float = 0.05,
                   concurrency_list=(1, 2, 4, 8, 16, 32)
                   ) -> List[Dict[str, Any]]:
    """I/O 边界扩展（远程 LLM/工具型 Agent 的真实受益区间）。

    用 asyncio.sleep 模拟每个 Agent 等待远程响应（不占 CPU）。这是云端数千
    子 Agent 的工作形态：fan-out 受并发额度而非本地核数约束，近线性直到并发
    上限。标 simulation（非真实 LLM 时延），但调度机制与协调器完全一致。
    """
    records = []

    async def job(sem):
        async with sem:
            await asyncio.sleep(latency_s)

    async def main():
        for n in concurrency_list:
            sem = asyncio.Semaphore(n)
            t0 = time.perf_counter()
            await asyncio.gather(*(job(sem) for _ in range(n_tasks)))
            dt = time.perf_counter() - t0
            records.append({"agents": n, "tasks": n_tasks,
                            "wall_s": round(dt, 4),
                            "tasks_per_s": round(n_tasks / dt, 2),
                            "ideal_wall_s": round(n_tasks * latency_s / n, 4)})
    asyncio.run(main())
    base = records[0]["wall_s"]
    for r in records:
        r["speedup_vs_1"] = round(base / r["wall_s"], 3)
    return records


def fit_quality_saturation(records: List[Dict[str, Any]]):
    """对加权集成点拟合 MSE(k)=a+b/k（最小二乘），给出饱和值 a 与增益 b。"""
    pts = [(r["agents"], r["weighted_ensemble_mse"]) for r in records
           if r.get("weighted_ensemble_mse") is not None]
    if len(pts) < 2:
        return None
    ks = np.array([k for k, _ in pts], dtype=np.float64)
    ys = np.array([y for _, y in pts], dtype=np.float64)
    X = np.stack([np.ones_like(ks), 1.0 / ks], axis=1)
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    return {"model": "mse(k)=a+b/k", "saturation_a": float(coef[0]),
            "gain_b": float(coef[1])}
