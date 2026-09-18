"""协同协议：讨论 → 提名 → 投票 → 碰撞裁决 → 选优 → 合并。

每一步都是纯函数式、可单测；协调器只负责异步调度，不把决策规则藏进流程里。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .memory import SharedMemory
from .types import (Claim, Collision, Decision, Task, TaskKind, Vote,
                    WorkProduct)
from .workers import CriticExpert, ValidatorExpert


def validate_products(task: Task, products: List[WorkProduct],
                      validator: ValidatorExpert) -> List[WorkProduct]:
    """有客观真值/测试跑分则客观打分；否则保留置信度。"""
    out = []
    for p in products:
        if p.output is None:
            continue
        out.append(validator.validate(p, task))
    return out


def discuss(task: Task, products: List[WorkProduct],
            critic: CriticExpert) -> Dict[str, Any]:
    return critic.review(products, task)


def nominate(products: List[WorkProduct],
             threshold: float = 0.0) -> List[WorkProduct]:
    ranked = sorted([p for p in products if p.score is not None],
                    key=lambda p: p.score, reverse=True)
    return [p for p in ranked if (p.score or 0.0) >= threshold]


def tally(products: List[WorkProduct],
          votes: Optional[List[Vote]] = None,
          weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """最终分 = 验证客观分（0.7）+ 加权投票归一化（0.3）。

    无投票时完全由验证分决定（可证伪、可复算）；投票只作有限修正，
    避免“声音大的人决定结果”。
    """
    weights = weights or {}
    scores = {p.id: float(p.score or 0.0) for p in products}
    result = dict(scores)
    if votes:
        by_target: Dict[str, List[Tuple[float, float]]] = {}
        for v in votes:
            by_target.setdefault(v.target_id, []).append((v.value, v.weight))
        for pid, lst in by_target.items():
            if pid not in result:
                continue
            wsum = sum(w for _, w in lst) or 1.0
            vmean = sum(val * w for val, w in lst) / wsum
            result[pid] = 0.7 * scores[pid] + 0.3 * float(vmean)
    return result


def detect_claim_collisions(products: List[WorkProduct]) -> List[Collision]:
    """同一任务里多个产物对同一资源写 → 碰撞（供选优裁决）。"""
    collisions: List[Collision] = []
    by_resource: Dict[str, List[WorkProduct]] = {}
    for p in products:
        for c in p.claims:
            if c.mode == "write":
                by_resource.setdefault(c.resource, []).append(p)
    for resource, ps in by_resource.items():
        if len({p.author for p in ps}) > 1:
            collisions.append(Collision(
                resource=resource,
                agents=sorted({p.author for p in ps}),
                kind="write_write", resolution="unresolved"))
    return collisions


def select_best(task: Task, products: List[WorkProduct], mem: SharedMemory,
                validator: ValidatorExpert, critic: CriticExpert,
                votes: Optional[List[Vote]] = None,
                nominate_threshold: float = 0.0
                ) -> Tuple[Optional[WorkProduct], Decision]:
    products = validate_products(task, products, validator)
    review = discuss(task, products, critic)
    nominated = nominate(products, nominate_threshold)
    collisions = detect_claim_collisions(products)
    tally_map = tally(nominated or products, votes)

    winner: Optional[WorkProduct] = None
    method = "best_score"
    if tally_map:
        wid = max(tally_map, key=lambda k: tally_map[k])
        winner = next((p for p in products if p.id == wid), None)

    if winner is None:
        method = "escalated"
    elif collisions:
        # 选优即裁决：最高分产物获得资源，其余写碰撞标记 rejected
        for c in collisions:
            c.resolution = "winner_took" if winner and winner.author in c.agents \
                else "serialized"

    note = ""
    if review.get("needs_human"):
        note = "低共识(consensus_spread=%.3g)，建议人工复核" % review[
            "consensus_spread"]

    decision = mem.record_decision(Decision(
        task_id=task.id,
        winner_id=winner.id if winner else None,
        method=method, tally=tally_map, collisions=collisions, note=note))

    return winner, decision


def merge_winning_patch(main_files: Dict[str, str],
                        winner: WorkProduct) -> Dict[str, str]:
    """代码补丁任务：仅把获胜分支合并进主线文件，败者分支丢弃（PR 语义）。"""
    if winner.kind != TaskKind.CODE_PATCH or not winner.output:
        return main_files
    merged = dict(main_files)
    target = winner.output["target_file"]
    merged[target] = winner.output["candidate"]
    return merged


def weighted_ensemble(products: List[WorkProduct],
                      weights: Dict[str, float]) -> Optional[np.ndarray]:
    """预测任务的加权集成（无真值时的生产路径）。权重按作者归一化。"""
    arrs, ws = [], []
    for p in products:
        if p.output is None:
            continue
        arrs.append(np.asarray(p.output, dtype=np.float32))
        ws.append(max(weights.get(p.author, 1.0), 1e-6))
    if not arrs:
        return None
    w = np.asarray(ws, dtype=np.float32)
    w = w / w.sum()
    return float(w[0]) * 0 + np.sum([w[i] * arrs[i] for i in range(len(arrs))],
                                    axis=0)
