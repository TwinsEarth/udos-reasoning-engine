"""
UDOS v4.5.0.dev3 多专家在思考深度上的协作 (与 4.4 耦合)
================================================================
性质: 规则+聚合, 零梯度外挂, 不入主 state_dict, 不训练。

机制 (analogy, not reproduction):
    隐式阶段, 各"专家"以 Agent-as-Tool 对 K 条潜路径分别打分/给候选;
    Orchestrator 聚合各专家 -> 选路; 计算**专家间分歧**;
    专家分歧大则**升级** (隐式->显式, star->handoff/chain),
    形成"难度 x 拓扑 x effort"联合策略。

专家为 UDOS 内部合成打分器 (非 LLM agent):
    - convergence_expert: 偏好高 CTM 终态置信
    - stability_expert: 偏好路径分歧小 (稳定)
    - parsimony_expert: 偏好低内部 tick (省算力)
分歧大 -> 升级拓扑与 effort, 并把决策写入 TraceChain (dev5 统一)。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from .collab_governance import TraceChain
from .collab_topology import Topology

logger = logging.getLogger("udos.latent_collab")


# --------------------------------------------------------------------------- #
# 专家打分器: 输入单条路径特征 dict, 返回 [0,1] 分数
# --------------------------------------------------------------------------- #
def score_by_convergence(path: Dict[str, Any]) -> float:
    """专家1: 偏好高 CTM 收敛置信。"""
    return float(path.get("convergence", 0.0))


def score_by_stability(path: Dict[str, Any], global_disagreement: float) -> float:
    """专家2: 偏好路径分歧小 (越稳定越好)。"""
    return 1.0 - max(0.0, min(1.0, global_disagreement))


def score_by_parsimony(path: Dict[str, Any], max_ticks: int) -> float:
    """专家3: 偏好低内部 tick (省算力), 归一化。"""
    ticks = float(path.get("ticks_used", 0))
    if max_ticks <= 0:
        return 1.0
    return 1.0 - min(1.0, ticks / float(max_ticks))


EXPERTS: Dict[str, Callable[..., float]] = {
    "convergence": score_by_convergence,
    "stability": score_by_stability,
    "parsimony": score_by_parsimony,
}


class LatentCollaborator:
    """多专家在思考深度上协作: 专家打分 -> 聚合选路 -> 分歧升级。"""

    DISAGREE_THRESHOLD = 0.25   # 专家间分歧超过此值 -> 升级

    def __init__(self, allow_mesh: bool = False):
        self.allow_mesh = allow_mesh
        self.traces = TraceChain()

    def collaborate(self, task_id: str,
                    paths: List[Dict[str, Any]],
                    global_disagreement: float,
                    base_effort: str, base_topology: str) -> Dict[str, Any]:
        """对 K 条潜路径做多专家评分聚合, 并据分歧决定是否升级。

        paths: [{"path":i,"convergence":c,"ticks_used":t}, ...]
        返回: 选路结果 + 专家分歧 + 升级后的 joint 策略。
        """
        if not paths:
            raise ValueError("paths 不能为空")

        max_ticks = max((p.get("ticks_used", 0) for p in paths), default=1)
        # 各专家对每条路径打分
        votes: Dict[str, List[float]] = {}
        for name, fn in EXPERTS.items():
            scores = []
            for p in paths:
                if name == "stability":
                    s = fn(p, global_disagreement)
                elif name == "parsimony":
                    s = fn(p, max_ticks)
                else:
                    s = fn(p)
                scores.append(float(max(0.0, min(1.0, s))))
            votes[name] = scores

        # 聚合: 各路径专家均分
        n_exp = len(EXPERTS)
        path_mean = [
            sum(votes[e][i] for e in EXPERTS) / n_exp
            for i in range(len(paths))
        ]
        best = int(max(range(len(paths)), key=lambda i: path_mean[i]))

        # 专家间分歧: 各专家"首选路径"是否一致 (异质度)
        expert_picks = [int(max(range(len(paths)),
                                key=lambda i: votes[e][i])) for e in EXPERTS]
        pick_set = set(expert_picks)
        expert_disagreement = 1.0 - (1.0 / len(pick_set))  # 1=全部分歧

        # 专家分歧大 -> 升级
        escalated = expert_disagreement >= self.DISAGREE_THRESHOLD
        effort = base_effort
        topology = base_topology
        rationale: List[str] = []
        if escalated:
            # 隐式->显式
            if effort in ("none", "low"):
                effort = "high"
            elif effort == "high":
                effort = "max"
            # star -> chain(handoff); 若已 chain 且 allow_mesh -> mesh
            if topology == Topology.STAR.value:
                topology = Topology.CHAIN.value
            elif topology == Topology.CHAIN.value and self.allow_mesh:
                topology = Topology.MESH.value
            rationale.append(
                f"专家分歧={round(expert_disagreement,3)} >= "
                f"{self.DISAGREE_THRESHOLD} -> 升级 effort {base_effort}->{effort}, "
                f"拓扑 {base_topology}->{topology}")
        else:
            rationale.append(
                f"专家分歧={round(expert_disagreement,3)} < "
                f"{self.DISAGREE_THRESHOLD} -> 维持 {base_effort}/{base_topology}")

        # 写 trace (dev5 与 4.4 Trace 统一)
        trace_id = f"latent-trace-{task_id}"
        self.traces.append(trace_id, "orchestrator", "multi_expert_vote",
                           detail={"best_path": best,
                                   "expert_picks": expert_picks,
                                   "expert_disagreement":
                                       round(expert_disagreement, 4)})
        closer = self.traces.append(
            trace_id, "orchestrator", "close",
            detail={"effort": effort, "topology": topology})

        return {
            "task_id": task_id,
            "selected_path": best,
            "path_mean_scores": [round(x, 4) for x in path_mean],
            "expert_votes": {e: [round(x, 4) for x in v]
                             for e, v in votes.items()},
            "expert_picks": expert_picks,
            "expert_disagreement": round(expert_disagreement, 4),
            "escalated": escalated,
            "joint_strategy": {"difficulty_effort": effort,
                               "topology": topology},
            "rationale": rationale,
            "trace_integrity": self.traces.integrity_report(trace_id),
        }
