"""四拓扑统一接口 + 拓扑选择器决策树 —— collab_topology.py (v4.4.0)
======================================================================
对应蓝图图1 选拓扑决策树 (先看控制需求再选拓扑, 没有银弹只有匹配):
    ①流程清晰可规划 -> star   (中心调度/统一状态/易审计)
    ②需不同专业分工 -> chain   (按专业交接, 上下文完整传递)
    ③能力可封装稳定接口 -> tool (稳定接口/可复用/降耦)
    ④需自主协作动态探索 -> mesh (去中心化/动态组队/治理成本高)
    四条都不满足 = 控制需求不足 -> **拒绝自治** (默认退回 star 或抛错)。
**默认从 star 起步**, 再按需放开自治。选择器输出**可解释**的选型理由。

设计纪律: 纯决策函数、确定性、零梯度、opt-in; 空/非法显式 ValueError。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("udos.collab_topology")


class Topology(str, Enum):
    """四种协作拓扑 (统一接口标识)。"""

    STAR = "star"      # Orchestrator-Worker
    CHAIN = "chain"    # Handoff Triage->Specialist->Return
    TOOL = "tool"      # Agent-as-Tool
    MESH = "mesh"      # Swarm (默认关 opt-in)


# 对外统一执行接口 (四拓扑各自实现 run(task, bundle, registry, gov) -> dict)。
# 本模块只定义契约与选择器; 具体执行在 collab_orchestrator / collab_handoff /
# collab_swarm / collab_agents 中实现。

@dataclass
class Selection:
    """拓扑选择结果 (含可解释理由)。"""

    topology: Topology
    rationale: str
    accepted: bool           # False = 控制需求不足, 拒绝自治
    reasons: Dict[str, Any]  # 四问各命中情况, 供审计

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology.value,
            "accepted": self.accepted,
            "rationale": self.rationale,
            "questions": self.reasons,
        }


class TopologySelector:
    """图1 决策树: 用四个二元问题选拓扑。

    四问 (对应图1):
        q_plan   流程是否清晰、可事前规划?            -> True => star
        q_expert 是否需要不同专业分工/按专业移交?       -> True => chain
        q_tool   候选能力是否可封装为稳定 I/O 接口?     -> True => tool
        q_autonomy 是否需要自主协作/动态探索?           -> True => mesh
    优先级: star > chain > tool > mesh (先受控后自治); 全 False => 控制需求不足,
    拒绝自治 (accepted=False, 不擅自选 mesh)。
    """

    def __init__(self, default: Topology = Topology.STAR,
                 allow_mesh: bool = False) -> None:
        # allow_mesh=False => 即使 q_autonomy=True 也不放开 mesh (swarm 默认关 opt-in)
        self.default = Topology(default)
        self.allow_mesh = bool(allow_mesh)

    def select(self, q_plan: bool = False, q_expert: bool = False,
               q_tool: bool = False, q_autonomy: bool = False,
               task_hint: str = "") -> Selection:
        for name, v in (("q_plan", q_plan), ("q_expert", q_expert),
                        ("q_tool", q_tool), ("q_autonomy", q_autonomy)):
            if not isinstance(v, bool):
                raise ValueError(f"{name} 须为 bool")
        reasons = {"q_plan": q_plan, "q_expert": q_expert,
                   "q_tool": q_tool, "q_autonomy": q_autonomy}

        if q_plan:
            return Selection(
                Topology.STAR,
                "流程清晰可规划 -> 中心调度(star): 统一状态、流程可控、易审计",
                True, reasons)
        if q_expert:
            return Selection(
                Topology.CHAIN,
                "需不同专业分工 -> 链式 handoff: 按专业路由、Transfer Bundle 完整交接",
                True, reasons)
        if q_tool:
            return Selection(
                Topology.TOOL,
                "能力可封装稳定接口 -> Agent-as-Tool: 主调度只看 I/O schema, 降耦复用",
                True, reasons)
        if q_autonomy:
            if not self.allow_mesh:
                # 图1: 控制需求不足时拒绝自治 —— 不擅自放 swarm, 退回默认 star 并显式告警
                return Selection(
                    self.default,
                    "需要动态自治但 mesh 未 opt-in(默认关) -> 拒绝自治, 退回默认 star",
                    True, reasons)
            return Selection(
                Topology.MESH,
                "需自主协作动态探索 -> 网状 swarm(已 opt-in): 灵活但治理成本高",
                True, reasons)
        # 四问全不满足 = 控制需求不足: 不擅自选任何自治拓扑
        return Selection(
            self.default,
            "四问均不满足 = 控制需求不足: 放任自治=不可控风险, 拒绝自治, 退回默认 star",
            False, reasons)
