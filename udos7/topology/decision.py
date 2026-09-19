"""拓扑决策树（v7.4.5）：先看控制需求，再选拓扑。

流程是否明确 → 专家接力 → 能力可封装 → 开放探索；
默认从 Orchestrator 起步，自治不足时要求先明确目标，不默认放任 Swarm。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskProfile:
    flow_explicit: bool          # 流程是否明确
    expert_relay: bool = False   # 是否需要多领域专家接力
    encapsulable: bool = False   # 能力是否可封装为稳定工具
    open_exploration: bool = False
    risk: str = "medium"         # low / medium / high


@dataclass
class TopologyDecision:
    choice: str                  # orchestrator/handoff/agent_as_tool/swarm/escalate
    rationale: str
    autonomy_level: int          # 0 中心控制 → 3 完全自治


def choose_topology(p: TaskProfile) -> TopologyDecision:
    if p.flow_explicit:
        return TopologyDecision(
            "orchestrator",
            "流程明确、目标清晰：中心调度，状态统一、可审计、责任唯一", 0)
    if p.expert_relay:
        return TopologyDecision(
            "handoff",
            "需要多领域专家接力：链式交接，Transfer Bundle 完整传递上下文与责任", 1)
    if p.encapsulable:
        return TopologyDecision(
            "agent_as_tool",
            "能力可封装：稳定工具接口、可复用、低耦合，主 Agent 经信封调用", 1)
    if p.open_exploration:
        if p.risk == "high":
            return TopologyDecision(
                "escalate",
                "高风险开放探索不放任自治：先加 Orchestrator 护栏/熔断再放开 Swarm",
                0)
        return TopologyDecision(
            "swarm",
            "开放探索：去中心化协商，灵活演化；必须配套 Owner/Trace/Stop 治理", 2)
    return TopologyDecision(
        "escalate",
        "控制需求不足：先明确目标或降低自由度，不进入多 Agent 协作", 0)


def choose_layered(p: TaskProfile, scale: int) -> Optional[TopologyDecision]:
    """规模足够大且非纯流程任务时，建议分层混合（v7.4.6 落地）。"""
    if scale >= 1000 and (p.open_exploration or p.expert_relay):
        return TopologyDecision(
            "layered_hybrid",
            "顶层 Orchestrator 战略调度 / 中层 Handoff 专业接力 / 底层 Swarm 并行探索",
            2)
    return None
