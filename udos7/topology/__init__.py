"""UDOS v7.4–v7.5 多智能体拓扑与治理层。

三种基础拓扑（Orchestrator 星型 / Handoff 链式 / Swarm 网状）+ Agent-as-Tool
+ Transfer Bundle + Owner/Trace/Stop Condition 治理 + 拓扑决策树
+ 分层混合 b 叉聚合树 + BFT-lite 停止共识 + 多层熔断/回滚
+ 内部市场结算 + Stigmergy 环境媒介协作 + 百万级矩阵集成。

全部为确定性 CPU 模拟：Agent 是带能力声明与可注入故障模式的处理器，不是
LLM 进程；规模数字是消息/tick 级模拟，证据 cpu-proto。
"""
from __future__ import annotations

from .base import (WorkOrder, AgentSpec, AgentReport, TraceEvent, RunMetrics,
                   Orchestrator, Handoff, Swarm, build_default_fleet,
                   standard_workload, triage_handler, specialist_handler,
                   qa_handler, FAULTS, DOMAINS, STAGES, TOPOLOGIES)
from .agent_tool import (ToolInput, ToolEnvelope, AgentTool, MainAgent,
                         ERROR_TYPES)
from .transfer import (TransferBundle, BundleError, make_bundle, advance,
                       handoff, replay)
from .governance import (TraceLedger, StopCondition, GovernanceReport,
                         audit_run, hop_guarded_run)
from .decision import TaskProfile, TopologyDecision, choose_topology, \
    choose_layered
from .hierarchy import (LayeredMatrix, ExplicitSim, star_routing,
                        scale_benchmark, fit_depth)
from .consensus import StopConsensus, Ballot, sign, STOP, CONTINUE, NO_QUORUM
from .circuit_breaker import (AgentBreaker, GroupBreaker, KillSwitch,
                              ResilienceGrid, LedgerSnapshot, snapshot,
                              rollback, CLOSED, OPEN, HALF_OPEN)
from .market import Bid, TaskAuction, ContributionLedger
from .stigmergy import Blackboard, WorkItem, run_stigmergy, \
    contract_net_message_count
from .matrix import MatrixConfig, build_matrix, run_mission

__all__ = [
    # base
    "WorkOrder", "AgentSpec", "AgentReport", "TraceEvent", "RunMetrics",
    "Orchestrator", "Handoff", "Swarm", "build_default_fleet",
    "standard_workload", "triage_handler", "specialist_handler",
    "qa_handler", "FAULTS", "DOMAINS", "STAGES", "TOPOLOGIES",
    # agent-as-tool
    "ToolInput", "ToolEnvelope", "AgentTool", "MainAgent", "ERROR_TYPES",
    # transfer bundle
    "TransferBundle", "BundleError", "make_bundle", "advance", "handoff",
    "replay",
    # governance
    "TraceLedger", "StopCondition", "GovernanceReport", "audit_run",
    "hop_guarded_run",
    # decision
    "TaskProfile", "TopologyDecision", "choose_topology", "choose_layered",
    # hierarchy
    "LayeredMatrix", "ExplicitSim", "star_routing", "scale_benchmark",
    "fit_depth",
    # consensus
    "StopConsensus", "Ballot", "sign", "STOP", "CONTINUE", "NO_QUORUM",
    # circuit breaker
    "AgentBreaker", "GroupBreaker", "KillSwitch", "ResilienceGrid",
    "LedgerSnapshot", "snapshot", "rollback", "CLOSED", "OPEN", "HALF_OPEN",
    # market
    "Bid", "TaskAuction", "ContributionLedger",
    # stigmergy
    "Blackboard", "WorkItem", "run_stigmergy", "contract_net_message_count",
    # integration
    "MatrixConfig", "build_matrix", "run_mission",
]
