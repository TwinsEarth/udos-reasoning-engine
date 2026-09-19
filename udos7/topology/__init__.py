"""UDOS v7.4–v7.5 多智能体拓扑与治理层。

三种基础拓扑（Orchestrator 星型 / Handoff 链式 / Swarm 网状）+ Agent-as-Tool
+ Transfer Bundle + Owner/Trace/Stop Condition 治理 + 分层混合百万级矩阵。

全部为确定性 CPU 模拟：Agent 是带能力声明与可注入故障模式的处理器，不是
LLM 进程；规模数字是消息/ tick 级模拟，证据 cpu-proto。
"""
from __future__ import annotations

from .base import (WorkOrder, AgentSpec, AgentReport, TraceEvent, RunMetrics,
                   Orchestrator, Handoff, Swarm, build_default_fleet,
                   standard_workload, triage_handler, specialist_handler,
                   qa_handler, FAULTS)

__all__ = ["WorkOrder", "AgentSpec", "AgentReport", "TraceEvent", "RunMetrics",
           "Orchestrator", "Handoff", "Swarm", "build_default_fleet",
           "standard_workload", "triage_handler", "specialist_handler",
           "qa_handler", "FAULTS"]
