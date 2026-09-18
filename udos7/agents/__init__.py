"""UDOS v7 多 Agent 协同层（v7.1 协同内核 / v7.2 军团与扩展律）。

公开接口：
- SharedMemory / CollisionError：共享记忆与碰撞
- CoordinatorAgent：不写核心代码的协调者（拆解→异步派发→验证/讨论/投票→选优合并）
- 专家 Agent：PredictionExpert / ActionCompareExpert / ValidatorExpert /
  CriticExpert / ClassifierExpert / DecomposerExpert / CodePatchExpert
- 协议：select_best / weighted_ensemble / hierarchical_select
- 军团：build_org / throughput_curve / quality_curve / fit_quality_saturation
- 自动化分级：AL、CapabilityGate、GateError、baseline_assessment
- 共享上下文传输：LocalTransport（可用）/ CloudTransport（未配置即门禁）
"""
from __future__ import annotations

from .automation import (AL, AL_NAMES, CapabilityAssessment, CapabilityGate,
                         GateError, baseline_assessment)
from .cloud import CloudTransport, LocalTransport, Transport
from .coordinator import CoordinatorAgent, GoalReport
from .legion import (OrgNode, build_org, fit_quality_saturation,
                     hierarchical_select, io_bound_curve, level_headcounts,
                     org_level_sizes, quality_curve, throughput_curve)
from .memory import CollisionError, SharedMemory
from .protocols import (detect_claim_collisions, discuss, merge_winning_patch,
                        nominate, select_best, tally, validate_products,
                        weighted_ensemble)
from .types import (Claim, Collision, Decision, Event, Task, TaskKind,
                    TaskStatus, Vote, WorkProduct)
from .workers import (ActionCompareExpert, Agent, AgentContext,
                      ClassifierExpert, CodePatchExpert, CriticExpert,
                      DecomposerExpert, PredictionExpert, ValidatorExpert)

__all__ = [
    # memory / transport
    "SharedMemory", "CollisionError", "Transport", "LocalTransport",
    "CloudTransport",
    # coordinator / agents
    "CoordinatorAgent", "GoalReport", "AgentContext", "Agent",
    "PredictionExpert", "ActionCompareExpert", "ValidatorExpert",
    "CriticExpert", "ClassifierExpert", "DecomposerExpert", "CodePatchExpert",
    # protocols
    "select_best", "weighted_ensemble", "validate_products", "discuss",
    "nominate", "tally", "merge_winning_patch", "detect_claim_collisions",
    "hierarchical_select",
    # legion
    "OrgNode", "build_org", "level_headcounts", "org_level_sizes",
    "throughput_curve", "io_bound_curve",
    "quality_curve", "fit_quality_saturation",
    # automation levels
    "AL", "AL_NAMES", "CapabilityGate", "GateError",
    "CapabilityAssessment", "baseline_assessment",
    # types
    "Task", "TaskKind", "TaskStatus", "WorkProduct", "Vote", "Claim",
    "Collision", "Decision", "Event",
]
