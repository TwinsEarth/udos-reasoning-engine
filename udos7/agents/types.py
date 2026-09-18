"""多 Agent 协同层的数据契约（v7.1+）。

全部为可序列化的普通对象：任务、工作产物、投票、碰撞、决策。
设计原则：
- 只描述 *行为与可观察结果*，不依赖任何在线 LLM；LLM 能力是可选后端（见 automation 门禁）。
- 每个工作产物都带作者、证据、资源声明与父事件 id，形成可审计血缘。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TaskKind(str, Enum):
    PREDICT = "predict"              # 用世界模型做轨迹预测
    ACTION_COMPARE = "action_compare"  # 多候选动作推演排序
    VERIFY = "verify"                # 验证/跑分/打分
    CODE_PATCH = "code_patch"        # 受约束的代码/配置改进（候选集内）
    REVIEW = "review"                # 批评/碰撞审查
    DECOMPOSE = "decompose"          # 拆解（协调器内部）
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """协调器派发的最小工作单元。deps 为必须先完成的 Task.id。"""
    goal: str
    kind: TaskKind = TaskKind.CUSTOM
    payload: Dict[str, Any] = field(default_factory=dict)
    deps: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("task"))
    assignee: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    branch: Optional[str] = None       # 隔离工作区/分支标识
    retries: int = 0

    def is_ready(self, done: set) -> bool:
        return all(d in done for d in self.deps)


@dataclass
class Claim:
    """资源占用声明：读可共享；同一资源上的两个写锁构成碰撞。"""
    resource: str                # 例如 "file:udos7/model.py" 或 "param:gate"
    mode: str = "read"           # read / write
    owner: str = ""

    def conflicts(self, other: "Claim") -> bool:
        if self.resource != other.resource:
            return False
        return self.mode == "write" and other.mode == "write"


@dataclass
class WorkProduct:
    """Agent 的工作产物（一个候选答案/补丁/预测）。"""
    task_id: str
    author: str
    output: Any = None
    kind: TaskKind = TaskKind.CUSTOM
    score: Optional[float] = None        # 验证者给出的 0..1（越大越好）
    metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    evidence_grade: str = "cpu-proto"
    rationale: str = ""
    claims: List[Claim] = field(default_factory=list)
    branch: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("prod"))
    event_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Vote:
    voter: str
    target_id: str           # WorkProduct.id
    weight: float = 1.0
    value: float = 0.0       # 0..1 偏好/评分
    rationale: str = ""


@dataclass
class Collision:
    resource: str
    agents: List[str]
    kind: str = "write_write"
    resolution: str = "unresolved"   # unresolved / serialized / winner_took / rejected


@dataclass
class Decision:
    task_id: str
    winner_id: Optional[str]
    method: str                       # nominate_vote / best_score / escalated
    tally: Dict[str, float] = field(default_factory=dict)
    collisions: List[Collision] = field(default_factory=list)
    note: str = ""
    id: str = field(default_factory=lambda: new_id("dec"))
    created_at: float = field(default_factory=time.time)


@dataclass
class Event:
    """只追加事件日志中的一条，构成全局血缘。"""
    kind: str
    author: str
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("evt"))
    ts: float = field(default_factory=time.time)
