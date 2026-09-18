"""埋点与零代码自动 instrumentation。

- run_pipeline：对 RAG/Agent 标准流水线（意图→重写→检索→重排→压缩→生成→校验）
  建立嵌套 Span。
- AgentRegistry：登记 Agent 名字/用途/状态/进程。
- observe_goal：非侵入地包裹一次 CoordinatorAgent.run。
- enable_autoinstrument：环境变量 UDOS_OTEL_AUTO=1 时零代码给协调器 run 自动埋点。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from .semconv import (EXT_AGENT_NAME, EXT_AGENT_PURPOSE, EXT_AGENT_STATUS,
                      EXT_STAGE, GEN_AI_OPERATION_NAME, PIPELINE_STAGES)
from .tracing import Tracer

_AUTO_TRACER: Optional[Tracer] = None


def run_pipeline(tracer: Tracer, stages: Dict[str, Callable[[Any], Any]],
                 payload: Any, root_name: str = "udos.rag_pipeline"):
    """按 PIPELINE_STAGES 顺序执行存在的阶段，返回 (最终payload, root_span)。"""
    with tracer.span(root_name, kind="pipeline") as root:
        data = payload
        for stage in PIPELINE_STAGES:
            if stage not in stages:
                continue
            with tracer.span(stage, kind="stage", **{EXT_STAGE: stage}) as sp:
                try:
                    data = stages[stage](data)
                except Exception as exc:
                    sp.add_event("stage_error", error=repr(exc))
                    raise
            root.add_event(f"{stage}.done")
    return data, root


class AgentRegistry:
    """Agent 名字、用途、状态、进程的登记表（进程内）。"""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, purpose: str, kind: str = "worker",
                 pid: Optional[int] = None) -> Dict[str, Any]:
        rec = {"name": name, "purpose": purpose, "kind": kind,
               "status": "provisioned", "pid": pid or os.getpid()}
        self._agents[name] = rec
        return rec

    def update_status(self, name: str, status: str) -> None:
        if name not in self._agents:
            raise KeyError(f"未登记的 Agent: {name}")
        self._agents[name]["status"] = status

    def get(self, name: str) -> Dict[str, Any]:
        return self._agents[name]

    def list(self) -> List[Dict[str, Any]]:
        return list(self._agents.values())

    def by_status(self, status: str) -> List[Dict[str, Any]]:
        return [a for a in self._agents.values() if a["status"] == status]


async def observe_goal(tracer: Tracer, coord, goal: str,
                       payload: Optional[Dict[str, Any]] = None,
                       votes=None):
    """非侵入包裹一次协调器高阶目标执行，记录阶段、决策与任务失败数。"""
    with tracer.span("agent.coordinator.run", kind="agent",
                     **{EXT_AGENT_NAME: getattr(coord, "project", "coordinator"),
                        EXT_AGENT_PURPOSE: "classify->decompose->fanout->select",
                        GEN_AI_OPERATION_NAME: "agent.run"}) as sp:
        report = await coord.run(goal, payload, votes)
        mem = getattr(coord, "memory", None)
        if mem is not None:
            try:
                decisions = len(mem.decisions())
                failed = len(mem.events(kind="task_failed"))
                sp.set_attribute("udos.decisions", decisions)
                sp.set_attribute("udos.tasks_failed", failed)
            except Exception:
                pass
        sp.add_event("goal_complete", goal=goal)
        return report


def get_tracer() -> Tracer:
    global _AUTO_TRACER
    if _AUTO_TRACER is None:
        _AUTO_TRACER = Tracer()
    return _AUTO_TRACER


def enable_autoinstrument() -> bool:
    """UDOS_OTEL_AUTO=1 时零代码自动埋点 CoordinatorAgent.run。"""
    global _AUTO_TRACER
    if os.environ.get("UDOS_OTEL_AUTO", "0").lower() not in ("1", "true", "yes"):
        return False
    from ..agents.coordinator import CoordinatorAgent
    if getattr(CoordinatorAgent.run, "_udos_auto", False):
        return True
    _AUTO_TRACER = Tracer()
    original = CoordinatorAgent.run

    async def wrapped_run(self, goal, payload=None, votes=None):
        with _AUTO_TRACER.span(
                "agent.coordinator.run.auto", kind="agent",
                **{EXT_AGENT_NAME: getattr(self, "project", "coordinator"),
                   EXT_AGENT_STATUS: "running",
                   GEN_AI_OPERATION_NAME: "agent.run"}):
            return await original(self, goal, payload, votes)

    wrapped_run._udos_auto = True          # type: ignore[attr-defined]
    wrapped_run.__wrapped__ = original     # type: ignore[attr-defined]
    CoordinatorAgent.run = wrapped_run
    return True


def disable_autoinstrument() -> bool:
    from ..agents.coordinator import CoordinatorAgent
    run = CoordinatorAgent.run
    original = getattr(run, "__wrapped__", None)
    if original is not None:
        CoordinatorAgent.run = original
        return True
    return False
