"""Transfer Bundle（v7.4.3）：交接即责任转移。

完整交接包五要素：Goal / Context / Done / Todo / Trace(+Owner)。
不完整交接会导致状态丢失；validate_bundle 在交接前机械校验，
handoff() 只在包完整时转移 Owner，并返回类型化缺口。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_KEYS = ("goal", "context", "done", "todo", "trace", "owner")


class BundleError(ValueError):
    def __init__(self, missing: List[str], stale: List[str]):
        self.missing = missing
        self.stale = stale
        super().__init__(f"missing={missing} stale={stale}")


@dataclass
class TransferBundle:
    goal: str
    context: Dict[str, Any]
    done: List[str] = field(default_factory=list)
    todo: List[str] = field(default_factory=list)
    trace: List[int] = field(default_factory=list)
    owner: Optional[str] = None

    def validate(self, required_context: Tuple[str, ...] = ()) -> List[str]:
        problems = []
        if not self.goal:
            problems.append("goal")
        for k in required_context:
            if k not in self.context:
                problems.append(f"context.{k}")
        if not self.todo:
            problems.append("todo_empty")
        if not self.trace:
            problems.append("trace")
        if self.owner is None:
            problems.append("owner")
        # done 与 todo 不得重叠（责任边界清晰）
        if set(self.done) & set(self.todo):
            problems.append("done_todo_overlap")
        return problems

    def to_dict(self):
        return asdict(self)


def make_bundle(goal, context, stages, trace0):
    return TransferBundle(goal=goal, context=dict(context), done=[],
                          todo=list(stages), trace=[trace0], owner=None)


def advance(bundle: TransferBundle, stage: str, outputs: Dict[str, Any],
            event_seq: int, new_owner: str) -> TransferBundle:
    """阶段完成：登记 done、产出写入 context、责任暂交下一棒（调用方 handoff）。"""
    if stage not in bundle.todo:
        raise BundleError([], [stage])
    b = TransferBundle(**bundle.to_dict())
    b.todo.remove(stage)
    b.done.append(stage)
    b.context.update(outputs)
    b.trace.append(event_seq)
    b.owner = new_owner
    return b


def handoff(bundle: TransferBundle, next_owner: str,
            required_context: Tuple[str, ...] = ()
            ) -> Tuple[Optional[TransferBundle], List[str]]:
    """责任转移：校验完整性 → 换 Owner。失败返回 (None, problems)，不转移。"""
    problems = bundle.validate(required_context)
    if problems:
        return None, problems
    b = TransferBundle(**bundle.to_dict())
    b.owner = next_owner
    return b, []


def replay(bundle: TransferBundle) -> Dict[str, Any]:
    """新接手方仅凭 bundle 重建工作状态（验证上下文充分性）。"""
    return {"remaining": list(bundle.todo), "facts": dict(bundle.context),
            "completed": list(bundle.done), "owner": bundle.owner,
            "trace_len": len(bundle.trace)}
