"""Transfer Bundle 五要素交接包 —— transfer_bundle.py (v4.4.0, 多智能体协作线)
======================================================================
对应蓝图图3/图7: Handoff = **转移责任不是转发消息**, 交接必须带五要素:
    * Goal    最终目标 + 成功标准
    * Context 用户约束 / 历史 / 关键事实
    * Done    已完成步骤与结果
    * Todo    下一步与风险
    * Trace   trace_id 与责任人

设计纪律 (与全工程一致):
    * 纯数据结构, 无可训练参数, 零梯度, 确定性; opt-in (不构造则旧路径逐位一致)。
    * 缺关键要素 (Goal / Trace) **拒绝交接**并显式 ValueError (非静默降级) —— 直接治理
      图2"状态丢失"失败。
    * 空 / 非法输入显式 ValueError; 序列化/反序列化对称可逆。
    * 日志沿用 logging_config (默认 WARNING, stderr), 不写 stdout / HTTP 体。
第二引擎一律称 GPM。analogy, not reproduction —— Bundle 为内存 dict 类比, 非线上消息协议。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("udos.transfer_bundle")

# 五要素中**缺一即拒绝交接**的关键要素 (Goal 定义"去哪", Trace 定义"谁负责")
REQUIRED_KEYS = ("goal", "trace")
# 全部五要素
ALL_KEYS = ("goal", "context", "done", "todo", "trace")


def _require_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Bundle 要素 {name} 须为非空字符串")
    return v


@dataclass
class TransferBundle:
    """Handoff 交接包五要素。

    Attributes:
        goal:    最终目标 + 成功标准 (关键要素, 不可缺)。
        trace:   trace_id + 责任人 (关键要素, 不可缺); dict 形如
                 {"trace_id": str, "owner": str}。
        context: 用户约束 / 历史 / 关键事实 (可空字符串表示无)。
        done:    已完成步骤与结果列表 (list[str]); 空列表表示未开始。
        todo:    下一步与风险列表 (list[str])。
    """

    goal: str
    trace: Dict[str, Any]
    context: str = ""
    done: List[str] = field(default_factory=list)
    todo: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_str(self.goal, "goal")
        if not isinstance(self.trace, dict):
            raise ValueError("trace 要素须为 dict {trace_id, owner}")
        if not isinstance(self.done, list) or not all(isinstance(x, str) for x in self.done):
            raise ValueError("done 须为 list[str]")
        if not isinstance(self.todo, list) or not all(isinstance(x, str) for x in self.todo):
            raise ValueError("todo 须为 list[str]")
        if not isinstance(self.context, str):
            raise ValueError("context 须为 str")

    # -- 完整性校验 -------------------------------------------------- #
    def validate(self) -> "TransferBundle":
        """交接前硬校验: 缺关键要素拒绝交接 (返回 self 便于链式调用)。

        Raises:
            ValueError: goal 空 / trace 缺 trace_id 或 owner。
        """
        _require_str(self.goal, "goal")
        tr = self.trace
        if not isinstance(tr, dict):
            raise ValueError("trace 要素须为 dict")
        if not isinstance(tr.get("trace_id"), str) or not tr["trace_id"].strip():
            raise ValueError("Bundle 缺关键要素 trace.trace_id —— 拒绝交接 (状态丢失风险)")
        if not isinstance(tr.get("owner"), str) or not tr["owner"].strip():
            raise ValueError("Bundle 缺关键要素 trace.owner —— 拒绝交接 (责任不清风险)")
        return self

    def is_complete(self) -> bool:
        """软判断: 五要素是否齐全 (goal/trace 非空, 其余键存在)。"""
        try:
            self.validate()
        except ValueError:
            return False
        return True

    # -- 序列化 ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "context": self.context,
            "done": list(self.done),
            "todo": list(self.todo),
            "trace": dict(self.trace),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransferBundle":
        if not isinstance(d, dict):
            raise ValueError("Bundle 须从 dict 重建")
        missing = [k for k in ("goal", "trace") if k not in d]
        if missing:
            raise ValueError(f"Bundle 重建缺关键要素: {missing}")
        return cls(
            goal=d["goal"],
            trace=d["trace"],
            context=d.get("context", ""),
            done=list(d.get("done", [])),
            todo=list(d.get("todo", [])),
        )

    # -- 交接时增量更新 --------------------------------------------- #
    def record_done(self, step: str) -> None:
        """记录一步已完成 (交接方把进展带给接手方, 防重复劳动)。"""
        if not isinstance(step, str) or not step.strip():
            raise ValueError("done 步骤须为非空字符串")
        self.done.append(step)

    def reassign_owner(self, new_owner: str, reason: str = "") -> None:
        """交接 = 责任转移: 更新 trace.owner 并记录转移原因到 trace 附属字段。"""
        _require_str(new_owner, "new_owner")
        prev = self.trace.get("owner")
        self.trace["owner"] = new_owner
        self.trace.setdefault("owner_history", [])
        if prev is not None:
            self.trace["owner_history"].append(
                {"from": prev, "to": new_owner, "reason": reason})
