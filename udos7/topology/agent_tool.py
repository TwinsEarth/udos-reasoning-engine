"""Agent-as-Tool（v7.4.2）：把专家 Agent 包成主 Agent 可调用的工具。

契约：输入 ToolInput（code/language/context/options 的同构泛化版本），
输出固定信封 {report, confidence, error_type}。主 Agent 不接触专家内部
推理轨迹，上下文负担有上界；专家内部失败被转成类型化错误而非异常逃逸。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

ERROR_TYPES = (None, "bad_input", "capability_gap", "internal_error",
               "timeout", "low_confidence")


@dataclass
class ToolInput:
    payload: Dict[str, Any]
    language: str = "generic"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEnvelope:
    tool: str
    report: Optional[Any] = None
    confidence: float = 0.0
    error_type: Optional[str] = None
    trace_ref: Optional[str] = None       # 内部 trace 只回传引用，不回传全文

    def as_dict(self):
        return {"tool": self.tool, "report": self.report,
                "confidence": self.confidence, "error_type": self.error_type,
                "trace_ref": self.trace_ref}


ExpertFn = Callable[[ToolInput], tuple]   # -> (report, confidence, trace_ref)


class AgentTool:
    """专家 Agent 的稳定接口；内部实现可独立迭代，schema 不变即兼容。"""

    def __init__(self, name: str, required_fields: tuple, expert: ExpertFn,
                 confidence_floor: float = 0.0, max_report_bytes: int = 4096):
        self.name = name
        self.required_fields = required_fields
        self._expert = expert
        self.confidence_floor = confidence_floor
        self.max_report_bytes = max_report_bytes
        self.calls = 0

    def invoke(self, tin: ToolInput) -> ToolEnvelope:
        self.calls += 1
        missing = [k for k in self.required_fields if k not in tin.payload]
        if missing:
            return ToolEnvelope(self.name, error_type="bad_input",
                                report={"missing": missing})
        try:
            report, conf, ref = self._expert(tin)
        except TimeoutError:
            return ToolEnvelope(self.name, error_type="timeout")
        except Exception as e:                          # 内部异常不外泄
            return ToolEnvelope(self.name, error_type="internal_error",
                                report={"exc": type(e).__name__})
        if conf < self.confidence_floor:
            return ToolEnvelope(self.name, report=report, confidence=conf,
                                error_type="low_confidence", trace_ref=ref)
        return ToolEnvelope(self.name, report=report, confidence=conf,
                            trace_ref=ref)


class MainAgent:
    """主 Agent：只通过工具信封获得结果，上下文增量有固定上界。"""

    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}
        self.context_log: list = []

    def register(self, tool: AgentTool):
        self.tools[tool.name] = tool

    def call(self, tool_name: str, tin: ToolInput) -> ToolEnvelope:
        if tool_name not in self.tools:
            env = ToolEnvelope(tool_name, error_type="capability_gap")
        else:
            env = self.tools[tool_name].invoke(tin)
        # 只写入信封（固定大小），不写入专家内部 trace
        self.context_log.append(env.as_dict())
        return env

    def context_size(self) -> int:
        return sum(len(repr(x)) for x in self.context_log)
