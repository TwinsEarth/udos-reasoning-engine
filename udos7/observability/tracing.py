"""全链路追踪 + 结构化日志 + PII 脱敏 + 尾采样（零依赖，verified）。

Trace(请求) -> 嵌套 Span(阶段) 树；每个 Span 可挂属性、事件、Token 用量、状态。
可导出 JSONL 结构化日志（强制 trace_id/span_id/step/latency_ms），供 Loki/ES 采集。
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .semconv import (GEN_AI_COMPLETION, GEN_AI_PROMPT, GEN_AI_USAGE_INPUT_TOKENS,
                      GEN_AI_USAGE_OUTPUT_TOKENS)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------
# PII 脱敏（Redaction Processor 的本地等价物）
# --------------------------------------------------------------------------
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")
_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def redact(text: Any) -> Any:
    """对邮箱/手机号/身份证号做确定性脱敏；非字符串原样返回。"""
    if not isinstance(text, str):
        return text
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    text = _IDCARD.sub("[REDACTED_ID]", text)
    return text


class Span:
    __slots__ = ("name", "kind", "span_id", "trace_id", "parent_id",
                 "attrs", "events", "status", "start_ns", "end_ns", "children")

    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None,
                 kind: str = "internal", attrs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.kind = kind
        self.span_id = new_id()
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.attrs: Dict[str, Any] = dict(attrs or {})
        self.events: List[Dict[str, Any]] = []
        self.status = "unset"
        self.start_ns = time.perf_counter_ns()
        self.end_ns: Optional[int] = None
        self.children: List["Span"] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.attrs[GEN_AI_USAGE_INPUT_TOKENS] = (
            self.attrs.get(GEN_AI_USAGE_INPUT_TOKENS, 0) + int(input_tokens))
        self.attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = (
            self.attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS, 0) + int(output_tokens))

    def add_event(self, name: str, **fields: Any) -> None:
        ev = {"ts": round(time.time(), 6), "name": name}
        ev.update(fields)
        self.events.append(ev)

    def end(self, status: str = "ok") -> None:
        if self.end_ns is None:
            self.end_ns = time.perf_counter_ns()
        self.status = status

    @property
    def latency_ms(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return round((end - self.start_ns) / 1e6, 3)

    def to_dict(self, redact_pii: bool = True) -> Dict[str, Any]:
        def conv(v: Any) -> Any:
            if isinstance(v, str):
                return redact(v) if redact_pii else v
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            return v
        return {
            "name": self.name, "kind": self.kind,
            "trace_id": self.trace_id, "span_id": self.span_id,
            "parent_id": self.parent_id, "status": self.status,
            "latency_ms": self.latency_ms,
            "attributes": conv(self.attrs),
            "events": conv(self.events),
            "children": [c.to_dict(redact_pii) for c in self.children],
        }


class Tracer:
    """进程内追踪器。span() 支持嵌套（contextvars 维护父栈）。"""

    def __init__(self, redact_pii: bool = True, service: str = "udos-engine"):
        self.redact_pii = redact_pii
        self.service = service
        self.spans: List[Span] = []
        self._stack: contextvars.ContextVar[List[Span]] = contextvars.ContextVar(
            "udos_span_stack", default=[])

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs: Any):
        stack = list(self._stack.get())
        parent = stack[-1] if stack else None
        trace_id = parent.trace_id if parent else new_id()
        s = Span(name, trace_id, parent.span_id if parent else None, kind, attrs)
        if parent:
            parent.children.append(s)
        self.spans.append(s)
        token = self._stack.set(stack + [s])
        try:
            yield s
        except Exception:
            s.end("error")
            raise
        else:
            if s.status == "unset":
                s.end("ok")
        finally:
            self._stack.reset(token)

    def traces(self) -> Dict[str, List[Span]]:
        out: Dict[str, List[Span]] = {}
        for s in self.spans:
            out.setdefault(s.trace_id, []).append(s)
        return out

    def root_spans(self) -> List[Span]:
        return [s for s in self.spans if s.parent_id is None]

    def trace_tree(self, trace_id: str) -> Optional[Dict[str, Any]]:
        for s in self.root_spans():
            if s.trace_id == trace_id:
                return s.to_dict(self.redact_pii)
        return None

    def structured_logs(self) -> List[Dict[str, Any]]:
        """每个 Span 一条 JSON 结构化日志记录。"""
        records = []
        for s in self.spans:
            records.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "service": self.service,
                "trace_id": s.trace_id, "span_id": s.span_id,
                "parent_id": s.parent_id, "step": s.name,
                "status": s.status, "latency_ms": s.latency_ms,
                "token_usage": {
                    "input": s.attrs.get(GEN_AI_USAGE_INPUT_TOKENS, 0),
                    "output": s.attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS, 0),
                },
                "attributes": (redact(s.attrs) if self.redact_pii else s.attrs),
            })
        return records

    def export_jsonl(self, path: str, only_trace_ids: Optional[List[str]] = None) -> int:
        keep = set(only_trace_ids) if only_trace_ids else None
        n = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.structured_logs():
                if keep is not None and rec["trace_id"] not in keep:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n


def _stable01(trace_id: str) -> float:
    h = int(hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:12], 16)
    return (h % 1_000_000) / 1_000_000.0


def tail_sample_keep(spans: List[Span], base_rate: float = 0.10,
                     latency_ms_threshold: float = 1000.0) -> bool:
    """尾采样决策：错误 / 超延迟 / 命中 PII 的 trace 100% 保留；其余按 base_rate。

    用 trace_id 稳定哈希做概率采样，保证同一 trace 的所有 span 同留同弃。
    """
    if not spans:
        return False
    trace_id = spans[0].trace_id
    for s in spans:
        if s.status == "error":
            return True
        if s.latency_ms >= latency_ms_threshold:
            return True
        if s.attrs.get("pii_detected"):
            return True
    return _stable01(trace_id) < base_rate
