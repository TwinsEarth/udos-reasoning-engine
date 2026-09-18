"""v7.3.3 可观测性加深层：会话关联 / 护栏 / 成本归因 / 智能异常与根因 /
拓扑与火焰图 / OTLP-JSON 导出。零第三方依赖，CPU 可测（verified）。

真实 gRPC 推送到 OTel Collector、eBPF、GPU 剖析、LLM-as-judge 仍属部署/资源闸门，
本模块只产出可被这些后端摄取的确定性数据结构，不冒充已对接。
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .evaluation import FormatJudge, GroundingJudge, ToolCorrectnessJudge, ToxicityJudge
from .semconv import (EXT_AGENT_NAME, GEN_AI_COMPLETION, GEN_AI_OPERATION_NAME,
                      GEN_AI_PROMPT, GEN_AI_REQUEST_MODEL,
                      GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS)
from .tracing import Span, Tracer, redact


# ==========================================================================
# 1. Session Correlation：Session -> Trace -> Span 会话关联
# ==========================================================================
def session_index(tracer: Tracer, default_session: str = "uncorrelated") -> Dict[str, Dict[str, Any]]:
    """以根 Span 的 session_id 属性把多条 trace 聚成会话（生产里等价 Session Correlation）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for root in tracer.root_spans():
        sid = str(root.attrs.get("session_id", default_session))
        rec = out.setdefault(sid, {"session_id": sid, "traces": [], "span_count": 0,
                                   "error_traces": 0, "total_latency_ms": 0.0})
        rec["traces"].append(root.trace_id)
        all_spans = tracer.traces().get(root.trace_id, [])
        rec["span_count"] += len(all_spans)
        rec["total_latency_ms"] += round(rec["total_latency_ms"] + root.latency_ms, 3)
        if any(s.status == "error" for s in all_spans):
            rec["error_traces"] += 1
    return out


# ==========================================================================
# 2. Guardrails：输出到达用户前的实时护栏，拦截事件本身可观测
# ==========================================================================
@dataclass
class GuardrailResult:
    allowed: bool
    violations: List[Dict[str, str]] = field(default_factory=list)
    checks: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "violations": self.violations,
                "checks": self.checks}


class Guardrails:
    """确定性护栏：PII 泄露 / 毒性 / 格式违规 / 未接地(幻觉代理) / 工具错误。

    LLM 毒性/幻觉商业模型属 AL4 闸门；这里的关键词与 grounding 为 cpu-proto 口径。
    """

    def __init__(self, require_json: bool = False, block_pii: bool = True,
                 block_toxic: bool = True, grounding_min: float = 0.5,
                 registry: Any = None):
        self.require_json = require_json
        self.block_pii = block_pii
        self.block_toxic = block_toxic
        self.grounding_min = grounding_min
        self.registry = registry
        self._fmt = FormatJudge()
        self._ground = GroundingJudge()
        self._toxic = ToxicityJudge()
        self._tool = ToolCorrectnessJudge()

    def _count(self, kind: str) -> None:
        if self.registry is not None:
            self.registry.counter("gen_ai_guardrail_blocks_total",
                                  "Guardrail blocks by type").inc(1, {"type": kind})

    def inspect(self, record: Dict[str, Any], span: Optional[Span] = None) -> GuardrailResult:
        violations: List[Dict[str, str]] = []
        checks: Dict[str, float] = {}
        out = str(record.get("output", ""))

        if self.block_pii and redact(out) != out:
            violations.append({"type": "pii_leak", "detail": "输出含未脱敏邮箱/手机/身份证"})
        if self.block_toxic:
            q = self._toxic.score(record)
            checks["toxicity"] = q.overall
            if q.overall <= 0:
                violations.append({"type": "toxic_content", "detail": "命中毒性阻断词"})
        if self.require_json or record.get("format") == "json":
            q = self._fmt.score({"output": out, "format": "json"})
            checks["format"] = q.overall
            if q.overall < 1:
                violations.append({"type": "format_violation", "detail": "期望 JSON 但解析失败"})
        if record.get("claims") and record.get("context") is not None:
            q = self._ground.score(record)
            checks["grounding"] = q.overall
            if q.overall < self.grounding_min:
                violations.append({"type": "ungrounded",
                                   "detail": f"接地率 {q.overall:.2f} 低于 {self.grounding_min}"})
        if record.get("expected_tool"):
            q = self._tool.score(record)
            checks["tool_correctness"] = q.overall
            if q.overall < 1:
                violations.append({"type": "tool_misuse", "detail": "工具或参数不正确"})

        res = GuardrailResult(allowed=not violations, violations=violations, checks=checks)
        for v in violations:
            self._count(v["type"])
        if span is not None:
            span.add_event("guardrail", allowed=res.allowed,
                           violations=[v["type"] for v in violations])
            if not res.allowed:
                span.set_attribute("udos.guardrail_blocked", True)
        return res


# ==========================================================================
# 3. 成本归因 + Token 效率（USD by model/user/feature；截断浪费）
# ==========================================================================
# 每 1K token 美元单价（可由构造参数覆盖；示例价，部署前必须替换为真实合同价）。
DEFAULT_PRICE_USD_PER_1K: Dict[str, Tuple[float, float]] = {
    "default": (0.0010, 0.0020),   # (input, output)
    "udos-cpu": (0.0, 0.0),        # 本地自建推理按 0 计，成本走硬件摊销
}


class CostAccounting:
    def __init__(self, price_table: Optional[Dict[str, Tuple[float, float]]] = None,
                 registry: Any = None):
        self.prices = dict(price_table or DEFAULT_PRICE_USD_PER_1K)
        self.registry = registry
        self.totals: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int,
               user: str = "unknown", feature: str = "unknown",
               truncated: bool = False) -> float:
        pin, pout = self.prices.get(model, self.prices["default"])
        cost = input_tokens / 1000.0 * pin + output_tokens / 1000.0 * pout
        key = (model, user, feature)
        t = self.totals.setdefault(key, {"calls": 0, "input_tokens": 0,
                                         "output_tokens": 0, "cost_usd": 0.0,
                                         "truncated_calls": 0})
        t["calls"] += 1
        t["input_tokens"] += input_tokens
        t["output_tokens"] += output_tokens
        t["cost_usd"] = round(t["cost_usd"] + cost, 8)
        if truncated:
            t["truncated_calls"] += 1
        if self.registry is not None:
            lab = {"model": model, "user": user, "feature": feature}
            self.registry.counter("gen_ai_cost_usd_total",
                                  "Attributed cost (USD)").inc(cost, lab)
        return cost

    def attach_tracer(self, tracer: Tracer) -> None:
        """从根 Span 的 gen_ai.usage.* 属性回填成本（用于离线 trace 重算）。"""
        for root in tracer.root_spans():
            model = str(root.attrs.get(GEN_AI_REQUEST_MODEL, "default"))
            it = int(root.attrs.get(GEN_AI_USAGE_INPUT_TOKENS, 0))
            ot = int(root.attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS, 0))
            user = str(root.attrs.get("udos.user", "unknown"))
            feature = str(root.attrs.get("udos.feature", "unknown"))
            trunc = bool(root.attrs.get("udos.truncated", False))
            if it or ot:
                self.record(model, it, ot, user, feature, trunc)

    def report(self) -> List[Dict[str, Any]]:
        return [{"model": k[0], "user": k[1], "feature": k[2], **v}
                for k, v in sorted(self.totals.items())]


# ==========================================================================
# 4. 智能分析：流式异常检测（Welford + EWMA）与根因提示
# ==========================================================================
class _Series:
    __slots__ = ("n", "mean", "m2", "ewma")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.ewma = 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0


class AnomalyDetector:
    """对每个指标序列做在线均值/方差与 EWMA；超 k 个标准差即异常。

    属确定性统计异常检测（verified 机制）；AI 根因/预测性告警的 ML 模型是后续闸门。
    """

    def __init__(self, warmup: int = 5, k: float = 3.0, ewma_alpha: float = 0.3):
        self.warmup = warmup
        self.k = k
        self.alpha = ewma_alpha
        self._s: Dict[str, _Series] = {}

    def observe(self, key: str, value: float) -> Optional[Dict[str, Any]]:
        s = self._s.setdefault(key, _Series())
        # 先用“纳入当前点之前”的统计量判异常，避免离群点把均值/方差拉大而自我稀释
        prev_n, prev_mean, prev_std = s.n, s.mean, s.std
        anomaly = None
        if prev_n > self.warmup and prev_std > 0:
            z = (value - prev_mean) / prev_std
            if abs(z) >= self.k:
                anomaly = {"key": key, "value": round(value, 4),
                           "mean": round(prev_mean, 4), "zscore": round(z, 2),
                           "direction": "spike" if z > 0 else "drop"}
        # 再更新 Welford 均值/方差与 EWMA
        s.n += 1
        d = value - s.mean
        s.mean += d / s.n
        s.m2 += d * (value - s.mean)
        s.ewma = value if s.n == 1 else self.alpha * value + (1 - self.alpha) * s.ewma
        return anomaly

    @staticmethod
    def root_cause(root: Span, children: List[Span]) -> Dict[str, Any]:
        """在一条 trace 内定位首个错误阶段，否则定位耗时占比最大的阶段。"""
        errs = [c for c in children if c.status == "error"]
        if errs:
            c = errs[0]
            return {"type": "error_stage", "stage": c.name, "latency_ms": c.latency_ms}
        if children:
            total = sum(c.latency_ms for c in children) or 1e-9
            c = max(children, key=lambda x: x.latency_ms)
            return {"type": "latency_hotspot", "stage": c.name,
                    "latency_ms": c.latency_ms,
                    "share": round(c.latency_ms / total, 3)}
        return {"type": "insufficient_spans", "stage": None, "latency_ms": 0.0}


# ==========================================================================
# 5. 调用拓扑 + 火焰图（自时间）聚合
# ==========================================================================
def trace_topology(spans: List[Span]) -> Dict[str, Dict[str, Any]]:
    by_id = {s.span_id: s for s in spans}
    nodes: Dict[str, Dict[str, Any]] = {}
    for s in spans:
        n = nodes.setdefault(s.name, {"name": s.name, "calls": 0, "total_ms": 0.0,
                                      "errors": 0, "children": set()})
        n["calls"] += 1
        n["total_ms"] = round(n["total_ms"] + s.latency_ms, 3)
        if s.status == "error":
            n["errors"] += 1
        if s.parent_id and s.parent_id in by_id:
            n["children"].add(by_id[s.parent_id].name)
    for n in nodes.values():
        n["children"] = sorted(n["children"])
    return nodes


def flame_profile(spans: List[Span]) -> List[Dict[str, Any]]:
    """自时间 self_ms = 总时长 − 直接子节点时长；depth 由父链长度得到。"""
    by_id = {s.span_id: s for s in spans}
    children_of: Dict[str, List[Span]] = {}
    for s in spans:
        if s.parent_id:
            children_of.setdefault(s.parent_id, []).append(s)

    def depth(s: Span) -> int:
        d, cur = 0, s
        seen = set()
        while cur.parent_id and cur.parent_id in by_id and cur.span_id not in seen:
            seen.add(cur.span_id)
            cur = by_id[cur.parent_id]
            d += 1
        return d

    rows = []
    for s in spans:
        child_ms = sum(c.latency_ms for c in children_of.get(s.span_id, []))
        self_ms = max(0.0, round(s.latency_ms - child_ms, 3))
        rows.append({"name": s.name, "depth": depth(s),
                     "total_ms": s.latency_ms, "self_ms": self_ms,
                     "status": s.status})
    return sorted(rows, key=lambda r: (r["depth"], -r["total_ms"]))


# ==========================================================================
# 6. OTLP/JSON 导出（文件形态可被 Collector/Tempo 摄取；gRPC 推送为闸门）
# ==========================================================================
def _otlp_value(v: Any) -> Dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": v}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": redact(str(v))}


def to_otlp_json(tracer: Tracer, service: str = "udos-engine") -> Dict[str, Any]:
    now_wall = time.time_ns()
    now_perf = time.perf_counter_ns()
    otlp_spans = []
    for s in tracer.spans:
        # perf_counter 相对时刻锚到墙上时间（Span 未存 wall clock，导出时换算）
        start_ns = now_wall - (now_perf - s.start_ns)
        end_ns = start_ns + int(s.latency_ms * 1_000_000)
        attrs = [{"key": k, "value": _otlp_value(v)} for k, v in s.attrs.items()]
        otlp_spans.append({
            "traceId": s.trace_id, "spanId": s.span_id,
            "parentSpanId": s.parent_id or "",
            "name": s.name, "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(start_ns), "endTimeUnixNano": str(end_ns),
            "status": {"code": "STATUS_CODE_ERROR" if s.status == "error"
                       else "STATUS_CODE_OK"},
            "attributes": attrs,
            "events": [{"name": e.get("name", "event"),
                        "timeUnixNano": str(int(e.get("ts", 0) * 1e9))}
                       for e in s.events],
        })
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name",
                                     "value": {"stringValue": service}}]},
        "scopeSpans": [{"scope": {"name": "udos.observability"},
                        "spans": otlp_spans}]}]}


def export_otlp_json(tracer: Tracer, path: str, service: str = "udos-engine") -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = to_otlp_json(tracer, service)
    n = len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return n
