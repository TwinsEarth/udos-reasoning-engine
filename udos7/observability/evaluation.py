"""在线质量评估（Online Evaluation）。

确定性 judge（格式/grounding/工具正确性/毒性）零依赖、可单测（verified，cpu-proto 口径）；
LLM-as-a-judge 需要供应商 API key（AL4），未配置即 GateError，绝不用规则冒充 LLM 评分。
质量回归：滚动窗口均值相对基线显著下滑时告警（“质量回归”先于用户投诉）。
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents.automation import GateError


@dataclass
class QualityScore:
    overall: float
    dims: Dict[str, float] = field(default_factory=dict)
    evidence: str = "cpu-proto"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"overall": round(self.overall, 4), "dims": self.dims,
                "evidence": self.evidence, "notes": self.notes}


class Judge:
    name = "base"

    def score(self, record: Dict[str, Any]) -> QualityScore:
        raise NotImplementedError


class FormatJudge(Judge):
    """结构化输出合规：要求 JSON 可解析（或声明 format=text 即跳过）。"""
    name = "format"

    def score(self, record: Dict[str, Any]) -> QualityScore:
        if record.get("format") == "text":
            return QualityScore(1.0, {self.name: 1.0}, "verified", ["text 免校验"])
        text = record.get("output", "")
        try:
            json.loads(text)
            return QualityScore(1.0, {self.name: 1.0}, "verified")
        except (TypeError, json.JSONDecodeError):
            return QualityScore(0.0, {self.name: 0.0}, "verified", ["JSON 解析失败"])


class GroundingJudge(Judge):
    """接地性：声明的关键 claim 有多少能在检索上下文里找到（朴素关键词，cpu-proto）。"""
    name = "grounding"

    def score(self, record: Dict[str, Any]) -> QualityScore:
        context = str(record.get("context", ""))
        claims = record.get("claims") or []
        if not claims:
            return QualityScore(0.5, {self.name: 0.5}, "cpu-proto", ["无 claim，中性分"])
        hit = sum(1 for c in claims if str(c).strip() and str(c).strip() in context)
        v = hit / len(claims)
        return QualityScore(v, {self.name: round(v, 4)}, "cpu-proto",
                            [f"{hit}/{len(claims)} claim 命中上下文"])


class ToolCorrectnessJudge(Judge):
    """工具调用正确性：工具名与关键参数是否与期望一致。"""
    name = "tool_correctness"

    def score(self, record: Dict[str, Any]) -> QualityScore:
        expected = record.get("expected_tool")
        if not expected:
            return QualityScore(0.5, {self.name: 0.5}, "cpu-proto", ["无期望工具"])
        calls = record.get("tool_calls") or []
        names = [c.get("name") for c in calls if isinstance(c, dict)]
        if expected not in names:
            return QualityScore(0.0, {self.name: 0.0}, "verified",
                                [f"未调用 {expected}"])
        exp_params = record.get("expected_params") or {}
        for c in calls:
            if c.get("name") != expected:
                continue
            params = c.get("params", {})
            if all(params.get(k) == v for k, v in exp_params.items()):
                return QualityScore(1.0, {self.name: 1.0}, "verified", ["工具与参数正确"])
        return QualityScore(0.5, {self.name: 0.5}, "verified", ["工具对、参数不符"])


class ToxicityJudge(Judge):
    """毒性/不安全输出：极简阻断词表（cpu-proto，非商用内容审核）。"""
    name = "toxicity"
    BLOCKLIST = ("暴恐", "儿童色情", "制毒", "制造炸药")

    def score(self, record: Dict[str, Any]) -> QualityScore:
        text = str(record.get("output", ""))
        hit = [w for w in self.BLOCKLIST if w in text]
        if hit:
            return QualityScore(0.0, {self.name: 0.0}, "cpu-proto",
                                [f"命中阻断词 {hit}"])
        return QualityScore(1.0, {self.name: 1.0}, "cpu-proto")


class LLMJudge(Judge):
    """LLM-as-a-judge：需供应商 key（AL4）。无 key 直接门禁，不提供伪实现。"""
    name = "llm_judge"

    def __init__(self, model: str = "judge-v1"):
        self.model = model
        self.key = (os.environ.get("UDOS_LLM_JUDGE_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("ANTHROPIC_API_KEY"))
        if not self.key:
            raise GateError(
                "LLM-as-a-judge 需要配置 UDOS_LLM_JUDGE_KEY/OPENAI_API_KEY（AL4）；"
                "未配置时请使用确定性 judge（Format/Grounding/Tool/Toxicity）。")

    def score(self, record: Dict[str, Any]) -> QualityScore:
        # 真实后端调用属部署闸门（需网络+计费），此处不伪装。
        raise GateError("LLM-as-a-judge 真实评分后端未接入（AL4，unverified）。")


class OnlineEvaluator:
    """对采样流量持续跑 judge，维护各维度滚动窗口并检测质量回归。"""

    def __init__(self, judges: Optional[List[Judge]] = None,
                 sample_rate: float = 0.15, window: int = 50,
                 regression_margin: float = 0.1, baseline_min: int = 10):
        self.judges = judges or [FormatJudge(), GroundingJudge(),
                                 ToolCorrectnessJudge(), ToxicityJudge()]
        self.sample_rate = sample_rate
        self.window = window
        self.margin = regression_margin
        self.baseline_min = baseline_min
        self._series: Dict[str, deque] = {}
        self.evaluated = 0

    def evaluate(self, record: Dict[str, Any], force: bool = False) -> Optional[Dict[str, Any]]:
        import hashlib
        rid = str(record.get("request_id", ""))
        if not force and rid:
            h = int(hashlib.sha256(rid.encode()).hexdigest()[:10], 16) % 1000 / 1000
            if h >= self.sample_rate:
                return None
        dims: Dict[str, float] = {}
        notes: List[str] = []
        evidence = "verified"
        for j in self.judges:
            q = j.score(record)
            dims[j.name] = q.overall
            notes.extend(q.notes)
            if q.evidence != "verified":
                evidence = "cpu-proto"
        overall = round(sum(dims.values()) / len(dims), 4)
        for k, v in dims.items():
            self._series.setdefault(k, deque(maxlen=self.window)).append(v)
        self.evaluated += 1
        return {"overall": overall, "dims": dims, "evidence": evidence, "notes": notes}

    def quality_regression(self, dim: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """比较前 baseline_min 个基线与最近窗口：均值下滑超过 margin 则告警。"""
        keys = [dim] if dim else list(self._series.keys())
        for k in keys:
            xs = list(self._series.get(k, []))
            if len(xs) < 2 * self.baseline_min:
                continue
            base = sum(xs[:self.baseline_min]) / self.baseline_min
            recent = sum(xs[-self.baseline_min:]) / self.baseline_min
            if base - recent >= self.margin:
                return {"dim": k, "baseline": round(base, 4),
                        "recent": round(recent, 4), "drop": round(base - recent, 4)}
        return None
