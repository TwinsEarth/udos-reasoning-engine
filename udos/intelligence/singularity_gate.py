"""奇点门(口径同网站 singularity_gate.py)。

换基跳变仅当全部满足: impact>=9 且 category∈核心技术 且
independent_sources>=2 且 human_confirmed=True。单点/单源/LLM 自称不触发。
"""
from __future__ import annotations

GATE_FIELDS = {"core_tech", "paradigm_shift"}
MIN_IMPACT = 9.0
MIN_SOURCES = 2


def singularity_gate(impact_score, category, independent_sources=1,
                     human_confirmed=False):
    reasons = []
    if impact_score < MIN_IMPACT:
        reasons.append(f"impact {impact_score} < {MIN_IMPACT}")
    if category not in GATE_FIELDS:
        reasons.append(f"category '{category}' 非核心技术")
    if independent_sources < MIN_SOURCES:
        reasons.append(f"独立来源 {independent_sources} < {MIN_SOURCES}")
    if not human_confirmed:
        reasons.append("human_confirmed=false; LLM 不可自证奇点")
    triggered = (impact_score >= MIN_IMPACT and category in GATE_FIELDS
                 and independent_sources >= MIN_SOURCES
                 and bool(human_confirmed))
    if triggered:
        reasons = ["ALL conditions met"]
    return triggered, reasons
