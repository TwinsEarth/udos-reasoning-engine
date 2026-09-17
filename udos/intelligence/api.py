"""IntelAPI: 组装 9 个锁定端点响应(确定性离线 seed)。"""
from __future__ import annotations
from typing import Dict, List

from .coefficient import ewma_next, coefficient_today, FORMULA_VERSION
from .singularity_gate import singularity_gate
from .capability import CapabilityVector
from .s_curve_eta import s_curve_eta
from .prediction_scoring import score_predictions
from .asi_rsi import agi_to_asi_years, aggregate_rsi_score
from .model_aggregate import model_aggregate
from .risk_signals import RISK_SEED
from .seed import OFFLINE_SEED

try:
    from udos import __version__ as ENGINE_VERSION
except Exception:  # pragma: no cover
    ENGINE_VERSION = "5.1.0"


class NoData(Exception):
    """对应 HTTP 409。"""


class IntelAPI:
    def __init__(self):
        self._step_series: List[Dict] = []

    def health(self):
        return {"service": "udos-intel", "engine_version": ENGINE_VERSION,
                "mode": "offline-deterministic"}

    def coefficient(self, days=None):
        series = list(OFFLINE_SEED["coefficient"])
        if self._step_series:
            series += self._step_series
        return {"series": series}

    def coefficient_step(self, body):
        events = body.get("events")
        if not events:
            raise ValueError("缺少 events")
        human = bool(body.get("human_confirmed", False))
        prev = self._step_series[-1]["weighted_score"] if self._step_series else None
        for ev in events:
            impact = float(ev.get("impact", 0))
            cred = float(ev.get("credibility", 0.5))
            cat = ev.get("category", "")
            srcs = int(ev.get("independent_sources", 1))
            raw = impact * 10
            weighted = coefficient_today(raw, cred)
            smoothed = ewma_next(weighted, prev)
            prev = smoothed
            trig, reasons = singularity_gate(impact, cat, srcs, human)
            self._step_series.append({
                "date": ev.get("date", ""), "raw_score": raw,
                "weighted_score": smoothed, "formula_version": FORMULA_VERSION,
                "milestone_flag": trig, "confidence": cred,
                "sources": ev.get("sources", []),
            })
        last = self._step_series[-1]
        return {"weighted_score": last["weighted_score"],
                "formula_version": FORMULA_VERSION,
                "gate": {"triggered": trig, "reasons": reasons,
                         "human_required": True}}

    def capability(self):
        cv = CapabilityVector(scores=dict(OFFLINE_SEED["capability"]))
        return {"dimensions": cv.dimensions_list()}

    def routes(self):
        out = []
        for r in OFFLINE_SEED["routes"]:
            eta = s_curve_eta([(2022, r["progress_pct"] / 2),
                               (2024, r["progress_pct"] * 0.8),
                               (2026, r["progress_pct"])])
            out.append({
                "route": r["route"], "orgs": r["orgs"], "people": r["people"],
                "progress_pct": r["progress_pct"],
                "eta_median": eta["eta_median"],
                "eta_earliest": eta["eta_earliest"],
                "eta_latest": eta["eta_latest"],
                "confidence": 0.7, "milestones": r["milestones"],
            })
        return {"routes": out}

    def predictions(self, status=None):
        preds = list(OFFLINE_SEED["predictions"])
        if status:
            preds = [p for p in preds if p["status"] == status]
        return {"predictions": preds}

    def predictions_score(self, body):
        preds = body.get("predictions", [])
        return {"scored": score_predictions(preds)}

    def models(self):
        return {"models": model_aggregate(OFFLINE_SEED["models"])}

    def asi(self):
        ind = OFFLINE_SEED["rsi"]
        rsi = aggregate_rsi_score(ind)
        interval = agi_to_asi_years(rsi)
        return {
            "rsi_indicators": [{"name": k, "value": v} for k, v in ind.items()],
            "takeoff_scenarios": [
                {"name": "fast", "interval_years": max(0.0, interval - 2)},
                {"name": "slow", "interval_years": interval + 5}],
            "asi_eta": {"median": round(interval, 2),
                        "earliest": round(max(0.0, interval - 2), 2),
                        "latest": round(interval + 5, 2),
                        "confidence": 0.4,
                        "disclaimer": "假设非事实"},
            "risk_signals": RISK_SEED,
        }
