"""ASI RSI(口径同网站 rsi.py): AGI→ASI 间隔随 RSI 增大而缩短。"""
from __future__ import annotations

BASE = 20.0
SPREAD = 18.0


def agi_to_asi_years(rsi_score, base=BASE, spread=SPREAD):
    if not 0.0 <= rsi_score <= 100.0:
        raise ValueError("rsi 须在 [0,100]")
    return round(max(0.0, base - (rsi_score / 100.0) * spread), 3)


def aggregate_rsi_score(indicators):
    if not indicators:
        return 0.0
    vals = [max(0.0, min(100.0, float(v))) for v in indicators.values()]
    return round(sum(vals) / len(vals), 2)
