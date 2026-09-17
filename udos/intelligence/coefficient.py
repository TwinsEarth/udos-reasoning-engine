"""系数递推(口径同网站 coefficient.py, v1.0)。

raw_score 0..100; weighted = raw * credibility(0..1)。
EWMA: new = alpha*weighted + (1-alpha)*prev, alpha=0.3。
"""
from __future__ import annotations

FORMULA_VERSION = "v1.0"
ALPHA = 0.3


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


def coefficient_today(raw_score: float, credibility: float) -> float:
    if not 0.0 <= credibility <= 1.0:
        raise ValueError("credibility 须在 [0,1]")
    return round(_clip(raw_score) * credibility, 4)


def ewma_next(raw_score: float, prev_score, alpha: float = ALPHA) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 须在 [0,1]")
    raw = _clip(raw_score)
    if prev_score is None:
        return raw
    return round(_clip(alpha * raw + (1.0 - alpha) * prev_score), 4)
