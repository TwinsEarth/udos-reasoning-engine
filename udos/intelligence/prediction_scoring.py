"""预言卡 Brier 记分(口径同网站 brier.py)。"""
from __future__ import annotations


def brier_single(p, outcome):
    if not 0.0 <= p <= 1.0:
        raise ValueError("p 须在 [0,1]")
    if outcome not in (0, 1):
        raise ValueError("outcome 须为 0/1")
    return round((p - outcome) ** 2, 6)


def score_predictions(predictions):
    """按 {probability∈[0,1], outcome∈{0,1}|缺失} 真正算 Brier。

    outcome=1 -> verified; outcome=0 -> falsified; 缺失 -> pending/score=null。
    probability 越界或非数 -> ValueError(->400)。
    """
    if not predictions:
        raise ValueError("predictions 缺失或为空")
    out = []
    for pr in predictions:
        rec = dict(pr)
        p = pr.get("probability")
        outcome = pr.get("outcome")
        if p is None:
            raise ValueError("缺 probability")
        try:
            p = float(p)
        except (TypeError, ValueError):
            raise ValueError("probability 非数")
        if not 0.0 <= p <= 1.0:
            raise ValueError("probability 须在 [0,1]")
        rec["probability"] = p
        if outcome not in (0, 1):
            rec["outcome"] = outcome
            rec["status"] = "pending"
            rec["score"] = None
        else:
            rec["outcome"] = outcome
            rec["status"] = "verified" if outcome == 1 else "falsified"
            rec["score"] = brier_single(p, outcome)
        out.append(rec)
    return out
