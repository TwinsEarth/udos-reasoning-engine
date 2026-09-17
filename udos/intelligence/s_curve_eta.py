"""S 曲线三时界(口径同网站 s_curve.py; 无 scipy, 确定性网格拟合)。

严格命名: eta_earliest=最乐观最早(target=90), eta_median=基准(target=95),
eta_latest=最悲观最晚(target=99)。保证 earliest<=median<=latest。
"""
from __future__ import annotations
import math


def _ym(year):
    year = max(1900.0, min(2200.0, year))
    y = int(math.floor(year)); m = int(round((year - y) * 12.0)) + 1
    if m > 12: y += 1; m = 1
    return f"{y:04d}-{max(1,m):02d}"


def _log(t, k, t0, L=100.0):
    return L / (1.0 + math.exp(-k * (t - t0)))


def _fit(pts):
    pts = sorted(pts)
    if len(pts) < 2:
        yr = pts[0][0] if pts else 2026.0
        return 0.5, yr + 2.0
    years = [float(y) for y, _ in pts]
    prog = [min(99.0, max(1.0, float(p))) for _, p in pts]
    best = None; be = None
    for k in [i / 100.0 for i in range(5, 500, 5)]:
        for t0 in [years[0] + (years[-1] - years[0]) * j / 20.0
                   for j in range(0, 21)]:
            err = sum((_log(y, k, t0) - p) ** 2 for y, p in zip(years, prog))
            if be is None or err < be:
                be, best = err, (k, t0)
    return best


def _yat(k, t0, p, L=100.0):
    r = L / p - 1.0
    return t0 if r <= 0 else t0 - math.log(r) / k


def s_curve_eta(pts, target=95.0):
    if not pts:
        raise ValueError("需要至少一个进度样本")
    k, t0 = _fit(pts)
    return {
        "eta_earliest": _ym(_yat(k, t0, 90.0)),
        "eta_median": _ym(_yat(k, t0, target)),
        "eta_latest": _ym(_yat(k, t0, 99.0)),
        "k": round(k, 4),
        "inflection": _ym(t0),
    }
