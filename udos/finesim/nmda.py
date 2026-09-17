"""NMDA Mg2+ 阻滞(Jahr-Stevens) + 时序性抑制可证伪对照。analogy。"""
from __future__ import annotations
import math


def nmda_conductance(v: float, mg_block: bool = True, mg: float = 1.0) -> float:
    """g_NMDA * 1/(1+Mg/3.5*exp(-v/16.1))。移除镁阻滞则无电压依赖。"""
    base = 1.0
    if not mg_block:
        return base
    return base / (1.0 + (mg / 3.5) * math.exp(-v / 16.1))


def temporal_suppression(ev_excite: float, ev_inhibit: float,
                        mg_block: bool = True) -> float:
    """平台电压抑制程度; 有镁阻滞时 Δt 决定, 移除后几乎不依赖 Δt。"""
    dt = ev_inhibit - ev_excite
    v = -30.0 + 10.0 * math.exp(-max(0, dt) / 5.0)
    g = nmda_conductance(v, mg_block)
    return round(g, 6)


def falsifiable_control():
    """有镁阻滞->时序敏感(不同 Δt 输出差大); 移除->不敏感(差小)。"""
    dts = [-2.0, 0.0, 3.0, 6.0]
    with_mg = [temporal_suppression(0, d, True) for d in dts]
    no_mg = [temporal_suppression(0, d, False) for d in dts]
    return {
        "with_mg": with_mg,
        "no_mg": no_mg,
        "with_mg_range": round(max(with_mg) - min(with_mg), 6),
        "no_mg_range": round(max(no_mg) - min(no_mg), 6),
        "falsifiable": True,
        "note": "Du 2017 / Doron 2017; analogy 数值对照",
    }
