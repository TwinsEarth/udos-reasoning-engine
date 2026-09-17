"""树突计算: 兴奋/抑制门、重合检测、NMDA 式非线性、时序性抑制 Δt 扫描。analogy。"""
from __future__ import annotations


def coincidence_detect(events_a: list, events_b: list, window: int = 2) -> int:
    """两列事件时间戳; 落在 ±window 内即重合一次。"""
    count = 0
    for a in events_a:
        for b in events_b:
            if abs(a - b) <= window:
                count += 1
                break
    return count


def nmda_amplify(voltage: float, theta: float = 1.0) -> float:
    """NMDA 式非线性放大: 超阈值后 S 型放大(确定性)。"""
    if voltage <= 0:
        return 0.0
    return round(voltage * (1.0 + (voltage / (voltage + theta))), 6)


def temporal_inhibition(ev_excite: float, ev_inhibit: float,
                        tau: float = 5.0, max_gate: float = 1.0) -> float:
    """时序性抑制: 抑制相对兴奋的时间差 Δt 决定门控输出。
    Δt>0(抑制晚) 门控衰减; Δt<0(抑制先) 几乎不抑制。"""
    dt = ev_inhibit - ev_excite
    if dt <= 0:
        return max_gate
    return round(max_gate * pow(2.718281828, -dt / tau), 6)


def inhibition_curve(dt_scan=range(-5, 11), tau: float = 5.0) -> list:
    """Δt 扫描成确定门控曲线(ev_excite=0, ev_inhibit=dt)。"""
    return [{"dt": dt, "gate": temporal_inhibition(0.0, float(dt), tau)}
            for dt in dt_scan]
