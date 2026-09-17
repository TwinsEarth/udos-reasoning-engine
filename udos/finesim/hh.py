"""Hodgkin-Huxley 简化门控积分(确定性 Euler), 阶跃电流产生动作电位。analogy。"""
from __future__ import annotations
import math


def _alpha_m(v): return 0.1 * (v + 40) / (1 - math.exp(-(v + 40) / 10) + 1e-9)
def _beta_m(v): return 4.0 * math.exp(-(v + 65) / 18)
def _alpha_h(v): return 0.07 * math.exp(-(v + 65) / 20)
def _beta_h(v): return 1.0 / (1 + math.exp(-(v + 35) / 10))
def _alpha_n(v): return 0.01 * (v + 55) / (1 - math.exp(-(v + 55) / 10) + 1e-9)
def _beta_n(v): return 0.125 * math.exp(-(v + 65) / 80)


def simulate(I=10.0, T=50.0, dt=0.05):
    v = -65.0; m=0.05; h=0.6; n=0.32
    C=1.0; gNa=120.0; gK=36.0; gL=0.3
    ENa=50.0; EK=-77.0; EL=-54.387
    spikes = 0; prev = v; crossed = False
    steps = int(T / dt); peak_v = v; min_v = v
    for _ in range(steps):
        Iinj = I
        Iion = gNa*m**3*h*(v-ENa) + gK*n**4*(v-EK) + gL*(v-EL)
        dv = (Iinj - Iion) / C
        v += dt * dv
        m += dt * (_alpha_m(v)*(1-m) - _beta_m(v)*m)
        h += dt * (_alpha_h(v)*(1-h) - _beta_h(v)*h)
        n += dt * (_alpha_n(v)*(1-n) - _beta_n(v)*n)
        peak_v = max(peak_v, v); min_v = min(min_v, v)
        if prev < 0 and v >= 0 and not crossed:
            spikes += 1; crossed = True
        if v < -20:
            crossed = False
        prev = v
    return {"spikes": spikes, "peak_mV": round(peak_v, 2),
            "min_mV": round(min_v, 2), "current_uA": I,
            "note": "Hodgkin-Huxley 1952 简化; analogy"}


def f_I_curve(currents=(2.0, 5.0, 10.0, 20.0)):
    return [{"I": i, "Hz": simulate(i)["spikes"]} for i in currents]
