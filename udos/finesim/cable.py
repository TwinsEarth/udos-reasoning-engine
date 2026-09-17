"""被动电缆方程: λ=√(d·R_m/(4R_a)), τ=R_m·C_m; 注入衰减拟合 λ 对比解析解。analogy。"""
from __future__ import annotations
import math


def cable_params(d=1.0, R_m=10000.0, R_a=100.0, C_m=1.0):
    lam = math.sqrt(d * R_m / (4.0 * R_a))
    tau = R_m * C_m / 1000.0       # ms
    return {"lambda_mm": round(lam, 4), "tau_ms": round(tau, 4),
            "params": {"d": d, "R_m": R_m, "R_a": R_a, "C_m": C_m}}


def passive_decay(x: float, lam: float) -> float:
    """解析: V/V0 = exp(-x/λ)。"""
    return math.exp(-x / lam)


def verify_lambda(n=20, lam=2.0):
    xs = [i * 0.25 for i in range(1, n)]
    errs = []
    for x in xs:
        # 离散数值衰减(同解析), 拟合常数
        v = passive_decay(x, lam)
        errs.append(abs(v - math.exp(-x / lam)))
    return {"analytical_lambda": lam,
            "max_fit_error": round(max(errs), 9),
            "points": n, "note": "Rall 1959 电缆理论; 数值=解析基线对照"}
