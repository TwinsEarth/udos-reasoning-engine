"""v5.5.2 报告: 四类运动路由 A/B + 类型条件 conformal 扇形覆盖率。

校准 seed=314, held-out seed=2026 (均与训练 seed=42 独立)。CPU 诚实档。
用法: PYTHONPATH=. python3 scripts/kind_fan_coverage.py
"""
from __future__ import annotations

import json
import os

import torch

from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.scene_head import load_scene_head
from udos.scene_fan import monte_carlo_rollout, coverage_fraction
from udos.dynamics_router import (
    CLASS_NAMES, classify_dynamics, routed_scene_params,
    KindSpecificParamErrorModel, apply_kind_inflation,
    fit_kind_conformal_inflation,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER_KIND = 96
N_SAMPLES = 48
NOMINAL = 0.80


def rollout_mse(pred, X, Y, sp=None):
    with torch.no_grad():
        out = pred.rollout(X, 4, scene_params=sp) if sp is not None \
            else pred.rollout(X, 4)
    return float(((out - Y) ** 2).mean())


def main() -> None:
    pred, _ = load_predictor(os.path.join(HERE, "checkpoints",
                                          "predictor_v4.3.9.pt"))
    pred.eval()
    head, _ = load_scene_head(os.path.join(HERE, "checkpoints",
                                           "scene_head_v5.5.0.pt"))
    cal = build_parametric_dataset(n_per_kind=N_PER_KIND, seed=314)
    held = build_parametric_dataset(n_per_kind=N_PER_KIND, seed=2026)

    true_held = torch.tensor([CLASS_NAMES.index(k) for k in held.kinds])
    true_cal = torch.tensor([CLASS_NAMES.index(k) for k in cal.kinds])
    r_held = classify_dynamics(held.X, held.dt)
    r_cal = classify_dynamics(cal.X, cal.dt)

    # 1) 混淆矩阵 (行=true, 列=pred)
    confusion = {}
    for ti, tn in enumerate(CLASS_NAMES):
        m = true_held == ti
        cnt = torch.bincount(r_held.labels[m], minlength=4).float()
        confusion[tn] = {CLASS_NAMES[pj]: round(float(v), 4)
                         for pj, v in enumerate(cnt / m.sum().float())}

    # 2) 路由经典估计 A/B (确定性, 不用学习头)
    routed = routed_scene_params(held.X, held.dt, r_held)
    ab = {}
    for k in CLASS_NAMES:
        m = held.kind_mask(k)
        blind = rollout_mse(pred, held.X[m], held.Y[m])
        explicit = rollout_mse(pred, held.X[m], held.Y[m], held.P[m])
        le = rollout_mse(pred, held.X[m], held.Y[m], routed[m])
        ab[k] = {"blind": round(blind, 4), "explicit": round(explicit, 4),
                 "routed": round(le, 4),
                 "recovery": round((blind - le) / (blind - explicit), 4)}

    # 3) 类型相关误差模型 (按真实类拟合于 calib)
    with torch.no_grad():
        p_cal = head(cal.X)
        p_held = head(held.X)
    kind_model = KindSpecificParamErrorModel.fit(p_cal, cal.P, true_cal)
    infl = fit_kind_conformal_inflation(
        pred, cal, r_cal.labels, true_cal, head=head,
        kind_error_model=kind_model, nominal=NOMINAL, n_samples=N_SAMPLES,
        generator=torch.Generator().manual_seed(7))

    # 4) held 扇形 (按预测类型绑定) + 类型条件膨胀
    bound = kind_model.bind(r_held.labels)
    fan = monte_carlo_rollout(pred, held.X, p_held, bound,
                              n_samples=N_SAMPLES, horizon=4,
                              generator=torch.Generator().manual_seed(8))
    raw_cov = coverage_fraction(fan.low, fan.high, held.Y)
    lo, hi = apply_kind_inflation(fan.low, fan.median, fan.high,
                                  infl, r_held.labels)
    cal_cov = coverage_fraction(lo, hi, held.Y)
    per_kind = {}
    for k in CLASS_NAMES:
        m = held.kind_mask(k)
        per_kind[k] = {
            "raw": round(float(((held.Y[m] >= fan.low[m])
                                & (held.Y[m] <= fan.high[m])).float().mean()), 4),
            "calibrated": round(float(((held.Y[m] >= lo[m])
                                       & (held.Y[m] <= hi[m])).float().mean()), 4),
            "inflation": round(float(infl[CLASS_NAMES.index(k)]), 4),
        }

    report = {
        "version": "5.5.2",
        "nominal_coverage": NOMINAL,
        "n_per_kind": N_PER_KIND,
        "n_samples": N_SAMPLES,
        "calib_seed": 314,
        "heldout_seed": 2026,
        "confusion_matrix_held": confusion,
        "routed_classical_ab_rollout4": ab,
        "fan_coverage": {
            "raw_pooled": round(raw_cov, 4),
            "calibrated_pooled": round(cal_cov, 4),
            "per_kind": per_kind,
        },
    }
    out = os.path.join(HERE, "reports", "v552_kind_fan.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
