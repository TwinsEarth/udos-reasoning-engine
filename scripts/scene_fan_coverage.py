#!/usr/bin/env python3
"""
v5.5.1 蒙特卡洛轨迹扇形覆盖率报告
==================================
- 校准集 (seed=314): 拟合参数误差模型 + split-conformal 膨胀因子
- held-out (seed=2026): 报告原始 MC 与 conformal 校准后的 pooled/分类型/分步
  经验覆盖率 (名义 80% 带 = p10..p90)。
产物: reports/v551_fan_coverage.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from udos.dynamics import build_parametric_dataset
from udos.persistence import load_predictor
from udos.scene_fan import (ParamErrorModel, calibrated_band,
                            coverage_fraction, fit_conformal_inflation,
                            monte_carlo_rollout, per_step_coverage)
from udos.scene_head import load_scene_head


def evaluate(predictor, head, ds, em, inflation, n_samples, horizon, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        phat = head(ds.X)
    fan = monte_carlo_rollout(predictor, ds.X, phat, em, n_samples,
                              horizon, quantiles=(0.1, 0.5, 0.9), generator=g)
    lo, hi = calibrated_band(fan, inflation)
    out = {
        "raw_mc_coverage": round(coverage_fraction(fan.low, fan.high, ds.Y), 4),
        "calibrated_coverage": round(coverage_fraction(lo, hi, ds.Y), 4),
        "calibrated_per_step": [round(c, 4) for c in
                                per_step_coverage(lo, hi, ds.Y)],
        "per_kind": {},
    }
    for k in ds.class_names:
        m = ds.kind_mask(k)
        out["per_kind"][k] = {
            "raw_mc_coverage": round(
                coverage_fraction(fan.low[m], fan.high[m], ds.Y[m]), 4),
            "calibrated_coverage": round(
                coverage_fraction(lo[m], hi[m], ds.Y[m]), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-per-kind", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--nominal", type=float, default=0.80)
    args = ap.parse_args()

    predictor, _ = load_predictor(REPO / "checkpoints" / "predictor_v4.3.9.pt")
    predictor.eval()
    head, _ = load_scene_head(REPO / "checkpoints" / "scene_head_v5.5.0.pt")

    cal = build_parametric_dataset(n_per_kind=args.n_per_kind, seed=314)
    held = build_parametric_dataset(n_per_kind=args.n_per_kind, seed=2026)
    em = ParamErrorModel.fit(head, cal)
    inflation = fit_conformal_inflation(
        predictor, head, cal, em, n_samples=args.n_samples,
        horizon=args.horizon, nominal=args.nominal,
        generator=torch.Generator().manual_seed(7))

    report = {
        "nominal_coverage": args.nominal,
        "n_samples": args.n_samples, "horizon": args.horizon,
        "param_bias": [round(x, 4) for x in em.bias.tolist()],
        "param_std": [round(x, 4) for x in em.std.tolist()],
        "conformal_inflation": round(inflation, 4),
        "calib_seed": 314, "heldout_seed": 2026,
        "heldout": evaluate(predictor, head, held, em, inflation,
                            args.n_samples, args.horizon, 8),
        "calib_self": evaluate(predictor, head, cal, em, inflation,
                               args.n_samples, args.horizon, 9),
    }
    (REPO / "reports").mkdir(exist_ok=True)
    out = REPO / "reports" / "v551_fan_coverage.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report["heldout"], ensure_ascii=False, indent=2))
    print("inflation", report["conformal_inflation"], "written:", out)


if __name__ == "__main__":
    main()
