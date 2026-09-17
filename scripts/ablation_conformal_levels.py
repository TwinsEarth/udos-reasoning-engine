#!/usr/bin/env python3
"""
v2.4.11 多名义水平 split-conformal 区间覆盖率验证基准 (80/90/95)
================================================================
唯一变量 = 名义覆盖水平 alpha∈{0.2,0.1,0.05} (即 80%/90%/95%)。
同合同: 每个 seed 同一模型权重、同一校准集拟合三水平半宽、同一**独立测试集**评估,
仅 alpha 不同。报告:
  - coverage_overall: 测试集真值落入区间的比例 (应随 alpha 减小而不下降)
  - mean_width:       全维平均区间宽度 (应随 alpha 减小而严格增宽)
split-conformal 的名义覆盖率在有限样本上通常**保守** (实测 >= 名义), 这里如实记录。

输出: benchmarks/results/conformal_levels_v2.4.11.json
用法: python3 scripts/ablation_conformal_levels.py [--quick] [--seeds 42 7 123]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (  # noqa: E402
    build_parametric_dataset, PhysicsPredictor, CTMTrainer, TrainConfig,
    CTMConfig,
)
from udos.calibration import fit_predictor_calibration  # noqa: E402

ALPHAS = (0.2, 0.1, 0.05)
LEVEL_NAME = {0.2: "80%", 0.1: "90%", 0.05: "95%"}


def small_ctm_config():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def train_one(seed, n_per_kind, epochs, horizon=4):
    torch.manual_seed(seed)
    tr_ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                     window=6, horizon=horizon, dt=0.5,
                                     seed=1000 + seed)
    tr, _ = tr_ds.split(0.8)
    model = PhysicsPredictor(small_ctm_config(), scene_param_dim=4)
    CTMTrainer(model, TrainConfig(epochs=epochs, seed=seed)).train(tr, None)
    model.eval()
    return model


def eval_levels(model, cal_ds, test_ds):
    # 校准集拟合三水平半宽 (report 里 conformal_by_alpha 为 list)
    _, rep, _ = fit_predictor_calibration(model, cal_ds)
    cba = {float(a): [torch.tensor(q) for q in hw]
           for a, hw in rep["conformal_by_alpha"].items()}
    rows = []
    # 临时挂三水平半宽 (residual_quantiles=None -> predict_interval 走 conformal_by_alpha)
    model.conformal_by_alpha = cba
    model.residual_quantiles = None
    for a in ALPHAS:
        iv = model.predict_interval(test_ds.X, test_ds.Y.size(1),
                                    scene_params=test_ds.P, alpha=a)
        Y = test_ds.Y
        inside = ((Y >= iv["lower"] - 1e-9) & (Y <= iv["upper"] + 1e-9))
        cov = float(inside.float().mean())
        width = float((iv["upper"] - iv["lower"]).mean())
        rows.append({"alpha": a, "level": LEVEL_NAME[a],
                     "coverage": round(cov, 4),
                     "mean_width": round(width, 6)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)

    epochs = args.epochs or (12 if args.quick else 40)
    n_per_kind = 16 if args.quick else 32
    horizon = 4
    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        model = train_one(seed, n_per_kind, epochs, horizon)
        cal_ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                          window=6, horizon=horizon, dt=0.5,
                                          seed=3000 + seed)
        test_ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                            window=6, horizon=horizon, dt=0.5,
                                            seed=4000 + seed)
        _, te = test_ds.split(0.8)
        rows = eval_levels(model, cal_ds, te)
        per_seed.append({"seed": seed, "runs": rows})
        print(f"seed={seed} " + " | ".join(
            f"{r['level']}:cov={r['coverage']:.3f},w={r['mean_width']:.4f}"
            for r in rows))

    def mean(a, k):
        return sum(r[k] for ps in per_seed for r in ps["runs"] if r["alpha"] == a) \
            / len(per_seed)

    summary = {}
    for a in ALPHAS:
        summary[LEVEL_NAME[a]] = {
            "alpha": a,
            "coverage": round(mean(a, "coverage"), 4),
            "nominal_coverage": round(1.0 - a, 4),
            "coverage_minus_nominal": round(mean(a, "coverage") - (1.0 - a), 4),
            "mean_width": round(mean(a, "mean_width"), 6),
        }
    # 非减性: 宽度随 alpha 减小而增宽 (跨种子一致性)
    w80, w90, w95 = (summary["80%"]["mean_width"], summary["90%"]["mean_width"],
                     summary["95%"]["mean_width"])
    summary["width_ordering_80<90<95"] = bool(w80 < w90 < w95)

    result = {
        "experiment": "v2.4.11 conformal multi-level coverage (80/90/95)",
        "alphas": list(ALPHAS), "metric": "independent test set coverage & width",
        "epochs": epochs, "n_per_kind": n_per_kind, "horizon": horizon,
        "seeds": args.seeds, "per_seed": per_seed, "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "benchmarks" / "results" / "conformal_levels_v2.4.11.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nsummary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {out}  ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
