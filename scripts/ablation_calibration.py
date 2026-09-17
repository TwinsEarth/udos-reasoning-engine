#!/usr/bin/env python3
"""
v2.4.6 校准方法 同合同 A/B: PAVA vs Temperature vs None
=========================================================
唯一变量 = 置信校准方法。同合同: 对每个 seed, 同一训练好的模型权重、同一校准集、
同一独立测试集, 仅校准映射不同。比较:
  - ECE (回归期望校准误差, 越低越好)
  - Spearman(置信, 误差) (期望为负: 置信越高误差越低, 绝对值越大越好)
  - split-conformal 区间覆盖率 (与置信方法无关, 作为参照)
跨多种子看方向是否一致 (吸取既往 A/B 收益随种子反转的教训)。

证据诚实: 若某方法跨种子不稳或更差, 照实写入 JSON, 不硬凑。
输出: benchmarks/results/calibration_ablation_v2.4.6.json
用法: python3 scripts/ablation_calibration.py [--quick] [--seeds 42 7 123]
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
from udos.calibration import (  # noqa: E402
    ConfidenceCalibrator, TemperatureScaling, reliability, spearman,
    collect_predictor_outputs, _conservative_upper_quantile,
)

METHODS = ("pava", "temperature", "none")


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


def eval_method(model, cal_ds, test_ds, method):
    # 在校准集上取原始置信/误差用于拟合
    conf_cal, err_cal, _, _ = collect_predictor_outputs(model, cal_ds)
    # 在测试集上取原始置信/误差
    conf_t, err_t, _, resid = collect_predictor_outputs(model, test_ds)

    if method == "pava":
        cal = ConfidenceCalibrator().fit(conf_cal, err_cal)
        scale = cal.scale
        ece = reliability(cal.transform(conf_t), err_t, scale, 5)["ece"]
    elif method == "temperature":
        cal = TemperatureScaling().fit(conf_cal, err_cal)
        scale = cal.scale
        ece = reliability(cal.transform(conf_t), err_t, scale, 5)["ece"]
    else:  # none: 原始置信
        scale = float(err_cal.mean()) or 1e-8
        ece = reliability(conf_t, err_t, scale, 5)["ece"]

    sp = spearman(conf_t, err_t)
    # split-conformal 覆盖率 (置信方法无关, 三方法共用同一残差半宽)
    H = resid.size(1)
    rq = [torch.stack([_conservative_upper_quantile(
        resid[:, h, r].abs(), 0.9) for r in range(resid.size(2))], dim=0)
        for h in range(H)]
    roll = model.rollout(test_ds.X, H, scene_params=test_ds.P)
    covs = []
    for h in range(H):
        lo, hi = roll[:, h, :] - rq[h], roll[:, h, :] + rq[h]
        covs.append(float(((test_ds.Y[:, h, :] >= lo - 1e-9) &
                            (test_ds.Y[:, h, :] <= hi + 1e-9)).float().mean()))
    return {"method": method, "ece": round(ece, 6),
            "spearman_conf_err": round(sp, 4),
            "coverage_overall": round(sum(covs) / len(covs), 4)}


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
        rows = [eval_method(model, cal_ds, te, m) for m in METHODS]
        per_seed.append({"seed": seed, "runs": rows})
        print(f"seed={seed} " + " | ".join(
            f"{r['method']}:ece={r['ece']:.4f},sp={r['spearman_conf_err']}"
            for r in rows))

    def mean(m, k):
        return sum(r[k] for ps in per_seed for r in ps["runs"]
                   if r["method"] == m) / len(per_seed)

    summary = {m: {"ece": round(mean(m, "ece"), 6),
                   "spearman_conf_err": round(mean(m, "spearman_conf_err"), 4),
                   "coverage_overall": round(mean(m, "coverage_overall"), 4)}
               for m in METHODS}
    # 方向一致性: pava/temperature 的 ECE 是否在多数种子低于 none
    for m in ("pava", "temperature"):
        wins = sum(1 for ps in per_seed
                   if next(r for r in ps["runs"] if r["method"] == m)["ece"]
                   < next(r for r in ps["runs"] if r["method"] == "none")["ece"])
        summary[m]["lower_ece_vs_none_seeds"] = f"{wins}/{len(per_seed)}"

    result = {
        "experiment": "v2.4.6 calibration method A/B (PAVA vs Temperature vs None)",
        "epochs": epochs, "n_per_kind": n_per_kind, "horizon": horizon,
        "seeds": args.seeds, "per_seed": per_seed, "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "benchmarks" / "results" / "calibration_ablation_v2.4.6.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nsummary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {out}  ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
