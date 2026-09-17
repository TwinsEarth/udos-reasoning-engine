#!/usr/bin/env python3
"""
v2.6 新特性推理延迟基准 (小批量 32 样本)
==========================================
测量 hybrid / counterfactual / identify / adaptive / risk 的单次推理延迟
(CPU, 小批量 32), 落 benchmarks/results/feature_latency_v2.6.0.json。
纯前向、确定性、不重训; 基于 v2.6.0 checkpoint。

用法:
    python3 scripts/benchmark_v26_features.py
    python3 scripts/benchmark_v26_features.py --repeats 15 --warmup 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from udos import __version__
from udos.persistence import load_predictor
from udos.hybrid import HybridPhysicsCorrector
from udos.counterfactual import CounterfactualEngine
from udos.identification import SceneParameterIdentifier
from udos.adaptive import adaptive_rollout
from udos.decision import RiskGrader
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.6.0.pt"
OUT = Path("benchmarks/results/feature_latency_v2.6.0.json")


def _time(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - t0)
    lat.sort()
    n = len(lat)
    return {
        "mean_ms": round(sum(lat) / n * 1e3, 4),
        "p50_ms": round(lat[n // 2] * 1e3, 4),
        "min_ms": round(lat[0] * 1e3, 4),
        "max_ms": round(lat[-1] * 1e3, 4),
        "repeats": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    predictor, _ = load_predictor(CKPT)
    predictor.eval()

    ds = build_parametric_dataset(n_per_kind=args.batch, n_steps=14,
                                  window=6, horizon=4, dt=0.5, seed=777)
    X, P = ds.X, ds.P

    # 为 hybrid 段挂一个 (未训练) 修正器, 仅测前向开销
    predictor.attach_hybrid(HybridPhysicsCorrector(raw_dim=predictor.raw_dim))
    cf_engine = CounterfactualEngine(predictor)
    ident = SceneParameterIdentifier(predictor, grid_size=3)
    grader = RiskGrader()

    results = {
        "version": __version__,
        "checkpoint": CKPT,
        "device": "cpu",
        "batch_size": args.batch,
        "n_window": X.size(1),
        "features": {
            "predict_next_baseline":
                _time(lambda: predictor.predict_next(X, scene_params=P),
                      args.warmup, args.repeats),
            "hybrid_correction":
                _time(lambda: predictor.predict_next(X, scene_params=P,
                                                     hybrid=True),
                      args.warmup, args.repeats),
            "counterfactual_h2":
                _time(lambda: cf_engine.counterfactual(
                    X, horizon=2, scene_params=P,
                    intervention={"scene_params": {"2": 1.1}}),
                    args.warmup, args.repeats),
            "identify_grid3":
                _time(lambda: ident.identify(X[:1], horizon=2),
                      args.warmup, args.repeats),
            "adaptive_rollout_h2":
                _time(lambda: adaptive_rollout(predictor, X, max_horizon=2,
                                              scene_params=P),
                      args.warmup, args.repeats),
            "risk_grade":
                _time(lambda: grader.grade(predictor, X, scene_params=P,
                                            horizon=1),
                      args.warmup, args.repeats),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写入 {OUT}")
    for k, v in results["features"].items():
        print(f"  {k:24s} mean {v['mean_ms']:8.3f} ms  p50 {v['p50_ms']:8.3f} ms")


if __name__ == "__main__":
    main()
