#!/usr/bin/env python3
"""
v2.7 新特性推理延迟基准 (小批量 32 样本, CPU)
==============================================
测量 policy(MPC) / online(observe+detection) / active(score) /
lite(INT8 量化后推理) / hierarchical(分层 rollout) 的单次推理延迟,
落 benchmarks/results/feature_latency_v2.7.0.json。纯前向、确定性、不重训;
基于 v2.7.0 checkpoint。

用法:
    python3 scripts/benchmark_v27_features.py
    python3 scripts/benchmark_v27_features.py --repeats 15 --warmup 3
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
from udos.policy import MPCActionSelector
from udos.online import OnlineAdapter
from udos.active_learning import UncertaintySampler
from udos.lite import DynamicQuantizer
from udos.hierarchical import HierarchicalRollout
from udos.dynamics import build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"
OUT = Path("benchmarks/results/feature_latency_v2.7.0.json")


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
    w1 = X[:1]
    p1 = P[:1]

    mpc = MPCActionSelector(predictor, horizon=2, lambda_risk=1.0)
    actions = [{}, {"state_perturbation": [0.0] * 6}, {"scene_param": p1.reshape(-1)}]
    sampler = UncertaintySampler()
    # 在线适配器: 用分布内参考构造, 仅测 observe+detection 开销
    adapter = OnlineAdapter(X, drift_window=32)
    hier = HierarchicalRollout(predictor, coarse_factor=4)
    quantized = DynamicQuantizer.quantize(predictor)

    results = {
        "version": __version__,
        "checkpoint": CKPT,
        "device": "cpu",
        "batch_size": args.batch,
        "features": {
            "predict_next_baseline":
                _time(lambda: predictor.predict_next(X, scene_params=P),
                      args.warmup, args.repeats),
            "policy_mpc_3actions":
                _time(lambda: mpc.select(w1, scene_params=p1,
                                        candidate_actions=actions),
                      args.warmup, args.repeats),
            "active_score_pool":
                _time(lambda: sampler.score_samples(predictor, X, P),
                      args.warmup, args.repeats),
            "online_observe_detect":
                _time(lambda: (adapter.observe(w1), adapter.is_drifted()),
                      args.warmup, args.repeats),
            "hierarchical_rollout_h8":
                _time(lambda: hier.rollout(w1, horizon=8, scene_params=p1),
                      args.warmup, args.repeats),
            "lite_int8_predict":
                _time(lambda: quantized.predict_next(X, scene_params=P),
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
