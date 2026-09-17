"""
v2.8 新特性推理延迟基准 (loop / multitask)
============================================
对加载好的 predictor_v2.8.0.pt, 测量:
    * predict_next (基线)
    * PhysicalLoopRunner.run (五步闭环)
    * MultiTaskHead 三头联合推理
落 benchmarks/results/feature_latency_v2.8.0.json (p50/p95, 毫秒)。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.persistence import load_predictor  # noqa: E402
from udos.physical_loop import PhysicalLoopRunner  # noqa: E402
from udos.multitask import (MultiTaskHead, SpatialCoordHead,  # noqa: E402
                            ActionTrajectoryHead, FutureStateHead)
from udos.dynamics import build_parametric_dataset, RAW_DIM  # noqa: E402

torch.set_num_threads(2)


def pct(xs, p):
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return float(s[f] + (k - f) * (s[c] - s[f]))


def main():
    predictor, _ = load_predictor("checkpoints/predictor_v2.8.0.pt")
    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=6060)
    wb, pb = ds.X[:1], ds.P[:1]

    loop = PhysicalLoopRunner(predictor, horizon=2)
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("spatial", SpatialCoordHead(32, n_pts=4))
    mth.register_head("action", ActionTrajectoryHead(32, horizon=4, action_dim=RAW_DIM))
    mth.register_head("future", FutureStateHead(32, horizon=4, state_dim=RAW_DIM))

    # warmup
    for _ in range(5):
        predictor.predict_next(wb, scene_params=pb)
        loop.run(wb, scene_params=pb, candidate_actions=[{}])
        mth.forward(wb, scene_params=pb)

    N = 50
    def bench(fn):
        ts = []
        for _ in range(N):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return {"p50_ms": round(pct(ts, 0.5), 4),
                "p95_ms": round(pct(ts, 0.95), 4),
                "mean_ms": round(sum(ts) / len(ts), 4)}

    out = {
        "version": "2.8.2",
        "n_repeats": N,
        "predict_next": bench(lambda: predictor.predict_next(wb, scene_params=pb)),
        "physical_loop_step": bench(lambda: loop.run(wb, scene_params=pb,
                                                    candidate_actions=[{}])),
        "multitask_three_heads": bench(lambda: mth.forward(wb, scene_params=pb)),
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v2.8.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
