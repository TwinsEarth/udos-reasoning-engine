"""
v2.9 新特性推理延迟基准 (retargeting / affordance / spatial_relation)
=====================================================================
对加载好的 predictor_v2.9.0.pt, 测量:
    * predict_next            (基线)
    * ActionRetargeter.retarget (60->4 DOF)
    * AffordanceScorer.score
    * MultiTaskHead + SpatialRelationHead
落 benchmarks/results/feature_latency_v2.9.0.json (p50/p95, 毫秒)。
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
from udos.retargeting import MorphologyLibrary, ActionRetargeter  # noqa: E402
from udos.affordance import AffordanceScorer  # noqa: E402
from udos.multitask import MultiTaskHead, SpatialRelationHead  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402

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
    predictor, _ = load_predictor("checkpoints/predictor_v2.9.0.pt")
    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=9090)
    wb, pb = ds.X[:1], ds.P[:1]

    lib = MorphologyLibrary()
    rt = ActionRetargeter(lib.get("prime_u_60dof"), lib.get("gripper_4dof"))
    src_actions = torch.randn(1, 60)
    scorer = AffordanceScorer(reach_radius=2.0)
    state = torch.zeros(1, 6)
    obj = torch.randn(1, 4, 6)

    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("spatial_rel", SpatialRelationHead(32, n_obj=4))

    # warmup
    for _ in range(5):
        predictor.predict_next(wb, scene_params=pb)
        rt.retarget(src_actions)
        scorer.score(state, obj)
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
        "version": "2.9.2",
        "n_repeats": N,
        "predict_next": bench(lambda: predictor.predict_next(wb, scene_params=pb)),
        "retarget_convert": bench(lambda: rt.retarget(src_actions)),
        "affordance_score": bench(lambda: scorer.score(state, obj)),
        "spatial_relation_head": bench(lambda: mth.forward(wb, scene_params=pb)),
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v2.9.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
