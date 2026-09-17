"""
v3.0 新特性推理延迟基准 (future_multimodal / eval5d)
=====================================================
对加载好的 predictor_v3.0.0.pt, 测量:
    * predict_next            (基线)
    * MultiTaskHead + FutureMultimodalHead (三模态代理)
    * RGBProxyHead / DepthProxyHead / MaskProxyHead (独立头)
    * FiveDimensionEvaluator.dim1_state_reconstruction (单维评测, 轻量)
落 benchmarks/results/feature_latency_v3.0.0.json (p50/p95, 毫秒)。
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
from udos.multitask import MultiTaskHead  # noqa: E402
from udos.future_multimodal import (FutureMultimodalHead, RGBProxyHead,
                                    DepthProxyHead, MaskProxyHead)  # noqa: E402
from udos.eval_suite import FiveDimensionEvaluator  # noqa: E402
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
    predictor, _ = load_predictor("checkpoints/predictor_v3.0.0.pt")
    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=9191)
    wb, pb = ds.X[:1], ds.P[:1]

    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    fmh = FutureMultimodalHead(32, horizon=4)
    mth.register_head("future_mm", fmh)

    rgb = RGBProxyHead(32, horizon=4)
    dep = DepthProxyHead(32, horizon=4)
    msk = MaskProxyHead(32, horizon=4)
    latent = mth.encode(wb, scene_params=pb)

    ev = FiveDimensionEvaluator(predictor, seed=2025, n_per_kind=8)

    # warmup
    for _ in range(5):
        predictor.predict_next(wb, scene_params=pb)
        mth.forward(wb, scene_params=pb)
        rgb(latent); dep(latent); msk(latent)

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
        "version": "3.0.1",
        "n_repeats": N,
        "predict_next": bench(lambda: predictor.predict_next(wb, scene_params=pb)),
        "future_multimodal_head": bench(lambda: mth.forward(wb, scene_params=pb)),
        "rgb_proxy_head": bench(lambda: rgb(latent)),
        "depth_proxy_head": bench(lambda: dep(latent)),
        "mask_proxy_head": bench(lambda: msk(latent)),
        "eval5d_dim1": bench(lambda: ev.dim1_state_reconstruction()),
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v3.0.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
