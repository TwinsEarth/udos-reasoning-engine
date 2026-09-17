"""
PWM 世界模型特征延迟基准 (v3.6.2) — 落 benchmarks/results/feature_latency_v3.6.0.json
================================================================================
在合成 rollout 上测各 WM 模块单次延迟 (CPU, torch 2 线程), 与 3.5 线同口径。
analogy, not reproduction —— 合成低维潜在代理。

用法: python3 scripts/wm_feature_latency_v36.py
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.world_model import LatentWorldModel  # noqa: E402
from udos.wm_events import ContactPredictor  # noqa: E402
from udos.wm_conservation import ConservationChecker  # noqa: E402

torch.set_num_threads(2)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    ckpt = ROOT / "checkpoints" / "predictor_v3.6.0.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "predictor_v3.5.0.pt"
    model, _ = load_predictor(str(ckpt))
    model.eval()
    wm = LatentWorldModel(model)
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=61)
    wm.fit(ds, epochs=10)
    te = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=62)
    X, P = te.X[:8], te.P[:8]

    cd = ContactPredictor()
    chk = ConservationChecker(mass=1.0)
    real = model.rollout(X, 4, scene_params=P)

    lat = {
        "wm_encode_latent_ms": round(bench(lambda: wm.encode_latent(X, scene_params=P)), 4),
        "wm_transit_step_ms": round(bench(lambda: wm.transit_step(wm.encode_latent(X, scene_params=P))), 4),
        "wm_imagine_h4_ms": round(bench(lambda: wm.imagine(X, 4, scene_params=P)), 4),
        "wm_imagine_uncertain_ms": round(
            bench(lambda: wm.imagine_uncertain(X, 4, n_samples=5, noise_scale=0.1,
                                              scene_params=P)), 4),
        "wm_contact_predict_ms": round(bench(lambda: cd.predict(real)), 4),
        "wm_conservation_check_ms": round(bench(lambda: chk.check(real)), 4),
        "baseline_predictor_rollout_h4_ms": round(
            bench(lambda: model.rollout(X, 4, scene_params=P)), 4),
    }

    out = {
        "feature": "pwm_world_model_latency",
        "version": "3.6.2",
        "cpu_threads": 2,
        "wm_params": wm.n_params,
        "main_params": sum(p.numel() for p in model.parameters()),
        "latency_ms": lat,
    }
    p = ROOT / "benchmarks" / "results" / "feature_latency_v3.6.0.json"
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(lat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
