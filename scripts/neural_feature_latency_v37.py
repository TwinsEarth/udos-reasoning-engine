"""分层神经控制特征延迟基准 (v3.7.2) — 落 feature_latency_v3.7.0.json。

在合成窗口上测分层神经控制各组件单次延迟 (CPU, torch 2 线程):
cortex 规划 / cerebellum 跟踪 / spinal 反射 / 完整 step / 基线 predict_next。
analogy, not reproduction —— 合成可测延迟, 非真机 WBC 延迟。

用法: python3 scripts/neural_feature_latency_v37.py
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.neural_control import HierarchicalController  # noqa: E402

torch.set_num_threads(2)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    ckpt = ROOT / "checkpoints" / "predictor_v3.7.0.pt"
    model, _ = load_predictor(str(ckpt))
    model.eval()
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=88)
    X, P = ds.X[:1], ds.P[:1]

    ctrl = HierarchicalController(model, cortex_every=1)

    lat = {
        "baseline_predict_next_ms": round(
            bench(lambda: model.predict_next(X, scene_params=P)), 4),
        "neural_full_step_ms": round(bench(lambda: ctrl.step(X, scene_params=P)), 4),
        "cortex_plan_ms": round(
            bench(lambda: ctrl.cortex.profile(
                X[:, -1, :], {"window": X, "scene_params": P})), 4),
        "cerebellum_track_ms": round(
            bench(lambda: ctrl.cerebellum.profile(
                X[:, -1, :], {"target": ctrl._cached_target})), 4),
        "spinal_reflex_ms": round(
            bench(lambda: ctrl.spinal.profile(X[:, -1, :], {})), 4),
    }
    out = {
        "feature": "hierarchical_neural_control_latency",
        "version": __version__,
        "cpu_threads": 2,
        "main_params": sum(p.numel() for p in model.parameters()),
        "priority": {"spinal": 0, "cerebellum": 1, "cortex": 2},
        "analogy_not_reproduction": True,
        "latency_ms": lat,
    }
    p = ROOT / "benchmarks" / "results" / "feature_latency_v3.7.0.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(lat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
