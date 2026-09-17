"""v3.7.0.dev6 分层神经控制延迟预算量化 A/B。

量化大脑(规划)/小脑(跟踪)/脊髓(反射)三层**合成可测**单次延迟, 并做
频率-精度-延迟三维 A/B (扫 cortex_every 即大脑频率)。

落 benchmarks/results/neural_latency_v3.7.0.json。
analogy, not reproduction —— 延迟为 CPU 合成实测, 非真机总线延迟。

用法:
    python3 scripts/neural_latency_v370.py
"""
import json
import statistics
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

CKPT = ROOT / "checkpoints" / "predictor_v3.7.0.pt"


def _stats(xs):
    if not xs:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "max_ms": None}
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean_ms": round(statistics.mean(xs), 4),
        "p50_ms": round(statistics.median(xs), 4),
        "max_ms": round(max(xs), 4),
    }


def measure(predictor, window, sp, cortex_every, steps=60):
    """跑 steps 步, 收集三层单次延迟 (仅实际执行的层)。"""
    ctrl = HierarchicalController(predictor, cortex_every=cortex_every)
    cortex_ms, cereb_ms, spinal_ms = [], [], []
    for _ in range(steps):
        out = ctrl.step(window, scene_params=sp)
        if out["cortex"].get("ran"):
            cortex_ms.append(out["cortex"]["elapsed_ms"])
        cereb_ms.append(out["cerebellum"]["elapsed_ms"])
        spinal_ms.append(out["spinal"]["elapsed_ms"])
    return {
        "cortex_every": cortex_every,
        "cortex_hz_planned_per_step": round(1.0 / cortex_every, 4),
        "cortex_plan_count": ctrl.run_counts["cortex"],
        "cortex": _stats(cortex_ms),
        "cerebellum": _stats(cereb_ms),
        "spinal": _stats(spinal_ms),
    }


def main():
    predictor, _ = load_predictor(str(CKPT))
    predictor.eval()
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=77)
    window, sp = ds.X[:1], ds.P[:1]

    t0 = time.perf_counter()
    # 三维 A/B: 扫大脑频率 (cortex_every=1,2,5,10)
    sweep = [measure(predictor, window, sp, every)
             for every in (1, 2, 5, 10)]
    wall_s = time.perf_counter() - t0

    # 合同延迟预算 vs 实测 (p50)
    ctrl = HierarchicalController(predictor)
    budget = {
        "cortex_budget_ms": ctrl.cortex.latency_budget_ms,
        "cerebellum_budget_ms": ctrl.cerebellum.latency_budget_ms,
        "spinal_budget_ms": ctrl.spinal.latency_budget_ms,
    }
    fastest = sweep[0]
    within_budget = {
        "cortex_p50_within_budget": (fastest["cortex"]["p50_ms"]
                                     <= budget["cortex_budget_ms"]),
        "cerebellum_p50_within_budget": (fastest["cerebellum"]["p50_ms"]
                                         <= budget["cerebellum_budget_ms"]),
        "spinal_p50_within_budget": (fastest["spinal"]["p50_ms"]
                                     <= budget["spinal_budget_ms"]),
    }

    summary = {
        "version": __version__,
        "feature": "hierarchical_neural_control_latency",
        "analogy_not_reproduction": True,
        "main_params": 52191,
        "wall_seconds": round(wall_s, 2),
        "latency_budget_contract": budget,
        "within_budget_p50": within_budget,
        "frequency_sweep_cortex_every": sweep,
        "priority": {"spinal": 0, "cerebellum": 1, "cortex": 2},
    }
    out = ROOT / "benchmarks" / "results" / "neural_latency_v3.7.0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "version": summary["version"],
        "within_budget_p50": within_budget,
        "cortex_p50_med": fastest["cortex"]["p50_ms"],
        "cerebellum_p50_med": fastest["cerebellum"]["p50_ms"],
        "spinal_p50_med": fastest["spinal"]["p50_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
