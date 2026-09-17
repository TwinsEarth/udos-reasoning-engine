"""
ICM 上下文预算 延迟-收益 A/B (v3.4.0.dev4)
================================================================
在同合成基准上扫描 budget=4/8/16/32, 记录每档的 ICM k-shot MSE 与
单次推理延迟 (ms), 落 benchmarks/results/icm_budget_ab_v3.4.0.json。
呼应 "8000 步 token 不能全上云" —— 预算越大收益边际递减, 延迟上升。
analogy, not reproduction。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)

from udos import load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator  # noqa: E402
from udos.icm_budget import ContextBudgetManager  # noqa: E402


def main():
    m, _ = load_predictor("checkpoints/predictor_v3.4.0.pt")
    tr = build_parametric_dataset(n_per_kind=32, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4242)
    te = build_parametric_dataset(n_per_kind=24, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=7777)
    mem = DemonstrationMemory()
    agg = ICMAggregator(m, temperature=1.0, lamb=1.0)
    for i in range(len(tr)):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                 scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])

    budgets = [4, 8, 16, 32]
    rows = []
    idx = torch.arange(min(60, len(te)))
    for b in budgets:
        bm = ContextBudgetManager(budget=b)
        k_eff = bm.cap_k(8, mem.size)
        # 延迟: 预热 + 计时
        agg.predict(te.X[0], memory=mem, k=k_eff, scene_params=te.P[0:1])
        t0 = time.perf_counter()
        mse = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=k_eff,
                           scene_params=te.P[idx])
        dt = (time.perf_counter() - t0) / len(idx) * 1000.0
        rows.append({"budget": b, "k_eff": k_eff,
                     "mse": round(mse, 6),
                     "latency_ms_per_query": round(dt, 4)})
        print(f"budget={b:3d} k={k_eff:2d} mse={mse:.6f} lat={dt:.3f}ms")

    out = {
        "feature": "icm_context_budget",
        "analogy_not_reproduction": True,
        "note": "budget=演示数上限; 越大收益边际递减、延迟上升",
        "budgets": rows,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/icm_budget_ab_v3.4.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved benchmarks/results/icm_budget_ab_v3.4.0.json")


if __name__ == "__main__":
    main()
