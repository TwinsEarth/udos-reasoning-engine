"""
ICM 三路线对照实验 (v3.4.0.dev6)
================================================================
同合成基准上对照三条 scaling 路线 (明确合成类比, 非复现 LLM scaling law):
    1. 数据 Scaling   : 增加训练数据量 (n_per_kind=16/32/48) -> eval_mse;
    2. 思维链 Scaling : CTM 内部 tick 数 (iterations=4/8/16) -> 精度/延迟;
    3. 上下文 Scaling : ICM 演示数 k=0/1/3/5/10 -> 精度/延迟。
样本-性能-延迟三维对照, 落 benchmarks/results/icm_three_route_v3.4.0.json。
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

from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos import load_predictor  # noqa: E402
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator  # noqa: E402


def small_cfg(iterations=8):
    return CTMConfig(iterations=iterations, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def quick_train(n_per_kind=32, iterations=8, epochs=25):
    tr_full = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                       window=6, horizon=4, dt=0.5, seed=42)
    tr, te = tr_full.split(0.8)
    m = PhysicsPredictor(small_cfg(iterations), scene_param_dim=4)
    CTMTrainer(m, TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                              patience=12, step_weight_scheme="front",
                              hybrid_weight=0.0)).train(tr, te)
    ind = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                   window=6, horizon=4, dt=0.5, seed=2718)
    t0 = time.perf_counter()
    ev = evaluate_predictor(m, ind)
    lat = (time.perf_counter() - t0) / max(len(ind), 1) * 1000.0
    return ev["single_step_mse"], lat


def main():
    # ---- 路线 1: 数据 Scaling ----
    data_route = []
    for npk in (16, 32, 48):
        mse, lat = quick_train(n_per_kind=npk, iterations=8)
        data_route.append({"n_per_kind": npk, "mse": round(mse, 6),
                           "latency_ms": round(lat, 4)})
        print(f"[data] n_per_kind={npk} mse={mse:.5f}")

    # ---- 路线 2: 思维链 (tick) Scaling ----
    cot_route = []
    for it in (4, 8, 16):
        mse, lat = quick_train(n_per_kind=32, iterations=it)
        cot_route.append({"iterations": it, "mse": round(mse, 6),
                          "latency_ms": round(lat, 4)})
        print(f"[cot] iterations={it} mse={mse:.5f} lat={lat:.2f}ms")

    # ---- 路线 3: 上下文 Scaling (ICM k) ----
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
    idx = torch.arange(min(60, len(te)))
    ctx_route = []
    for k in (0, 1, 3, 5, 10):
        t0 = time.perf_counter()
        mse = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=k,
                           scene_params=te.P[idx])
        lat = (time.perf_counter() - t0) / len(idx) * 1000.0
        ctx_route.append({"k": k, "mse": round(mse, 6),
                          "latency_ms": round(lat, 4)})
        print(f"[ctx] k={k} mse={mse:.5f} lat={lat:.2f}ms")

    out = {
        "experiment": "icm_three_route",
        "analogy_not_reproduction": True,
        "note": "合成数据上三路线对照; 非 LLM scaling law 复现",
        "data_route": data_route,
        "cot_route": cot_route,
        "context_route": ctx_route,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/icm_three_route_v3.4.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved benchmarks/results/icm_three_route_v3.4.0.json")


if __name__ == "__main__":
    main()
