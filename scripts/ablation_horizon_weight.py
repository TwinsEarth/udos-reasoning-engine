#!/usr/bin/env python3
"""
v2.3 F2 多时域损失权重 同合同 A/B
=================================
唯一变量 = TrainConfig.step_weight_scheme ∈ {front(2.2.1 基线), uniform, back}。
同合同: 对每个 seed, 三种方案从**同一初始 state_dict**、同一训练/测试数据与同一
batch 顺序出发, 仅远期步权重不同。比较单步 MSE、rollout 后段(step2..H-1) MSE 与
累积率 growth_x, 跨多种子看方向是否一致 (吸取 v2.2 SS 收益随种子反转的教训)。

输出: benchmarks/results/horizon_weight_ablation_v2.3.0.json
用法: python3 scripts/ablation_horizon_weight.py [--quick] [--seeds 42 7 123]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (  # noqa: E402
    build_parametric_dataset, PhysicsPredictor, CTMTrainer, TrainConfig,
    CTMConfig, evaluate_predictor,
)

SCHEMES = ("front", "uniform", "back")


def small_ctm_config():
    # 与服务 /train、build_v221 同一小配置, 保证 CPU 可复现
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def fresh_model(scene_param_dim=4):
    return PhysicsPredictor(small_ctm_config(), scene_param_dim=scene_param_dim)


def run_one(init_state, train_ds, test_ds, scheme, epochs, seed):
    model = fresh_model()
    model.load_state_dict(copy.deepcopy(init_state))  # 同一初始权重
    cfg = TrainConfig(epochs=epochs, seed=seed, step_weight_scheme=scheme)
    CTMTrainer(model, cfg).train(train_ds, test_ds, verbose=False)
    rep = evaluate_predictor(model, test_ds)
    curve = rep["rollout_mse_curve"]
    tail = curve[2:]  # 后段远期步
    return {
        "scheme": scheme,
        "single_step_mse": rep["single_step_mse"],
        "rollout_curve": curve,
        "tail_mse": round(sum(tail) / max(len(tail), 1), 6),
        "growth_x": rep["rollout_growth_x"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    epochs = args.epochs or (12 if args.quick else 45)
    n_per_kind = 16 if args.quick else 32
    horizon = 4
    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        train_ds = build_parametric_dataset(
            n_per_kind=n_per_kind, n_steps=14, window=6,
            horizon=horizon, dt=0.5, seed=1000 + seed)
        tr, _ = train_ds.split(0.8)
        test_ds = build_parametric_dataset(
            n_per_kind=n_per_kind, n_steps=14, window=6,
            horizon=horizon, dt=0.5, seed=2000 + seed)
        _, te = test_ds.split(0.8)
        base = fresh_model()                       # 该 seed 的共享初始权重
        init_state = copy.deepcopy(base.state_dict())
        rows = [run_one(init_state, tr, te, s, epochs, seed) for s in SCHEMES]
        per_seed.append({"seed": seed, "runs": rows})
        print(f"seed={seed} " + " | ".join(
            f"{r['scheme']}:single={r['single_step_mse']:.4f},"
            f"tail={r['tail_mse']:.4f},growth={r['growth_x']:.2f}"
            for r in rows))

    # 跨种子方向裁决: back/uniform 相对 front 的 tail/growth 是否一致更低
    def mean_over_seeds(scheme, key):
        vals = [r[key] for ps in per_seed for r in ps["runs"]
                if r["scheme"] == scheme]
        return sum(vals) / len(vals)

    summary = {s: {"single_step_mse": mean_over_seeds(s, "single_step_mse"),
                   "tail_mse": mean_over_seeds(s, "tail_mse"),
                   "growth_x": mean_over_seeds(s, "growth_x")}
               for s in SCHEMES}
    front_tail = summary["front"]["tail_mse"]
    for s in ("uniform", "back"):
        summary[s]["tail_delta_vs_front_pct"] = round(
            100 * (summary[s]["tail_mse"] - front_tail) / max(front_tail, 1e-12), 2)
        # 方向一致的种子数 (tail 严格更低)
        wins = 0
        for ps in per_seed:
            rf = next(r for r in ps["runs"] if r["scheme"] == "front")
            rx = next(r for r in ps["runs"] if r["scheme"] == s)
            wins += int(rx["tail_mse"] < rf["tail_mse"])
        summary[s]["tail_better_seeds"] = f"{wins}/{len(per_seed)}"

    result = {
        "experiment": "v2.3 F2 horizon step-weight ablation",
        "single_delta": "step_weight_scheme only; shared init state/data/order",
        "epochs": epochs, "n_per_kind": n_per_kind, "horizon": horizon,
        "seeds": args.seeds, "per_seed": per_seed, "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "benchmarks" / "results" / "horizon_weight_ablation_v2.3.0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nsummary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {out}  ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
