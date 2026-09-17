#!/usr/bin/env python3
"""
v2.3 F2 补充: 多时域损失权重的"训练口径敏感性"对照
==================================================
ablation_horizon_weight.py 在【口径A: 无早停 / 45ep / train=1000+seed / test=2000+seed
/ n=32】下得到 uniform 3/3 更优。本脚本在【口径B: 早停 patience=12 / 60ep /
train=seed42 split / 独立 test=seed2718 / n=48】(即正式 build 口径) 下, 让 front 与
uniform 从**同一初始 state_dict** 对照, 检验 uniform 优势是否跨口径成立。

结论先行(实测): 口径B 下 front 反超 uniform —— 权重方案优劣对训练口径敏感、不稳健,
因此 v2.3 不把 uniform 设为默认, 三方案均 opt-in, 默认 front 逐位复现 2.2.1。

输出: benchmarks/results/horizon_weight_sensitivity_v2.3.0.json
用法: python3 scripts/sensitivity_step_weight_pipeline.py [--quick]
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


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def run(scheme, init_state, tr, te, epochs, patience):
    m = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    m.load_state_dict(copy.deepcopy(init_state))
    CTMTrainer(m, TrainConfig(epochs=epochs, patience=patience,
                              step_weight_scheme=scheme)).train(tr, te)
    r = evaluate_predictor(m, te)
    curve = r["rollout_mse_curve"]
    return {"single_step_mse": r["single_step_mse"],
            "tail_mse": round(sum(curve[2:]) / len(curve[2:]), 6),
            "growth_x": r["rollout_growth_x"], "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    epochs = 12 if args.quick else 60
    patience = None if args.quick else 12
    n = 16 if args.quick else 48
    t0 = time.time()

    trf = build_parametric_dataset(n_per_kind=n, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=42)
    tr, _ = trf.split(0.8)
    tef = build_parametric_dataset(n_per_kind=n, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=2718)
    _, te = tef.split(0.8)
    init = copy.deepcopy(PhysicsPredictor(small_cfg(),
                                          scene_param_dim=4).state_dict())
    runs = {s: run(s, init, tr, te, epochs, patience) for s in ("front", "uniform")}
    f, u = runs["front"], runs["uniform"]
    pipeline_b = {
        "setting": "early-stop patience=12 / 60ep / train=seed42 split / "
                   "indep-test=seed2718 / n_per_kind=48",
        "front": f, "uniform": u,
        "uniform_vs_front_single_pct": round(
            100 * (u["single_step_mse"] - f["single_step_mse"])
            / max(f["single_step_mse"], 1e-12), 2),
        "uniform_vs_front_tail_pct": round(
            100 * (u["tail_mse"] - f["tail_mse"]) / max(f["tail_mse"], 1e-12), 2),
        "uniform_better_in_pipeline_B": u["tail_mse"] < f["tail_mse"],
    }
    # 引用口径A(无早停多种子)结论做跨口径对比
    ab_path = ROOT / "benchmarks" / "results" / \
        "horizon_weight_ablation_v2.3.0.json"
    pipeline_a_summary = None
    if ab_path.exists():
        ab = json.loads(ab_path.read_text())
        pipeline_a_summary = {
            "setting": "no-early-stop / 45ep / train=1000+seed / "
                       "test=2000+seed / n=32",
            "summary": ab["summary"],
            "uniform_tail_better_seeds":
                ab["summary"]["uniform"]["tail_better_seeds"]}

    cross_consistent = (pipeline_b["uniform_better_in_pipeline_B"] and
                        pipeline_a_summary is not None and
                        pipeline_a_summary["uniform_tail_better_seeds"].split("/")[0]
                        == pipeline_a_summary["uniform_tail_better_seeds"].split("/")[1])
    result = {
        "experiment": "v2.3 F2 step-weight cross-pipeline sensitivity",
        "single_delta": "step_weight_scheme only, shared init within pipeline",
        "pipeline_A_no_earlystop_multiseed": pipeline_a_summary,
        "pipeline_B_build_earlystop": pipeline_b,
        "uniform_advantage_consistent_across_pipelines": bool(cross_consistent),
        "decision": ("uniform 优势仅在口径A成立、口径B被front反超 -> 口径敏感、不稳健; "
                     "默认保持 front(=2.2.1), uniform/back opt-in"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "benchmarks" / "results" / \
        "horizon_weight_sensitivity_v2.3.0.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"pipeline_B": pipeline_b,
                      "cross_consistent": result[
                          "uniform_advantage_consistent_across_pipelines"],
                      "decision": result["decision"],
                      "elapsed_sec": result["elapsed_sec"]},
                     ensure_ascii=False, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
