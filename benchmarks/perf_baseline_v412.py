#!/usr/bin/env python3
"""UDOS v4.1.2 自规划自监督线性能基准 (只读诊断, 不改主权重)。

测量并落盘 benchmarks/results/perf_baseline_v412.json:
  1. 课程生成 (CurriculumGenerator.make_lesson)
  2. 可解性自验证 (SolvabilityVerifier.verify)
  3. PWM 一致性伪标签 (PWMConsistencyPseudoLabeler)
  4. 自规划目标分解 (GoalDecomposer.decompose)
  5. 停止/纠正判据 (StopCorrectController.decide)
主参恒 52191; 全部外挂零梯度。
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from udos import (__version__, load_predictor,  # noqa: E402
                  CurriculumGenerator, SolvabilityVerifier,
                  PWMConsistencyPseudoLabeler, GoalDecomposer,
                  StopCorrectController)
from udos.dynamics import build_parametric_dataset  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "predictor_v4.1.0.pt")


def pct(vals, p):
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return float(s[f] if f == c else s[f] + (k - f) * (s[c] - s[f]))


def bench(fn, warmup=2, iters=15):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": round(sum(samples) / len(samples), 3),
            "p50_ms": round(pct(samples, 0.5), 3),
            "p95_ms": round(pct(samples, 0.95), 3),
            "iters": iters}


def main():
    torch.set_num_threads(2)
    model, _ = load_predictor(CKPT)
    n_params = sum(p.numel() for p in model.parameters())
    gen = CurriculumGenerator(base_seed=4120)
    ver = SolvabilityVerifier(model)
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=4122)
    w, sp = ds.X[0:1], ds.P[0:1]
    pl = PWMConsistencyPseudoLabeler(model, fit_dataset=ds, fit_epochs=10)
    de = GoalDecomposer(model, horizon=2)
    scc = StopCorrectController(de, confidence_fn=lambda x: 0.9)
    goal = ds.Y[0, -1, :]

    result = {
        "version": __version__, "checkpoint": CKPT,
        "n_params": n_params, "main_params_untouched": True,
        "latency_ms": {
            "curriculum_generate": bench(lambda: gen.make_lesson(2)),
            "solvability_verify": bench(lambda: ver.verify(w, scene_params=sp,
                                                           horizon=2)),
            "pwm_pseudo_label": bench(lambda: pl.pseudo_label(w, horizon=3,
                                                              scene_params=sp)),
            "goal_decompose": bench(lambda: de.decompose(w, goal, n_subgoals=3,
                                                          scene_params=sp)),
            "stopcorrect_decide": bench(lambda: scc.decide(w, goal, n_subgoals=3,
                                                           scene_params=sp)),
        },
    }
    out = os.path.join(ROOT, "benchmarks", "results",
                       "perf_baseline_v412.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["latency_ms"], ensure_ascii=False, indent=2))
    print("n_params", n_params)


if __name__ == "__main__":
    main()
