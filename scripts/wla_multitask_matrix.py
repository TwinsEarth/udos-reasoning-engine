"""
3.9.2 多任务统一头评测矩阵 (WLA 16-benchmark 缩微类比)
================================================================
在合成任务族 (动力学 kind) 上, 用 EmbodiedReasoningHead 统一头跑每族结构化代理,
产出 "任务族 × ER 代理" 评测矩阵 (对应官方广口径多任务评测的缩微类比)。
落 benchmarks/results/wla_multitask_matrix.json。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.embodied import EmbodiedReasoningHead  # noqa: E402

torch.set_num_threads(2)


def main():
    from udos import load_predictor
    model, _ = load_predictor("checkpoints/predictor_v3.9.0.pt")
    head = EmbodiedReasoningHead(model, enable=True)

    ds = build_parametric_dataset(n_per_kind=32, n_steps=14, window=6,
                                 horizon=4, dt=0.5, seed=2718)
    kinds = list(ds.kinds)
    uniq = sorted(set(kinds))

    matrix = []
    for k in uniq:
        idx = torch.tensor([i for i, x in enumerate(kinds) if x == k])
        if idx.numel() == 0:
            continue
        W = ds.X[idx]
        out = head(W, scene_params=ds.P[idx])
        matrix.append({
            "task_kind": k, "n_samples": int(idx.numel()),
            "spatial_rel_norm": round(float(out["spatial_relation"].norm(dim=-1).mean()), 4),
            "target_point_norm": round(float(out["target_point"].norm(dim=-1).mean()), 4),
            "traj_proxy_norm": round(float(out["trajectory_proxy"].norm(dim=-1).mean()), 4),
        })

    overall = evaluate_predictor(model, ds)
    summary = {
        "n_task_kinds": len(matrix),
        "matrix": matrix,
        "overall_single_step_mse": overall["single_step_mse"],
        "main_params_untouched": True,
        "analogy_not_reproduction": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/wla_multitask_matrix.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"n_task_kinds": len(matrix),
                      "overall_single_step_mse": overall["single_step_mse"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
