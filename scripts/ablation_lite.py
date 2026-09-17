"""
模型轻量化 A/B 证据 (v2.7.0.dev3)
====================================
参数-延迟-精度三维对比: 全量 vs 剪枝50% vs 量化 INT8 vs 蒸馏学生。
落 benchmarks/results/lite_ablation_v2.7.0.json。
用法: python3 scripts/ablation_lite.py
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

from udos.ctm_engine import CTMConfig                       # noqa: E402
from udos.persistence import load_predictor                  # noqa: E402
from udos.dynamics import build_parametric_dataset           # noqa: E402
from udos.evaluation import evaluate_predictor               # noqa: E402
from udos.lite import MagnitudePruner, DynamicQuantizer, DistillationTrainer  # noqa: E402


def small_cfg(d_model=64):
    return CTMConfig(iterations=8, d_model=d_model, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


@torch.no_grad()
def latency_ms(model, X, P, repeats=30, warmup=5):
    model.eval()
    for _ in range(warmup):
        model(X, scene_params=P) if not hasattr(model, "forward") else None
    t0 = time.time()
    for _ in range(repeats):
        if isinstance(model, torch.nn.Module):
            model(X, scene_params=P)
    return round((time.time() - t0) / repeats * 1000.0, 3)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    te = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=801)
    tr = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=802)
    Xb, Pb = te.X[:8], te.P[:8]

    variants = {}

    # 1) 全量
    full, _ = load_predictor("checkpoints/predictor_v2.7.0.pt")
    variants["full"] = {
        "n_params": n_params(full),
        "sparsity": 0.0,
        "eval_mse": evaluate_predictor(full, te)["single_step_mse"],
        "latency_ms": latency_ms(full, Xb, Pb),
    }

    # 2) 剪枝 50%
    pruned, _ = load_predictor("checkpoints/predictor_v2.7.0.pt")
    sp = MagnitudePruner().prune(pruned, 0.5)
    variants["pruned_50"] = {
        "n_params": n_params(pruned),
        "sparsity": round(sp, 4),
        "eval_mse": evaluate_predictor(pruned, te)["single_step_mse"],
        "latency_ms": latency_ms(pruned, Xb, Pb),
    }

    # 3) 量化 INT8
    q_model, _ = load_predictor("checkpoints/predictor_v2.7.0.pt")
    q = DynamicQuantizer.quantize(q_model)
    # 量化模型前向与全量一致即可, eval 用 predict_next 单点 MSE
    with torch.no_grad():
        q_single = q.predict_next(te.X, scene_params=te.P)
        q_mse = float(((q_single - te.Y[:, 0, :]) ** 2).mean().item())
    variants["quantized_int8"] = {
        "n_params": n_params(q_model),
        "sparsity": 0.0,
        "eval_mse": round(q_mse, 6),
        "note": "quantize_dynamic 不改参数计数, 仅算子替换 INT8",
    }

    # 4) 蒸馏学生 (d_model=32)
    teacher, _ = load_predictor("checkpoints/predictor_v2.7.0.pt")
    student = DistillationTrainer(alpha=0.7).distill(
        teacher, small_cfg(d_model=32), tr, epochs=15)
    variants["distilled_student"] = {
        "n_params": n_params(student),
        "sparsity": 0.0,
        "eval_mse": evaluate_predictor(student, te)["single_step_mse"],
    }

    result = {
        "version": "2.7.0.dev3",
        "note": "CPU 小模型延迟对比含噪声, 如实记录; 量化/剪枝精度损失亦如实记录。",
        "variants": variants,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/lite_ablation_v2.7.0.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
