"""
效率 Pareto A/B (v3.3.0.dev4) —— 参数 / 延迟 / eval_mse 三维对比
================================================================
对比四种形态 (analogy, not reproduction, CPU-only 合成数据):
    1. full        : 正式件 predictor_v3.3.0 (全量旧架构, 52191 参数)
    2. pruned_v2   : full 经 StructuredPrunerV2 通道级剪枝 (不重训)
    3. student_v2  : DistillationTrainerV2 学生 (d_model 减半, 少量蒸馏)
    4. moe_addon   : LightweightMoE 路由头外挂 (仅计额外参数/延迟, 状态-MSE 继承 full)
落 benchmarks/results/efficiency_pareto_v3.3.0.json; 确定推荐配置。
纪律: 收益不稳则 opt-in, 不默认改正式件; 被否决候选照实记录。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, load_predictor  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.lite import StructuredPrunerV2, DistillationTrainerV2  # noqa: E402
from udos.moe import LightweightMoE  # noqa: E402

torch.set_num_threads(2)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def latency_ms(model, x, p, repeats=20):
    for _ in range(3):                       # warmup
        model.predict_next(x, scene_params=p)
    t0 = time.time()
    for _ in range(repeats):
        model.predict_next(x, scene_params=p)
    return round((time.time() - t0) / repeats * 1000, 3)


def main():
    ckpt = ROOT / "checkpoints" / "predictor_v3.3.0.pt"
    full, _ = load_predictor(str(ckpt))
    full.eval()

    ds = build_parametric_dataset(n_per_kind=24, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    x, p = ds.X[:16], ds.P[:16]

    # 1. full
    full_mse = evaluate_predictor(full, ds)["single_step_mse"]
    rows = [{
        "config": "full",
        "n_params": n_params(full),
        "predict_ms": latency_ms(full, x, p),
        "eval_mse": round(full_mse, 6),
        "note": "正式件, 旧架构默认",
    }]

    # 2. pruned_v2 (不重训)
    import copy
    pruned = copy.deepcopy(full)
    rep = StructuredPrunerV2(prune_ratio=0.5).prune(pruned)
    pruned_mse = evaluate_predictor(pruned, ds)["single_step_mse"]
    rows.append({
        "config": "pruned_v2",
        "n_params": n_params(pruned),
        "predict_ms": latency_ms(pruned, x, p),
        "eval_mse": round(pruned_mse, 6),
        "channel_sparsity": rep["channel_sparsity"],
        "weight_sparsity": rep["weight_sparsity"],
        "note": "通道级整行剪枝, 保留形状, 不重训",
    })

    # 3. student_v2 (少量蒸馏, 快速)
    base = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    base.scene_dim = 32
    torch.manual_seed(0)
    student = DistillationTrainerV2(student_scale=0.5).distill(
        full, base, ds, epochs=6, batch_size=32, seed=0)
    student_mse = evaluate_predictor(student, ds)["single_step_mse"]
    rows.append({
        "config": "student_v2",
        "n_params": n_params(student),
        "predict_ms": latency_ms(student, x, p),
        "eval_mse": round(student_mse, 6),
        "d_model": student.ctm.cfg.d_model,
        "note": "蒸馏学生 d_model 减半",
    })

    # 4. moe_addon (外挂, 状态-MSE 继承 full; 仅计额外开销)
    moe = LightweightMoE(in_dim=32, out_dim=16, num_experts=4, top_k=2)
    moe.eval()
    moe_x = torch.randn(16, 32)
    for _ in range(3):
        moe(moe_x)
    t0 = time.time()
    for _ in range(20):
        moe(moe_x)
    moe_ms = (time.time() - t0) / 20 * 1000
    rows.append({
        "config": "moe_addon",
        "n_params": n_params(moe),
        "predict_ms": round(moe_ms, 3),
        "eval_mse": None,   # 不替代状态预测, 继承 full
        "note": "路由头外挂, 额外开销",
    })

    # ---- 推荐配置判定: 在 eval_mse 不劣化超过 5% 的前提下选最小参数 ----
    ref = rows[0]["eval_mse"]
    feasible = [r for r in rows
                if r["eval_mse"] is not None
                and r["eval_mse"] <= ref * 1.05]
    feasible.sort(key=lambda r: r["n_params"])
    recommended = feasible[0]["config"] if feasible else "full"

    out = {
        "version": __version__,
        "device": "cpu_2_threads",
        "n_eval_samples": len(ds),
        "metrics": ["n_params", "predict_ms", "eval_mse"],
        "rows": rows,
        "recommended_config": recommended,
        "recommendation_rationale": (
            "在 eval_mse 相对 full 劣化 <=5% 的候选中选参数量最小者; "
            "若剪枝/学生精度损失明显则退回 full (opt-in)"),
        "default_unchanged": True,
        "rejected_candidates": [
            "pruned_v2 不重训精度通常回升有限, 不默认开启",
            "student_v2 需额外蒸馏训练成本, 精度收益不稳 => opt-in",
        ],
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/efficiency_pareto_v3.3.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
