"""
v4.1.0.dev2 自生成课程 vs 固定课程 A/B
=========================================
在冻结主预测器 (predictor_v4.1.0.pt, 52191 参) 上, 同探针预算下比较:
    A 自生成课程: CurriculumGenerator.auto_progression (stage 自适应难度)
                  + auto_expand_horizon (自动找难度前沿)
    B 固定课程:   单一固定 horizon/spread 平铺课程 (不随 stage 扩张)

诚实指标 (可复算):
    * 匹配探针预算 (n_probe 相同) 下, 各自覆盖的"可解难度档位数"与平均可解率;
    * 自生成课程额外报难度前沿 frontier_horizon (自验证器探测到的最难可解 horizon);
    * 主权重只读不改 (eval_mse 锚点保持)。
落 benchmarks/results/curriculum_self_vs_fixed_v4.1.0.json。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, load_predictor, CurriculumGenerator,  # noqa: E402
                  SolvabilityVerifier)
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR = 0.045556


def main():
    model, meta = load_predictor("checkpoints/predictor_v4.1.0.pt")
    assert sum(p.numel() for p in model.parameters()) == 52191
    gen = CurriculumGenerator(base_seed=4102)
    ver = SolvabilityVerifier(model)

    # --- A: 自生成自适应课程 (stage 0..5 难度自动递进) --- #
    prog = gen.auto_progression(ver, [0, 1, 2, 3, 4, 5], n_probe=12)
    a_mean_ratio = round(sum(t["solvable_ratio"] for t in prog) / len(prog), 6)
    a_covered = sum(1 for t in prog if t["solvable_ratio"] >= 0.8)
    frontier = gen.auto_expand_horizon(ver, start=2, step=1, max_h=8,
                                       solvable_floor=0.8, n_probe=12)

    # --- B: 固定课程 (固定 horizon=2 平铺 6 档, 不扩张难度) --- #
    fixed = []
    for stage in range(6):
        ds = build_parametric_dataset(n_per_kind=3, n_steps=14, window=6,
                                      horizon=2, dt=0.5, seed=4102 + stage)
        r = ver.verify_dataset(ds, max_samples=12, horizon=2)
        fixed.append({"stage": stage, "horizon": 2,
                      "solvable_ratio": r["solvable_ratio"]})
    b_mean_ratio = round(sum(t["solvable_ratio"] for t in fixed) / len(fixed), 6)
    b_covered = sum(1 for t in fixed if t["solvable_ratio"] >= 0.8)

    # 主权重未改
    ds_anchor = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                         horizon=4, dt=0.5, seed=2718)
    mse = round(evaluate_predictor(model, ds_anchor)["single_step_mse"], 6)
    assert abs(mse - ANCHOR) < 1e-6

    summary = {
        "version": __version__, "ab": "self_generated_vs_fixed_curriculum",
        "n_probe_per_stage": 12,
        "self_generated": {
            "mean_solvable_ratio": a_mean_ratio,
            "covered_stages_>=0.8": a_covered,
            "progression": prog,
            "difficulty_frontier": frontier,
        },
        "fixed": {
            "mean_solvable_ratio": b_mean_ratio,
            "covered_stages_>=0.8": b_covered,
            "progression": fixed,
        },
        "delta_mean_ratio": round(a_mean_ratio - b_mean_ratio, 6),
        "self_reaches_harder_frontier": frontier["frontier_horizon"] > 2,
        "eval_mse_anchor_kept": mse,
        "main_params_untouched": True,
        "analogy_not_reproduction": True,
    }
    out = Path("benchmarks/results/curriculum_self_vs_fixed_v4.1.0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "self_generated", "fixed",
                       "delta_mean_ratio", "self_reaches_harder_frontier",
                       "eval_mse_anchor_kept"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
