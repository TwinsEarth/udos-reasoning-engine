"""
构建 v4.1.9 最终正式物理预测器 checkpoint (自规划自监督线终点)
====================================================================
红线: 不重训不改主权重。内化冻结 v4.1.0 主预测器 (eval_mse 锚点 0.045556,
主参 52191)。落 predictor_v4.1.9.pt (第 29 代)。
全线新机制 (课程/自验证器/伪信号/目标分解/停止判据) 纯前向外挂零梯度自检,
不入主 state_dict; 29 代 backcompat 清单 + 逐代可加载校验。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, save_predictor, load_predictor,  # noqa: E402
                  CurriculumGenerator, SolvabilityVerifier,
                  PWMConsistencyPseudoLabeler, GoalDecomposer,
                  StopCorrectController)
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR_EVAL_MSE = 0.045556


def main():
    model, meta = load_predictor("checkpoints/predictor_v4.1.0.pt")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 52191

    ds = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    eval_mse = round(evaluate_predictor(model, ds)["single_step_mse"], 6)
    assert abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6

    # --- 全线机制零梯度终检 (只读主预测器) --- #
    gen = CurriculumGenerator(base_seed=4190)
    lesson = gen.make_lesson(1)
    ver = SolvabilityVerifier(model)
    sol = ver.verify_dataset(lesson["dataset"], max_samples=16, horizon=2)
    pl = PWMConsistencyPseudoLabeler(model, fit_dataset=lesson["dataset"],
                                     fit_epochs=8)
    plout = pl.pseudo_label(lesson["dataset"].X[0:1], horizon=2,
                            scene_params=lesson["dataset"].P[0:1])
    de = GoalDecomposer(model, horizon=2)
    plan = de.decompose(lesson["dataset"].X[0:1], lesson["dataset"].Y[0, -1, :],
                        n_subgoals=3, scene_params=lesson["dataset"].P[0:1])
    scc = StopCorrectController(de, confidence_fn=lambda w: 0.9)
    dec = scc.decide(lesson["dataset"].X[0:1], lesson["dataset"].Y[0, -1, :],
                     n_subgoals=3, scene_params=lesson["dataset"].P[0:1])
    assert abs(evaluate_predictor(model, ds)["single_step_mse"] - eval_mse) < 1e-9

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v4.1.9.pt"
    save_predictor(model, ckpt, metrics={
        "evaluation": evaluate_predictor(model, ds),
        "inherits_from": "checkpoints/predictor_v4.1.0.pt"})

    loaded, meta2 = load_predictor(ckpt)
    assert abs(evaluate_predictor(loaded, ds)["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__
    assert sum(p.numel() for p in loaded.parameters()) == 52191

    # 全 29 代 backcompat: 逐件可加载 + 主参一致
    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    assert len(ckpt_list) == 29, f"expected 29 checkpoints, got {len(ckpt_list)}"
    loadable = []
    for name in ckpt_list:
        m, _ = load_predictor(f"checkpoints/{name}")
        assert sum(p.numel() for p in m.parameters()) == 52191
        loadable.append(name)

    summary = {
        "version": __version__, "n_params": n_params,
        "inherits_frozen_main_from": "predictor_v4.1.0.pt",
        "eval_mse": eval_mse, "anchor_eval_mse": ANCHOR_EVAL_MSE,
        "eval_mse_matches_anchor": abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_all_loadable": len(loadable) == len(ckpt_list),
        "backcompat_list": ckpt_list,
        "line_selfcheck": {
            "solvable_ratio": sol["solvable_ratio"],
            "pwm_mean_consistency": plout["mean_consistency"],
            "goal_chain_len": len(plan["subgoal_chain"]),
            "stopcorrect_decision": dec["decision"]},
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v4.1.9.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints",
                       "backcompat_all_loadable", "line_selfcheck"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
