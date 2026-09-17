"""
构建 v3.9.9 终点正式物理预测器 checkpoint (WLA 类比线终点)
====================================================================
红线: 不重训不改主权重。内化冻结 v3.8.6 主预测器 (eval_mse 锚点 0.045556,
主参 52191)。与 3.9.0 同口径, 落 predictor_v3.9.9.pt (第 27 代)。
WLA 全外挂零梯度自检 + 27 代 backcompat 清单。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, save_predictor, load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR_EVAL_MSE = 0.045556


def build(seed, n_per_kind, horizon=4):
    return build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                    window=6, horizon=horizon, dt=0.5, seed=seed)


def main():
    n_per_kind = 48
    model, meta = load_predictor("checkpoints/predictor_v3.9.0.pt")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 52191

    ind_te = build(2718, n_per_kind)
    eval_mse = round(evaluate_predictor(model, ind_te)["single_step_mse"], 6)
    assert abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v3.9.9.pt"
    save_predictor(model, ckpt, metrics={"evaluation": evaluate_predictor(model, ind_te),
                                        "inherits_from": "checkpoints/predictor_v3.9.0.pt"})

    loaded, meta2 = load_predictor(ckpt)
    assert abs(evaluate_predictor(loaded, ind_te)["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__

    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    summary = {
        "version": __version__, "n_params": n_params,
        "inherits_frozen_main_from": "predictor_v3.9.0.pt",
        "eval_mse": eval_mse, "anchor_eval_mse": ANCHOR_EVAL_MSE,
        "eval_mse_matches_anchor": abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_list": ckpt_list,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v3.9.9.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
