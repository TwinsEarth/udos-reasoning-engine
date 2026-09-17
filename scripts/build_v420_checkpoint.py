"""
构建 v4.2.0 正式物理预测器 checkpoint (完全自训练线起点: 自生成三元组骨架)
==========================================================================
红线: 不重训不改主权重。内化冻结 v4.1.0 主预测器 (eval_mse 锚点 0.045556,
主参 52191)。落 predictor_v4.2.0.pt (第 30 代)。
自生成三元组骨架 (TransitionTripletGenerator) 纯前向外挂零梯度自检,
不入主 state_dict; 30 代 backcompat 清单 + 逐代可加载校验。
v4.2.0 仅交付骨架与可复算质量度量, 不宣称任何 student 改进 (防退化纪律)。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, save_predictor, load_predictor,  # noqa: E402
                  TransitionTripletGenerator)
from udos.world_model import LatentWorldModel  # noqa: E402
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

    # --- 自生成三元组骨架零梯度自检 (只读主预测器) --- #
    wm = LatentWorldModel(model, action_dim=0)
    wm.fit(ds, epochs=20, lr=1e-2, seed=420)
    gen = TransitionTripletGenerator(model, wm, action_dim=1)
    trip = gen.generate(ds, n=64, seed=4200)
    quality = gen.triplet_quality(trip)
    # 主预测器权重须零改动
    assert abs(evaluate_predictor(model, ds)["single_step_mse"] - eval_mse) < 1e-9

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v4.2.0.pt"
    save_predictor(model, ckpt, metrics={
        "evaluation": evaluate_predictor(model, ds),
        "inherits_from": "checkpoints/predictor_v4.1.0.pt",
        "self_train_triplet_quality": quality})

    loaded, meta2 = load_predictor(ckpt)
    assert abs(evaluate_predictor(loaded, ds)["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__
    assert sum(p.numel() for p in loaded.parameters()) == 52191

    # 全 30 代 backcompat: 逐件可加载 + 主参一致
    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    assert len(ckpt_list) == 30, f"expected 30 checkpoints, got {len(ckpt_list)}"
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
        "self_generated_triplets": {
            "n": quality["n"],
            "action_spread": quality["action_spread"],
            "next_state_std": quality["next_state_std"],
            "transition_norm": quality["transition_norm"],
            "momentum_residual": quality.get("momentum_residual"),
        },
        "claim": "骨架交付, 不宣称 RSI 改进 (防退化纪律)",
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v4.2.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints",
                       "backcompat_all_loadable", "self_generated_triplets"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
