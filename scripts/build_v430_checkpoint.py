"""
构建 v4.3.0 正式物理预测器 checkpoint (完全自进化线起点: 基础设施自优化)
==========================================================================
红线: 不重训不改主权重。内化冻结 v4.1.0 主预测器 (eval_mse 锚点 0.045556,
主参 52191)。落 predictor_v4.3.0.pt (第 32 代)。

v4.3 全线机制 (配置自优化搜索器/搜索-验证-选用闭环/配置 A/B) 纯前向外挂零梯度
终检, 不入主 state_dict; 配置搜索须过"输出保真"硬门 (不允许靠降质换速度);
32 代 backcompat 清单 + 逐代可加载校验。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, save_predictor, load_predictor,  # noqa: E402
                  ConfigSpec, ConfigEvaluator, ConfigSearcher,
                  SearchVerifySelectLoop, ConfigAB)  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR_EVAL_MSE = 0.045556
EXPECTED_N_CKPTS = 32


def main():
    model, meta = load_predictor("checkpoints/predictor_v4.1.0.pt")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 52191

    ds = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    eval_mse = round(evaluate_predictor(model, ds)["single_step_mse"], 6)
    assert abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6

    # --- v4.3 基础设施自优化机制零梯度终检 (只读主预测器) --- #
    windows = ds.X[:24]
    evaluator = ConfigEvaluator(model, windows, repeats=3,
                                atol_tol=1e-4)

    # 1) 默认配置评估
    default_res = evaluator.evaluate(ConfigSpec.default())

    # 2) 配置自优化搜索器 (4.3.0)
    searcher = ConfigSearcher(evaluator)
    sres = searcher.search()

    # 3) 搜索-验证-选用闭环 (dev1)
    loop = SearchVerifySelectLoop(evaluator, n_rounds=2,
                                  search_mode="grid").run()

    # 4) 配置 A/B (dev2): 搜索后 vs 默认
    searched_spec = ConfigSpec.from_dict(sres["selected"])
    ab = ConfigAB(evaluator, searched=searched_spec).run()

    # 主预测器权重须零改动
    assert abs(evaluate_predictor(model, ds)["single_step_mse"] - eval_mse) < 1e-9

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v4.3.0.pt"
    save_predictor(model, ckpt, metrics={
        "evaluation": evaluate_predictor(model, ds),
        "inherits_from": "checkpoints/predictor_v4.1.0.pt"})

    loaded, meta2 = load_predictor(ckpt)
    assert abs(evaluate_predictor(loaded, ds)["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__
    assert sum(p.numel() for p in loaded.parameters()) == 52191

    # 全 backcompat
    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    assert len(ckpt_list) == EXPECTED_N_CKPTS, \
        f"expected {EXPECTED_N_CKPTS} checkpoints, got {len(ckpt_list)}"
    for name in ckpt_list:
        m, _ = load_predictor(f"checkpoints/{name}")
        assert sum(p.numel() for p in m.parameters()) == 52191

    summary = {
        "version": __version__, "n_params": n_params,
        "inherits_frozen_main_from": "predictor_v4.1.0.pt",
        "eval_mse": eval_mse, "anchor_eval_mse": ANCHOR_EVAL_MSE,
        "eval_mse_matches_anchor": abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6,
        "backcompat_checkpoints": len(ckpt_list),
        "line_430": {
            "default_cost": default_res["composite_cost"],
            "search_n_candidates": sres["n_candidates"],
            "search_n_acceptable": sres["n_acceptable"],
            "search_best_cost": sres["best_cost"],
            "search_improved": sres["improved"],
            "search_selected": sres["selected"],
            "loop_n_archive": loop["n_archive"],
            "ab_A_wins": ab["A_wins"],
            "ab_cost_saving_pct": ab["cost_saving_pct"],
        },
        "honest_conclusion": (
            "配置自优化在固定基准负载上, 仅采纳'输出保真(<=atol)且成本严格更低'"
            f"的配置; search_improved={sres['improved']}, "
            f"ab_A_wins={ab['A_wins']}; 主预测器零改动, eval_mse 锚点不变。"),
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v4.3.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints",
                       "line_430", "honest_conclusion"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
