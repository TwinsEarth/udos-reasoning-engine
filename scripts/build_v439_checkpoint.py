"""
构建 v4.3.9 最终物理预测器 checkpoint (完全自进化线全局终点)
==========================================================================
红线: 不重训不改主权重。内化冻结 v4.1.0 主预测器 (eval_mse 锚点 0.045556,
主参 52191)。落 predictor_v4.3.9.pt (第 33 代)。

v4.3 全线机制 (配置自优化搜索/搜索-验证-选用闭环/配置 A/B/自进化 orchestrator/
多代曲线/停止纠正/长程闭环) 纯前向外挂零梯度终检, 不入主 state_dict;
配置搜索须过"输出保真"硬门; 33 代 backcompat 清单 + 逐代可加载校验。
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
                  SearchVerifySelectLoop, ConfigAB,
                  GlobalStopCorrectCriterion, LongHorizonLoop)
from udos.world_model import LatentWorldModel  # noqa: E402
from udos.curriculum import CurriculumGenerator  # noqa: E402
from udos.self_train import TransitionTripletGenerator  # noqa: E402
from udos.self_evolution import (SelfEvolutionOrchestrator,  # noqa: E402
                                  MultiGenerationRunner)
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR_EVAL_MSE = 0.045556
EXPECTED_N_CKPTS = 33


def main():
    model, meta = load_predictor("checkpoints/predictor_v4.1.0.pt")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 52191

    ds = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    eval_mse = round(evaluate_predictor(model, ds)["single_step_mse"], 6)
    assert abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6

    # --- v4.3 全线机制零梯度终检 (只读主预测器) --- #
    windows = ds.X[:24]
    evaluator = ConfigEvaluator(model, windows, repeats=3, atol_tol=1e-4)

    # 4.3.0 配置搜索器
    sres = ConfigSearcher(evaluator).search()
    # dev1 闭环
    loop = SearchVerifySelectLoop(evaluator, n_rounds=2,
                                  search_mode="grid").run()
    # dev2 A/B
    ab = ConfigAB(evaluator,
                  searched=ConfigSpec.from_dict(sres["selected"])).run()

    # dev3+dev4: orchestrator + 多代曲线 (三维串联)
    wm = LatentWorldModel(model, action_dim=0)
    wm.fit(ds, epochs=12, lr=1e-2, seed=420)
    gen = TransitionTripletGenerator(model, wm, action_dim=1)
    cur = CurriculumGenerator()
    orch = SelfEvolutionOrchestrator(model, ds, evaluator, wm, cur, gen)
    multigen = MultiGenerationRunner(orch, n_generations=2).run()

    # dev5 停止/纠正
    gsc = GlobalStopCorrectCriterion(patience=2, min_delta=1e-3)
    stop = gsc.check(multigen["cost_curve"], [True, True])

    # dev6 长程闭环
    lh = LongHorizonLoop(model, horizon=4, n_sub=2).run(seed=42)

    # 主预测器权重须零改动
    assert abs(evaluate_predictor(model, ds)["single_step_mse"] - eval_mse) < 1e-9

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v4.3.9.pt"
    save_predictor(model, ckpt, metrics={
        "evaluation": evaluate_predictor(model, ds),
        "inherits_from": "checkpoints/predictor_v4.1.0.pt"})

    loaded, meta2 = load_predictor(ckpt)
    assert abs(evaluate_predictor(loaded, ds)["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__
    assert sum(p.numel() for p in loaded.parameters()) == 52191

    # 全 33 代 backcompat
    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    assert len(ckpt_list) == EXPECTED_N_CKPTS, \
        f"expected {EXPECTED_N_CKPTS} checkpoints, got {len(ckpt_list)}"
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
        "line_439": {
            "config_default_cost": sres["default_cost"],
            "config_best_cost": sres["best_cost"],
            "config_improved": sres["improved"],
            "ab_A_wins": ab["A_wins"],
            "ab_cost_saving_pct": ab["cost_saving_pct"],
            "loop_n_archive": loop["n_archive"],
            "multigen_cost_curve": multigen["cost_curve"],
            "multigen_honest_verdict": multigen["honest_verdict"],
            "global_stop_action": stop["recommended_action"],
            "long_horizon_verified": lh["verified"],
        },
        "honest_conclusion": (
            "v4.3 基础设施自优化在固定负载上仅采纳'输出保真且更省'的配置"
            f"(config_improved={sres['improved']}, ab_A_wins={ab['A_wins']}, "
            f"降本 {ab['cost_saving_pct']:.2f}%); 多代成本曲线诚实记录为 "
            f"{multigen['honest_verdict']}; 全局推荐动作={stop['recommended_action']}; "
            "主预测器零改动, eval_mse 锚点不变, 不宣称飞轮必然提升。"),
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v4.3.9.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints",
                       "backcompat_all_loadable", "line_439",
                       "honest_conclusion"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
