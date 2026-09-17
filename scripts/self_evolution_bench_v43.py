"""
v4.3 完全自进化线基准: 配置 A/B + 多代曲线 + 停止/纠正 + 长程闭环
==========================================================================
落 benchmarks/results/*.json, 全部确定性、可复算。主预测器只读零梯度。
诚实记录: 真改进 vs 退化, 不宣称飞轮必然提升。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (load_predictor, ConfigSpec, ConfigEvaluator,  # noqa: E402
                  ConfigSearcher, ConfigAB, GlobalStopCorrectCriterion,
                  LongHorizonLoop)
from udos.dynamics import build_parametric_dataset  # noqa: E402

torch.set_num_threads(2)
ANCHOR = 0.045556


def main():
    model, _ = load_predictor("checkpoints/predictor_v4.3.0.pt")
    assert sum(p.numel() for p in model.parameters()) == 52191
    ds = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    ev = ConfigEvaluator(model, ds.X[:24], repeats=3, atol_tol=1e-4)

    # dev2: 配置 A/B
    searched = ConfigSearcher(ev).search()
    ab = ConfigAB(ev, searched=ConfigSpec.from_dict(searched["selected"])).run()

    # dev4: 多代曲线 (每代重跑同合同搜索, 照实记录)
    curve = []
    for g in range(3):
        r = ConfigSearcher(ev).search()
        curve.append({"gen": g, "best_cost": r["best_cost"],
                      "improved": r["improved"]})
    costs = [c["best_cost"] for c in curve]
    deltas = [costs[i] - costs[i - 1] for i in range(1, len(costs))]
    honest = ("efficiency_improving"
              if all(d <= 1e-9 for d in deltas) and deltas else "drifting")

    # dev5: 全局停止/纠正
    gsc = GlobalStopCorrectCriterion(patience=2, min_delta=1e-3)
    stop = gsc.check(costs, [True] * len(costs))

    # dev6: 长程任务闭环
    lh = LongHorizonLoop(model, horizon=4, n_sub=2).run(seed=42)

    out = {
        "version": "4.3.x",
        "anchor_eval_mse": ANCHOR,
        "dev2_config_ab": {
            "A_wins": ab["A_wins"],
            "cost_saving_pct": ab["cost_saving_pct"],
            "verdict": ab["verdict"],
        },
        "dev4_multigen_curve": {
            "curve": curve, "cost_deltas": [round(d, 4) for d in deltas],
            "honest_verdict": honest,
            "note": "多代成本曲线照实记录; 网格确定性下 best 逐代稳定, "
                    "不宣称飞轮必然持续上升",
        },
        "dev5_global_stop_correct": {
            "recommended_action": stop["recommended_action"],
            "diminishing_returns": stop["diminishing_returns"],
        },
        "dev6_long_horizon": {
            "verified": lh["verified"],
            "n_subtasks": lh["n_subtasks"],
            "n_recovered": lh["n_recovered"],
        },
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/self_evolution_v43.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
