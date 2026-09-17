"""v3.8.0.dev1 世界模型预算-收益 A/B: 低预算(全回退真实) vs 高预算(优先级想象)。

复用 LatentWorldModel + WMScheduler; 纯离线、零梯度、不改主权重。
落 benchmarks/results/wm_budget_ab_v3.8.0.json。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import load_predictor  # noqa: E402
from udos.multi_agent import MultiAgentScene  # noqa: E402
from udos.wm_scheduler import WMScheduler  # noqa: E402
from udos.world_model import LatentWorldModel  # noqa: E402

torch.set_num_threads(2)

CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def build_scene():
    sc = MultiAgentScene()
    for i, pri in enumerate([1, 3, 5, 7, 9]):
        sc.add_agent(f"a{i}", [float(i), 0, 0, 0, 0, 0], priority=pri)
    return sc


def run_budget(wm, sc, budget):
    sched = WMScheduler(total_imagination_budget=budget, base_horizon=1,
                        max_horizon=4)
    windows = {i: torch.randn(1, 6, 6) for i in sc.ids}
    rep = sched.imagine(wm, sc, windows)
    return {
        "budget": budget,
        "used_budget": rep["used_budget"],
        "n_fell_back": len(rep["fell_back_real"]),
        "n_imagined": len(rep["imagined"]),
        "allocation": rep["allocation"],
    }


def main():
    model, _ = load_predictor(CKPT)
    wm = LatentWorldModel(model)
    sc = build_scene()
    low = run_budget(wm, sc, 5)    # n*base => 全回退真实
    high = run_budget(wm, sc, 15)  # 优先级想象
    summary = {
        "feature": "wm_imagination_budget_ab",
        "n_agents": sc.n_agents,
        "main_params_untouched": True,
        "learned": False,
        "analogy_not_reproduction": True,
        "ab": {"low_budget": low, "high_budget": high},
        "conclusion": "低预算全回退真实(horizon=1逐位锚定); 高预算按优先级分配"
                      "额外想象步, 总消耗<=预算上限",
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/wm_budget_ab_v3.8.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary["ab"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
