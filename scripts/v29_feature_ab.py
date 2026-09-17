"""
v2.9.1 2.9 线特性 A/B 汇总
=============================
汇总两类 A/B 到 benchmarks/results/v29_feature_ab.json:
    1. retargeting A/B (重定向 vs 截断) —— 复用 retarget_ab_v2.9.0.json;
    2. affordance A/B (有 affordance 引导 vs 无引导) —— 动作成功率代理指标:
       在 N 个随机场景里, 仅一个物体可达 (target), 其余为干扰远物;
       * guided   : AffordanceScorer 选最近可达部位;
       * unguided : 朴素恒选 part_0 (无引导)。
       成功率 = best_part == target 的比例。
analogy, not reproduction: 成功率为合成代理指标, 非真机抓取成功率。
零重依赖, CPU-only。收益不稳则 opt-in (如实记录)。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.affordance import AffordanceScorer, AffordanceActionPlanner  # noqa


def affordance_ab(n=200, n_parts=4, seed=42):
    g = torch.Generator().manual_seed(seed)
    scorer = AffordanceScorer(reach_radius=2.0)
    planner = AffordanceActionPlanner(scorer)
    guided_ok = unguided_ok = 0
    for _ in range(n):
        # 随机选一个可达 target, 其余为远干扰
        tgt = int(torch.randint(0, n_parts, (1,), generator=g))
        pos = torch.zeros(n_parts, 6)
        # target: 距原点 ~0.5 随机方向
        ang = torch.rand(1, generator=g) * 6.2832
        pos[tgt, 0] = 0.5 * torch.cos(ang)
        pos[tgt, 1] = 0.5 * torch.sin(ang)
        # 其余: 距离 ~5 的远物
        far = torch.rand(n_parts, generator=g) * 4 + 5
        for k in range(n_parts):
            if k != tgt:
                pos[k, 0] = far[k]
        state = torch.zeros(1, 6)
        obj = pos.unsqueeze(0)
        res = planner.plan(state, obj)
        guided_best = int(res["best_part"][0])
        guided_ok += int(guided_best == tgt)
        unguided_ok += int(0 == tgt)        # 无引导恒选 part_0
    return {"n_trials": n, "n_parts": n_parts,
            "guided_success": guided_ok / n,
            "unguided_success": unguided_ok / n,
            "guided_better": guided_ok > unguided_ok}


def main():
    retarget_path = ROOT / "benchmarks" / "results" / "retarget_ab_v2.9.0.json"
    retarget = {}
    if retarget_path.exists():
        retarget = json.load(open(retarget_path))

    aff = affordance_ab()
    result = {
        "version": "2.9.1",
        "analogy_not_reproduction": True,
        "retarget_ab": {
            "eval_mse_retarget": retarget.get("retarget", {}).get("eval_mse_vs_ref"),
            "eval_mse_truncate": retarget.get("truncate", {}).get("eval_mse_vs_ref"),
            "retarget_better": retarget.get("retarget_better_mse"),
            "conclusion": retarget.get("conclusion"),
        },
        "affordance_ab": aff,
        "opt_in_default": True,   # 收益为合成代理, 维持 opt-in
        "rejected_candidates": [
            "retarget/affordance/spatial_relation 进旧 predict_next 默认路径",
            "affordance 成功率宣称真机抓取指标",
        ],
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    out = ROOT / "benchmarks" / "results" / "v29_feature_ab.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
