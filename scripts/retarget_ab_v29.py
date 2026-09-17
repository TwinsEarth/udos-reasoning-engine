"""
v2.9.0.dev3 零样本形态切换 A/B: 重定向 vs 直接截断
====================================================
analogy, not reproduction —— 在合成数据上比较两种把"源形态动作策略"搬到目标形态的方式:
    * retarget : 端点对齐 DOF 映射 (map_dof) + 关节限幅;
    * truncate : 朴素截断 (直接取源前 T 维) + 关节限幅 (naive baseline)。

参考目标 ref = 端点对齐映射但**不** clamp 的"理想目标动作"。两法最终都 clamp,
故最终违例率均为 0; 区分度指标 = 对 ref 的重建 MSE (retarget 仅 clamp 误差,
truncate 另有投影误差)。落 benchmarks/results/retarget_ab_v2.9.0.json。

零重依赖, CPU-only。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.retargeting import MorphologyConfig, ActionRetargeter, map_dof  # noqa


def main():
    torch.manual_seed(42)
    S, T = 60, 4
    src = MorphologyConfig(S, 100.0, [[-1.57, 1.57]] * S, name="prime_u_60dof")
    tgt = MorphologyConfig(T, 1000.0, [[-1.0, 1.0]] * T, name="gripper_4dof")
    rt = ActionRetargeter(src, tgt)

    # 合成源动作 (尺度超过目标限位, 以触发 clamp)
    X = torch.randn(2048, S) * 2.0
    # 理想参考: 端点对齐映射 (不 clamp)
    ref = map_dof(X, S, T)
    lo, hi = tgt.low, tgt.high

    # retarget: 端点对齐 + clamp
    out_ret = rt.retarget(X)
    # truncate: 朴素取前 T 维 + clamp
    out_trunc = torch.clamp(X[:, :T], lo, hi)

    def mse(a, b):
        return float(((a - b) ** 2).mean())

    res_ret = rt.zero_shot_transfer(X)
    result = {
        "version": "2.9.0.dev3",
        "analogy_not_reproduction": True,
        "src_dof": S, "tgt_dof": T, "n_samples": 2048,
        "reference": "endpoint_aligned_map_no_clamp",
        "retarget": {
            "method": "endpoint_aligned_map+clamp",
            "eval_mse_vs_ref": round(mse(out_ret, ref), 6),
            "final_within_limits": bool(((out_ret >= lo) & (out_ret <= hi)).all()),
            "pre_clamp_violation_rate": round(res_ret["violation_rate"], 4),
        },
        "truncate": {
            "method": "naive_first_T+clamp",
            "eval_mse_vs_ref": round(mse(out_trunc, ref), 6),
            "final_within_limits": bool(((out_trunc >= lo) & (out_trunc <= hi)).all()),
        },
    }
    ret_mse = result["retarget"]["eval_mse_vs_ref"]
    trunc_mse = result["truncate"]["eval_mse_vs_ref"]
    result["retarget_better_mse"] = bool(ret_mse < trunc_mse)
    result["mse_reduction_x"] = round(trunc_mse / ret_mse, 2) if ret_mse > 0 else None
    # 诚实结论: 合成随机数据下端点对齐投影天然优于朴素截断; 不宣称任务精度收益
    result["conclusion"] = (
        "synthetic_proxy_only" if result["retarget_better_mse"]
        else "no_gain_opt_in")

    os.makedirs("benchmarks/results", exist_ok=True)
    out = "benchmarks/results/retarget_ab_v2.9.0.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
