"""
F1 可复现实验: Scheduled Sampling (SS) 对照 A/B
================================================
同合同配对: 同种子 / 同数据 / 同初始权重 / 同 epoch, 仅 SS 课程不同。
比较 teacher-forcing (ss_max=0) 与 SS 的自由 rollout 逐步误差、单步误差。

诚实说明 (见 docs/VERSION_PLAN_2.2.md §3.2 与 VERIFICATION_v2.2.1.md):
本沙箱小模型 + 合成数据规模下, SS 对长时程 rollout 的收益方向不稳 (依种子而变),
故 SS 只作为 opt-in 特性 (默认 ss_max=0), 不宣称确定增益。本脚本即证据复现入口。

用法:
    python3 scripts/ablation_scheduled_sampling.py --quick     # 小规模, 约 1 分钟
    python3 scripts/ablation_scheduled_sampling.py             # 中等规模, 约数分钟
结果落 benchmarks/results/ss_ablation_v2.2.0.json
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.ctm_engine import CTMConfig
from udos.dynamics import build_parametric_dataset
from udos.training import (PhysicsPredictor, CTMTrainer, TrainConfig, set_seed)
from udos.evaluation import evaluate_predictor

torch.set_num_threads(2)


def _fresh():
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    args = ap.parse_args()
    n_per_kind = 16 if args.quick else 36
    epochs = 12 if args.quick else 45
    horizon = 4
    # (名称, ss_max, ss_start, ss_warmup); TF 为对照基线
    variants = [("teacher_forcing", 0.0, 0, 10),
                ("ss_0.30_s15_w25", 0.30, 15, 25)]
    out = {"config": {"n_per_kind": n_per_kind, "epochs": epochs,
                      "horizon": horizon, "seeds": args.seeds,
                      "note": "paired A/B; SS opt-in, gain not robust at this scale"},
           "results": {}}
    t0 = time.time()
    for seed in args.seeds:
        ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                      window=6, horizon=horizon, dt=0.5, seed=seed)
        tr, te = ds.split(0.8)
        m0 = _fresh(); set_seed(seed)
        sd = copy.deepcopy(m0.state_dict())
        out["results"][f"seed{seed}"] = {}
        for name, ss, st, wm in variants:
            m = _fresh(); m.load_state_dict(sd)  # 配对: 相同初始权重
            CTMTrainer(m, TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                                      ss_max=ss, ss_start=st, ss_warmup=wm,
                                      seed=seed)).train(tr, te)
            rep = evaluate_predictor(m, te)
            roll = rep["rollout_mse_curve"]
            out["results"][f"seed{seed}"][name] = {
                "single_step_mse": rep["single_step_mse"],
                "rollout_mse_curve": roll,
                "late_steps_mean": sum(roll[2:]) / max(len(roll[2:]), 1),
                "rollout_growth_x": rep["rollout_growth_x"]}
            print(f"seed{seed} {name:18s} single={roll[0]:.4f} "
                  f"roll={[round(x,3) for x in roll]}")
    os.makedirs("benchmarks/results", exist_ok=True)
    path = "benchmarks/results/ss_ablation_v2.2.0.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s, saved {path}")


if __name__ == "__main__":
    main()
