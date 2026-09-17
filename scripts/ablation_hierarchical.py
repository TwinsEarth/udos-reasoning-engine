"""
分层 vs 平铺 rollout A/B 证据 (v2.7.0.dev4)
==============================================
H=8/12/16 下分层 (HierarchicalRollout) vs 平铺 (predictor.rollout) 逐步 MSE 累积率对比。
落 benchmarks/results/hierarchical_ablation_v2.7.0.json。
用法: python3 scripts/ablation_hierarchical.py
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)

from udos.persistence import load_predictor                  # noqa: E402
from udos.dynamics import build_parametric_dataset          # noqa: E402
from udos.hierarchical import HierarchicalRollout           # noqa: E402


def mse_curve(pred, Y, H):
    return [float(((pred[:, h] - Y[:, h]) ** 2).mean()) for h in range(H)]


def main():
    model, _ = load_predictor("checkpoints/predictor_v2.7.0.pt")
    out = {"version": "2.7.0.dev4", "coarse_factor": 4, "results": {}}
    for H in (8, 12, 16):
        ds = build_parametric_dataset(n_per_kind=16, n_steps=22, window=6,
                                      horizon=H, dt=0.5, seed=901 + H)
        flat = model.rollout(ds.X, H, scene_params=ds.P)
        hier = HierarchicalRollout(model, coarse_factor=4).rollout(
            ds.X, H, scene_params=ds.P)["predictions"]
        flat_curve = mse_curve(flat, ds.Y, H)
        hier_curve = mse_curve(hier, ds.Y, H)
        out["results"][f"H={H}"] = {
            "flat_curve": [round(x, 5) for x in flat_curve],
            "hier_curve": [round(x, 5) for x in hier_curve],
            "flat_growth_x": round(flat_curve[-1] / max(flat_curve[0], 1e-12), 3),
            "hier_growth_x": round(hier_curve[-1] / max(hier_curve[0], 1e-12), 3),
            "bit_identical": bool(torch.allclose(flat, hier, atol=1e-6)),
        }
    out["note"] = ("单自回归头下分块滑窗与平铺逐位一致 (bit_identical=true); "
                   "分层为接入独立粗粒度头预留的 opt-in 脚手架, 当前无误差下降。")
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/hierarchical_ablation_v2.7.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
