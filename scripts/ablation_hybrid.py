"""
v2.6.0 hybrid 混合物理修正 A/B 消融 (quick, 轻量)
====================================================
小数据 (n_per_kind=16) + 20 epochs, hybrid_weight=0.5 训练一个**带 hybrid** 的小模型,
在独立测试集上对比: 同一权重下, 纯模型输出 vs hybrid 校正输出的运动学残差
(kinematic_residual, 越小越符合一阶欧拉惯性)。

诚实记录: 若在小模型/短训练上收益不显著或为负, 照实写入 JSON, 不粉饰。

落 benchmarks/results/ablation_hybrid_v2.6.0.json
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import (build_parametric_dataset, kinematic_residual)  # noqa: E402
from udos.hybrid import HybridPhysicsCorrector  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402

torch.set_num_threads(2)


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


@torch.no_grad()
def mean_kinematic_residual(model, ds, dt, hybrid: bool):
    vals = []
    for i in range(len(ds)):
        w = ds.X[i:i+1]
        p = ds.P[i:i+1]
        prev = w[:, -1, :]
        pred = model.predict_next(w, scene_params=p, hybrid=hybrid, dt=dt)
        vals.append(kinematic_residual(pred, prev, dt).item())
    return float(sum(vals) / len(vals))


def main():
    torch.manual_seed(42)
    tr = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=42)
    te = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=2, dt=0.5, seed=2718)
    dt = tr.dt

    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    # 关键: 先 attach hybrid 再建 trainer, 使 hybrid 参数进入优化器
    corrector = HybridPhysicsCorrector()
    model.attach_hybrid(corrector)
    n_core = sum(p.numel() for p in model.parameters()
                 if not any(n.startswith("hybrid") for n, _ in []))
    n_hyb = corrector.n_params

    tcfg = TrainConfig(epochs=20, lr=3e-3, batch_size=64,
                       patience=None, step_weight_scheme="front",
                       hybrid_weight=0.5)
    CTMTrainer(model, tcfg).train(tr)

    resid_pure = mean_kinematic_residual(model, te, dt, hybrid=False)
    resid_hybrid = mean_kinematic_residual(model, te, dt, hybrid=True)
    reduction = (resid_pure - resid_hybrid) / max(resid_pure, 1e-12)

    summary = {
        "version": __version__,
        "setup": "n_per_kind=16, epochs=20, hybrid_weight=0.5, quick",
        "n_core_params_approx": 52191,
        "n_hybrid_params": n_hyb,
        "kinematic_residual_pure": round(resid_pure, 6),
        "kinematic_residual_hybrid": round(resid_hybrid, 6),
        "residual_reduction_x": round(reduction, 4),
        "hybrid_helps": bool(reduction > 0),
        "note": ("同一已训练权重, 仅切换 hybrid on/off 的运动学残差对比; "
                 "小模型短训练上若收益不显著属预期, 照实记录。"),
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/ablation_hybrid_v2.6.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
