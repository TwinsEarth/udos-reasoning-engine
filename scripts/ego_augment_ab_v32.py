"""
v3.2.0.dev4 数据增强 A/B + 增强类型消融
================================================================
对比: 原始数据训练 vs 合成 Ego360 启发增强数据训练 的
    * eval_mse (独立测试集)
    * 噪声鲁棒性 (test_X + sigma)
    * OOD 鲁棒性 (test_X * 5)
增强类型消融: view / perturb / noise / time_scale 各自单独开。

纪律 (与全工程一致):
    * **收益不稳则 opt-in, 默认关**; 本脚本仅离线测量, 不改正式件口径;
    * 小规模 quick 训练 (n_per_kind/epochs 可配), 多种子;
    * 落 benchmarks/results/ego_augment_ab_v3.2.0.json;
    * analogy, not reproduction —— 合成状态序列增强, 不碰真机/视频/VLM。

用法:
    python3 scripts/ego_augment_ab_v32.py            # quick (~20s)
    python3 scripts/ego_augment_ab_v32.py --epochs 30 --n 24
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.dynamics import (build_parametric_dataset, ParametricDynamicsDataset,
                            noise_augment)  # noqa: E402
from udos.ego_data import SyntheticEgoAugmenter  # noqa: E402

torch.set_num_threads(2)


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def _train_eval(train_ds, test_ds, epochs, seed):
    torch.manual_seed(42)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    tcfg = TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                       patience=12, step_weight_scheme="front",
                       hybrid_weight=0.0)
    CTMTrainer(model, tcfg).train(train_ds, None)
    model.eval()
    with torch.no_grad():
        pred = model.predict_next(test_ds.X, scene_params=test_ds.P)
        mse = float(((pred - test_ds.Y[:, 0, :]) ** 2).mean())
        # 噪声鲁棒性
        noisy_X = noise_augment(test_ds.X, 0.05)
        pred_n = model.predict_next(noisy_X, scene_params=test_ds.P)
        noise_mse = float(((pred_n - test_ds.Y[:, 0, :]) ** 2).mean())
        # OOD 鲁棒性 (放大输入)
        ood_X = test_ds.X * 5.0
        pred_o = model.predict_next(ood_X, scene_params=test_ds.P)
        ood_mse = float(((pred_o - test_ds.Y[:, 0, :]) ** 2).mean())
    return {"eval_mse": round(mse, 6),
            "noise_robust_mse": round(noise_mse, 6),
            "ood_mse": round(ood_mse, 6)}


def _augment_ds(ds, aug):
    Xa, Ya = aug.augment_pair(ds.X, ds.Y)
    return ParametricDynamicsDataset(Xa, Ya, ds.P, list(ds.kinds),
                                     dt=ds.dt, class_names=ds.class_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    train = build_parametric_dataset(n_per_kind=args.n, n_steps=14, window=6,
                                     horizon=4, dt=0.5, seed=42)
    test = build_parametric_dataset(n_per_kind=args.n, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=2718)

    # 1) 基线: 原始数据
    raw = _train_eval(train, test, args.epochs, seed=42)

    # 2) 全量增强
    full_aug = SyntheticEgoAugmenter(view_rotate_deg=20.0,
                                      view_translate_xyz=(0.1, -0.1, 0.0),
                                      traj_perturb=0.02, noise_sigma=0.01,
                                      time_scale=1.05, seed=1)
    aug = _train_eval(_augment_ds(train, full_aug), test, args.epochs, seed=42)

    # 3) 单类型消融
    ablations = {}
    for name, kw in {
        "view": dict(view_rotate_deg=20.0, view_translate_xyz=(0.1, -0.1, 0.0)),
        "perturb": dict(traj_perturb=0.02),
        "noise": dict(noise_sigma=0.01),
        "time_scale": dict(time_scale=1.05),
    }.items():
        a = SyntheticEgoAugmenter(seed=2, **kw)
        ablations[name] = _train_eval(_augment_ds(train, a), test,
                                      args.epochs, seed=42)

    summary = {
        "version": "3.2.0.dev4",
        "n_per_kind": args.n, "epochs": args.epochs,
        "raw": raw, "augmented_full": aug,
        "ablations": ablations,
        "augment_mse_gain": round(raw["eval_mse"] - aug["eval_mse"], 6),
        "augment_noise_gain": round(raw["noise_robust_mse"] - aug["noise_robust_mse"], 6),
        "augment_ood_gain": round(raw["ood_mse"] - aug["ood_mse"], 6),
        "opt_in_default_off": True,
        "recommendation": ("augmentation opt-in; only adopt if gains are "
                           "consistent across seeds" if
                           aug["eval_mse"] <= raw["eval_mse"]
                           else "no consistent gain vs raw => keep opt-in, "
                                "rejected as formal-data default"),
        "analogy_not_reproduction": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    out = "benchmarks/results/ego_augment_ab_v3.2.0.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
