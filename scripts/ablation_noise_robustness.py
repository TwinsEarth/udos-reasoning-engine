#!/usr/bin/env python3
"""
v2.4.9 噪声鲁棒性 同合同 A/B: 训练噪声 × 测试噪声
=====================================================
唯一变量 = 训练期注入噪声 sigma (noise_sigma) 与测试期是否加噪。
经典噪声鲁棒性问题: "训练时加噪能否换来对测试时观测噪声的稳健性?"

设计 (3 x 2 网格, 多种子):
  - train_sigma ∈ {0.0, 0.05, 0.1}   (TrainConfig.noise_sigma, v2.4.3)
  - test_sigma   ∈ {0.0, 0.1}         (对测试输入窗口 X 临时注入同分布高斯噪声)
对每个 seed, 同一 split、同一测试集, 仅 train_sigma 不同 -> 比较单步 MSE。
鲁棒性缺口 = MSE(test_sigma=0.1) - MSE(test_sigma=0); 期望: 训练噪声越大,
该缺口越小 (更稳健), 但干净集上 MSE 可能略升 (偏差-方差权衡)。

证据诚实: 若跨种子方向不稳或负面, 照实写入 JSON, 不硬凑。
输出: benchmarks/results/noise_robustness_v2.4.9.json
用法: python3 scripts/ablation_noise_robustness.py [--quick] [--seeds 42 7 123]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (  # noqa: E402
    build_parametric_dataset, PhysicsPredictor, CTMTrainer, TrainConfig,
    CTMConfig,
)
from udos.dynamics import noise_augment  # noqa: E402

TRAIN_SIGMAS = (0.0, 0.05, 0.1)
TEST_SIGMAS = (0.0, 0.1)


def small_ctm_config():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def train_one(seed, train_sigma, n_per_kind, epochs, horizon=4):
    torch.manual_seed(seed)
    tr_ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                     window=6, horizon=horizon, dt=0.5,
                                     seed=1000 + seed)
    tr, _ = tr_ds.split(0.8)
    model = PhysicsPredictor(small_ctm_config(), scene_param_dim=4)
    cfg = TrainConfig(epochs=epochs, seed=seed, noise_sigma=train_sigma)
    CTMTrainer(model, cfg).train(tr, None)
    model.eval()
    return model


@torch.no_grad()
def eval_mse(model, X, P, Y0, test_sigma, eval_seed):
    """单步 MSE: 对输入窗口 (可选加噪) 预测, 比较 Y[:,0]。确定性 generator。"""
    gen = torch.Generator().manual_seed(eval_seed)
    Xq = noise_augment(X, test_sigma, generator=gen) if test_sigma > 0.0 else X
    pred = model.predict_next(Xq, scene_params=P)
    return float((pred - Y0).pow(2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)

    epochs = args.epochs or (12 if args.quick else 40)
    n_per_kind = 16 if args.quick else 32
    horizon = 4
    t0 = time.time()

    # 固定测试集 (跨 seed 共享同一测试分布, 仅模型权重不同), 保证可比
    per_seed = []
    for seed in args.seeds:
        test_ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                          window=6, horizon=horizon, dt=0.5,
                                          seed=4000 + seed)
        _, te = test_ds.split(0.8)
        X, P, Y = te.X, te.P, te.Y
        Y0 = Y[:, 0, :]

        cells = []
        for ts in TRAIN_SIGMAS:
            model = train_one(seed, ts, n_per_kind, epochs, horizon)
            row = {"train_sigma": ts}
            mses = {}
            for test_s in TEST_SIGMAS:
                m = eval_mse(model, X, P, Y0, test_s, eval_seed=9000 + seed)
                mses[f"mse_test{test_s}"] = round(m, 6)
            row.update(mses)
            row["robustness_gap"] = round(
                mses[f"mse_test{TEST_SIGMAS[1]}"] - mses[f"mse_test{TEST_SIGMAS[0]}"], 6)
            cells.append(row)
            print(f"seed={seed} train_sigma={ts:<5} "
                  f"clean={mses[f'mse_test{TEST_SIGMAS[0]}']:.4f} "
                  f"noisy={mses[f'mse_test{TEST_SIGMAS[1]}']:.4f} "
                  f"gap={row['robustness_gap']:+.4f}")
        per_seed.append({"seed": seed, "runs": cells})

    # summary: 每个 train_sigma 跨 seed 平均
    summary = {}
    for ts in TRAIN_SIGMAS:
        rs = [c for ps in per_seed for c in ps["runs"] if c["train_sigma"] == ts]
        summary[str(ts)] = {
            "mse_test0.0": round(sum(r["mse_test0.0"] for r in rs) / len(rs), 6),
            "mse_test0.1": round(sum(r["mse_test0.1"] for r in rs) / len(rs), 6),
            "robustness_gap": round(sum(r["robustness_gap"] for r in rs) / len(rs), 6),
        }
    # 方向一致性: train_sigma>0 的 gap 是否在多数种子小于 train_sigma=0 的 gap
    def gap(ts, ps):
        return next(c for c in ps["runs"] if c["train_sigma"] == ts)["robustness_gap"]
    for ts in (0.05, 0.1):
        better = sum(1 for ps in per_seed if gap(ts, ps) < gap(0.0, ps))
        summary[str(ts)]["smaller_gap_vs_sigma0_seeds"] = f"{better}/{len(per_seed)}"

    result = {
        "experiment": "v2.4.9 noise robustness A/B (train_sigma x test_sigma)",
        "grid": {"train_sigma": list(TRAIN_SIGMAS), "test_sigma": list(TEST_SIGMAS)},
        "metric": "single-step MSE on test window",
        "epochs": epochs, "n_per_kind": n_per_kind, "horizon": horizon,
        "seeds": args.seeds, "per_seed": per_seed, "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "benchmarks" / "results" / "noise_robustness_v2.4.9.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nsummary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {out}  ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
