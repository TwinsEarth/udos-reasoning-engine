"""
主动学习 A/B 证据 (v2.7.0.dev2)
=================================
同种子、同初始训练集大小: UncertaintySampler 主动选点 vs 随机选点,
在同样本增量下比较独立测试集 eval_mse。
落 benchmarks/results/active_learning_ablation_v2.7.0.json。
用法: python3 scripts/ablation_active_learning.py
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)

from udos.ctm_engine import CTMConfig                      # noqa: E402
from udos.dynamics import build_parametric_dataset         # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig, set_seed  # noqa: E402
from udos.evaluation import evaluate_predictor             # noqa: E402
from udos.calibration import fit_predictor_calibration     # noqa: E402
from udos.ood import DistributionDriftDetector              # noqa: E402
from udos.active_learning import UncertaintySampler         # noqa: E402


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def train_and_eval(tr, cal, te, seed, epochs=25):
    set_seed(seed)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    tcfg = TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                       patience=8, step_weight_scheme="front", hybrid_weight=0.0)
    CTMTrainer(model, tcfg).train(tr)
    ood = DistributionDriftDetector(ridge=1e-3, alpha=0.05).fit(tr.X)
    model.attach_ood_detector(ood)
    fit_predictor_calibration(model, cal)
    rep = evaluate_predictor(model, te)
    return model, rep["single_step_mse"]


def subset(ds, idx):
    from udos.dynamics import ParametricDynamicsDataset
    return ParametricDynamicsDataset(ds.X[idx], ds.Y[idx], ds.P[idx],
                                     [ds.kinds[i] for i in idx.tolist()],
                                     ds.dt, ds.class_names)


def main():
    torch.manual_seed(42)
    # 初始小训练集 + 待标注池 + 独立测试集 (不同 seed 隔离)
    seed_tr = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                       horizon=4, dt=0.5, seed=701)
    pool = build_parametric_dataset(n_per_kind=24, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=702)
    te = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=703)
    cal = build_parametric_dataset(n_per_kind=12, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=704)

    # 1) 仅初始小训练集基线
    _, mse_base = train_and_eval(seed_tr, cal, te, seed=11)

    # 先在 seed_tr 上训一个"预热"模型, 用于不确定性打分 (打分模型不参与对比训练)
    warm, _ = train_and_eval(seed_tr, cal, te, seed=11, epochs=20)
    sampler = UncertaintySampler(alpha=0.4, beta=0.3, gamma=0.3)

    # 2) 主动: 选池里最不确定的 k=128 条加入训练
    topi, tops = sampler.select_top_k(warm, pool.X, k=128, scene_params=pool.P)
    active_idx = torch.arange(pool.X.size(0))
    active_mask = torch.zeros(pool.X.size(0), dtype=torch.bool)
    active_mask[topi] = True
    active_pool = subset(pool, active_mask.nonzero(as_tuple=True)[0])
    from udos.dynamics import ParametricDynamicsDataset
    active_tr = ParametricDynamicsDataset(
        torch.cat([seed_tr.X, active_pool.X]),
        torch.cat([seed_tr.Y, active_pool.Y]),
        torch.cat([seed_tr.P, active_pool.P]),
        list(seed_tr.kinds) + list(active_pool.kinds), seed_tr.dt, seed_tr.class_names)
    _, mse_active = train_and_eval(active_tr, cal, te, seed=12)

    # 3) 随机: 同数量 k=128 条 (固定随机种子)
    g = torch.Generator().manual_seed(123)
    rand_idx = torch.randperm(pool.X.size(0), generator=g)[:128]
    random_pool = subset(pool, rand_idx)
    random_tr = ParametricDynamicsDataset(
        torch.cat([seed_tr.X, random_pool.X]),
        torch.cat([seed_tr.Y, random_pool.Y]),
        torch.cat([seed_tr.P, random_pool.P]),
        list(seed_tr.kinds) + list(random_pool.kinds), seed_tr.dt, seed_tr.class_names)
    _, mse_random = train_and_eval(random_tr, cal, te, seed=13)

    result = {
        "version": "2.7.0.dev2",
        "seed": 42,
        "n_seed": len(seed_tr),
        "n_added": int(active_pool.X.size(0)),
        "eval_mse_baseline_seed_only": round(mse_base, 6),
        "eval_mse_active": round(mse_active, 6),
        "eval_mse_random": round(mse_random, 6),
        "active_minus_random": round(mse_active - mse_random, 6),
        "active_better": bool(mse_active < mse_random),
        "note": "CPU 小模型/小样本上主动选点增益不稳健属预期; 如实记录 A/B 差值。",
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/active_learning_ablation_v2.7.0.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
