"""
构建 v2.2.1 物理预测器 checkpoint (可复现)
=========================================
- 参数化合成动力学训练 (默认 teacher-forcing, 与 v2.1 稳健可比);
  加 --ss 可开启 opt-in Scheduled Sampling (收益不稳健, 默认关闭)。
- v2.2 增强评估: rollout 累积率、置信度分层校准、场景消融、运动学诊断。
- 落 checkpoints/predictor_v2.2.1.pt 与 benchmarks/results/training_v2.2.1.json,
  并做"保存后重新载入 + 结果一致"校验。

用法:
    python3 scripts/build_v221_checkpoint.py --quick
    python3 scripts/build_v221_checkpoint.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, save_predictor, load_predictor
from udos.ctm_engine import CTMConfig
from udos.dynamics import build_parametric_dataset
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig
from udos.evaluation import evaluate_predictor

torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ss", action="store_true",
                    help="开启 opt-in scheduled sampling (默认关闭=teacher-forcing)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    n_per_kind = 16 if args.quick else 48
    epochs = args.epochs or (15 if args.quick else 60)
    torch.manual_seed(42)

    ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                  window=6, horizon=4, dt=0.5, seed=42)
    tr, te = ds.split(0.8)
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    model = PhysicsPredictor(cfg, scene_param_dim=4)
    n_params = sum(p.numel() for p in model.parameters())

    ss_kwargs = dict(ss_max=0.30, ss_start=15, ss_warmup=25) if args.ss else {}
    tcfg = TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                       patience=12, **ss_kwargs)  # v2.2 早停: 12 轮无改善即停
    t0 = time.time()
    hist = CTMTrainer(model, tcfg).train(tr, te)
    train_s = time.time() - t0

    before = evaluate_predictor(PhysicsPredictor(
        CTMConfig(iterations=8, d_model=64, d_input=32, heads=4, n_synch_out=16,
                  n_synch_action=8, memory_length=8, nlm_hidden=16, out_dims=32,
                  certainty_threshold=0.0), scene_param_dim=4), te)
    rep = evaluate_predictor(model, te)
    naive = ((te.X[:, -1, :] - te.Y[:, 0, :]) ** 2).mean().item()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v2.2.1.pt"
    save_predictor(model, ckpt, metrics={"training": hist.final(),
                                         "evaluation": rep, "naive_mse": naive})
    # 重新载入校验一致
    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, te)
    assert abs(rep2["single_step_mse"] - rep["single_step_mse"]) < 1e-9

    summary = {
        "version": __version__, "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "epochs_run": hist.final()["epochs_run"],
        "stopped_early": hist.stopped_early, "best_epoch": hist.best_epoch,
        "scheduled_sampling": {"enabled": bool(args.ss),
                               "ss_max": tcfg.ss_max, "ss_start": tcfg.ss_start,
                               "ss_warmup": tcfg.ss_warmup},
        "untrained_single_mse": round(before["single_step_mse"], 6),
        "naive_mse": round(naive, 6),
        "trained_single_mse": rep["single_step_mse"],
        "reduction_vs_untrained_x": round(
            before["single_step_mse"] / max(rep["single_step_mse"], 1e-12), 2),
        "reduction_vs_naive_x": round(naive / max(rep["single_step_mse"], 1e-12), 2),
        "condition_gain_x": rep["ablation"]["condition_gain_x"],
        "rollout_mse_curve": rep["rollout_mse_curve"],
        "rollout_growth_x": rep["rollout_growth_x"],
        "confidence_stratification": rep["confidence_stratification"],
        "checkpoint": ckpt,
        "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v2.2.1.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
