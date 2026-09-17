"""
构建 v2.6.0 物理预测器 checkpoint (可复现, 2.6 线正式件)
================================================================
与 v2.5.2 同口径 (front 默认 + PAVA 校准 + conformal 区间 + OOD 拟合)。
v2.6.0 引入 hybrid 混合物理修正模块 (默认关闭, hybrid_weight=0), 故主 checkpoint
**不挂载 hybrid**, 主模型参数量仍为 52191 (hybrid 为可选外挂, 见 ablation 脚本)。

落 checkpoints/predictor_v2.6.0.pt 与 benchmarks/results/training_v2.6.0.json。

用法:
    python3 scripts/build_v260_checkpoint.py --quick
    python3 scripts/build_v260_checkpoint.py
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

from udos import __version__, save_predictor, load_predictor  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.calibration import fit_predictor_calibration  # noqa: E402
from udos.ood import DistributionDriftDetector  # noqa: E402

torch.set_num_threads(2)


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def build(seed, n_per_kind, horizon=4):
    return build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                    window=6, horizon=horizon, dt=0.5, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    n_per_kind = 16 if args.quick else 48
    epochs = args.epochs or (15 if args.quick else 60)
    torch.manual_seed(42)

    train_full = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                          window=6, horizon=4, dt=0.5, seed=42)
    tr, te_es = train_full.split(0.8)
    cal_te = build(314, n_per_kind)
    ind_te = build(2718, n_per_kind)

    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    # v2.6.0: 主 checkpoint 不挂载 hybrid (hybrid_weight=0 默认关, 与 v2.5.2 逐位同口径)
    n_params = sum(p.numel() for p in model.parameters())
    tcfg = TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                       patience=12, step_weight_scheme="front",
                       hybrid_weight=0.0)
    t0 = time.time()
    hist = CTMTrainer(model, tcfg).train(tr, te_es)
    train_s = time.time() - t0

    ood = DistributionDriftDetector(ridge=1e-3, alpha=0.05).fit(tr.X)
    model.attach_ood_detector(ood)

    before = evaluate_predictor(PhysicsPredictor(small_cfg(), scene_param_dim=4),
                                ind_te)
    calibrator, fitted_report, rq = fit_predictor_calibration(model, cal_te)
    model.attach_calibration(calibrator, rq)
    cal_eval = evaluate_predictor(model, ind_te)
    naive = ((ind_te.X[:, -1, :] - ind_te.Y[:, 0, :]) ** 2).mean().item()

    id_scores = model.ood_score(ind_te.X)
    ood_X = ind_te.X * 5.0
    ood_scores = model.ood_score(ood_X)
    ood_diag = {
        "threshold": round(float(ood.threshold_), 6),
        "id_score_mean": round(float(id_scores.mean()), 6),
        "id_score_max": round(float(id_scores.max()), 6),
        "ood_score_mean": round(float(ood_scores.mean()), 6),
        "ood_score_max": round(float(ood_scores.max()), 6),
        "ood_hit_rate": round(float((ood_scores > ood.threshold_).float().mean()), 4),
        "id_false_alarm_rate": round(
            float((id_scores > ood.threshold_).float().mean()), 4),
    }

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v2.6.0.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval,
        "naive_mse": naive, "ood": ood_diag, "hybrid_attached": False})

    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated
    assert loaded.hybrid is None          # 主件不挂 hybrid
    assert rep2["interval"]["coverage_overall"] == \
        cal_eval["interval"]["coverage_overall"]
    assert loaded.has_ood_detector
    assert torch.allclose(loaded.ood_score(ind_te.X), id_scores, atol=1e-4)

    # v2.5.0: 批量推理一致性自检
    batch_out = loaded.predict_batch(ind_te.X, scene_params=ind_te.P)
    single_out = torch.stack([
        loaded.predict_next(ind_te.X[i:i+1], scene_params=ind_te.P[i:i+1])[0]
        for i in range(min(20, len(ind_te)))])
    batch_max_diff = (batch_out[:len(single_out)] - single_out).abs().max().item()
    assert batch_max_diff < 1e-5

    cal = cal_eval["calibration"]
    iv = cal_eval["interval"]
    summary = {
        "version": __version__, "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "epochs_run": hist.final()["epochs_run"],
        "stopped_early": hist.stopped_early, "best_epoch": hist.best_epoch,
        "step_weight_scheme": "front",
        "hybrid_weight": 0.0, "hybrid_attached": False,
        "final_loss": round(hist.final()["train_loss"], 6),
        "eval_mse": cal_eval["single_step_mse"],
        "naive_mse": round(naive, 6),
        "untrained_single_mse": round(before["single_step_mse"], 6),
        "ece": cal["calibrated"]["ece"],
        "coverage": iv["coverage_overall"],
        "interval_width": round(sum(iv["width_by_step"]) / len(iv["width_by_step"]), 6),
        "params_count": n_params,
        "condition_gain_x": cal_eval["ablation"]["condition_gain_x"],
        "rollout_mse_curve": cal_eval["rollout_mse_curve"],
        "rollout_growth_x": cal_eval["rollout_growth_x"],
        "batch_inference_max_diff": round(batch_max_diff, 8),
        "calibration": {
            "independent_test_raw_ece": cal["raw"]["ece"],
            "independent_test_calibrated_ece": cal["calibrated"]["ece"],
            "ece_reduction_x": cal["ece_reduction_x"],
            "ranking_informative": cal["ranking_informative"],
            "num_segments": cal["num_segments"],
        },
        "interval": iv,
        "ood": ood_diag,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v2.6.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
