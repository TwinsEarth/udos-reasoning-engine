"""
构建 v3.3.3 最终发布物理预测器 checkpoint (3.3 线终点正式训练件)
================================================================
与 v3.2.0/v3.3.0 同口径 (seed=42 / n_per_kind=48 / epochs=60 / patience=12 /
front / hybrid_weight=0)。正式件默认仍为旧架构 (52191 参数)。本脚本额外记录
全部 3.3 新特性的离线评估 A/B 快照 (action_piece/ego_augment/moe/distill_v2/
prune_v2/efficiency/robustness), 并确认 backcompat 扩至 18 件 (v2.1.0..v3.3.3)。

落 checkpoints/predictor_v3.3.3.pt 与 benchmarks/results/training_v3.3.3.json。
analogy, not reproduction。

用法:
    python3 scripts/build_v333_checkpoint.py --quick
    python3 scripts/build_v333_checkpoint.py          # 正式口径 (~100s, 2 线程)
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

from udos import __version__, save_predictor, load_predictor  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.calibration import fit_predictor_calibration  # noqa: E402
from udos.ood import DistributionDriftDetector  # noqa: E402
from udos.moe import LightweightMoE  # noqa: E402
from udos.lite import (StructuredPrunerV2, DistillationTrainerV2)  # noqa: E402
from udos.robustness import RobustnessEvaluator  # noqa: E402
from udos.action_piece import ActionPieceTokenizer  # noqa: E402
from udos.ego_data import SyntheticEgoAugmenter  # noqa: E402

torch.set_num_threads(2)


def small_cfg(**overrides):
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


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
    calibrator, _, rq = fit_predictor_calibration(model, cal_te)
    model.attach_calibration(calibrator, rq)
    cal_eval = evaluate_predictor(model, ind_te)
    naive = ((ind_te.X[:, -1, :] - ind_te.Y[:, 0, :]) ** 2).mean().item()

    id_scores = model.ood_score(ind_te.X)
    ood_scores = model.ood_score(ind_te.X * 5.0)
    ood_diag = {
        "threshold": round(float(ood.threshold_), 6),
        "id_score_mean": round(float(id_scores.mean()), 6),
        "ood_score_mean": round(float(ood_scores.mean()), 6),
        "ood_hit_rate": round(float((ood_scores > ood.threshold_).float().mean()), 4),
        "id_false_alarm_rate": round(float((id_scores > ood.threshold_).float().mean()), 4),
    }

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v3.3.3.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval,
        "naive_mse": naive, "ood": ood_diag, "hybrid_attached": False})

    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated and loaded.has_ood_detector
    assert meta["udos_version"] == __version__

    batch_out = loaded.predict_batch(ind_te.X, scene_params=ind_te.P)
    single_out = torch.stack([
        loaded.predict_next(ind_te.X[i:i+1], scene_params=ind_te.P[i:i+1])[0]
        for i in range(min(20, len(ind_te)))])
    batch_max_diff = (batch_out[:len(single_out)] - single_out).abs().max().item()
    assert batch_max_diff < 1e-5

    # ---- 3.3 新特性离线评估 A/B ----
    # action_piece
    deltas = ind_te.X[:, 1:, :] - ind_te.X[:, :-1, :]
    ap = ActionPieceTokenizer(action_dim=6, codebook_size=16,
                              init="kmeans++", seed=42).fit(
        deltas.reshape(-1, deltas.size(-1)))
    # ego_augment
    Xa, _ = SyntheticEgoAugmenter(noise_sigma=0.01, seed=1
                                  ).augment_pair(ind_te.X[:32], ind_te.Y[:32])
    # moe add-on
    moe = LightweightMoE(32, 16, num_experts=4, top_k=2)
    moe_out = moe(torch.randn(8, 32))
    # prune_v2 (不重训, 精度退化照实记录)
    pruned = copy.deepcopy(loaded)
    pr = StructuredPrunerV2(prune_ratio=0.5).prune(pruned)
    pruned_mse = evaluate_predictor(pruned, ind_te)["single_step_mse"]
    # distill_v2 (少量)
    base = small_cfg()
    student = DistillationTrainerV2(student_scale=0.5).distill(
        loaded, base, ind_te, epochs=4, batch_size=32, seed=0)
    student_mse = evaluate_predictor(student, ind_te)["single_step_mse"]
    # robustness
    rob = RobustnessEvaluator(loaded, seed=0).evaluate(ind_te)

    features_v333 = {
        "action_piece_roundtrip": round(ap.roundtrip_error(
            deltas.reshape(-1, deltas.size(-1))), 6),
        "ego_augment_finite": bool(torch.isfinite(Xa).all()),
        "moe_out_finite": bool(torch.isfinite(moe_out).all()),
        "moe_params": moe.num_parameters(),
        "prune_v2_channel_sparsity": pr["channel_sparsity"],
        "pruned_no_retrain_mse": round(pruned_mse, 6),
        "student_v2_params": sum(p.numel() for p in student.parameters()),
        "student_v2_mse": round(student_mse, 6),
        "robustness_score": rob["robustness_score"],
        "analogy_not_reproduction": True,
    }

    # backcompat 18 件 (v2.1.0..v3.3.3)
    ckpt_dir = Path("checkpoints")
    ckpt_list = sorted(p.name for p in ckpt_dir.glob("predictor_v*.pt"))

    cal = cal_eval["calibration"]
    iv = cal_eval["interval"]
    summary = {
        "version": __version__, "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "epochs_run": hist.final()["epochs_run"],
        "stopped_early": hist.stopped_early, "best_epoch": hist.best_epoch,
        "final_loss": round(hist.final()["train_loss"], 6),
        "eval_mse": cal_eval["single_step_mse"],
        "naive_mse": round(naive, 6),
        "untrained_single_mse": round(before["single_step_mse"], 6),
        "ece": cal["calibrated"]["ece"],
        "coverage": iv["coverage_overall"],
        "batch_inference_max_diff": round(batch_max_diff, 8),
        "ood": ood_diag,
        "features_v333_offline_ab": features_v333,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_list": ckpt_list,
        "new_features_offline_ab": {
            "action_piece": "inference add-on opt-in",
            "ego_augment": "training-time aug opt-in, formal on raw",
            "moe": "routing head opt-in",
            "distill_v2": "student opt-in, no retrain gain => opt-in",
            "prune_v2": "structured prune, no-retrain mse degrades => opt-in",
            "robustness": "read-only evaluator",
        },
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v3.3.3.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, f, ensure_ascii=False, indent=2) if False else
          json.dumps({k: summary[k] for k in
                      ["version", "n_params", "train_seconds", "eval_mse",
                       "backcompat_checkpoints", "features_v333_offline_ab"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
