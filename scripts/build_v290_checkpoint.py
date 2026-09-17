"""
构建 v2.9.0 物理预测器 checkpoint (2.9 线首个正式训练件)
================================================================
与 v2.8.0 / v2.7.3 同口径 (seed=42 / n_per_kind=48 / epochs=60 / patience=12 /
front / hybrid_weight=0): 主 checkpoint **不挂载 hybrid / PhysicalLoop / retargeting**,
主模型参数量仍为 52191。ActionRetargeter 为推理时外挂重定向层 (纯前向只读),
不参与训练、不改变训练口径与主模型结构。

流程: 训练 -> OOD 拟合 -> 校准拟合 -> 独立集评估 -> save -> reload 验证 -> 批量一致性
      -> 2.7 特性离线快照 -> 2.8 PhysicalLoop 离线快照 -> 2.9 retargeting 离线快照。
落 checkpoints/predictor_v2.9.0.pt 与 benchmarks/results/training_v2.9.0.json。

analogy, not reproduction: retargeting 为受 Human-as-Humanoid 启发的轻量化类比层,
用不同维动作向量代理不同机器人形态。

用法:
    python3 scripts/build_v290_checkpoint.py --quick   # 小规模快速复现
    python3 scripts/build_v290_checkpoint.py          # 正式口径 (~100s, 2 线程)
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
from udos.policy import MPCActionSelector  # noqa: E402
from udos.active_learning import UncertaintySampler  # noqa: E402
from udos.hierarchical import HierarchicalRollout  # noqa: E402
from udos.physical_loop import PhysicalLoopRunner  # noqa: E402
from udos.retargeting import MorphologyConfig, ActionRetargeter  # noqa: E402

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
    ckpt = "checkpoints/predictor_v2.9.0.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval,
        "naive_mse": naive, "ood": ood_diag, "hybrid_attached": False})

    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated
    assert loaded.hybrid is None
    assert rep2["interval"]["coverage_overall"] == \
        cal_eval["interval"]["coverage_overall"]
    assert loaded.has_ood_detector
    assert torch.allclose(loaded.ood_score(ind_te.X), id_scores, atol=1e-4)
    assert meta["udos_version"] == "2.9.0", \
        f"checkpoint 版本 {meta['udos_version']} != 2.9.0"

    # 批量推理一致性自检
    batch_out = loaded.predict_batch(ind_te.X, scene_params=ind_te.P)
    single_out = torch.stack([
        loaded.predict_next(ind_te.X[i:i+1], scene_params=ind_te.P[i:i+1])[0]
        for i in range(min(20, len(ind_te)))])
    batch_max_diff = (batch_out[:len(single_out)] - single_out).abs().max().item()
    assert batch_max_diff < 1e-5

    # ---- 2.7 特性离线快照 (外挂只读, 不改主模型) ----
    w1, p1 = ind_te.X[:1], ind_te.P[:1]
    mpc = MPCActionSelector(loaded, horizon=2)
    mpc_out = mpc.select(w1, scene_params=p1,
                         candidate_actions=[{}, {"state_perturbation": [0.0] * 6}])
    active_scores = UncertaintySampler().score_samples(loaded, ind_te.X[:8],
                                                       ind_te.P[:8])
    hier = HierarchicalRollout(loaded, coarse_factor=4)
    h8 = hier.rollout(w1, horizon=8, scene_params=p1)
    flat8 = loaded.rollout(w1, 8, scene_params=p1)
    hier_bit_identical = bool(torch.equal(h8["predictions"], flat8))
    features = {
        "policy_best_index": mpc_out["best_index"],
        "policy_no_valid_action": mpc_out["no_valid_action"],
        "active_score_mean": round(float(active_scores.mean()), 6),
        "active_score_max": round(float(active_scores.max()), 6),
        "hierarchical_h8_shape": list(h8["predictions"].shape),
        "hierarchical_bit_identical_to_flat": hier_bit_identical,
    }

    # ---- 2.8 PhysicalLoop 离线快照 (推理时外挂, 不改主模型) ----
    loop = PhysicalLoopRunner(loaded, horizon=2)
    loop_out = loop.run(w1, scene_params=p1)
    direct = loaded.predict_next(w1, scene_params=p1)
    loop_bit_identical = bool(torch.equal(loop_out["prediction"], direct))
    features_v28 = {
        "loop_prediction_shape": list(loop_out["prediction"].shape),
        "loop_steps": [s["name"] for s in loop_out["loop_state"]["steps"]],
        "loop_bit_identical_to_predict_next": loop_bit_identical,
        "loop_run_count": loop_out["loop_state"]["run_count"],
    }

    # ---- 2.9 retargeting 离线快照 (推理时外挂, 不改主模型) ----
    src_morph = MorphologyConfig(dof=6, control_freq=60.0,
                                 joint_limits=[[-2.0, 2.0]] * 6,
                                 name="prime_u_proxy")
    tgt_morph = MorphologyConfig(dof=4, control_freq=120.0,
                                 joint_limits=[[-0.5, 0.5]] * 4,
                                 name="gripper_proxy")
    rt = ActionRetargeter(src_morph, tgt_morph)
    src_actions = ind_te.X[:4, -1, :]            # [4,6] 代理源动作轨迹
    tgt_actions = rt.retarget(src_actions)
    within_limits = bool(
        (tgt_actions >= -0.5).all() and (tgt_actions <= 0.5).all())
    features_v29 = {
        "src_dof": 6, "tgt_dof": 4,
        "retargeted_shape": list(tgt_actions.shape),
        "within_target_limits": within_limits,
        "retargeted_abs_max": round(float(tgt_actions.abs().max()), 6),
        "analogy_not_reproduction": True,
    }

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
        "train_loss_curve": [round(x, 6) for x in hist.train_loss],
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
            "raw_ece": cal["raw"]["ece"],
            "calibrated_ece": cal["calibrated"]["ece"],
            "ece_reduction_x": cal["ece_reduction_x"],
            "ranking_informative": cal["ranking_informative"],
            "num_segments": cal["num_segments"],
        },
        "interval": {
            "coverage_overall": iv["coverage_overall"],
            "coverage_by_step": iv["coverage_by_step"],
            "width_by_step": iv["width_by_step"],
        },
        "ood": ood_diag,
        "features_v27": features,
        "features_v28": features_v28,
        "features_v29": features_v29,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v2.9.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
