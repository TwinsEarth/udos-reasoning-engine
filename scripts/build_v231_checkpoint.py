"""
构建 v2.3.1 物理预测器 checkpoint (可复现)
=========================================
在 v2.2.1 同口径训练(默认 front 权重=teacher-forcing 等价)之上新增 v2.3 可信能力:
- 在**独立校准集**上拟合保序置信校准 + 每步残差分位 (split-conformal);
- 在**另一个独立测试集**上对照 校准前/后 ECE 与区间覆盖率 (唯一变量=校准映射);
- 落 checkpoints/predictor_v2.3.1.pt (校准器随件持久化) 与
  benchmarks/results/training_v2.3.1.json, 并多维校验保存/重载一致。

用法:
    python3 scripts/build_v231_checkpoint.py --quick
    python3 scripts/build_v231_checkpoint.py
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

torch.set_num_threads(2)


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def build(seed, n_per_kind, horizon=4):
    # 独立校准/测试集: 整份使用 (它们已按独立种子生成, 无需再 train/val 切分;
    # 切分只会让逐维 conformal 半宽的估计样本变少、区间不稳)。
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

    # 训练(含早停验证) / 独立校准集 / 独立测试集, 三集种子互不相同
    train_full = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                          window=6, horizon=4, dt=0.5, seed=42)
    tr, te_es = train_full.split(0.8)
    cal_te = build(314, n_per_kind)      # 校准集
    ind_te = build(2718, n_per_kind)     # 独立测试集

    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    n_params = sum(p.numel() for p in model.parameters())
    tcfg = TrainConfig(epochs=epochs, lr=3e-3, batch_size=64,
                       patience=12, step_weight_scheme="front")
    t0 = time.time()
    hist = CTMTrainer(model, tcfg).train(tr, te_es)
    train_s = time.time() - t0

    before = evaluate_predictor(PhysicsPredictor(small_cfg(), scene_param_dim=4),
                                ind_te)
    # 1) 独立校准集拟合; 2) 校准前在独立测试集留底; 3) 挂载后独立测试集对照
    calibrator, fitted_report, rq = fit_predictor_calibration(model, cal_te)
    raw_eval = evaluate_predictor(model, ind_te)
    model.attach_calibration(calibrator, rq)
    cal_eval = evaluate_predictor(model, ind_te)
    naive = ((ind_te.X[:, -1, :] - ind_te.Y[:, 0, :]) ** 2).mean().item()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v2.3.1.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval, "naive_mse": naive})

    # 多维重载一致性: 权重逐位 + 校准器挂载 + 校准输出 + 区间覆盖
    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated, "重载后校准器丢失"
    assert rep2["calibration"]["calibrated"]["ece"] == \
        cal_eval["calibration"]["calibrated"]["ece"]
    assert rep2["interval"]["coverage_overall"] == \
        cal_eval["interval"]["coverage_overall"]

    cal = cal_eval["calibration"]
    summary = {
        "version": __version__, "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "epochs_run": hist.final()["epochs_run"],
        "stopped_early": hist.stopped_early, "best_epoch": hist.best_epoch,
        "step_weight_scheme": "front",
        "untrained_single_mse": round(before["single_step_mse"], 6),
        "naive_mse": round(naive, 6),
        "trained_single_mse": cal_eval["single_step_mse"],
        "reduction_vs_untrained_x": round(
            before["single_step_mse"] / max(cal_eval["single_step_mse"], 1e-12), 2),
        "reduction_vs_naive_x": round(
            naive / max(cal_eval["single_step_mse"], 1e-12), 2),
        "condition_gain_x": cal_eval["ablation"]["condition_gain_x"],
        "rollout_mse_curve": cal_eval["rollout_mse_curve"],
        "rollout_growth_x": cal_eval["rollout_growth_x"],
        # v2.3 F1: 校准前/后 ECE (独立测试集), 及排序信息量
        "calibration": {
            "fitted_on_calibration_set": fitted_report,
            "independent_test_raw_ece": cal["raw"]["ece"],
            "independent_test_calibrated_ece": cal["calibrated"]["ece"],
            "ece_reduction_x": cal["ece_reduction_x"],
            "ranking_informative": cal["ranking_informative"],
            "num_segments": cal["num_segments"],
            "raw_spearman_conf_err": cal["raw"]["spearman_conf_err"],
            "calibrated_spearman_conf_err":
                cal["calibrated"]["spearman_conf_err"],
        },
        # v2.3 F3: split-conformal 区间覆盖率/宽度
        "interval": cal_eval["interval"],
        # 校准前的分层诊断 (对照 2.2.1 的非单调问题)
        "raw_confidence_stratification":
            raw_eval["confidence_stratification"],
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v2.3.1.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
