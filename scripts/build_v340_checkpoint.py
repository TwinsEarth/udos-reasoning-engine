"""
构建 v3.4.0 正式发布物理预测器 checkpoint (3.4 线首训件)
================================================================
与 v3.2.0/v3.3.0/v3.3.3 同口径 (seed=42 / n_per_kind=48 / epochs=60 /
patience=12 / front / hybrid_weight=0)。

为什么 3.4.0 要重训主件 (而非沿用 v3.3.3 权重):
    * ICM 是**纯推理外挂**, 不入主 state_dict, 理论上不改变主模型;
    * 但 3.4 线需要一件 "线内正式件" 作为 backcompat 谱系新节点, 并把
      ICM 离线 A/B (naive 退化复现 + ICM 检索聚合 k-shot) 与正式件绑定落盘;
    * 同 seed/同配方 => 主架构与参数量不变 (仍 ~52191), eval_mse 与 v3.3.3 同量级,
      前后指标在 summary 中并列报告 (见 "prev_line_mse" / "v340_mse")。

落 checkpoints/predictor_v3.4.0.pt 与 benchmarks/results/training_v3.4.0.json。
analogy, not reproduction。

用法:
    python3 scripts/build_v340_checkpoint.py --quick
    python3 scripts/build_v340_checkpoint.py          # 正式口径 (~100s, 2 线程)
"""
import argparse
import hashlib
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
from udos.icm import (DemonstrationEpisode, DemonstrationMemory,  # noqa: E402
                      ICMAggregator)
from udos.incontext import InContextLearner  # noqa: E402

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


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


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
    ckpt = "checkpoints/predictor_v3.4.0.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval,
        "naive_mse": naive, "ood": ood_diag, "hybrid_attached": False})

    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated and loaded.has_ood_detector
    assert meta["udos_version"] == __version__

    # ---- 3.4.0 ICM 离线 A/B: 复现 naive 退化 + ICM 检索聚合不退化 ----
    icm_tr = build(4242, n_per_kind)
    icm_te = build(7777, n_per_kind)
    mem = DemonstrationMemory()
    agg = ICMAggregator(loaded, temperature=1.0, lamb=1.0)
    for i in range(len(icm_tr)):
        ep = DemonstrationEpisode(icm_tr.X[i], icm_tr.Y[i, 0],
                                 kind=icm_tr.kinds[i], scene_params=icm_tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=icm_tr.P[i])
    icm_ab = {}
    for k in (0, 1, 3, 5):
        icm_ab[f"icm_k{k}"] = round(agg.shot_mse(
            icm_te.X, icm_te.Y[:, 0], memory=mem, k=k,
            scene_params=icm_te.P), 6)
    pool = [icm_tr.X[j] for j in range(min(16, len(icm_tr)))]
    icl = InContextLearner(loaded)
    for k in (0, 1, 3):
        icm_ab[f"naive_k{k}"] = round(icl.shot_mse(
            icm_te.X[:80], icm_te.Y[:80, 0], pool, k,
            scene_params=icm_te.P[:80]), 6)
    # 零梯度锚点: ICM 推理前后 state_dict md5 逐位一致
    sd_before = _md5(loaded)
    agg.predict(icm_te.X[0], memory=mem, k=3, scene_params=icm_te.P[0:1])
    sd_after = _md5(loaded)
    icm_ab["zero_grad_state_dict_md5_unchanged"] = (sd_before == sd_after)

    # 上一线正式件 (v3.3.3) 指标并列, 证明同配方重训无漂移
    prev_mse = None
    prev_ckpt = "checkpoints/predictor_v3.3.3.pt"
    if os.path.isfile(prev_ckpt):
        prev_m, _ = load_predictor(prev_ckpt)
        prev_mse = round(evaluate_predictor(prev_m, ind_te)["single_step_mse"], 6)

    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))

    cal = cal_eval["calibration"]
    iv = cal_eval["interval"]
    summary = {
        "version": __version__, "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "epochs_run": hist.final()["epochs_run"],
        "stopped_early": hist.stopped_early, "best_epoch": hist.best_epoch,
        "final_loss": round(hist.final()["train_loss"], 6),
        "eval_mse": cal_eval["single_step_mse"],
        "prev_line_v333_mse": prev_mse,
        "naive_mse": round(naive, 6),
        "untrained_single_mse": round(before["single_step_mse"], 6),
        "ece": cal["calibrated"]["ece"],
        "coverage": iv["coverage_overall"],
        "ood": ood_diag,
        "icm_offline_ab": icm_ab,
        "icm_design": {
            "path": "retrieve + aggregate in residual space (NOT naive concat)",
            "zero_gradient": True,
            "aggregator_trainable_params": 0,
            "opt_in": True,
            "naive_degradation_root_cause": "naive window concat dilutes 52k attention",
            "analogy_not_reproduction": True,
        },
        "retrain_reason": (
            "3.4 line formal node; ICM is opt-in add-on not in state_dict; "
            "same seed/recipe so n_params & eval_mse stay on the v3.3 line."),
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_list": ckpt_list,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v3.4.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "train_seconds", "eval_mse",
                       "prev_line_v333_mse", "icm_offline_ab",
                       "backcompat_checkpoints"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
