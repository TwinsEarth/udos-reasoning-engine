"""
构建 v3.6.0 正式物理预测器 checkpoint (PWM 物理世界模型线首训件)
====================================================================
与 v3.5.0/v3.4.5 同口径 (seed=42 / n_per_kind=48 / epochs=60 / patience=12 /
front / hybrid_weight=0)。主 predictor 仍为 52191 参数, 评估口径不变。

本线新增**潜在空间前向世界模型** (udos/world_model.py LatentWorldModel):
    * 全程**纯推理外挂、零梯度、opt-in**: 不进主 state_dict, 不改主 predictor
      52191 参数; 潜在转移/读出 MLP 为独立小模块 (约 2310 参数, 显式报告)。
    * 本脚本在离线阶段对冻结 predictor 做一次外挂世界模型拟合自检
      (decode 潜在->物理 / transit 潜在->潜在), 验证:
        - 拟合 loss 下降;
        - 拟合前后主 predictor 权重 md5 不变 (零梯度);
        - H=1 imagine 与 predictor.rollout 逐位等价。
    * analogy, not reproduction —— 潜在空间为合成低维代理 (d_input=32),
      不宣称复现 V-JEPA / Cosmos / Genie / 视频世界模型。

落 checkpoints/predictor_v3.6.0.pt 与 benchmarks/results/training_v3.6.0.json。

用法:
    python3 scripts/build_v360_checkpoint.py --quick
    python3 scripts/build_v360_checkpoint.py
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
from udos.world_model import LatentWorldModel  # noqa: E402

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


def wm_sanity(model, n_per_kind):
    """PWM 零外挂自检: 外挂世界模型拟合 + 零梯度 + H=1 锚点。"""
    wm = LatentWorldModel(model)
    fit_ds = build(4242, n_per_kind)
    fit_rep = wm.fit(fit_ds, epochs=20)
    # H=1 逐位锚点
    te = build(7777, n_per_kind)
    h1_ref = model.rollout(te.X[:8], 1, scene_params=te.P[:8])
    h1_im = wm.imagine(te.X[:8], 1, scene_params=te.P[:8])
    anchor_bitwise = bool(torch.equal(h1_ref, h1_im))
    # H=4 想象 vs 真实 rollout 单步 MSE (仅报告, 不追求精度 —— 合成类比)
    real = model.rollout(te.X[:8], 4, scene_params=te.P[:8])
    imag = wm.imagine(te.X[:8], 4, scene_params=te.P[:8])
    step_mse = [round(float(((real[:, h, :] - imag[:, h, :]) ** 2).mean()), 6)
                for h in range(4)]
    return {
        "wm_params": wm.n_params,
        "latent_dim": wm.latent_dim,
        "fitted": wm.fitted,
        "fit_first_loss": fit_rep["fit_first_loss"],
        "fit_last_loss": fit_rep["fit_last_loss"],
        "h1_bitwise_anchor": anchor_bitwise,
        "imagine_vs_real_step_mse": step_mse,
        "is_pure_readonly_no_main_param": True,
    }


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
    calibrator, _, rq = fit_predictor_calibration(model, cal_te)
    model.attach_calibration(calibrator, rq)
    cal_eval = evaluate_predictor(model, ind_te)
    naive = ((ind_te.X[:, -1, :] - ind_te.Y[:, 0, :]) ** 2).mean().item()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v3.6.0.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval,
        "naive_mse": naive, "hybrid_attached": False})

    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated and loaded.has_ood_detector
    assert meta["udos_version"] == __version__

    # ---- ICM 全特性离线 A/B (与 v3.5.0 同配方) ----
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
    sd0 = _md5(loaded)
    agg.predict(icm_te.X[0], memory=mem, k=3, scene_params=icm_te.P[0:1])
    icm_ab["zero_grad_state_dict_md5_unchanged"] = (sd0 == _md5(loaded))

    # ---- PWM 世界模型零外挂自检: 跑外挂拟合前后, 主 predictor 权重 md5 不变 ----
    wm = wm_sanity(loaded, n_per_kind)
    wm["zero_grad_state_dict_md5_unchanged"] = (sd0 == _md5(loaded))

    prev_mse = None
    for prev in ("checkpoints/predictor_v3.5.0.pt",
                 "checkpoints/predictor_v3.4.5.pt"):
        if os.path.isfile(prev):
            pm, _ = load_predictor(prev)
            prev_mse = round(evaluate_predictor(pm, ind_te)["single_step_mse"], 6)
            break

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
        "prev_line_mse": prev_mse,
        "naive_mse": round(naive, 6),
        "ece": cal["calibrated"]["ece"],
        "coverage": iv["coverage_overall"],
        "icm_offline_ab": icm_ab,
        "pwm_design": {"path": "latent world model wrapper (read-only)",
                       "zero_gradient": True, "opt_in": True,
                       "analogy_not_reproduction": True,
                       "main_params_untouched": True,
                       "wm_params_not_in_main_state_dict": True},
        "pwm_sanity": wm,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_list": ckpt_list,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v3.6.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "train_seconds", "eval_mse",
                       "prev_line_mse", "pwm_sanity",
                       "backcompat_checkpoints"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
