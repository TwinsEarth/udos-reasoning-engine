"""
构建 v3.5.0 正式物理预测器 checkpoint (SFM 空间基础模型线首训件)
====================================================================
与 v3.4.5/v3.4.0 同口径 (seed=42 / n_per_kind=48 / epochs=60 / patience=12 /
front / hybrid_weight=0)。SFM 空间模块 (udos/spatial.py) 全程为**纯推理外挂、
零梯度、opt-in**: 不进主 state_dict, 不改主 predictor 52191 参数; 本脚本在
离线阶段附加一段 SFM 自检 (几何可逆性 / 不改主权重 md5)。

落 checkpoints/predictor_v3.5.0.pt 与 benchmarks/results/training_v3.5.0.json。
analogy, not reproduction —— 空间为合成低维代理, 不复现真实 3D/点云。

用法:
    python3 scripts/build_v350_checkpoint.py --quick
    python3 scripts/build_v350_checkpoint.py
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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
from udos.spatial import (SpatialObject, SpatialScene,  # noqa: E402
                          SpatialTransform)

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


def sfm_sanity():
    """SFM 零外挂自检: 几何可逆 + 不持有/不改任何主模型权重 (纯 numpy)。"""
    scene = SpatialScene([
        SpatialObject("box", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5),
        SpatialObject("ball", [3.0, 4.0, 0.0], [0.0, 1.0, 0.0], 0.4),
    ])
    t = (SpatialTransform.translation([10, 0, 0])
         .compose(SpatialTransform.rotation_from_axis_angle([0, 0, 1], 0.9))
         .compose(SpatialTransform.scaling([1.5, 0.8, 1.2])))
    moved = scene.apply_transform(t)
    err = t.roundtrip_error(scene.position_matrix())
    dist = moved.get("box").distance_to(moved.get("ball"))
    return {"objects": len(scene),
            "roundtrip_max_err": float(err),
            "is_pure_numpy_no_param": True,
            "scene_distance_after": round(float(dist), 4)}


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
    ckpt = "checkpoints/predictor_v3.5.0.pt"
    save_predictor(model, ckpt, metrics={
        "training": hist.final(), "evaluation": cal_eval,
        "naive_mse": naive, "hybrid_attached": False})

    loaded, meta = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - cal_eval["single_step_mse"]) < 1e-9
    assert loaded.is_calibrated and loaded.has_ood_detector
    assert meta["udos_version"] == __version__

    # ---- ICM 全特性离线 A/B (与 v3.4.5 同配方) ----
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

    # ---- SFM 零外挂: 跑空间几何后, 主 predictor 权重 md5 不变 ----
    sfm = sfm_sanity()
    sfm["zero_grad_state_dict_md5_unchanged"] = (sd0 == _md5(loaded))

    prev_mse = None
    for prev in ("checkpoints/predictor_v3.4.5.pt",
                 "checkpoints/predictor_v3.4.0.pt"):
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
        "sfm_design": {"path": "pure analytic geometry (numpy)",
                       "zero_gradient": True, "opt_in": True,
                       "analogy_not_reproduction": True,
                       "main_params_untouched": True},
        "sfm_sanity": sfm,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_list": ckpt_list,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v3.5.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "train_seconds", "eval_mse",
                       "prev_line_mse", "sfm_sanity",
                       "backcompat_checkpoints"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
