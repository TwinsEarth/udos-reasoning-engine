#!/usr/bin/env python3
"""
v5.5.0 学习型场景估计头: 训练 + 四方 A/B + 独立权重
====================================================
冻结发布主预测员 (predictor_v4.3.9.pt, 52191 参数锚点不动), 端到端训练
SceneEstimationHead (观测窗口 -> 4 维隐藏参数), 在严格 held-out (seed=2026)
上比较四种场景条件:

    blind      场景盲
    explicit   真值参数 (上界)
    classical  v5.4.8 经典运动学估计器 (不可观测槽位置 0)
    learned    v5.5.0 学习型场景头

产物:
    checkpoints/scene_head_v5.5.0.pt   (独立权重, 几 KB)
    reports/v550_head_ab.json          (四方 A/B)
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from udos.dynamics import build_parametric_dataset
from udos.persistence import load_predictor
from udos.scene_estimator import estimate_scene_params
from udos.scene_head import save_scene_head, train_scene_head


def mse(a, b):
    return float(((a - b) ** 2).mean())


def triplet_block(pred, X, Y, P, P_classical, P_learned, H):
    with torch.no_grad():
        b1 = pred.predict_next(X)
        e1 = pred.predict_next(X, scene_params=P)
        c1 = pred.predict_next(X, scene_params=P_classical)
        l1 = pred.predict_next(X, scene_params=P_learned)
        bH = pred.rollout(X, H)
        eH = pred.rollout(X, H, scene_params=P)
        cH = pred.rollout(X, H, scene_params=P_classical)
        lH = pred.rollout(X, H, scene_params=P_learned)

    def rec(blind, ref, x):
        d = blind - ref
        return round((blind - x) / d, 4) if abs(d) > 1e-12 else None

    def pack(blind, expl, cls, lrn):
        return {
            "blind": round(blind, 6), "explicit": round(expl, 6),
            "classical": round(cls, 6), "learned": round(lrn, 6),
            "classical_recovery": rec(blind, expl, cls),
            "learned_recovery": rec(blind, expl, lrn),
            "learned_over_blind": round(lrn / blind, 4) if blind > 1e-12 else None,
        }

    m = slice(None)
    return {
        "single_step": pack(mse(b1[m], Y[m, 0]), mse(e1[m], Y[m, 0]),
                            mse(c1[m], Y[m, 0]), mse(l1[m], Y[m, 0])),
        f"rollout_{H}": pack(mse(bH[m], Y[m]), mse(eH[m], Y[m]),
                             mse(cH[m], Y[m]), mse(lH[m], Y[m])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-per-kind", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout-seed", type=int, default=2026)
    args = ap.parse_args()

    ckpt = REPO / "checkpoints" / "predictor_v4.3.9.pt"
    predictor, meta = load_predictor(ckpt)
    predictor.eval()

    train_ds = build_parametric_dataset(
        n_per_kind=args.n_per_kind, window=6, horizon=args.horizon,
        dt=0.5, seed=args.seed)
    held = build_parametric_dataset(
        n_per_kind=128, window=6, horizon=args.horizon, dt=0.5,
        seed=args.heldout_seed)

    result = train_scene_head(
        predictor, train_ds, epochs=args.epochs, horizon=args.horizon,
        seed=args.seed, verbose=True)
    head = result.head

    P_classical = estimate_scene_params(held.X, 0.5).values_filled(0.0)
    with torch.no_grad():
        P_learned = head(held.X)

    overall = triplet_block(predictor, held.X, held.Y, held.P,
                            P_classical, P_learned, args.horizon)
    per_kind = {}
    for k in held.class_names:
        m = held.kind_mask(k)
        per_kind[k] = triplet_block(
            predictor, held.X[m], held.Y[m], held.P[m],
            P_classical[m], P_learned[m], args.horizon)

    n_head = sum(p.numel() for p in head.parameters())
    report = {
        "model": ckpt.name, "train_seed": args.seed,
        "heldout_seed": args.heldout_seed, "horizon": args.horizon,
        "frozen_predictor_params": sum(p.numel() for p in predictor.parameters()),
        "head_trainable_params": n_head,
        "train_final_rollout_mse": round(result.loss_history[-1], 6),
        "train_loss_history": [round(x, 6) for x in result.loss_history],
        "overall": overall, "per_kind": per_kind,
    }
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "v550_head_ab.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_scene_head(
        REPO / "checkpoints" / "scene_head_v5.5.0.pt", head,
        meta={"version": "5.5.0", "frozen_predictor": ckpt.name,
              "train_seed": args.seed, "heldout_seed": args.heldout_seed,
              "horizon": args.horizon, "epochs": args.epochs,
              "head_params": n_head})
    print(json.dumps({"overall": overall, "per_kind": per_kind,
                      "head_params": n_head}, ensure_ascii=False, indent=2))
    print("written: checkpoints/scene_head_v5.5.0.pt, reports/v550_head_ab.json")


if __name__ == "__main__":
    main()
