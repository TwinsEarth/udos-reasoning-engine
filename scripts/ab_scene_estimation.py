#!/usr/bin/env python3
"""
v5.4.9 场景条件三方 A/B 报告生成
================================
对发布 checkpoint 在严格独立 held-out (训练 seed=42; 本报告 seed=2026) 上
比较 blind / explicit / estimated 三种场景条件, 结果落
reports/v549_scene_ab.json。

用法: python3 scripts/ab_scene_estimation.py [--seed 2026] [--n-per-kind 128]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from udos.dynamics import build_parametric_dataset
from udos.persistence import load_predictor
from udos.scene_estimation_ab import three_way_scene_ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-per-kind", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.5)
    args = ap.parse_args()

    ckpt = REPO / "checkpoints" / "predictor_v4.3.9.pt"
    predictor, meta = load_predictor(ckpt)
    predictor.eval()
    ds = build_parametric_dataset(
        n_per_kind=args.n_per_kind, window=6, horizon=args.horizon,
        dt=args.dt, seed=args.seed)
    report = three_way_scene_ab(predictor, ds, dt=args.dt, horizon=args.horizon)
    report["model"] = ckpt.name
    report["heldout_seed"] = args.seed
    report["train_seed"] = 42

    out_dir = REPO / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v549_scene_ab.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
