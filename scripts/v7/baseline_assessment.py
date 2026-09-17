#!/usr/bin/env python3
"""
v7.0.0 重构基线评估 (baseline-first, 无测量不结论)
=================================================
在重构任何代码前, 先对当前发布模型 predictor_v4.3.9.pt 在**严格独立 held-out**
(训练 seed=42; 本评估 seed=2026, 同分布但训练从未见过) 上测出真实数字:

  1. 四类运动 + 整体: 单步 / 4 步 rollout 的场景盲 vs 带场景 MSE 与增益倍数;
  2. split-conformal 预测区间在名义 80/90/95% 下的**经验边际覆盖率**
     (检验"区间是否真的覆盖真值", 而非只看区间宽度);
  3. CPU 推理延迟 (batch=1, 单步 / 4 步, 中位数), 记录线程数与 torch 版本;
  4. 参数量 / buffer 量, 钉死模型规模事实。

结果写到 scripts/v7/baseline_v4.3.9_heldout_seed2026.json, 不覆盖
benchmarks/ 历史快照。这是 v7 要"根治差距"的起点证据, 不是优化结论。
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from udos.dynamics import build_parametric_dataset
from udos.persistence import load_predictor

CKPT = REPO / "checkpoints" / "predictor_v4.3.9.pt"
SEED, N_PER_KIND, W, H, DT = 2026, 128, 6, 4, 0.5
ALPHAS = (0.2, 0.1, 0.05)  # 名义覆盖 80% / 90% / 95%
KINDS = ("uniform", "accel", "spring", "collision")


def mse(pred, true):
    return float(((pred - true) ** 2).mean())


def main():
    torch.manual_seed(0)
    predictor, meta = load_predictor(CKPT)
    predictor.eval()
    ds = build_parametric_dataset(n_per_kind=N_PER_KIND, n_steps=W + H + 4,
                                  window=W, horizon=H, dt=DT, seed=SEED)
    X, Y, P = ds.X, ds.Y, ds.P
    n_learn = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_buf = sum(b.numel() for b in predictor.buffers())

    report = {
        "model": "predictor_v4.3.9.pt",
        "heldout_seed": SEED, "train_seed": 42,
        "n_samples": int(X.size(0)), "window": W, "horizon": H, "dt": DT,
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "learnable_params": int(n_learn), "buffers": int(n_buf),
        "mse": {}, "coverage": {}, "latency_ms": {},
    }

    with torch.no_grad():
        # ---- 单步 / 多步, 盲 vs 带场景, 分运动类型 ----
        blind1 = predictor.predict_next(X)
        scene1 = predictor.predict_next(X, scene_params=P)
        blind4 = predictor.rollout(X, H)
        scene4 = predictor.rollout(X, H, scene_params=P)

        def block(pred1, pred4, mask=None):
            m = mask if mask is not None else slice(None)
            return {
                "single_step": round(mse(pred1[m], Y[m, 0]), 6),
                "rollout_4": round(mse(pred4[m], Y[m]), 6),
            }

        report["mse"]["overall_blind"] = block(blind1, blind4)
        report["mse"]["overall_scene"] = block(scene1, scene4)
        report["mse"]["gain_x"] = {
            "single_step": round(
                report["mse"]["overall_blind"]["single_step"]
                / report["mse"]["overall_scene"]["single_step"], 2),
            "rollout_4": round(
                report["mse"]["overall_blind"]["rollout_4"]
                / report["mse"]["overall_scene"]["rollout_4"], 2),
        }
        for k in KINDS:
            mk = ds.kind_mask(k)
            report["mse"][f"blind_{k}"] = block(blind1, blind4, mk)
            report["mse"][f"scene_{k}"] = block(scene1, scene4, mk)

        # ---- conformal 经验边际覆盖率 (带场景) ----
        for a in ALPHAS:
            try:
                iv = predictor.predict_interval(X, H, scene_params=P, alpha=a)
                lo, up = iv["lower"], iv["upper"]
                inside = ((Y >= lo) & (Y <= up)).float()
                nominal = 1.0 - a
                per_step = [round(float(inside[:, h].mean()), 4)
                            for h in range(H)]
                report["coverage"][f"alpha_{a}_nominal_{nominal:.2f}"] = {
                    "marginal_overall": round(float(inside.mean()), 4),
                    "per_step": per_step,
                    "nominal": round(nominal, 2),
                }
            except Exception as e:  # 区间未挂载等: 诚实记录, 不伪造
                report["coverage"][f"alpha_{a}"] = {"error": repr(e)}

        # ---- CPU 延迟 batch=1 (warmup 20, 计时 200) ----
        x1, p1 = X[:1], P[:1]
        for _ in range(20):
            predictor.predict_next(x1, scene_params=p1)
        t1 = []
        for _ in range(200):
            t0 = time.perf_counter()
            predictor.predict_next(x1, scene_params=p1)
            t1.append((time.perf_counter() - t0) * 1000)
        for _ in range(20):
            predictor.rollout(x1, H, scene_params=p1)
        t4 = []
        for _ in range(200):
            t0 = time.perf_counter()
            predictor.rollout(x1, H, scene_params=p1)
            t4.append((time.perf_counter() - t0) * 1000)
        report["latency_ms"] = {
            "predict_next_batch1_median": round(statistics.median(t1), 4),
            "rollout4_batch1_median": round(statistics.median(t4), 4),
        }

    out = REPO / "scripts" / "v7" / "baseline_v4.3.9_heldout_seed2026.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
