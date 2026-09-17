"""
v3.6.0.dev6 长 horizon 想象 vs 真实 rollout 对比实验
=====================================================
想象 rollout 在长 horizon (H=8/16/32) 下的误差累积 vs 真实 predictor rollout;
潜在空间压缩率; 落 benchmarks/results/wm_imagine_v3.6.0.json。

analogy, not reproduction —— 合成低维潜在代理 (d_input=32), 非视频世界模型复现;
如实报告长 horizon 误差累积, 不宣称想象可替代真实 rollout。

用法:
    python3 scripts/wm_imagine_ab_v36.py
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.world_model import LatentWorldModel  # noqa: E402

torch.set_num_threads(2)


def main():
    ckpt = ROOT / "checkpoints" / "predictor_v3.6.0.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "predictor_v3.5.0.pt"
    model, _ = load_predictor(str(ckpt))
    model.eval()
    wm = LatentWorldModel(model)
    fit_ds = build_parametric_dataset(n_per_kind=16, n_steps=16, window=6,
                                      horizon=8, dt=0.5, seed=4343)
    wm.fit(fit_ds, epochs=25)

    te = build_parametric_dataset(n_per_kind=16, n_steps=40, window=6,
                                  horizon=32, dt=0.5, seed=9191)
    X, P = te.X, te.P

    horizons = [8, 16, 32]
    per_horizon = {}
    with torch.no_grad():
        for H in horizons:
            real = model.rollout(X, H, scene_params=P)
            imag = wm.imagine(X, H, scene_params=P)
            step_mse = [round(float(((real[:, h, :] - imag[:, h, :]) ** 2)
                                    .mean()), 6) for h in range(H)]
            per_horizon[f"H={H}"] = {
                "step_mse": step_mse,
                "final_step_mse": step_mse[-1],
                "mean_step_mse": round(sum(step_mse) / len(step_mse), 6),
                "step0_mse_anchor": step_mse[0],
            }

    # 潜在空间压缩率: 窗口展平维 (W*raw) vs 潜在维
    W = X.size(1)
    window_flat = W * wm.raw_dim
    compression = round(window_flat / wm.latent_dim, 3)

    summary = {
        "version": __version__,
        "checkpoint": str(ckpt.name),
        "wm_params": wm.n_params,
        "main_params": sum(p.numel() for p in model.parameters()),
        "horizons": horizons,
        "per_horizon": per_horizon,
        "latent_dim": wm.latent_dim,
        "window_flat_dim": int(window_flat),
        "compression_ratio_window_over_latent": compression,
        "analogy_not_reproduction": True,
        "note": "合成低维潜在代理; 长 horizon 想象误差累积为预期现象, "
                "不宣称复现视频世界模型, 不替代主 rollout。",
    }
    out = ROOT / "benchmarks" / "results" / "wm_imagine_v3.6.0.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: per_horizon[k] for k in
                      ["H=8", "H=16", "H=32"]}, ensure_ascii=False, indent=2))
    print("compression_ratio_window_over_latent =", compression)


if __name__ == "__main__":
    main()
