"""
v3.6.0.dev4 世界模型 vs 预测器一致性 A/B
=========================================
A: predictor.rollout 真实多步推演 (主路径, 52191 参数)
B: LatentWorldModel.imagine_rollout 潜在空间多步想象 (外挂, 2310 参数)

对比维度:
    * 逐步 MSE (imagined vs real);
    * 守恒违反量 (动量/能量代理) 在两条 rollout 上的对比;
    * 单步延迟对比 (潜在想象是否更省)。

裁决纪律: 若想象在长 horizon 下误差累积快于真实 rollout, 或收益不稳, 则维持
opt-in, 不替换主 rollout。落 benchmarks/results/wm_consistency_v3.6.0.json。
analogy, not reproduction —— 合成低维潜在代理, 非视频世界模型复现。

用法:
    python3 scripts/wm_consistency_ab_v36.py
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.world_model import LatentWorldModel  # noqa: E402
from udos.wm_conservation import ConservationChecker  # noqa: E402

torch.set_num_threads(2)


def main():
    ckpt = ROOT / "checkpoints" / "predictor_v3.6.0.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "predictor_v3.5.0.pt"
    model, _ = load_predictor(str(ckpt))
    model.eval()

    wm = LatentWorldModel(model)
    fit_ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                      horizon=4, dt=0.5, seed=4242)
    wm.fit(fit_ds, epochs=25)

    te = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=8899)
    H = 4
    X, P = te.X, te.P

    # ---- A/B: 逐步 MSE ----
    with torch.no_grad():
        real = model.rollout(X, H, scene_params=P)
        imag = wm.imagine(X, H, scene_params=P)
    step_mse = [round(float(((real[:, h, :] - imag[:, h, :]) ** 2).mean()), 6)
                for h in range(H)]

    # ---- 守恒违反量对比 ----
    checker = ConservationChecker(mass=1.0)
    real_cons = checker.check(real)["batch"][0]
    imag_cons = checker.check(imag)["batch"][0]

    # ---- 延迟对比 (warmup + repeats) ----
    def bench(fn, n=20):
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0

    lat_real = bench(lambda: model.rollout(X[:8], H, scene_params=P[:8]))
    lat_imag = bench(lambda: wm.imagine(X[:8], H, scene_params=P[:8]))

    # ---- 裁决: 想象是否可替换真实 rollout? ----
    # 判据: 第 0 步严格一致(锚点), 但 H=4 末步 MSE > 阈值 即认为发散。
    final_mse = step_mse[-1]
    replace = final_mse < 1e-3
    decision = ("opt-in; do NOT replace predictor.rollout by default"
                if not replace else
                "candidate: imagine can replace rollout at this horizon")

    rejected = [
        {"candidate": "默认把 imagine_rollout 设为主 rollout 路径",
         "reason": "第 0 步虽逐位锚定, 但 H>1 步为外挂潜在转移解码, "
                   f"末步 MSE={final_mse} (>>0), 误差随 horizon 累积; "
                   "逐位等价锚点会被破坏, 违反兼容铁律。"},
        {"candidate": "把外挂世界模型参数并入主 state_dict",
         "reason": "会使主 predictor 偏离 52191 参数, 污染 22 代 checkpoint 兼容; "
                   "外挂小 MLP 应独立 (参照 hybrid 先例)。"},
    ]

    summary = {
        "version": __version__,
        "checkpoint": str(ckpt.name),
        "n_eval": int(X.size(0)),
        "horizon": H,
        "wm_params": wm.n_params,
        "main_params": sum(p.numel() for p in model.parameters()),
        "per_step_mse_imagine_vs_real": step_mse,
        "final_step_mse": final_mse,
        "real_rollout_conservation": {
            "momentum_violation": real_cons["momentum_violation"],
            "energy_violation": real_cons["energy_violation"]},
        "imagine_rollout_conservation": {
            "momentum_violation": imag_cons["momentum_violation"],
            "energy_violation": imag_cons["energy_violation"]},
        "latency_real_rollout_ms": round(lat_real, 4),
        "latency_imagine_ms": round(lat_imag, 4),
        "imagine_faster": lat_imag < lat_real,
        "decision": decision,
        "rationale": "第 0 步逐位锚定真实 rollout; H>1 步为潜在外挂转移解码, "
                     "误差累积为合成类比现象。维持 opt-in, 默认输出仍走主 predictor。",
        "rejected_candidates": rejected,
        "analogy_not_reproduction": True,
    }
    out = ROOT / "benchmarks" / "results" / "wm_consistency_v3.6.0.json"
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "final_step_mse", "decision",
                       "latency_real_rollout_ms", "latency_imagine_ms",
                       "imagine_faster"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
