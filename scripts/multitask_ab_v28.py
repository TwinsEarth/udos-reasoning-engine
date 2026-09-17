"""
多任务 A/B: 共享 backbone 三头 vs 独立头 (v2.8.1)
====================================================
对加载好的 predictor_v2.8.0.pt, 在独立测试窗口上比较:
    * 共享 backbone: encode 一次, spatial/action/future 三头共用同一 latent;
    * 独立头:        每个头各自 encode 一次 (3 次 backbone 前向)。
指标: 平均延迟 / 参数量 / FutureStateHead 输出 vs predictor.rollout 的代理 eval_mse。
诚实纪律: 头为随机初始化线性投影 (不参与训练), eval_mse 仅作"两头架构输出形状/数值可比"
代理, 不宣称任务精度收益; 收益不稳 => MultiTaskHead 默认 opt-in (enable=False)。
落 benchmarks/results/multitask_ab_v2.8.0.json。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.persistence import load_predictor  # noqa: E402
from udos.multitask import (MultiTaskHead, SpatialCoordHead,  # noqa: E402
                            ActionTrajectoryHead, FutureStateHead)
from udos.dynamics import build_parametric_dataset, RAW_DIM  # noqa: E402

torch.set_num_threads(2)


def _params(*mods):
    return sum(sum(p.numel() for p in m.parameters()) for m in mods)


def main():
    predictor, _ = load_predictor("checkpoints/predictor_v2.8.0.pt")
    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4040)
    wb, pb = ds.X[:8], ds.P[:8]
    H = 4
    LAT = 32

    torch.manual_seed(0)
    spatial = SpatialCoordHead(LAT, n_pts=4)
    action = ActionTrajectoryHead(LAT, horizon=H, action_dim=RAW_DIM)
    future = FutureStateHead(LAT, horizon=H, state_dim=RAW_DIM)
    heads = [spatial, action, future]

    mth = MultiTaskHead(predictor, latent_dim=LAT, enable=True)
    mth.register_head("spatial", spatial)
    mth.register_head("action", action)
    mth.register_head("future", future)

    # warmup
    for _ in range(3):
        z = mth.encode(wb, scene_params=pb)
        for h in heads:
            h(z)

    N = 30
    # 共享: encode 一次, 三头共用
    t0 = time.perf_counter()
    for _ in range(N):
        z = mth.encode(wb, scene_params=pb)
        for h in heads:
            h(z)
    shared_s = (time.perf_counter() - t0) / N

    # 独立: 每头各自 encode
    t0 = time.perf_counter()
    for _ in range(N):
        for h in heads:
            h(mth.encode(wb, scene_params=pb))
    indep_s = (time.perf_counter() - t0) / N

    # FutureStateHead 代理 eval_mse: 其输出轨迹 vs predictor.rollout (随机头, 仅可比形状)
    with torch.no_grad():
        z = mth.encode(wb, scene_params=pb)
        fut = future(z)["trajectory"]
        ref = predictor.rollout(wb, H, scene_params=pb)
        ab_mse = float(((fut - ref) ** 2).mean())

    result = {
        "version": "2.8.1",
        "latent_dim": LAT, "horizon": H, "n_heads": 3,
        "shared_backbone": {
            "encode_calls": 1,
            "latency_ms": round(shared_s * 1000.0, 4),
            "backbone_params_shared": _params(spatial, action, future),
        },
        "independent_heads": {
            "encode_calls": 3,
            "latency_ms": round(indep_s * 1000.0, 4),
            "backbone_params_per_head": _params(spatial, action, future),
        },
        "latency_speedup_x": round(indep_s / max(shared_s, 1e-9), 3),
        "future_head_proxy_mse_vs_rollout": round(ab_mse, 6),
        "conclusion": (
            "共享 backbone 省 encode 次数 (1 vs 3); 头为随机初始化线性投影, "
            "future_head_proxy_mse 不代表任务精度收益。MultiTaskHead 默认 opt-in "
            "(enable=False), 收益不稳不默认开启。"),
        "rejected_candidate": (
            "三头联合默认开启 —— 无训练头, 精度不带来源收益, 降级为 opt-in。"),
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/multitask_ab_v2.8.0.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
