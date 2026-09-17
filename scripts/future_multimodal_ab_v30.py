"""
未来多模态 vs 单模态未来头 A/B 证据 (v3.0.0.dev4)
====================================================
analogy, not reproduction —— 在 v3.0.0 正式件上做离线 A/B:
    * 单模态未来头: FutureStateHead (仅状态预测 [B,H,6]);
    * 多模态未来头: FutureMultimodalHead (RGB/深度/mask 三模态联合代理);
    * 对比 eval_mse / 延迟 / 参数量; 跨模态一致性损失开 vs 关。

诚实结论 (CPU 小模型, 两头发起时均为未训练线性投影):
    * 多模态头不输出状态, 不提升状态预测 MSE; 仅增加参数量与前向延迟;
    * 一致性损失 opt-in, 推理不计算, 不改变推理输出;
    * 收益不稳/无状态精度增益 => 默认 opt-in 关, 被否决/降级候选保留。

落 benchmarks/results/future_multimodal_ab_v3.0.0.json。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__  # noqa: E402
from udos.persistence import load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.multitask import MultiTaskHead, FutureStateHead  # noqa: E402
from udos.future_multimodal import (FutureMultimodalHead,
                                    CrossModalAlignmentLoss)  # noqa: E402

torch.set_num_threads(2)
CKPT = ROOT / "checkpoints" / "predictor_v3.0.0.pt"
OUT = ROOT / "benchmarks" / "results" / "future_multimodal_ab_v3.0.0.json"


def _p50(samples):
    s = sorted(samples)
    return s[len(s) // 2] * 1000.0  # -> ms


def main():
    torch.manual_seed(0)
    predictor, meta = load_predictor(str(CKPT))
    predictor.eval()

    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=7070)
    wb, pb = ds.X[:16], ds.P[:16]
    future_Y = ds.Y[:16]                    # [B,H,6] 真实未来状态

    # latent backbone 一次 encode
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    latent = mth.encode(wb, scene_params=pb)    # [B,32]

    # ---- 单模态未来头 (仅状态) ----
    torch.manual_seed(0)
    single = FutureStateHead(32, horizon=4, state_dim=6)
    single.eval()
    single_params = sum(p.numel() for p in single.parameters())
    with torch.no_grad():
        s_out = single(latent)["trajectory"]     # [B,H,6]
    single_mse = float(((s_out - future_Y) ** 2).mean())
    # 延迟
    for _ in range(5):
        single(latent)
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            single(latent)
        ts.append(time.perf_counter() - t0)

    # ---- 多模态未来头 (三模态联合) ----
    torch.manual_seed(0)
    multi = FutureMultimodalHead(32, horizon=4, rgb_dim=8, depth_dim=4,
                                 mask_dim=4)
    multi.eval()
    multi_params = sum(p.numel() for p in multi.parameters())
    with torch.no_grad():
        m_out = multi(latent, scene_params=pb)
    # 多模态不输出状态, 状态 MSE 记 n/a (诚实)
    for _ in range(5):
        multi(latent, scene_params=pb)
    tm = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            multi(latent, scene_params=pb)
        tm.append(time.perf_counter() - t0)

    # ---- 跨模态一致性损失 开 vs 关 ----
    align = CrossModalAlignmentLoss()
    with torch.no_grad():
        align_loss_value = float(align(m_out["rgb"], m_out["depth"], m_out["mask"]))
    # 关: 推理 forward 不计算损失; 输出与开损失时逐位一致
    before = multi(latent, scene_params=pb)
    # (一致性损失只在训练时调用; 推理路径不调用)
    infer_unchanged = all(
        torch.equal(before[k], m_out[k]) for k in ("rgb", "depth", "mask"))

    summary = {
        "version": __version__,
        "checkpoint": str(CKPT.name),
        "n_repeats": 50,
        "single_modal": {
            "kind": "FutureStateHead",
            "params": single_params,
            "state_mse_vs_future": round(single_mse, 6),
            "latency_p50_ms": round(_p50(ts), 4),
        },
        "multimodal": {
            "kind": "FutureMultimodalHead",
            "params": multi_params,
            "modalities": sorted(m_out.keys()),
            "state_mse_vs_future": "n/a (多模态头不输出状态, 仅代理向量)",
            "latency_p50_ms": round(_p50(tm), 4),
        },
        "alignment_loss": {
            "mode": "opt-in auxiliary, 推理不计算",
            "loss_value_on_proxy": round(align_loss_value, 6),
            "changes_inference_output": bool(not infer_unchanged),
        },
        "param_overhead_ratio": round(multi_params / max(single_params, 1), 3),
        "conclusion": (
            "多模态头不输出未来状态 => 无状态预测 MSE 增益, 仅增加参数与延迟; "
            "一致性损失 opt-in 不改变推理输出。=> 默认 opt-in 关。"),
        "rejected_candidates": [
            "默认开启多模态未来头 (无状态精度增益, 增加开销)",
            "将一致性损失挂入正式训练主损失 (训练口径 hybrid_weight=0 不变)",
        ],
    }
    os.makedirs(OUT.parent, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
