"""
ActionPiece tokenized vs 连续动作 A/B (v3.1.0.dev4)
================================================================
在合成动作上对比:
    * 连续基线: 直接用全局均值预测动作 (naive continuous baseline);
    * tokenized: k-means 码本量化 -> 反量化, 量化解码 MSE。
扫描码本大小 K in {8,16,32,64}, 记录 eval_mse / 推理延迟 / 参数量。

结论落 benchmarks/results/action_piece_ab_v3.1.0.json。
诚实纪律: 若 tokenized 相对连续基线无显著 MSE 收益, 照实标 opt-in / 被否决候选,
不夸大收益。analogy, not reproduction —— 合成动作, 非真机/VLM。

用法: python3 scripts/action_piece_ab_v31.py
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import ActionPieceTokenizer  # noqa: E402

torch.set_num_threads(2)


def make_synthetic_actions(n_per_cluster=80, n_clusters=6, dim=6, seed=7):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, dim, generator=g) * 3.0
    chunks = [c + 0.25 * torch.randn(n_per_cluster, dim, generator=g)
              for c in centers]
    return torch.cat(chunks, dim=0)


def main():
    torch.manual_seed(0)
    actions = make_synthetic_actions()
    N, D = actions.shape

    # 连续基线: 全局常数均值 (naive continuous predictor)
    global_mean = actions.mean(dim=0, keepdim=True)
    baseline_mse = float(((actions - global_mean) ** 2).mean())

    scan = []
    for K in (8, 16, 32, 64):
        tok = ActionPieceTokenizer(action_dim=D, codebook_size=K,
                                   init="kmeans++", seed=42).fit(actions)
        # 推理延迟: encode + decode (warmup 1 次)
        tok.encode(actions[:8])
        t0 = time.perf_counter()
        ids = tok.encode(actions)
        recon = tok.decode(ids)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        mse = float(((actions - recon) ** 2).mean())
        scan.append({
            "codebook_size": K,
            "quant_mse": round(mse, 6),
            "rel_vs_baseline": round(mse / max(baseline_mse, 1e-12), 4),
            "latency_ms": round(latency_ms, 3),
            "params": K * D,           # 码本参数量
            "utilization": round(tok.utilization(), 4),
        })

    best = min(scan, key=lambda r: r["quant_mse"])
    # 诚实判定: tokenized 是否真正优于连续均值基线
    tokenized_wins = best["quant_mse"] < baseline_mse * 0.5
    summary = {
        "benchmark": "action_piece_ab_v3.1.0",
        "analogy_not_reproduction": True,
        "n_actions": N, "action_dim": D,
        "continuous_baseline_mse": round(baseline_mse, 6),
        "scan": scan,
        "best_codebook": best["codebook_size"],
        "tokenized_quant_mse": best["quant_mse"],
        "tokenized_wins_over_continuous": bool(tokenized_wins),
        "verdict": ("tokenized quantization reduces action MSE, opt-in"
                    if tokenized_wins else
                    "no robust MSE gain over continuous baseline; kept as "
                    "opt-in rejected candidate"),
        "opt_in_default_off": True,
    }
    out = ROOT / "benchmarks" / "results" / "action_piece_ab_v3.1.0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
