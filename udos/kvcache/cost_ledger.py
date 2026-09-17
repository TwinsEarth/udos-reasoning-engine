"""保留 vs 重算账本: 对 keep/offload/evict/recompute 取期望总成本最小。"""
from __future__ import annotations


def decide_action(block_bytes: float,
                  readout_bytes: float, bandwidth_gbs: float,
                 tokens: int, cost_per_token: float,
                 keep_price_per_mb: float = 0.001) -> dict:
    """成本模型(显式假设, 单位任意相对):
        keep     = bytes * keep_price
        offload  = readout/bandwidth + decompress  (取回成本)
        recompute= tokens * cost_per_token
    选最小; 返回决策 + 各候选成本。
    """
    keep = block_bytes / 1e6 * keep_price_per_mb
    offload = (readout_bytes / 1e9) / bandwidth_gbs * 1e3
    recompute = tokens * cost_per_token
    cands = {"keep": keep, "offload": offload, "recompute": recompute}
    best = min(cands, key=cands.get)
    return {"decision": best, "costs": {k: round(v, 6) for k, v in cands.items()}}


TIERS = {"hbm", "ddr", "ssd", "remote"}


def cost_retain_vs_recompute(tier: str, hit_rate: float,
                             recall_cost: float, recompute_cost: float) -> dict:
    """盈亏平衡: 保留取回成本=hit_rate*recall_cost; 重算成本=recompute_cost。
    保留取回成本 < 重算成本 -> retain, 否则 recompute。
    盈亏平衡命中率 = recompute_cost / recall_cost。"""
    if tier not in TIERS:
        raise ValueError(f"未知介质层: {tier}")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError("命中率须在 [0,1]")
    if recall_cost < 0 or recompute_cost < 0:
        raise ValueError("成本须非负")
    retain = hit_rate * recall_cost
    breakeven = (recompute_cost / recall_cost) if recall_cost > 0 else 0.0
    retain_better = retain < recompute_cost
    return {
        "tier": tier, "hit_rate": hit_rate,
        "retain_cost": round(retain, 6),
        "recompute_cost": recompute_cost,
        "breakeven_hit_rate": round(min(1.0, breakeven), 4),
        "recommendation": "retain" if retain_better else "recompute",
    }
