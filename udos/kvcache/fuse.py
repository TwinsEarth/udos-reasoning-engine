"""KV Fuse: 多前缀/文档缓存融合, 只复用可重叠段。"""
from __future__ import annotations


def fuse_overlap(seq_a: list, seq_b: list) -> dict:
    """两段 token 序列; 返回重叠可复用区间 [lo, hi) 与唯一段。"""
    n = min(len(seq_a), len(seq_b))
    i = 0
    while i < n and seq_a[i] == seq_b[i]:
        i += 1
    return {"shared_len": i,
            "reuse": seq_a[:i],
            "unique_a": seq_a[i:], "unique_b": seq_b[i:]}


def fuse_savings(seqs: list, cost_per_token: float = 1.0) -> dict:
    """多文档融合: 公共前缀只 prefill 一次, 其余复用。

    无融合总重算 = sum(len(seq)); 有融合 = 公共前缀 + 各文档唯一段。
    """
    if not seqs:
        raise ValueError("需要至少一段")
    common = seqs[0]
    for s in seqs[1:]:
        k = 0
        while k < min(len(common), len(s)) and common[k] == s[k]:
            k += 1
        common = common[:k]
    common_len = len(common)
    unique_total = sum(max(0, len(s) - common_len) for s in seqs)
    with_fuse = common_len + unique_total
    without = sum(len(s) for s in seqs)
    saved = without - with_fuse
    return {
        "common_prefix": common,
        "recompute_without": without,
        "recompute_with_fuse": with_fuse,
        "recompute_units_saved": saved,
        "saved_cost": round(saved * cost_per_token, 4),
        "disclaimer": "analogy, not reproduction",
    }
