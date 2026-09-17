"""
ICM 上下文预算 / 检索压缩 (v3.4.0.dev4)
================================================================
analogy, not reproduction —— 呼应 "8000 步 token 不能全上云 / 边缘算力受限" 工程
现实的**轻量化类比实现**, 非复现:

演示记忆库可能很大, 但单次推理能承担的上下文预算有限。本模块提供
`ContextBudgetManager`:

    1. 存储预算: 记忆库 episode 数上限 (FIFO 淘汰, 复用 cache/batch 模式);
    2. 检索截断: top-k 被 cap 到预算内的 k_max;
    3. 原型压缩: 把检索到的 k 个残差原型按相似度分桶聚成 n_proto 个,
       降低聚合计算量 (零梯度、确定性均值)。

设计纪律:
    * 纯函数式、确定性、零参数、不调主模型;
    * opt-in: 默认 budget=None 不截断 (与 v3.4.0 逐位一致);
    * 预算过小退化为更少演示, 绝不报错 (k 自动 min(budget, available));
    * 第二引擎一律称 GPM; analogy, not reproduction。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch

from .dynamics import RAW_DIM

logger = logging.getLogger("udos.icm_budget")


class ContextBudgetManager:
    """ICM 上下文预算管理器 (存储上限 / 检索截断 / 原型压缩)。

    Parameters
    ----------
    budget : 单次推理可使用的演示数上限 (None 表示不限, opt-in 默认)。
    n_proto : 检索后原型压缩的桶数 (<=k; None 表示不压缩)。
    """

    def __init__(self, budget: Optional[int] = None,
                 n_proto: Optional[int] = None) -> None:
        if budget is not None and budget < 1:
            raise ValueError("budget 须 >=1 或 None")
        if n_proto is not None and n_proto < 1:
            raise ValueError("n_proto 须 >=1 或 None")
        self.budget = budget
        self.n_proto = n_proto

    # ------------------------------------------------------------------ #
    # 检索截断: 把请求的 k cap 到预算内
    # ------------------------------------------------------------------ #
    def cap_k(self, k: int, available: int) -> int:
        if k <= 0 or available <= 0:
            return 0
        k = min(k, available)
        if self.budget is not None:
            k = min(k, self.budget)
        return k

    # ------------------------------------------------------------------ #
    # 原型压缩: k 个 (residual[6], weight) -> n_proto 个桶均值
    # ------------------------------------------------------------------ #
    def compress(self, residuals: Sequence[torch.Tensor],
                 weights: Sequence[float]) -> Tuple[torch.Tensor, float]:
        """把检索到的 k 个残差原型压成 n_proto 个, 返回 (压缩后残差, 总权重)。

        策略: 按权重降序排序, 前 n_proto-1 桶各收 1 个原型, 剩余并入末桶
        (等权或按权重加权均值)。n_proto>=k 时原样返回 (不压缩)。
        """
        if not residuals:
            raise ValueError("无可压缩残差")
        if self.n_proto is None or self.n_proto >= len(residuals):
            stacked = torch.stack(list(residuals), dim=0)      # [k,6]
            w = torch.tensor(list(weights), dtype=torch.float32)
            tot = w.sum()
            if tot.item() < 1e-12:
                return stacked.mean(dim=0), 0.0
            return (w.unsqueeze(1) * stacked).sum(dim=0) / tot, float(tot)
        # 分桶压缩
        order = sorted(range(len(residuals)), key=lambda i: -weights[i])
        n_big = self.n_proto - 1
        buckets: List[torch.Tensor] = []
        for i in order[:n_big]:
            buckets.append(residuals[i])
        rest = [residuals[i] for i in order[n_big:]]
        if rest:
            buckets.append(torch.stack(rest, dim=0).mean(dim=0))
        merged = torch.stack(buckets, dim=0)                   # [n_proto,6]
        tot = float(sum(weights))
        return merged.mean(dim=0), tot


__all__ = ["ContextBudgetManager"]
