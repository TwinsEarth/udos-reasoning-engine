"""
轻量混合专家 (Lightweight Mixture-of-Experts) 任务路由 (v3.3.0.dev1)
====================================================================
analogy, not reproduction: 受混合专家 (MoE) 架构启发, 在合成特征上验证
"按输入特征路由到 top-k 小专家线性层" 的机制; 不涉及任何大规模预训练 MoE、
token 并行或专家并行, 纯 CPU 小参数量推理外挂。

设计:
    * N 个专家 = 彼此独立的小线性层 (W[e]: in_dim -> out_dim, batched 存为
      [E, in, out]), 总参数量 = E*(in*out+out) + 路由头, 完全可控;
    * 门控路由 = 一个 in_dim -> E 的线性打分, 选 top-k 专家, 对被选 logits 做 softmax
      作为混合权重;
    * 输出 = sum_{e in topk} w_e * (x @ W[e] + b_e);
    * **opt-in 默认关**: 本模块是独立推理外挂, 不挂载进 PhysicsPredictor 默认前向,
      不调用时主模型输出逐位不变。
第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.moe")


from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightMoE(nn.Module):
    """N 个专家线性层 + 门控 top-k 路由 (推理外挂, opt-in)。"""

    def __init__(self, in_dim: int, out_dim: int,
                 num_experts: int = 4, top_k: int = 2,
                 seed: int = 0) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts 需 >= 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} 需在 [1, num_experts={num_experts}]")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)

        g = torch.Generator().manual_seed(seed)
        # batched 专家权重: [E, in, out], 偏置 [E, out]
        self.W = nn.Parameter(torch.empty(num_experts, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_experts, out_dim))
        # 门控路由头
        self.router = nn.Linear(in_dim, num_experts)
        # 小初始化 (xavier 风格, 用生成器保证可复现)
        scale = 1.0 / (in_dim ** 0.5)
        with torch.no_grad():
            self.W.copy_(torch.randn(num_experts, in_dim, out_dim,
                                     generator=g) * scale)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> Dict[str, object]:
        return {
            "in_dim": self.in_dim, "out_dim": self.out_dim,
            "num_experts": self.num_experts, "top_k": self.top_k,
            "n_params": self.num_parameters(),
            "analogy_not_reproduction": True,
        }

    def forward(self, x: torch.Tensor,
                return_routing: bool = False
                ) -> torch.Tensor:
        """
        x: [B, in_dim] -> [B, out_dim]。
        return_routing=True 时额外返回 (selected_experts [B,k], gate_weights [B,k])。
        """
        if x.dim() != 2 or x.size(-1) != self.in_dim:
            raise ValueError(
                f"输入需为 [B, in_dim={self.in_dim}], 收到 shape={tuple(x.shape)}")
        if x.size(0) == 0:
            raise ValueError("空输入 (batch=0), 拒绝路由")
        gate_logits = self.router(x)                      # [B, E]
        top_val, top_idx = gate_logits.topk(self.top_k, dim=-1)
        gate_w = F.softmax(top_val, dim=-1)               # [B, k]
        sel_W = self.W[top_idx]                          # [B, k, in, out]
        sel_b = self.b[top_idx]                          # [B, k, out]
        expert_outs = torch.einsum("bi,bkio->bko", x, sel_W) + sel_b
        out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)   # [B, out]
        if return_routing:
            return out, top_idx.detach(), gate_w.detach()
        return out
