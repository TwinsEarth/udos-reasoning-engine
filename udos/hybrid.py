"""
混合物理修正 (v2.6.0) — learned-residual 混合物理修正
======================================================
思路: 纯数据驱动模型的预测常常在"惯性运动段"偏离解析物理。我们把
一阶欧拉匀速预测 (解析、零参数) 作为骨架, 再用一个极小的 MLP 学习
"物理骨架与真实下一状态之间的残差":

    euler_pred = [last_pos + last_vel*dt,  last_vel]      # 匀速假设解析预测
    residual   = MLP([model_pred, euler_pred, last_state]) # 学物理/真实的差距
    hybrid_pred = euler_pred + residual

设计纪律 (与全工程 opt-in 一致):
    * 默认不挂载 (predictor.hybrid is None) => predict_next 完全走旧路径, 逐位一致。
    * disable 模式下 forward 直接返回 model_pred 原样, 不计算 euler/residual。
    * 残差 MLP 极小: 输入 concat(model_pred[6], euler_pred[6], last_state[6])=18
      -> Linear(18,32)+bias -> GELU -> Linear(32,6)+bias, 参数量
      18*32+32 + 32*6+6 = 576+32+192+6 = 806 (约)。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.hybrid")


import torch
import torch.nn as nn

from .dynamics import RAW_DIM


class HybridPhysicsCorrector(nn.Module):
    """一阶欧拉物理骨架 + 学习残差的混合修正器 (opt-in, 极小参数量)。"""

    def __init__(self, raw_dim: int = RAW_DIM, hidden: int = 32):
        super().__init__()
        assert raw_dim == RAW_DIM, "混合修正器针对 [pos(3),vel(3)] 布局"
        self.raw_dim = raw_dim
        self.hidden = hidden
        # 输入 = concat(model_pred[R], euler_pred[R], last_state[R]) = 3*R
        self.mlp = nn.Sequential(
            nn.Linear(3 * raw_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, raw_dim),
        )

    # 参数量自检: Linear(18,32)+bias + Linear(32,6)+bias = 806
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def euler_prediction(last_state: torch.Tensor, dt: float) -> torch.Tensor:
        """一阶欧拉匀速预测: x' = x + v*dt; v' = v。布局 [pos(3), vel(3)]。"""
        pos = last_state[..., :3]
        vel = last_state[..., 3:6]
        epos = pos + vel * dt
        return torch.cat([epos, vel], dim=-1)

    def forward(self, model_pred: torch.Tensor, last_state: torch.Tensor,
                dt: float, enabled: bool = True) -> torch.Tensor:
        """
        model_pred [B,R], last_state [B,R], dt 标量。
        enabled=False (disable) 时原样返回 model_pred, 不计算 euler/residual,
        保证与未挂载时逐位一致。
        """
        if not enabled:
            return model_pred
        euler = self.euler_prediction(last_state, dt)
        inp = torch.cat([model_pred, euler, last_state], dim=-1)
        residual = self.mlp(inp)
        return euler + residual
