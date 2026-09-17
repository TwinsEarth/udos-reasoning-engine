"""v7 唯一预测员：残差循环世界模型（单一推理图，无第二引擎/无 LoRA）。

图：观测逐帧编码 →（每帧注入统一场景 ctx）GRU 时序核 → 残差解码
    next_state = last_state + Delta(h)
残差形式让“匀速延续”成为零学习成本基线，模型只需学加速度/振动/碰撞偏差，
显著优于旧 CTM tick 堆叠。确定性核不伪造概率；不确定性由 uncertainty.py 的
按 α conformal 与参数蒙特卡洛给出（M3），entropy 不再冒充校准概率。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .contracts import STATE_DIM, StateContract
from .scene import SceneChannel, SceneContext


class WorldModelCore(nn.Module):
    def __init__(self, window: int = 6, hidden: int = 256,
                 scene_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.window = window
        self.hidden = hidden
        self.contract = StateContract()
        self.scene = SceneChannel(window, scene_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.ctx_proj = nn.Linear(scene_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, STATE_DIM))
        # 残差头零初始化：初始模型严格预测“状态不变”，训练从稳定点起步
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_observations(self, window: torch.Tensor,
                            sc: SceneContext) -> torch.Tensor:
        self.contract.check_window(window)
        feats = self.obs_encoder(window)              # [B,W,H]
        ctx = self.ctx_proj(sc.ctx).unsqueeze(1)      # [B,1,H]
        return feats + ctx                            # 每帧统一场景注入

    def forward(self, window: torch.Tensor,
                explicit: Optional[torch.Tensor] = None,
                return_scene: bool = False):
        sc = self.scene(window, explicit)
        feats = self.encode_observations(window, sc)
        out, h_n = self.gru(feats)
        delta = self.decoder(out[:, -1, :])
        next_state = window[:, -1, :] + delta
        if return_scene:
            return next_state, sc
        return next_state

    @torch.no_grad()
    def rollout(self, window: torch.Tensor, horizon: int,
                explicit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """自回归多步（推理，无梯度）：ctx 由初始窗口估计一次并固定。"""
        self.eval()
        return self.rollout_grad(window, horizon, explicit)

    def rollout_grad(self, window: torch.Tensor, horizon: int,
                     explicit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """自回归多步（保留梯度，训练用）。"""
        sc = self.scene(window, explicit)
        cur = window
        preds = []
        for _ in range(horizon):
            feats = self.encode_observations(cur, sc)
            out, _ = self.gru(feats)
            nxt = cur[:, -1, :] + self.decoder(out[:, -1, :])
            preds.append(nxt)
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)              # [B,H,6]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
