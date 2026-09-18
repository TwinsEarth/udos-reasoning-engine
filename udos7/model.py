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

from .contracts import DT, STATE_DIM, StateContract
from .kinematics import KIN_DIM, kinematic_features
from .scene import SceneChannel, SceneContext


class WorldModelCore(nn.Module):
    def __init__(self, window: int = 6, hidden: int = 256,
                 scene_dim: int = 32, n_layers: int = 2,
                 use_kinematics: bool = True, dt: float = DT):
        super().__init__()
        self.window = window
        self.hidden = hidden
        self.use_kinematics = use_kinematics
        self.dt = dt
        self.contract = StateContract()
        self.scene = SceneChannel(window, scene_dim,
                                  use_kinematics=use_kinematics, dt=dt)
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
        # v7.0.3 解析运动学积分门控（仅运动学通道开启时构建）：
        # nxt = learned + g * (analytic - learned)。
        # 门控是一个**只吃确定性运动学特征的独立小头**（不读共享 GRU 隐状态），
        # 避免门在 accel/uniform 饱和后改变共享表征梯度、拖累 spring 的学习路径。
        # 软门 g = ca_conf · tanh(MLP(kin)/2)：末层零初始化 => tanh(0)=0，未训练
        # g≡0（严格恒等，0 点导数 0.5 不冻死）；ca_conf 先确定性屏蔽 spring/collision，
        # 门再在 uniform/accel 上学到信任、在“窗内无跳变但视界内将碰撞”的 collision
        # 窗上保持谨慎（这正是保留可学习门而非硬 g=ca_conf 的原因）。
        if use_kinematics:
            self.kin_gate = nn.Sequential(
                nn.Linear(KIN_DIM, 32), nn.GELU(),
                nn.Linear(32, STATE_DIM))
            nn.init.zeros_(self.kin_gate[-1].weight)
            nn.init.zeros_(self.kin_gate[-1].bias)
        else:
            self.kin_gate = None

    def encode_observations(self, window: torch.Tensor,
                            sc: SceneContext) -> torch.Tensor:
        self.contract.check_window(window)
        feats = self.obs_encoder(window)              # [B,W,H]
        ctx = self.ctx_proj(sc.ctx).unsqueeze(1)      # [B,1,H]
        return feats + ctx                            # 每帧统一场景注入

    def analytic_kinematic_step(self, last: torch.Tensor,
                                feats: torch.Tensor) -> torch.Tensor:
        """用**固定**的可观测加速度 a_lin 做一步恒定加速度解析积分。

        v_{t+1}=v_t+a·dt；p_{t+1}=p_t+v_t·dt+½·a·dt²。
        a 取自初始观测窗（rollout 全程固定：恒定加速度模型里 a 不随时间变），
        脱梯度视为观测；对 uniform/accel 为真值生成模型，spring/collision 由
        同样固定的 ca_conf 门控抑制，避免预测帧滑窗导致门中途误开。
        """
        a = feats[:, 3:6].detach()                       # [B,3]
        p, v = last[:, :3], last[:, 3:]
        v_next = v + a * self.dt
        p_next = p + v * self.dt + 0.5 * a * self.dt * self.dt
        return torch.cat([p_next, v_next], dim=1)        # [B,6]

    def _step(self, cur: torch.Tensor, sc: SceneContext,
              kin: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.encode_observations(cur, sc)
        out, _ = self.gru(feats)
        h = out[:, -1, :]
        learned = cur[:, -1, :] + self.decoder(h)
        if self.kin_gate is not None and kin is not None:
            analytic = self.analytic_kinematic_step(cur[:, -1, :], kin)
            # ca_conf 与 a 均固定自初始窗；软门 0 点可导、零初始化严格恒等
            ca = kin[:, 10:11].detach()                  # [B,1]
            g = ca * torch.tanh(self.kin_gate(kin) / 2.0)  # 独立小头，(-1,1)
            return learned + g * (analytic - learned)
        return learned

    def forward(self, window: torch.Tensor,
                explicit: Optional[torch.Tensor] = None,
                return_scene: bool = False):
        sc = self.scene(window, explicit)
        kin = kinematic_features(window, self.dt) if self.kin_gate is not None else None
        nxt = self._step(window, sc, kin)
        if return_scene:
            return nxt, sc
        return nxt

    @torch.no_grad()
    def rollout(self, window: torch.Tensor, horizon: int,
                explicit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """自回归多步（推理，无梯度）：场景 ctx、可观测 a/ca 均由初始窗算一次并固定。"""
        self.eval()
        return self.rollout_grad(window, horizon, explicit)

    def rollout_grad(self, window: torch.Tensor, horizon: int,
                     explicit: Optional[torch.Tensor] = None) -> torch.Tensor:
        """自回归多步（保留梯度，训练用）。"""
        sc = self.scene(window, explicit)
        kin = kinematic_features(window, self.dt) if self.kin_gate is not None else None
        cur = window
        preds = []
        for _ in range(horizon):
            nxt = self._step(cur, sc, kin)
            preds.append(nxt)
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)              # [B,H,6]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
