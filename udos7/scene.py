"""v7 统一场景通道（取代旧 scene_estimator/scene_head/gpm_bridge/router 多套割裂件）。

单一上下文向量 ctx（scene_dim），三个来源在**同一编码空间求和**：
    ctx = enc_params(p_use) + bridge(window)
其中
    p_use 逐槽位 = 显式参数（若提供且有限）否则估计器输出 p_hat；
    bridge(window) 是零初始化的加性“场景记忆”（旧 GPM 桥角色），
    接入瞬间对预测零扰动（零偏置短路），训练后才承载窗口侧残差信息。

不再有第二套未训练 CTM、TinyBaseModel 或“注入了却从不 forward”的 LoRA。
隐藏参数可观测性诚实建模：估计器同时输出每槽位可观测置信，碰撞 v2、
弹簧 ω 在短窗下弱可辨识（M2 逐参数评估，不假装能从窗口反推）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .contracts import DT, SCENE_DIM, STATE_DIM
from .kinematics import KIN_DIM, kinematic_features


class SceneEstimator(nn.Module):
    """从观测窗口估计隐藏场景参数 + 每槽位可观测置信（0..1）。"""

    def __init__(self, window: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(window * STATE_DIM, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
            nn.GELU())
        self.param_head = nn.Linear(hidden, SCENE_DIM)
        self.observ_head = nn.Linear(hidden, SCENE_DIM)  # sigmoid 可观测置信

    def forward(self, window: torch.Tensor):
        B = window.size(0)
        h = self.body(window.reshape(B, -1))
        return self.param_head(h), torch.sigmoid(self.observ_head(h))


class SceneBridge(nn.Module):
    """窗口侧加性场景记忆；末层零初始化 => 初始 ctx 贡献严格为 0。"""

    def __init__(self, window: int, scene_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window * STATE_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, scene_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        B = window.size(0)
        return self.net(window.reshape(B, -1))


@dataclass
class SceneContext:
    ctx: torch.Tensor                  # [B, scene_dim]
    params_used: torch.Tensor          # [B,4] 实际进入编码器的参数
    source: torch.Tensor               # [B,4] 1=显式 0=估计
    params_hat: torch.Tensor           # [B,4] 估计器输出
    observability: torch.Tensor        # [B,4] 可观测置信
    kinematics: Optional[torch.Tensor] = None  # [B,KIN_DIM] 确定性可观测运动学


class SceneChannel(nn.Module):
    def __init__(self, window: int, scene_dim: int = 32,
                 use_kinematics: bool = True, dt: float = DT):
        super().__init__()
        self.scene_dim = scene_dim
        self.use_kinematics = use_kinematics
        self.dt = dt
        self.param_encoder = nn.Sequential(
            nn.Linear(SCENE_DIM, scene_dim), nn.LayerNorm(scene_dim),
            nn.GELU(), nn.Linear(scene_dim, scene_dim))
        self.estimator = SceneEstimator(window)
        self.bridge = SceneBridge(window, scene_dim)
        if use_kinematics:
            # 确定性可观测运动学通道；末层零初始化 => 接入瞬间对预测零扰动，
            # 训练后才承载“速度斜率/弹簧频率/碰撞跳变”等可直接观测信息。
            self.kin_encoder = nn.Linear(KIN_DIM, scene_dim)
            nn.init.zeros_(self.kin_encoder.weight)
            nn.init.zeros_(self.kin_encoder.bias)
        else:
            self.kin_encoder = None

    def forward(self, window: torch.Tensor,
                explicit: Optional[torch.Tensor] = None) -> SceneContext:
        p_hat, observ = self.estimator(window)
        if explicit is not None:
            exp = torch.as_tensor(explicit, dtype=window.dtype,
                                  device=window.device)
            if exp.shape != p_hat.shape:
                raise ValueError(
                    f"显式场景参数需为 {tuple(p_hat.shape)}，实际 {tuple(exp.shape)}")
            # NaN 是“该槽未提供”的哨兵 => 回退估计；仅 inf/-inf 视为非法
            if bool(torch.isinf(exp).any()):
                raise ValueError("显式场景参数含 inf（仅允许 NaN 表示缺槽）")
            avail = torch.isfinite(exp)
            p_use = torch.where(avail, exp, p_hat)
            source = avail.float()
        else:
            p_use = p_hat
            source = torch.zeros_like(p_hat)
        ctx = self.param_encoder(p_use) + self.bridge(window)
        kin = None
        if self.kin_encoder is not None:
            kin = kinematic_features(window, self.dt)
            ctx = ctx + self.kin_encoder(kin)
        return SceneContext(ctx, p_use, source, p_hat, observ, kin)
