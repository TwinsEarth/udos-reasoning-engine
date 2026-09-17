"""
GPM 记忆桥 (v5.4.7)
===================
把 GPM 的场景嵌入 `scene_embedding`（维度 latent_size，默认 128）投影到主
预测员 PhysicsPredictor 内部 CTM 的场景条件维度（scene_dim，发布 checkpoint
为 32），作为 4 维显式物理参数之外的**加性场景记忆**：

    ctx = scene_encoder(scene_params)          # 训练过的显式参数通道 (4 -> 32)
    ctx = ctx + bridge(gpm.scene_embedding)     # 本模块, 零初始化可学习桥

设计纪律
--------
1. **零初始化、逐位兼容**：Linear 的权重与偏置全部零初始化，接入瞬间桥输出
   恒为 0，主预测与 v5.4.6 数值完全一致；真实增益在 v5.5.0 联合训练后产生。
2. **独立模块、不碰主锚点**：桥是独立 `nn.Module`，由推理引擎持有，不进入
   `PhysicsPredictor.state_dict`，主预测员 52191 个可学习参数锚点不变，可
   独立训练、独立存权重、独立回滚。
3. **反 LoRA 死路**：与"LoRA 注入演示基座却从不 forward"不同，桥输出在
   `reason()` 中真实传入训练过的主预测员前向；权重非零即可观察到预测变化。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GPMSceneBridge(nn.Module):
    """GPM 场景嵌入 -> 主预测员 CTM scene_dim 的零初始化线性加性桥。"""

    def __init__(self, latent_dim: int, scene_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.scene_dim = int(scene_dim)
        self.proj = nn.Linear(self.latent_dim, self.scene_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding [..., latent_dim] -> [..., scene_dim]，初始恒为 0。"""
        return self.proj(embedding)

    @torch.no_grad()
    def bridge_norm(self, embedding: torch.Tensor) -> float:
        """本次桥输出的 L2 范数；零初始化/未训练时为 0，用于可观测性。"""
        return float(self.forward(embedding).detach().norm().item())
