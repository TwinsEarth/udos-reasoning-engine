"""
统一多任务头骨架 (v2.8.0.dev5)
================================
analogy, not reproduction —— 借鉴 PhysBrain 1.5 统一自回归多任务头思想的轻量化类比：
在**不新增可训练权重、不改动现有 CTM 主路径**的前提下，提供一层"共享 backbone 接口 +
头注册机制"，把多个结构化预测头挂在同一个确定性 latent 之上。

设计纪律 (与全工程一致):
    * **纯接口层 + 确定性路由**: encode(window, scene_params) -> latent 完全复用
      predictor 已有 obs_encoder / scene_encoder, 不引入新参数;
    * **默认不挂载时预测路径不变**: MultiTaskHead.enable=False (默认), 主模型
      predict_next 不受任何影响;
    * **头注册机制**: register_head(name, head) / list_heads / get_head; 每个头是一个
      可调用对象 (接受 latent[B, latent_dim], 返回各自结构化输出);
    * latent_dim 可配置; 未注册任何头时 forward 返回空字典。
    * 不参与正式件训练 (2.8.0 主模型 52191 参数不变)。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.multitask")


from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


class MultiTaskHead:
    """共享 backbone 接口 + 头注册路由 (推理时外挂, 不改主模型)。"""

    def __init__(self, predictor, latent_dim: int = 32,
                 enable: bool = False) -> None:
        """
        predictor:    训练好的 PhysicsPredictor (只读 backbone)。
        latent_dim:   共享 latent 维度 (可配置; encode 会把 backbone 特征对齐到此维)。
        enable:       默认 False => forward 返回空、主路径不变; True 才跑已注册头。
        """
        if latent_dim < 1:
            raise ValueError("latent_dim 必须 >= 1")
        self.predictor = predictor
        self.latent_dim = int(latent_dim)
        self.enable = bool(enable)
        self._heads: Dict[str, Callable[[torch.Tensor], Any]] = {}

    # ------------------------------------------------------------------ #
    # 共享 backbone: 复用 predictor 特征提取, 确定性投影到 latent_dim
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, window: torch.Tensor,
               scene_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """window[B,W,6] -> latent[B, latent_dim] (确定性, 无新参数)。

        实现: obs_encoder 输出 [B,W,d_input] 在时间维平均池化 -> [B,d_input],
        再确定性地裁剪/零填充对齐到 latent_dim (scene_params 经 scene_encoder 注入
        backbone, 与主模型同一条特征路径)。
        """
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,6] 或 [B,W,6]")
        self.predictor.eval()
        feats = self.predictor.obs_encoder(w)               # [B,W,d_input]
        pooled = feats.mean(dim=1)                           # [B,d_input]
        d = pooled.size(-1)
        if d >= self.latent_dim:
            return pooled[:, :self.latent_dim]
        pad = pooled.new_zeros(pooled.size(0), self.latent_dim - d)
        return torch.cat([pooled, pad], dim=-1)

    # ------------------------------------------------------------------ #
    # 头注册机制
    # ------------------------------------------------------------------ #
    def register_head(self, name: str, head: Callable[[torch.Tensor], Any]) -> None:
        """注册一个结构化预测头 (callable: latent[B,latent_dim] -> 任意)。"""
        if not isinstance(name, str) or not name:
            raise ValueError("头名必须是非空字符串")
        if not callable(head):
            raise ValueError("head 必须是可调用对象")
        self._heads[name] = head

    def get_head(self, name: str):
        if name not in self._heads:
            raise KeyError(f"未注册的头: {name!r}; 已有 {self.list_heads()}")
        return self._heads[name]

    def list_heads(self) -> List[str]:
        return list(self._heads)

    def remove_head(self, name: str) -> None:
        self._heads.pop(name, None)

    # ------------------------------------------------------------------ #
    # 路由: enable 时对每个注册头跑一遍
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, window: torch.Tensor,
                scene_params: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """enable=False => 返回 {} (主路径不变); 否则 {head_name: head(latent)}。"""
        if not self.enable:
            return {}
        if not self._heads:
            return {}
        latent = self.encode(window, scene_params=scene_params)
        return {name: head(latent) for name, head in self._heads.items()}

    def __call__(self, window, scene_params=None):
        return self.forward(window, scene_params=scene_params)

    # ------------------------------------------------------------------ #
    # 配置序列化 (头名 + latent_dim + enable; 不含头权重, 纯元数据)
    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, Any]:
        return {"kind": "MultiTaskHead", "latent_dim": self.latent_dim,
                "enable": self.enable, "heads": self.list_heads()}

    def load_config(self, cfg: Dict[str, Any]) -> None:
        """恢复头名列表 / latent_dim / enable (头对象需调用方重新 register)。"""
        self.latent_dim = int(cfg.get("latent_dim", self.latent_dim))
        self.enable = bool(cfg.get("enable", self.enable))


# --------------------------------------------------------------------------- #
# 两个结构化预测头 (v2.8.0.dev6)
# analogy, not reproduction: 以下头为共享 latent 上的轻量线性投影代理,
# 不参与正式件训练, 仅验证共享 backbone + 多头路由的形状/延迟/隔离性质。
# --------------------------------------------------------------------------- #
class SpatialCoordHead(nn.Module):
    """从共享 latent 输出结构化空间坐标 [B, N_pts, 3] 代理。

    在合成数据里用 3 个输出通道代理 xyz 坐标 (analogy); 纯线性投影, 推理时外挂。
    """

    def __init__(self, latent_dim: int, n_pts: int = 4) -> None:
        super().__init__()
        if n_pts < 1:
            raise ValueError("n_pts 必须 >= 1")
        self.n_pts = int(n_pts)
        self.proj = nn.Linear(latent_dim, self.n_pts * 3)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        return self.proj(latent).view(B, self.n_pts, 3)


class ActionTrajectoryHead(nn.Module):
    """从共享 latent 输出动作轨迹 [B, H, action_dim] 代理。

    action_dim 默认对齐 RAW_DIM=6 (与 policy.state_perturbation 动作空间一致);
    纯线性投影, 推理时外挂。
    """

    def __init__(self, latent_dim: int, horizon: int = 4,
                 action_dim: int = 6) -> None:
        super().__init__()
        if horizon < 1 or action_dim < 1:
            raise ValueError("horizon / action_dim 必须 >= 1")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.proj = nn.Linear(latent_dim, self.horizon * self.action_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        return self.proj(latent).view(B, self.horizon, self.action_dim)


class FutureStateHead(nn.Module):
    """从共享 latent 输出未来状态代理 [B, H, 6] + 每步不确定性 [B, H, 1]。

    与 predictor.rollout 对齐 (H 步、state_dim=6); 纯线性投影代理, 推理时外挂。
    返回 dict {"trajectory", "uncertainty"}。
    """

    def __init__(self, latent_dim: int, horizon: int = 4,
                 state_dim: int = 6) -> None:
        super().__init__()
        if horizon < 1 or state_dim < 1:
            raise ValueError("horizon / state_dim 必须 >= 1")
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.traj_proj = nn.Linear(latent_dim, self.horizon * self.state_dim)
        self.unc_proj = nn.Linear(latent_dim, self.horizon)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = latent.size(0)
        traj = self.traj_proj(latent).view(B, self.horizon, self.state_dim)
        # 每步不确定性 (非负, 软plus), 与轨迹同 H
        unc = torch.nn.functional.softplus(
            self.unc_proj(latent)).view(B, self.horizon, 1)
        return {"trajectory": traj, "uncertainty": unc}


# --------------------------------------------------------------------------- #
# 空间关系头 (v2.9.0.dev5)
# analogy, not reproduction: 从共享 latent 投影出每物体 2D 坐标代理, 再由坐标几何
# 关系派生"左/右/上/下/接触"关系矩阵; 纯推理时外挂, 不参与正式件训练。
# --------------------------------------------------------------------------- #
class SpatialRelationHead(nn.Module):
    """从共享 latent 输出物体对空间关系矩阵 [B, N_obj, N_obj, rel_dim]。

    rel_dim=5 通道依次代理: [right, left, up, down, contact]。
    用 latent -> 每物体 2D 坐标的线性投影, 再由坐标差的符号/距离派生关系
    (analogy); 天然满足对称性: right(i,j) == left(j,i)。
    """

    REL_DIM = 5

    def __init__(self, latent_dim: int, n_obj: int = 4,
                 contact_thr: float = 0.5) -> None:
        super().__init__()
        if n_obj < 2:
            raise ValueError("n_obj 必须 >= 2 (单物体关系矩阵无定义, 见 2.9.3 边界)")
        self.n_obj = int(n_obj)
        self.contact_thr = float(contact_thr)
        self.coord_proj = nn.Linear(latent_dim, self.n_obj * 2)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        coords = self.coord_proj(latent).view(B, self.n_obj, 2)   # [B,N,2]
        xi = coords.unsqueeze(2)                                  # [B,N,1,2]
        xj = coords.unsqueeze(1)                                  # [B,1,N,2]
        d = xj - xi                                              # [B,N,N,2]
        dx, dy = d[..., 0], d[..., 1]
        right = (dx > 0).float()
        left = (dx < 0).float()
        up = (dy > 0).float()
        down = (dy < 0).float()
        contact = (d.norm(dim=-1) < self.contact_thr).float()
        return torch.stack([right, left, up, down, contact], dim=-1)  # [B,N,N,5]
