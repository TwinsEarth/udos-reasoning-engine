"""
未来状态多模态代理头 (v3.0.0 / dev1 / dev2)
==============================================
analogy, not reproduction —— 借鉴 PhysBrain 1.5 "未来状态三模态预测" 思想的
**轻量化类比**, 非复现:

UDOS 为 CPU-only ~52k 参数合成动力学小模型, 不涉及真实 RGB / 深度 / 实例 mask
图像。这里用**共享 latent 的不同线性投影**代理三种未来模态的低维统计向量:
    * RGB 代理   [B, H, rgb_dim]   —— 状态向量 -> 低维"颜色统计"代理 (dev1)
    * 深度代理   [B, H, depth_dim] —— 状态向量位置 -> "深度排序"代理 (dev1)
    * 对象 mask 代理 [B, H, mask_dim] —— 场景参数 -> "对象存在性"软 mask (dev2)

设计纪律 (与全工程一致):
    * **纯前向、确定性、推理时外挂**: 不参与正式件训练, 不改主模型 52191 参数;
    * **三模态共享 backbone**: 同一份 latent[B, latent_dim] 经不同线性投影;
    * **默认关**: 通过 MultiTaskHead(enable=False) 或本头 use_*=False 关闭,
      旧 predict_next 路径逐位一致;
    * **模态独立可开关 + 独立损失**: RGB/深度/mask 各自前向, 各自可独立 MSE 监督;
    * 不碰真实图像 / 不下载大权重 / CPU-only。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.future_multimodal")


from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class RGBProxyHead(nn.Module):
    """状态向量 -> 低维"颜色统计"代理 [B, H, rgb_dim] (dev1)。

    analogy, not reproduction: 不碰真实 RGB 图像; 用 latent 的线性投影代理一组
    "颜色统计"标量。rgb_dim 默认 8, 通道语义 (仅类比, 不对应真实像素):
        [mean_r, mean_g, mean_b, var, hist_bin0, hist_bin1, hist_bin2, hist_bin3]
    纯线性投影, 推理时外挂; 同一 latent -> 同一输出 (确定性仿射映射)。
    """

    def __init__(self, latent_dim: int, horizon: int = 4,
                 rgb_dim: int = 8) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim 必须 >= 1")
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1 (H=0 见 3.0.2 边界)")
        if rgb_dim < 1:
            raise ValueError("rgb_dim 必须 >= 1")
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.rgb_dim = int(rgb_dim)
        self.proj = nn.Linear(latent_dim, horizon * rgb_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent[B, latent_dim] -> rgb 代理 [B, H, rgb_dim]。"""
        if latent.dim() != 2:
            raise ValueError("latent 需为 [B, latent_dim]")
        B = latent.size(0)
        return self.proj(latent).view(B, self.horizon, self.rgb_dim)


class DepthProxyHead(nn.Module):
    """状态向量位置 -> "深度排序"代理 [B, H, depth_dim] (dev1)。

    analogy, not reproduction: 不碰真实深度图; 用 latent 的线性投影代理一组
    "深度排序"标量。depth_dim 默认 4, 通道语义 (仅类比):
        [rel_dist, depth_grad, near_ratio, far_ratio]
    纯线性投影, 推理时外挂。
    """

    def __init__(self, latent_dim: int, horizon: int = 4,
                 depth_dim: int = 4) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim 必须 >= 1")
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1 (H=0 见 3.0.2 边界)")
        if depth_dim < 1:
            raise ValueError("depth_dim 必须 >= 1")
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.depth_dim = int(depth_dim)
        self.proj = nn.Linear(latent_dim, horizon * depth_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent[B, latent_dim] -> depth 代理 [B, H, depth_dim]。"""
        if latent.dim() != 2:
            raise ValueError("latent 需为 [B, latent_dim]")
        B = latent.size(0)
        return self.proj(latent).view(B, self.horizon, self.depth_dim)


class MaskProxyHead(nn.Module):
    """状态向量的场景参数 -> "对象存在性"软 mask [B, H, mask_dim] (dev2)。

    analogy, not reproduction: 不碰真实实例 mask; 用 latent 线性投影 + 可选
    scene_params 阈值偏置, 经 sigmoid 生成 [0,1] 软 mask。mask_dim 默认 N_obj=4。
    mask 与 RGB/深度在同一 H 步对齐 (同 horizon)。

    Parameters
    ----------
    latent_dim:    共享 latent 维度。
    horizon:       未来步数 (与 RGB/深度对齐)。
    mask_dim:      对象数 (每步每对象一个存在性概率)。
    threshold:     scene_params 阈值 (超过则对该对象存在性 logit 加正偏置)。
    scene_bias_scale: scene_params 偏置强度。
    """

    def __init__(self, latent_dim: int, horizon: int = 4,
                 mask_dim: int = 4, threshold: float = 0.0,
                 scene_bias_scale: float = 1.0) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim 必须 >= 1")
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1 (H=0 见 3.0.2 边界)")
        if mask_dim < 1:
            raise ValueError("mask_dim 必须 >= 1 (mask_dim=0 见 3.0.2 边界)")
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.mask_dim = int(mask_dim)
        self.threshold = float(threshold)
        self.scene_bias_scale = float(scene_bias_scale)
        self.mask_proj = nn.Linear(latent_dim, horizon * mask_dim)

    def forward(self, latent: torch.Tensor,
                scene_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """latent[B, latent_dim] (可选 scene_params[B, spd]) -> 软 mask [B,H,mask_dim]∈[0,1]。"""
        if latent.dim() != 2:
            raise ValueError("latent 需为 [B, latent_dim]")
        B = latent.size(0)
        logits = self.mask_proj(latent).view(B, self.horizon, self.mask_dim)
        if scene_params is not None:
            sp = torch.as_tensor(scene_params, dtype=torch.float32)
            if sp.dim() == 1:
                sp = sp.unsqueeze(0)
            # 取前 mask_dim 维场景参数做阈值偏置, 广播到 H 步
            sp_c = sp[:, :self.mask_dim]                       # [B,mask_dim]
            bias = self.scene_bias_scale * (sp_c - self.threshold)
            logits = logits + bias.unsqueeze(1)                # [B,1,mask_dim]
        return torch.sigmoid(logits)                           # [B,H,mask_dim]∈[0,1]


class FutureMultimodalHead(nn.Module):
    """三模态未来代理头: 共享 latent -> RGB / 深度 / 对象 mask 低维向量。

    所有模态共享同一份 latent (backbone); RGB/深度分别委托 RGBProxyHead /
    DepthProxyHead, mask 用独立线性投影 (dev2 将抽出 MaskProxyHead)。输出 dict
    {name: tensor[B, H, dim]}, 仅包含被 use_* 开关打开的模态。

    Parameters
    ----------
    latent_dim:  共享 latent 维度 (与 MultiTaskHead.encode 对齐)。
    horizon:     未来步数 H (与 FutureStateHead 对齐, 默认 4)。
    rgb_dim:     RGB 代理维度 (颜色统计通道, 默认 8)。
    depth_dim:   深度代理维度 (深度排序通道, 默认 4)。
    mask_dim:    对象 mask 维度 (默认 N_obj=4; dev2 细化为软 mask [0,1])。
    use_rgb/use_depth/use_mask: 各模态独立开关 (默认全开)。
    """

    def __init__(self, latent_dim: int = 32, horizon: int = 4,
                 rgb_dim: int = 8, depth_dim: int = 4, mask_dim: int = 4,
                 use_rgb: bool = True, use_depth: bool = True,
                 use_mask: bool = True) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim 必须 >= 1")
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1 (H=0 见 3.0.2 边界)")
        if rgb_dim < 1 or depth_dim < 1:
            raise ValueError("rgb_dim / depth_dim 必须 >= 1")
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.rgb_dim = int(rgb_dim)
        self.depth_dim = int(depth_dim)
        self.mask_dim = int(mask_dim)
        self.use_rgb = bool(use_rgb)
        self.use_depth = bool(use_depth)
        self.use_mask = bool(use_mask)
        # 仅为开启的模态分配投影 (节省参数, 也便于逐模态开关)
        if self.use_rgb:
            self.rgb = RGBProxyHead(latent_dim, horizon, rgb_dim)
        if self.use_depth:
            self.depth = DepthProxyHead(latent_dim, horizon, depth_dim)
        if self.use_mask:
            self.mask = MaskProxyHead(latent_dim, horizon, mask_dim)

    # ------------------------------------------------------------------ #
    def forward(self, latent: torch.Tensor,
                scene_params: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """latent[B, latent_dim] (可选 scene_params[B,spd]) -> {modality: [B,H,dim]}。

        注: 通过 MultiTaskHead 注册时框架仅以 head(latent) 调用, scene_params 默认
        None; 直接调用时可传 scene_params 以启用 mask 的场景阈值偏置。
        """
        if latent.dim() != 2:
            raise ValueError("latent 需为 [B, latent_dim]")
        out: Dict[str, torch.Tensor] = {}
        if self.use_rgb:
            out["rgb"] = self.rgb(latent)
        if self.use_depth:
            out["depth"] = self.depth(latent)
        if self.use_mask:
            out["mask"] = self.mask(latent, scene_params=scene_params)
        return out

    # ------------------------------------------------------------------ #
    # 配置序列化 (供 save/load 兼容测试; 头权重随 state_dict 保存)
    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, Any]:
        return {"kind": "FutureMultimodalHead",
                "latent_dim": self.latent_dim, "horizon": self.horizon,
                "rgb_dim": self.rgb_dim, "depth_dim": self.depth_dim,
                "mask_dim": self.mask_dim,
                "use_rgb": self.use_rgb, "use_depth": self.use_depth,
                "use_mask": self.use_mask}

    def load_config(self, cfg: Dict[str, Any]) -> None:
        self.horizon = int(cfg.get("horizon", self.horizon))
        self.rgb_dim = int(cfg.get("rgb_dim", self.rgb_dim))
        self.depth_dim = int(cfg.get("depth_dim", self.depth_dim))
        self.mask_dim = int(cfg.get("mask_dim", self.mask_dim))
        self.use_rgb = bool(cfg.get("use_rgb", self.use_rgb))
        self.use_depth = bool(cfg.get("use_depth", self.use_depth))
        self.use_mask = bool(cfg.get("use_mask", self.use_mask))


# --------------------------------------------------------------------------- #
# 跨模态空间对齐一致性损失 (v3.0.0.dev3)
# analogy, not reproduction: 在合成代理向量上用相关性约束三模态在同一时间步的
# 空间一致性; 作为 multitask 训练的 opt-in 辅助损失, **推理时不计算**。
# --------------------------------------------------------------------------- #
def _batch_pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐 batch 计算 x[B,H] 与 y[B,H] 的 Pearson 相关 -> [B]。

    零方差/常数列时退化为 0 (NaN 防护), 不抛出。
    """
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    num = (x * y).sum(dim=-1)
    den = torch.sqrt((x * x).sum(dim=-1) * (y * y).sum(dim=-1) + eps)
    corr = num / den
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


class CrossModalAlignmentLoss(nn.Module):
    """跨模态空间一致性损失 (opt-in 训练辅助损失, 推理不计算)。

    约束三模态在同一时间步 (同一 h) 的一致性:
        1. RGB 统计 (rgb_dim 维均值轨迹) 与深度排序 (depth_dim 维均值轨迹)
           应相关 -> 损失 (1 - pearson) ;
        2. 对象 mask (mask_dim 维均值) 应与 RGB "对象区域活跃" (|rgb 均值|)
           对应 -> 损失 (1 - pearson) 。

    纯参数-free, 有限值; 任一输入零方差/常数时相关性退化为 0 (NaN 防护)。
    analogy, not reproduction: 不对应真实 RGB-D 几何对齐。
    """

    def __init__(self, rgb_depth_weight: float = 1.0,
                 mask_rgb_weight: float = 1.0) -> None:
        super().__init__()
        self.rgb_depth_weight = float(rgb_depth_weight)
        self.mask_rgb_weight = float(mask_rgb_weight)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """rgb[B,H,Cr], depth[B,H,Cd], mask[B,H,Q] -> 标量一致性损失 (>=0)。"""
        if rgb.dim() != 3 or depth.dim() != 3 or mask.dim() != 3:
            raise ValueError("rgb/depth/mask 需为 [B,H,C]")
        rgb_pool = rgb.mean(dim=-1)          # [B,H]
        dep_pool = depth.mean(dim=-1)        # [B,H]
        mask_pool = mask.mean(dim=-1)        # [B,H]
        rgb_act = rgb_pool.abs()             # |RGB 活跃| 代理对象区域

        corr_rd = _batch_pearson(rgb_pool, dep_pool)       # [B]
        corr_mr = _batch_pearson(mask_pool, rgb_act)       # [B]
        loss_rd = (1.0 - corr_rd).clamp(min=0.0).mean()
        loss_mr = (1.0 - corr_mr).clamp(min=0.0).mean()
        loss = self.rgb_depth_weight * loss_rd + self.mask_rgb_weight * loss_mr
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def __repr__(self) -> str:  # pragma: no cover - 可读性
        return (f"CrossModalAlignmentLoss(rgb_depth_weight="
                f"{self.rgb_depth_weight}, mask_rgb_weight={self.mask_rgb_weight})")

