"""
Ego360 启发的合成多视角数据增强 (v3.2.0)
==========================================
analogy, not reproduction —— 受 Ego360 全景人类数据 / 多视角自监督思想启发的
**轻量化类比实现**, 非复现:

UDOS 是 CPU-only、~52k 参数的合成参数化动力学小模型, 状态向量
raw = [pos(3), vel(3)] (RAW_DIM=6), 只在 x 轴产生非平凡运动。本模块在**合成
参数化数据**上做多视角/多情境数据增强, 仅验证以下工程事实:

    1. 视角几何变换 (xy 平面旋转 + 平移) 是刚体变换, 对位置/速度分量分别作用,
       且可逆 (R^T R = I);
    2. 轨迹扰动 / 噪声注入的幅度可控 (参数=0 时逐位不变, 越大偏离越大);
    3. 时间缩放沿时间轴做线性重采样, 形状保持 (长度不变);
    4. 增强 (X 历史窗口, Y 未来目标) 成对施加, 保证几何层面标签对齐
       (同一世界帧内连续轨迹整体变换);
    5. 空 / 非法输入守卫。

设计纪律 (与全工程一致):
    * **纯前向、确定性、不修改主模型权重** (训练时数据增强, opt-in;
      正式件仍用原始数据, 见训练纪律);
    * 纯 torch + 标准库, CPU-only;
    * 不涉及真实视频 / 第一人称 RGB / VLM;
    * 第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.ego_data")


import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .dynamics import RAW_DIM

# 状态向量槽位: pos = dims 0,1,2 ; vel = dims 3,4,5
_POS = slice(0, 3)
_VEL = slice(3, 6)


def _as_seq(x: torch.Tensor, name: str = "seq") -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.numel() == 0:
        raise ValueError(f"{name} 为空")
    if t.size(-1) != RAW_DIM:
        raise ValueError(
            f"{name} 最后一维 {t.size(-1)} 应为 RAW_DIM={RAW_DIM}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} 含 NaN/inf 非有限值")
    return t


def rotation_xy(angle_rad: float) -> torch.Tensor:
    """xy 平面 2x2 旋转矩阵 R, 满足 R^T R = I (可逆)。"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def _rotate_xy(state: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """对状态 [...,6] 的 (x,y) 位置与 (vx,vy) 速度分别左乘 R (刚体视图旋转)。"""
    out = state.clone()
    # 位置 xy (dims 0,1)
    pos_xy = out[..., 0:2]
    out[..., 0:2] = pos_xy @ R.T
    # 速度 xy (dims 3,4)
    vel_xy = out[..., 3:5]
    out[..., 3:5] = vel_xy @ R.T
    return out


def time_warp(seq: torch.Tensor, scale: float) -> torch.Tensor:
    """沿时间轴线性重采样 (形状保持)。

    输入 [..., T, 6] -> 同形 [..., T, 6]。输出第 t 帧 = 在旧时间轴上于 t*scale
    处线性插值 (scale>1 放慢 => 取更早帧; scale<1 加速 => 取更晚帧)。越界 clamp。
    scale=1 时逐位返回原序列。
    """
    if scale == 1.0:
        return seq
    if scale <= 0.0:
        raise ValueError("time_scale 必须 > 0")
    T = seq.size(-2)
    idx = torch.arange(T, dtype=torch.float32) * scale
    idx = idx.clamp(0.0, T - 1.0)
    lo = idx.floor().long()
    hi = torch.minimum(lo + 1, torch.tensor(T - 1))
    # frac 对齐到时间维 (-2): shape [1,1,...,T,1]
    view_shape = [1] * (seq.dim() - 2) + [T, 1]
    frac = (idx - lo).reshape(view_shape)
    return (1.0 - frac) * seq.index_select(-2, lo) + frac * seq.index_select(-2, hi)


class SyntheticEgoAugmenter:
    """合成数据多视角/多情境增强器 (训练时 opt-in)。

    Parameters
    ----------
    view_rotate_deg:
        视图绕 z 轴旋转角度 (度)。0 = 不旋转。刚体旋转, 位置/速度同步旋转。
    view_translate_xyz:
        视图平移 (加到位置 dims 0,1,2; 速度不变)。长度 3。
    traj_perturb:
        轨迹扰动幅度 (逐帧有界, 确定性抖动)。0 = 不扰动。
    noise_sigma:
        高斯噪声标准差 (零均值)。0 = 不加噪。
    time_scale:
        时间缩放因子 (>0)。1 = 不缩放。
    seed:
        确定性种子 (扰动 / 噪声用)。
    """

    def __init__(self,
                 view_rotate_deg: float = 0.0,
                 view_translate_xyz: Sequence[float] = (0.0, 0.0, 0.0),
                 traj_perturb: float = 0.0,
                 noise_sigma: float = 0.0,
                 time_scale: float = 1.0,
                 seed: int = 42) -> None:
        if traj_perturb < 0.0:
            raise ValueError("traj_perturb 必须 >= 0")
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma 必须 >= 0")
        if time_scale <= 0.0:
            raise ValueError("time_scale 必须 > 0")
        t = torch.as_tensor(view_translate_xyz, dtype=torch.float32).reshape(-1)
        if t.numel() != 3:
            raise ValueError("view_translate_xyz 长度须为 3")
        self.view_rotate_deg = float(view_rotate_deg)
        self.angle = math.radians(self.view_rotate_deg)
        self.R = rotation_xy(self.angle)
        self.translate = t.clone()
        self.traj_perturb = float(traj_perturb)
        self.noise_sigma = float(noise_sigma)
        self.time_scale = float(time_scale)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # 视图几何 (可逆)
    # ------------------------------------------------------------------ #
    def apply_view(self, seq: torch.Tensor) -> torch.Tensor:
        """施加视图旋转 + 平移: 先旋转 xy, 再平移位置。"""
        out = _rotate_xy(seq, self.R)
        out = out.clone()
        out[..., _POS] = out[..., _POS] + self.translate.to(out.device)
        return out

    def reverse_view(self, seq: torch.Tensor) -> torch.Tensor:
        """逆视图变换 (先减平移, 再用 R^T 旋转回去)。"""
        out = seq.clone()
        out[..., _POS] = out[..., _POS] - self.translate.to(out.device)
        return _rotate_xy(out, self.R.T)

    # ------------------------------------------------------------------ #
    # 单条序列增强
    # ------------------------------------------------------------------ #
    def augment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """对 [..., T, 6] 序列依次: 时间缩放 -> 视图几何 -> 轨迹扰动 -> 噪声注入。

        顺序固定: 时间缩放先 (改变采样时刻), 再做刚体视图变换, 最后加扰动/噪声。
        """
        s = _as_seq(seq, "seq")
        if s.size(-2) == 0:
            raise ValueError("序列时间维为空, 无法增强")
        # 1) 时间缩放 (形状保持)
        out = time_warp(s, self.time_scale)
        # 2) 视图刚体变换 (可逆)
        out = self.apply_view(out)
        g = torch.Generator().manual_seed(self.seed)
        # 3) 轨迹扰动 (有界逐帧抖动, 位置+速度都小幅扰动)
        if self.traj_perturb > 0.0:
            eps = (torch.randn(out.shape, generator=g, dtype=torch.float32)
                   * self.traj_perturb)
            out = out + eps
        # 4) 噪声注入 (零均值高斯)
        if self.noise_sigma > 0.0:
            eps = (torch.randn(out.shape, generator=g, dtype=torch.float32)
                   * self.noise_sigma)
            out = out + eps
        return out.to(torch.float32)

    # ------------------------------------------------------------------ #
    # (X, Y) 成对增强: 几何层面标签对齐
    # ------------------------------------------------------------------ #
    def augment_pair(self, X: torch.Tensor, Y: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 (历史窗口 X [B,W,6], 未来目标 Y [B,H,6]) 成对增强。

        把 X 与 Y 沿时间维拼成连续轨迹 [B, W+H, 6] 后整体增强, 再切回。
        刚体视图变换因此在 X 与 Y 之间严格一致 (同一世界帧连续变换), 标签对齐;
        扰动/噪声也施加在连续轨迹上, 不破坏 "同一轨迹的不同观测" 语义。
        """
        x = _as_seq(X, "X")
        y = _as_seq(Y, "Y")
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError("X/Y 须为 [B,T,6]")
        if x.size(0) != y.size(0):
            raise ValueError(f"X 批 {x.size(0)} 与 Y 批 {y.size(0)} 不一致")
        if x.size(0) == 0:
            raise ValueError("数据集为空, 无法增强")
        W = x.size(1)
        cont = torch.cat([x, y], dim=1)         # [B, W+H, 6]
        cont = self.augment_sequence(cont)
        return cont[:, :W, :].contiguous(), cont[:, W:, :].contiguous()

    # ------------------------------------------------------------------ #
    # 可读摘要
    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        return {
            "view_rotate_deg": self.view_rotate_deg,
            "view_translate_xyz": [round(float(v), 4) for v in self.translate],
            "traj_perturb": self.traj_perturb,
            "noise_sigma": self.noise_sigma,
            "time_scale": self.time_scale,
            "seed": self.seed,
            "analogy_not_reproduction": True,
        }


# --------------------------------------------------------------------------- #
# MultiViewGenerator (node 42 / v3.2.0.dev1): 多视图合成 + 视图一致性
# --------------------------------------------------------------------------- #
class MultiViewGenerator:
    """从同一场景状态序列参数化生成多个"视角", 验证跨视图物理一致性可互逆。

    analogy, not reproduction —— 受 Ego360 多视角思想启发, 在合成状态向量上用
    已知 xy 平面刚体旋转 R_k 生成第 k 个视角。视图间变换为解析已知的 2x2 矩阵,
    因此可严格互逆: view_k = R_k @ base, base = R_k^T @ view_k。

    Parameters
    ----------
    n_views: 视角数 K (>=2)。
    angle_step_deg: 相邻视角绕 z 轴的角度间隔 (度)。
    """

    def __init__(self, n_views: int = 4, angle_step_deg: float = 45.0) -> None:
        if n_views < 2:
            raise ValueError("n_views 必须 >= 2")
        self.n_views = int(n_views)
        self.angle_step_deg = float(angle_step_deg)
        # 各视角旋转矩阵 (相对世界/基准视角)
        self.Rs = [rotation_xy(math.radians(i * angle_step_deg))
                   for i in range(self.n_views)]

    def view_angles_deg(self) -> list:
        return [round(i * self.angle_step_deg, 4)
                for i in range(self.n_views)]

    def apply_view(self, seq: torch.Tensor, k: int) -> torch.Tensor:
        """把基准序列变换到第 k 视角 (仅 xy 旋转, 不平移)。"""
        self._check_k(k)
        s = _as_seq(seq, "seq")
        return _rotate_xy(s, self.Rs[k])

    def revert_view(self, seq: torch.Tensor, k: int) -> torch.Tensor:
        """第 k 视角还原回基准视角 (R_k^T)。"""
        self._check_k(k)
        s = _as_seq(seq, "seq")
        return _rotate_xy(s, self.Rs[k].T)

    def generate(self, base_seq: torch.Tensor) -> torch.Tensor:
        """基准序列 [T,6] -> 全部视角 [K, T, 6]。"""
        s = _as_seq(base_seq, "base_seq")
        if s.size(-2) == 0:
            raise ValueError("基准序列时间维为空")
        return torch.stack([self.apply_view(s, k)
                            for k in range(self.n_views)], dim=0)

    def _check_k(self, k: int) -> None:
        if not (0 <= k < self.n_views):
            raise ValueError(f"视角索引 k={k} 须在 [0,{self.n_views})")

    def correspondence(self, k: int, j: int) -> torch.Tensor:
        """视图间对应关系矩阵: 把第 k 视角坐标映射到第 j 视角 = R_j @ R_k^T。"""
        self._check_k(k)
        self._check_k(j)
        return self.Rs[j] @ self.Rs[k].T

    def cross_view_consistency(self, views: torch.Tensor) -> float:
        """跨视图一致性误差: 对所有 (k,j), 把 view_k 经对应矩阵映到 view_j,
        与直接生成的 view_j 的最大绝对差。应为 ~0 (解析已知变换)。"""
        if views.dim() != 3 or views.size(0) != self.n_views:
            raise ValueError(f"views 须为 [K={self.n_views}, T, 6]")
        err = 0.0
        for k in range(self.n_views):
            for j in range(self.n_views):
                C = self.correspondence(k, j)
                mapped = _rotate_xy(views[k], C)
                err = max(err, float((mapped - views[j]).abs().max()))
        return err

    def invertibility_error(self, base_seq: torch.Tensor) -> float:
        """每视角 应用再还原 的最大绝对误差 (应 ~0, 证明 R_k^T R_k = I)。"""
        s = _as_seq(base_seq, "base_seq")
        err = 0.0
        for k in range(self.n_views):
            back = self.revert_view(self.apply_view(s, k), k)
            err = max(err, float((back - s).abs().max()))
        return err


__all__ = ["SyntheticEgoAugmenter", "rotation_xy", "time_warp",
           "MultiViewGenerator"]
