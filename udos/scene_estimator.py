"""
场景隐藏参数估计器 (v5.4.8)
===========================
在没有 GPM 显式场景参数时, 仅凭观测窗口 [B,W,6]=[pos(3),vel(3)] 用确定性
经典运动学反演 4 个隐藏场景参数 (槽位与 udos.dynamics.SCENE_PARAM_NAMES 一致):

    (v0, accel_a, spring_omega, other_v2)

设计原则 (与 v5.4.5 显式承载、v5.4.6 主链接线互补):
  * 只输出物理上可从窗口辨识的量; 不可辨识就置 NaN 并把 observable 掩码置
    False, 绝不编造 (尤其碰撞对方速度 other_v2, 主体窗口永远看不到)。
  * 无需训练、确定性、可批量; 学习型估计头在 v5.5.0 联合训练引入。
  * 速度/加速度只取 x 轴 (合成数据仅 x 轴非平凡, 见 dynamics 模块说明)。

方法:
  v0       速度对时间最小二乘的窗口局部截距 (匀速=全局 v0; 匀加速=全局
           v0 + a*s*dt, 即窗口起点速度)。速度本就在 raw 中, 恒可观测。
  accel_a  速度最小二乘斜率; 仅当速度线性拟合优度 lin_r2>=LINEAR_R2_MIN
           (匀速 a=0、匀加速) 标记可观测; 弹簧/碰撞非常数加速置掩码 False。
  omega    简谐关系 acc=-w^2 x, 用位置二阶差分最小二乘反演 w2; 仅当
           (速度非线性 lin_r2<LINEAR_R2_MIN) 且 w2>=OMEGA2_MIN 且
           简谐拟合相对残差 rr<=OMEGA_RR_MAX 才标记可观测。
  other_v2 恒不可观测, 恒 NaN。

在生成器 (W=6, dt=0.5) 三 seed 实测: 弹簧 omega 召回约 0.78, 其余三类
omega 误报为 0 (精度优先工作点; 短窗弧度不足时诚实放弃)。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from .dynamics import RAW_DIM, SCENE_PARAM_DIM, SCENE_PARAM_NAMES

logger = logging.getLogger("udos.scene_estimator")

# 槽位索引 (与 SCENE_PARAM_NAMES 对齐)
_SLOT_V0 = SCENE_PARAM_NAMES.index("v0")
_SLOT_A = SCENE_PARAM_NAMES.index("accel_a")
_SLOT_OMEGA = SCENE_PARAM_NAMES.index("spring_omega")
_SLOT_V2 = SCENE_PARAM_NAMES.index("other_v2")

# 工作点门限 (由 scripts/v7 同类探针在生成器分布上标定, 见模块 docstring)
LINEAR_R2_MIN = 0.95     # 速度线性 (常数加速 regime) 优度门限
OMEGA2_MIN = 0.09        # omega>=~0.30, 排除近零曲率
OMEGA_RR_MAX = 0.10      # 简谐 acc=-w2*x 的相对残差上限
_EPS = 1e-12


@dataclass
class SceneEstimate:
    """估计结果。

    values:         [B,4] float32, 不可观测槽位为 NaN
    observable:     [B,4] bool, 该槽位是否物理可从窗口辨识
    linear_r2:      [B] 速度线性拟合优度 (常数加速判定依据)
    omega_residual: [B] 简谐拟合相对残差 rr (越小越像简谐, 可观测性依据)
    """

    values: torch.Tensor
    observable: torch.Tensor
    linear_r2: torch.Tensor
    omega_residual: torch.Tensor

    def values_filled(self, fill: float = 0.0) -> torch.Tensor:
        """把不可观测槽位的 NaN 替换为 fill (通常 0), 供只有 4 维入口、
        尚无掩码通道的主预测员消费; 可观测槽位逐位不变。"""
        if not math.isfinite(float(fill)):
            raise ValueError("fill 必须为有限数值")
        out = self.values.clone()
        out[~self.observable] = float(fill)
        return out


def _validate(window: torch.Tensor, dt: float) -> torch.Tensor:
    if not isinstance(window, torch.Tensor):
        raise ValueError("window 必须是 torch.Tensor")
    w = window
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.dim() != 3 or w.size(-1) != RAW_DIM:
        raise ValueError(
            f"window 形状须为 [W,{RAW_DIM}] 或 [B,W,{RAW_DIM}], 实际 {tuple(window.shape)}")
    if w.size(1) < 3:
        raise ValueError("估计 spring_omega 需要至少 3 帧做位置二阶差分")
    if not bool(torch.isfinite(w).all()):
        raise ValueError("window 含 NaN/inf, 拒绝估计")
    if not (isinstance(dt, (int, float)) and math.isfinite(float(dt)) and dt > 0):
        raise ValueError(f"dt 必须为正有限值, 实际 {dt!r}")
    return w


@torch.no_grad()
def estimate_scene_params(window: torch.Tensor, dt: float = 0.5, *,
                          linear_r2_min: float = LINEAR_R2_MIN,
                          omega2_min: float = OMEGA2_MIN,
                          omega_rr_max: float = OMEGA_RR_MAX
                          ) -> SceneEstimate:
    """从观测窗口反演 4 维隐藏场景参数 + 可观测性掩码 (确定性, 可批量)。"""
    w = _validate(window, dt).float()
    B, W, _ = w.shape
    device = w.device
    t = torch.arange(W, dtype=torch.float32, device=device) * float(dt)
    tm = t.mean()
    t_c = t - tm
    stt = float((t_c ** 2).sum().clamp_min(_EPS))

    x = w[:, :, 0]          # 位置 x
    vx = w[:, :, 3]         # 速度 vx
    vm = vx.mean(dim=1, keepdim=True)

    # --- 速度线性最小二乘: vx = c + a*t ---
    a_hat = ((t_c.unsqueeze(0)) * (vx - vm)).sum(dim=1) / stt          # [B]
    c_hat = (vm.squeeze(1) - a_hat * float(tm))                        # [B]
    v_pred = c_hat.unsqueeze(1) + a_hat.unsqueeze(1) * t.unsqueeze(0)
    ss_tot = ((vx - vm) ** 2).sum(dim=1)
    ss_res = ((vx - v_pred) ** 2).sum(dim=1)
    # 常速窗口 ss_tot≈0, 视为完美线性
    lin_r2 = torch.where(ss_tot > _EPS, 1.0 - ss_res / ss_tot.clamp_min(_EPS),
                         torch.ones_like(ss_tot))
    const_accel = lin_r2 >= linear_r2_min

    # --- 简谐反演: 位置二阶差分 acc, 拟合 acc = -w2*x ---
    acc = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / (float(dt) ** 2)
    xi = x[:, 1:-1]
    sxx = (xi ** 2).sum(dim=1).clamp_min(_EPS)
    sxa = (xi * acc).sum(dim=1)
    omega2 = -(sxa / sxx)                                             # [B]
    pa = -omega2.unsqueeze(1) * xi
    e_acc = (acc ** 2).sum(dim=1)
    rr = ((acc - pa) ** 2).sum(dim=1) / e_acc.clamp_min(_EPS)         # [B]
    omega_ok = (~const_accel) & (omega2 >= omega2_min) & (rr <= omega_rr_max)
    omega_val = torch.sqrt(omega2.clamp_min(0.0))

    # --- 组装 4 槽位 ---
    values = torch.full((B, SCENE_PARAM_DIM), float("nan"),
                        dtype=torch.float32, device=device)
    observable = torch.zeros((B, SCENE_PARAM_DIM), dtype=torch.bool,
                             device=device)
    values[:, _SLOT_V0] = c_hat
    observable[:, _SLOT_V0] = True
    values[:, _SLOT_A] = a_hat
    observable[:, _SLOT_A] = const_accel
    values[:, _SLOT_OMEGA] = torch.where(
        omega_ok, omega_val, torch.full_like(omega_val, float("nan")))
    observable[:, _SLOT_OMEGA] = omega_ok
    # other_v2: 主体窗口物理不可见, 恒 NaN / 不可观测
    values[:, _SLOT_V2] = float("nan")
    observable[:, _SLOT_V2] = False

    return SceneEstimate(values=values, observable=observable,
                         linear_r2=lin_r2, omega_residual=rr)
