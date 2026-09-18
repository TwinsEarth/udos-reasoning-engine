"""v7.0.2 确定性可观测运动学通道（root-cause：把“可直接观测的量”与“隐含量”分开）。

v7.0.1 的诚实负结果是：盲路径 accel 缺口最大，且标量 ``accel_a`` 辨识技能为负
（−0.525）。根因不是“加速度不可辨识”，而是**表征口径错误**：

- 状态向量本身就含三维速度，恒定加速度是速度对时间的最小二乘斜率，**三维加速度
  向量可从观测窗直接、精确反演**；
- 旧估计器被迫输出“沿未知随机三维方向 d 的带符号标量 a”，而 d 的符号与 a 的符号
  不可分离（d 翻转、(v0,a) 同时反号给出同一轨迹），这个**标量**天然弱可辨识。

本模块只做确定性、无需训练、零未来泄漏的运动学反演（仅用观测窗 W 帧）：

- ``v_c``[3]：窗心速度（速度均值，匀速即速度本身）；
- ``a_lin``[3]：速度对时间的最小二乘斜率 = 恒定加速度向量（匀速≈0，加速精确）；
- 弹簧 ``omega``+有效性：沿位置 PCA 主轴投影成标量 q，对内点做
  带截距最小二乘 ``q̈ = b1·q + b0``（b1=−ω²，截距吸收振动中心偏差），
  以“弹簧模型残差严格小于常加速度模型残差 + ω² 下限 + 残差门限”做**模型选择**，
  避免短窗二次轨迹被误判为简谐；
- 碰撞 ``jump``+有效性：x 轴帧间速度跳变的稳健检测（中位尺度）。
- ``ca_conf``[10]（v7.0.3）：**恒定加速度一致性置信** ∈[0,1]。速度对时间线性拟合
  的逐窗 R²，再乘以 (1−弹簧有效)(1−碰撞有效)；uniform/accel≈1，spring/collision≈0。
  用作解析积分门控的确定性选择量，避免门控在“解析有利/有害”的类型间收到冲突梯度。

无效特征置 0 且有效性标志置 0（干净输入），不编造。所有量仅来自观测窗；
rollout 每步用当前滑窗重测一次（无未来泄漏）。
"""
from __future__ import annotations

from typing import Dict

import torch

from .contracts import STATE_DIM, VELOCITY_SLICE, POSITION_SLICE

# 特征排布：v_c(3) a_lin(3) omega(1) omega_valid(1) jump(1) jump_valid(1) ca_conf(1)
KIN_DIM = 11
_VC = slice(0, 3)
_AL = slice(3, 6)
_OMEGA = 6
_OMEGA_VALID = 7
_JUMP = 8
_JUMP_VALID = 9
_CA_CONF = 10

# 模型选择门限（探针在 seed 42/1337/314/2026 上：弹簧召回 1.0、其余误报 0）
_OMEGA_MIN = 0.45          # ω² 下限（ω_floor）
_SPRING_REL_RESID = 0.25   # 弹簧拟合相对残差门限
_JUMP_MED_K = 3.0
_JUMP_BIAS = 0.35


@torch.no_grad()
def kinematic_features(window: torch.Tensor, dt: float) -> torch.Tensor:
    """window: [B,W,6] -> [B,KIN_DIM] 确定性可观测运动学特征（全有限）。"""
    if window.dim() != 3 or window.size(-1) != STATE_DIM:
        raise ValueError(
            f"窗口需为 [B,W,{STATE_DIM}]，实际 {tuple(window.shape)}")
    if not bool(torch.isfinite(window).all()):
        raise ValueError("窗口含 NaN/inf，拒绝计算运动学特征")
    B, W, _ = window.shape
    pos = window[..., POSITION_SLICE]
    vel = window[..., VELOCITY_SLICE]
    feat = torch.zeros(B, KIN_DIM, dtype=window.dtype, device=window.device)

    # --- 恒定运动：窗心速度 + 速度对时间的最小二乘斜率（三维加速度向量）---
    t = torch.arange(W, dtype=window.dtype, device=window.device) * dt
    tc = (t - t.mean()).view(1, W, 1)
    denom = float((tc.view(-1) ** 2).sum())
    vbar = vel.mean(dim=1)                                  # [B,3]
    a_lin = ((tc * (vel - vbar.unsqueeze(1))).sum(dim=1)
             / denom)                                       # [B,3]
    feat[:, _VC] = vbar
    feat[:, _AL] = a_lin

    # --- 弹簧：位置 PCA 主轴 -> 标量 q -> 带截距 q̈=b1 q+b0 最小二乘 ---
    if W >= 4:
        axis = _principal_axis(pos)
        q = ((pos - pos.mean(dim=1, keepdim=True))
             * axis.unsqueeze(1)).sum(dim=2)               # [B,W]
        qa = (q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]) / (dt * dt)
        qi = q[:, 1:-1]
        n = qi.size(1)
        s_qi = qi.sum(dim=1)
        s_qa = qa.sum(dim=1)
        det = (qi ** 2).sum(dim=1) * n - s_qi ** 2
        rhs1 = (qi * qa).sum(dim=1)
        b1 = (n * rhs1 - s_qi * s_qa) / det.clamp_min(1e-12)
        b0 = (s_qa - b1 * s_qi) / n
        pred = b1.unsqueeze(1) * qi + b0.unsqueeze(1)
        eps = 1e-9
        r_spring = ((qa - pred).norm(dim=1)
                    / (qa.norm(dim=1) + eps))
        r_const = ((qa - qa.mean(dim=1, keepdim=True)).norm(dim=1)
                   / (qa.norm(dim=1) + eps))
        w2 = -b1
        omega_valid = (w2 > _OMEGA_MIN ** 2) & (r_spring < _SPRING_REL_RESID) \
            & (r_spring < r_const)
        omega = torch.sqrt(w2.clamp_min(0.0))
        feat[:, _OMEGA] = omega * omega_valid.float()
        feat[:, _OMEGA_VALID] = omega_valid.float()

    # --- 碰撞：x 轴帧间速度跳变（稳健，中位尺度）---
    dvx = vel[:, 1:, 0] - vel[:, :-1, 0]                    # [B,W-1]
    abs_dv = dvx.abs()
    med = abs_dv.median(dim=1).values
    jidx = abs_dv.argmax(dim=1)
    jump = dvx[torch.arange(B), jidx]
    jump_valid = jump.abs() > _JUMP_MED_K * med + _JUMP_BIAS
    feat[:, _JUMP] = jump * jump_valid.float()
    feat[:, _JUMP_VALID] = jump_valid.float()

    # --- 恒定加速度一致性 ca_conf：速度线性拟合 R²，并排除弹簧/碰撞窗 ---
    vfit = vbar.unsqueeze(1) + tc * a_lin.unsqueeze(1)      # v(t)=v̄+a·t
    sse = ((vel - vfit) ** 2).sum(dim=(1, 2))
    sst = ((vel - vel.mean()) ** 2).sum() / B              # 全局基准（逐窗稳定）
    r2 = (1.0 - sse / sst.clamp_min(1e-12)).clamp(0.0, 1.0)
    if W >= 4:
        ca_conf = r2 * (1.0 - omega_valid.float()) * (1.0 - jump_valid.float())
    else:
        ca_conf = r2 * (1.0 - jump_valid.float())
    feat[:, _CA_CONF] = ca_conf
    return feat


def _principal_axis(pos: torch.Tensor) -> torch.Tensor:
    """pos [B,W,3] 的逐窗位置主方向（最大方差轴，PCA）。"""
    d = pos - pos.mean(dim=1, keepdim=True)
    S = torch.bmm(d.transpose(1, 2), d)                     # [B,3,3]
    ev, U = torch.linalg.eigh(S)                            # 升序特征值
    return U[:, :, -1]


def feature_names() -> Dict[str, object]:
    return {"KIN_DIM": KIN_DIM,
            "v_c": (0, 3), "a_lin": (3, 6),
            "omega": _OMEGA, "omega_valid": _OMEGA_VALID,
            "jump": _JUMP, "jump_valid": _JUMP_VALID,
            "ca_conf": _CA_CONF}
