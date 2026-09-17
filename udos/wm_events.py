"""
世界模型接触 / 碰撞事件预测 (v3.6.0.dev2) —— PWM 物理世界模型
====================================================================
analogy, not reproduction —— 从世界模型解码出的物理状态轨迹, 用**碰撞几何代理**
(非学习) 预测接触/碰撞事件发生时刻与序列。不接真实物理引擎 (无冲量/摩擦/解算);
与 `udos/collision.py` 的球-球代理判定同口径 (中心距 <= r1+r2)。

输入: 一条想象/真实 rollout 的物理状态轨迹 states [B, H, 6] = [pos(3), vel(3)]。

判定 (逐样本):
    * distance_t   : 主体位置到参考物体/墙的距离;
    * contact_flag : distance_t <= agent_r + partner_r  (球-球接触代理);
    * closing_speed: -(distance_{t+1}-distance_t)/dt  (负=接近中);
    * contact_prob : 几何代理软概率 ∈ [0,1] (接触时=1, 随接近单调增大);
    * event_step   : 新进入接触的步 (上一步未接触、本步接触), 对齐
                     CollisionDetector.continuous_contact 的"新接触"语义;
    * impact_step  : 速度分量符号反转的步 (合成 collision 数据集弹性碰撞代理)。

设计纪律: 纯前向、确定性、零梯度; 空轨迹/非法半径显式 ValueError; 不改主权重。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.wm_events")


from typing import Any, Dict, List, Optional, Sequence

import torch

from .dynamics import RAW_DIM


class ContactPredictor:
    """从预测物理状态轨迹预测接触/碰撞事件 (几何代理, 非学习)。"""

    def __init__(self, contact_tol: float = 1e-6) -> None:
        self.contact_tol = float(contact_tol)

    # ------------------------------------------------------------------ #
    # 输入守卫
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_states(states: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(states, dtype=torch.float32)
        if s.dim() == 2:
            s = s.unsqueeze(0)
        if s.dim() != 3 or s.size(-1) != RAW_DIM:
            raise ValueError(
                "states 需为 [H,RAW] 或 [B,H,RAW_DIM], 末维=6 [pos3,vel3]")
        if s.size(1) == 0:
            raise ValueError("轨迹时间维为空")
        if not bool(torch.isfinite(s).all()):
            raise ValueError("states 含 NaN/inf")
        return s

    # ------------------------------------------------------------------ #
    # 主接口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, states: torch.Tensor,
                partner_position: Optional[Sequence[float]] = None,
                agent_radius: float = 0.5, partner_radius: float = 0.5,
                dt: float = 0.5) -> Dict[str, Any]:
        """对一条 (或一批) 物理状态轨迹预测接触事件序列。

        partner_position=None => 以"原点墙" (x=y=z=0) 为参考; 否则为 [3] 参考点。
        返回逐样本 dict 列表 (长度 B), 每样本含:
            distances [H], contact_flags [H] bool, closing_speed [H],
            contact_prob [H] ∈[0,1], contact_steps list[int],
            impact_steps list[int], n_events int。
        """
        if agent_radius <= 0 or partner_radius <= 0:
            raise ValueError("半径须为正")
        if dt <= 0:
            raise ValueError("dt 须为正")
        s = self._as_states(states)
        B, H, _ = s.shape

        if partner_position is None:
            ref = torch.zeros(1, 1, 3)
        else:
            ref = torch.as_tensor(partner_position, dtype=torch.float32)
            if ref.numel() != 3:
                raise ValueError("partner_position 须为长度 3")
            ref = ref.view(1, 1, 3)

        pos = s[..., 0:3]                                   # [B,H,3]
        vel = s[..., 3:6]                                   # [B,H,3]
        dist = torch.linalg.norm(pos - ref, dim=-1)         # [B,H]
        threshold = float(agent_radius + partner_radius)
        contact_flag = dist <= threshold + self.contact_tol  # [B,H]

        # 接近速度: -(dist_{t+1}-dist_t)/dt; 末步用末速度沿位置方向投影近似
        d_diff = -dist.diff(dim=1) / dt                     # [B,H-1]
        closing = torch.cat([d_diff, d_diff[:, -1:]], dim=1)  # [B,H]

        # 几何代理软概率: 距离越近越接近 1; 接触时饱和到 1
        ratio = (dist / max(threshold, 1e-9)).clamp(0.0, 1.0)
        contact_prob = (1.0 - ratio).clamp(0.0, 1.0)
        contact_prob = torch.where(contact_flag, torch.ones_like(contact_prob),
                                   contact_prob)

        # 事件步: 新进入接触 (t 接触 且 t-1 未接触)
        # 速度反转代理: 任一速度分量相邻步异号 (且非零穿越)
        vx = vel[..., 0]                                    # [B,H]
        sign_change = (vx[:, 1:] * vx[:, :-1] < 0)        # [B,H-1]

        results: List[Dict[str, Any]] = []
        for b in range(B):
            cf = contact_flag[b]
            contact_steps = [int(t) for t in range(H)
                             if bool(cf[t]) and (t == 0 or not bool(cf[t - 1]))]
            imp = [int(t + 1) for t in range(H - 1) if bool(sign_change[b, t])]
            results.append({
                "distances": dist[b].tolist(),
                "contact_flags": [bool(x) for x in cf],
                "closing_speed": closing[b].tolist(),
                "contact_prob": [round(float(x), 6) for x in contact_prob[b]],
                "contact_steps": contact_steps,
                "impact_steps": imp,
                "n_events": len(contact_steps),
                "threshold": threshold,
            })
        return {"batch": results, "n_batch": B, "horizon": H,
                "threshold": threshold}

    @torch.no_grad()
    def has_contact(self, states: torch.Tensor, **kw) -> bool:
        """整条轨迹上是否发生过任一接触事件。"""
        out = self.predict(states, **kw)
        return any(r["n_events"] > 0 for r in out["batch"])
