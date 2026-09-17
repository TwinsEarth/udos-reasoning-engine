"""
物理守恒一致性检验 (v3.6.0.dev3) —— PWM 物理世界模型
====================================================================
analogy, not reproduction —— 在**合成低维代理**上检验世界模型 rollout 的物理
守恒性, 不宣称真实物理引擎/拉格朗日力学严格不变量; 输出违反量供 A/B 解读。

对一条 rollout 物理状态轨迹 states [B,H,6]=[pos(3),vel(3)]:
    * 动量代理   p_t = m * v_t            (矢量 [3]);
    * 动能代理   K_t = 0.5 * m * |v_t|^2;
    * 势能代理   U_t = spring_k * 0.5 * x^2   (可选弹簧势, x 轴)
                     + mass * g * z           (可选重力势, z 轴向上为正);
    * 总能量代理 E_t = K_t + U_t。

一致性 (违反量):
    * momentum_violation : 沿轨迹 |p_t| 相对首步的最大绝对漂移 (m*v 大小应守恒);
    * momentum_step_delta: 相邻步 |p| 差的最大值;
    * energy_violation    : 相对首步的最大**相对**能量漂移 |E_t-E_0|/(|E_0|+eps);
    * conserved           : 上述违反量均在 tol 内 => 近似守恒。

合成可验: 匀速直线段 m*v 与 0.5*m*v^2 严格不变 (违反≈0); 发散/加速的想象轨迹
违反量 >0。设计纪律: 纯前向、零梯度、确定性; 空序列/非法参数显式 ValueError。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.wm_conservation")


from typing import Any, Dict, Optional

import torch

from .dynamics import RAW_DIM


class ConservationChecker:
    """动量/能量代理量在 rollout 中的一致性检验 (纯解析, 零梯度)。"""

    def __init__(self, mass: float = 1.0,
                 spring_k: Optional[float] = None,
                 gravity: Optional[float] = None) -> None:
        if mass <= 0:
            raise ValueError("mass 须为正")
        self.mass = float(mass)
        self.spring_k = float(spring_k) if spring_k is not None else None
        self.gravity = float(gravity) if gravity is not None else None

    def _potential(self, pos: torch.Tensor) -> torch.Tensor:
        """势能代理 [B,H]。pos[...,0]=x, pos[...,2]=z。"""
        U = torch.zeros(pos.shape[:-1], dtype=pos.dtype)
        if self.spring_k is not None:
            x = pos[..., 0]
            U = U + 0.5 * self.spring_k * x * x
        if self.gravity is not None:
            z = pos[..., 2]
            U = U + self.mass * self.gravity * z
        return U

    @torch.no_grad()
    def check(self, states: torch.Tensor,
              momentum_tol: float = 1e-3,
              energy_tol: float = 1e-2) -> Dict[str, Any]:
        """检验一条 (或一批) rollout 轨迹的动量/能量一致性。

        states: [H,RAW] 或 [B,H,RAW]=[pos3,vel3]。
        返回逐样本 dict (长度 B):
            momentum_per_step [H], momentum_violation, momentum_step_delta,
            energy_per_step [H], energy_violation, conserved, tol。
        """
        s = torch.as_tensor(states, dtype=torch.float32)
        if s.dim() == 2:
            s = s.unsqueeze(0)
        if s.dim() != 3 or s.size(-1) != RAW_DIM:
            raise ValueError("states 需为 [H,RAW] 或 [B,H,RAW_DIM=6]")
        if s.size(1) < 2:
            raise ValueError("守恒检验至少需要 2 步轨迹")
        if not bool(torch.isfinite(s).all()):
            raise ValueError("states 含 NaN/inf")

        pos = s[..., 0:3]                       # [B,H,3]
        vel = s[..., 3:6]                       # [B,H,3]
        p = self.mass * vel                     # [B,H,3]
        p_norm = torch.linalg.norm(p, dim=-1)   # [B,H]
        ke = 0.5 * self.mass * (vel ** 2).sum(dim=-1)   # [B,H]
        pe = self._potential(pos)                          # [B,H]
        energy = ke + pe                                   # [B,H]

        mom_viol = (p_norm - p_norm[:, 0:1]).abs().max(dim=1).values      # [B]
        mom_delta = p_norm.diff(dim=1).abs().max(dim=1).values           # [B]
        e0 = energy[:, 0:1].abs().clamp_min(1e-8)
        e_viol = ((energy - energy[:, 0:1]).abs() / e0).max(dim=1).values  # [B]

        out_batch = []
        for b in range(s.size(0)):
            conserved = (float(mom_viol[b]) <= momentum_tol
                         and float(e_viol[b]) <= energy_tol)
            out_batch.append({
                "momentum_per_step": [round(float(x), 6) for x in p_norm[b]],
                "momentum_violation": round(float(mom_viol[b]), 6),
                "momentum_step_delta": round(float(mom_delta[b]), 6),
                "energy_per_step": [round(float(x), 6) for x in energy[b]],
                "energy_violation": round(float(e_viol[b]), 6),
                "conserved": bool(conserved),
            })
        return {"batch": out_batch, "n_batch": s.size(0),
                "horizon": s.size(1), "mass": self.mass,
                "spring_k": self.spring_k, "gravity": self.gravity,
                "tol": {"momentum": momentum_tol, "energy": energy_tol}}

    def is_conserved(self, states: torch.Tensor, **kw) -> bool:
        """整批是否全部近似守恒 (逐样本 conserved 全为 True)。"""
        out = self.check(states, **kw)
        return all(r["conserved"] for r in out["batch"])
