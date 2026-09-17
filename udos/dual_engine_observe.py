"""
双引擎可观测性 (v5.5.3)
=======================
让"CTM 主预测员 + GPM 场景记录员"的协同对调用方透明可审计。给定观测窗口与
本次实际使用的场景条件, 在同一份输入上并排跑两条 rollout:

    盲轨迹   blind       : 不喂任何场景参数/记忆
    感知轨迹 conditioned : 喂本次场景条件 (显式参数 / 学习头 / 经典路由 / GPM 桥)

并给出:
  * source          场景条件来源 (metadata/attributes/learned_head/classical_router/blind)
  * route_*         v5.5.2 运动类型路由结果与一个 [0,1] 的"判别果断度"启发量
                    (注意: 这是分类裕度, 不是概率)
  * gate_*_delta    盲-感知轨迹在位置/速度上的逐步差 (场景门的边际贡献),
                    以及均值/末步标量; 差越大说明场景记忆对预测的改变越大
  * fan_*_halfwidth 可选: v5.5.1/5.5.2 扇形的逐步位置半宽 (预测不确定性)

所有量都来自同输入、同冻结预测员的两次确定性前向, 可复现、无训练、无外部状态。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .dynamics import RAW_DIM
from .dynamics_router import (
    CLASS_NAMES, JUMP_K, JUMP_FLOOR, DynamicsRoute, classify_dynamics,
)
from .scene_estimator import _validate


@dataclass
class EngineObservation:
    source: str
    conditioned: bool
    route_label: int
    route_name: str
    route_confidence: float
    blind_trajectory: torch.Tensor          # [H,6]
    conditioned_trajectory: torch.Tensor    # [H,6]
    gate_position_delta: torch.Tensor       # [H] 每步位置向量范数
    gate_velocity_delta: torch.Tensor       # [H] 每步速度向量范数
    gate_contribution: float                # 位置逐步差的均值
    gate_final_delta: float                 # 末步位置差
    fan_position_halfwidth: Optional[torch.Tensor] = None  # [H] 或 None

    def as_dict(self, ndigits: int = 6) -> Dict[str, Any]:
        def rlist(t: torch.Tensor):
            return [round(float(v), ndigits) for v in t.tolist()]

        out = {
            "source": self.source,
            "conditioned": bool(self.conditioned),
            "route_label": int(self.route_label),
            "route_name": self.route_name,
            "route_confidence": round(float(self.route_confidence), ndigits),
            "gate_contribution": round(float(self.gate_contribution), ndigits),
            "gate_final_delta": round(float(self.gate_final_delta), ndigits),
            "gate_position_delta_per_step": rlist(self.gate_position_delta),
            "gate_velocity_delta_per_step": rlist(self.gate_velocity_delta),
        }
        if self.fan_position_halfwidth is not None:
            out["fan_position_halfwidth_per_step"] = rlist(
                self.fan_position_halfwidth)
        return out


def _route_confidence(route: DynamicsRoute) -> float:
    """把路由信号压成 [0,1] 的判别果断度 (启发量, 非概率)。取批内第 1 条。"""
    label = int(route.labels[0])
    name = CLASS_NAMES[label]
    if name == "collision":
        thr = JUMP_K * float(route.jump_med[0]) + JUMP_FLOOR
        ratio = float(route.jump_max[0]) / max(thr, 1e-12)
        return min(1.0, max(0.0, ratio - 1.0))
    if name == "spring":
        rc = float(route.const_rr[0])
        rr = float(route.harmonic_rr[0])
        return min(1.0, max(0.0, (rc - rr) / (rc + rr + 1e-12)))
    # uniform / accel: 速度线性优度即果断度
    return min(1.0, max(0.0, float(route.lin_r2[0])))


@torch.no_grad()
def observe_conditioning(predictor, window: torch.Tensor, *,
                         conditioned_params: Optional[torch.Tensor] = None,
                         source: str = "blind",
                         scene_bias: Optional[torch.Tensor] = None,
                         horizon: int = 4, dt: float = 0.5,
                         route: Optional[DynamicsRoute] = None,
                         fan=None,
                         precomputed_conditioned: Optional[torch.Tensor] = None
                         ) -> EngineObservation:
    """在同一窗口上并排比较盲轨迹与场景条件轨迹, 产出可观测性记录。"""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon 必须为 >=1 的整数")
    w = _validate(window, dt).float()
    if w.size(0) != 1:
        raise ValueError("observe_conditioning 仅支持单样本窗口 [1,W,6]")
    H = int(horizon)

    blind = predictor.rollout(w, H)[0]                        # [H,6]
    if precomputed_conditioned is not None:
        cond = precomputed_conditioned.detach().float()
        if cond.shape != blind.shape:
            raise ValueError("precomputed_conditioned 形状须为 [H,6]")
    elif conditioned_params is not None:
        cond = predictor.rollout(w, H,
                                 scene_params=conditioned_params,
                                 scene_bias=scene_bias)[0]
    else:
        cond = blind

    delta = (cond - blind)
    pos_delta = delta[..., :3].norm(dim=-1)                   # [H]
    vel_delta = delta[..., 3:6].norm(dim=-1)                  # [H]

    if route is None:
        route = classify_dynamics(w, dt)
    label = int(route.labels[0])

    fan_hw = None
    if fan is not None:
        low = fan.low[0] if fan.low.dim() == 3 else fan.low
        high = fan.high[0] if fan.high.dim() == 3 else fan.high
        fan_hw = (((high - low) / 2.0)[..., :3]).norm(dim=-1)  # [H]

    conditioned = conditioned_params is not None or scene_bias is not None
    return EngineObservation(
        source=str(source),
        conditioned=bool(conditioned),
        route_label=label,
        route_name=CLASS_NAMES[label],
        route_confidence=_route_confidence(route),
        blind_trajectory=blind,
        conditioned_trajectory=cond,
        gate_position_delta=pos_delta,
        gate_velocity_delta=vel_delta,
        gate_contribution=float(pos_delta.mean()),
        gate_final_delta=float(pos_delta[-1]),
        fan_position_halfwidth=fan_hw,
    )
