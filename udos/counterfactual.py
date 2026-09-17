"""
因果链 / 反事实推演 (v2.6.0+dev1)
====================================
在**不修改**已训练 predictor 状态、纯前向、确定性的前提下, 回答
"如果场景物理参数 / 初始状态 / 初始速度不同, 轨迹会怎样?" 的反事实问题。

    baseline      = 无干预 rollout (与 predictor.rollout 逐位一致)
    counterfactual= 施加 intervention 后的 rollout
    ate_by_step   = 逐步 |cf - baseline|^2 的均值 [H]  (平均处理效应 ATE)

intervention 协议 (dict):
    None / {}                                  => 基线 (逐位等价 predictor.rollout)
    {"scene_params": {index: value}}           => 覆盖指定 scene_param 槽位
    {"initial_state": delta_tensor[R]|[B,R]}   => 对窗口末帧加扰动
    {"velocity_override": value|tensor[3]|[B,3]} => 覆盖初始速度 (窗口末帧 vel)
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.counterfactual")


from typing import Any, Dict, Optional

import torch


class CounterfactualEngine:
    """持有 PhysicsPredictor 引用, 做干预式反事实 rollout (只读, 不训练)。"""

    def __init__(self, predictor):
        self.predictor = predictor

    @staticmethod
    def _validate_intervention(intervention: Dict[str, Any]) -> None:
        """v2.6.1: 干预值含 NaN/inf 时显式 ValueError (不静默污染 rollout)。"""
        def _check_scalar(name: str, v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return
            if fv != fv or fv == float("inf") or fv == float("-inf"):
                raise ValueError(
                    f"intervention[{name}]={v!r} 含 NaN/inf, 拒绝非有限干预")

        sp = intervention.get("scene_params")
        if sp:
            for idx, val in sp.items():
                _check_scalar(f"scene_params[{idx}]", val)
        for key in ("initial_state", "velocity_override"):
            val = intervention.get(key)
            if val is None:
                continue
            t = torch.as_tensor(val, dtype=torch.float32)
            if not bool(torch.isfinite(t).all()):
                raise ValueError(
                    f"intervention[{key}] 含 NaN/inf, 拒绝非有限干预")

    @torch.no_grad()
    def rollout_intervention(self, raw_window: torch.Tensor, horizon: int,
                             scene_params: Optional[torch.Tensor] = None,
                             intervention: Optional[Dict[str, Any]] = None
                             ) -> torch.Tensor:
        """按 intervention 修改输入后 rollout; None/空 => 基线逐位等价。"""
        window = raw_window
        sp = scene_params
        if intervention is None or len(intervention) == 0:
            return self.predictor.rollout(raw_window, horizon,
                                          scene_params=scene_params)
        # v2.6.1: 干预值含 NaN/inf 显式报错 (不静默污染 rollout)
        self._validate_intervention(intervention)
        if "scene_params" in intervention and intervention["scene_params"]:
            assert sp is not None, "scene_params 干预需要基线 scene_params"
            sp = sp.clone()
            for idx, val in intervention["scene_params"].items():
                sp[:, int(idx)] = float(val)
        if "initial_state" in intervention and intervention["initial_state"] is not None:
            delta = intervention["initial_state"]
            window = window.clone()
            window[:, -1, :] = window[:, -1, :] + delta.to(window.dtype)
        if "velocity_override" in intervention \
                and intervention["velocity_override"] is not None:
            v = intervention["velocity_override"]
            window = window.clone()
            window[:, -1, 3:6] = torch.as_tensor(v, dtype=window.dtype,
                                                 device=window.device)
        return self.predictor.rollout(window, horizon, scene_params=sp)

    @torch.no_grad()
    def counterfactual(self, raw_window: torch.Tensor, horizon: int,
                       scene_params: Optional[torch.Tensor] = None,
                       intervention: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        """返回 baseline / counterfactual / ate_by_step / ate_mean /
        final_state_diff / intervention。ATE = mean((cf-baseline)^2) 逐步。"""
        baseline = self.rollout_intervention(raw_window, horizon,
                                             scene_params=scene_params,
                                             intervention=None)
        cf = self.rollout_intervention(raw_window, horizon,
                                       scene_params=scene_params,
                                       intervention=intervention)
        per_step_se = ((cf - baseline) ** 2).mean(dim=(0, 2))     # [H]
        return {
            "baseline": baseline,
            "counterfactual": cf,
            "ate_by_step": per_step_se,                            # [H]
            "ate_mean": float(per_step_se.mean().item()),
            "final_state_diff": cf[:, -1, :] - baseline[:, -1, :],  # [B,R]
            "intervention": intervention,
        }
