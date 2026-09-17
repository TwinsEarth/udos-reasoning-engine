"""规划-执行-反馈全域闭环 —— ClosedLoopOrchestrator (v3.8.0.dev2)
======================================================================
把 3.7 三层神经控制 (大脑规划 -> 小脑执行 -> 脊髓反射) 与 3.6 世界模型 (WM 想象反馈)
以及空间感知更新串成**一步完整闭环**, 复用全部既有模块, 不加权重。

一步闭环 (step):
    1) 大脑规划 -> 小脑执行 -> 脊髓反射: HierarchicalController.step 出 command[6];
    2) WM 想象反馈: LatentWorldModel.imagine(window, horizon) 出多步想象轨迹,
       与本步 command 比较得反馈误差 (仅观测, 不回传改权重);
    3) 空间感知更新: 以 command 滑窗推进观测 (新窗 = [旧窗[1:], command]),
       供下一步感知消费 (状态机式闭环, 非一次性 rollout)。

设计纪律 (与全工程一致):
    * 纯前向、**确定性**、零梯度、**不改主 predictor 52191 参数** (只读外挂组合)。
    * opt-in: 不构造 ClosedLoopOrchestrator 时, 旧推理路径逐位一致。
    * 空 / 非法输入显式 ValueError。
    * analogy, not reproduction —— 闭环为合成可测, 非真机实时控制总线。

第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .neural_control import HierarchicalController
from .world_model import LatentWorldModel

logger = logging.getLogger("udos.closed_loop")


class ClosedLoopOrchestrator:
    """全域一步闭环编排器 (大脑->小脑->脊髓->WM想象反馈->空间感知更新)。"""

    def __init__(self, predictor, wm_horizon: int = 2,
                 position_bounds: Optional[float] = None,
                 collision_obstacles=None, collision_radius: float = 0.5,
                 speed_limit: Optional[float] = None) -> None:
        if wm_horizon < 1:
            raise ValueError("wm_horizon 需 >= 1")
        self.predictor = predictor
        self.wm_horizon = int(wm_horizon)
        self.ctrl = HierarchicalController(
            predictor, position_bounds=position_bounds,
            collision_obstacles=collision_obstacles,
            collision_radius=collision_radius, speed_limit=speed_limit)
        self.wm = LatentWorldModel(predictor)
        self.step_index = 0
        self.last_window: Optional[torch.Tensor] = None
        self.feedback_log = []

    def reset(self) -> None:
        self.ctrl.reset()
        self.step_index = 0
        self.last_window = None
        self.feedback_log = []

    @torch.no_grad()
    def step(self, window: torch.Tensor,
             scene_params: Optional[torch.Tensor] = None,
             candidate_actions: Optional[Any] = None) -> Dict[str, Any]:
        """执行一步全域闭环, 返回 command + 分层诊断 + WM 反馈 + 更新后感知窗。"""
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3 or w.size(0) != 1:
            raise ValueError("window 需为 [W,6] 或 [1,W,6] (闭环单智能体)")
        if not bool(torch.isfinite(w).all()):
            raise ValueError("window 含 NaN/inf 非有限值")

        # 1) 大脑->小脑->脊髓
        ctrl_out = self.ctrl.step(w, scene_params=scene_params,
                                  candidate_actions=candidate_actions)
        command = ctrl_out["command"].reshape(6)   # [6]

        # 2) WM 想象反馈 (仅观测; H=1 步逐位锚定真实单步)
        imagined = self.wm.imagine(w, self.wm_horizon,
                                   scene_params=scene_params)   # [1,H,6]
        next_real = imagined[:, 0, :].reshape(6)
        feedback_l1 = float((command - next_real).abs().mean().item())
        feedback_l2 = float(((command - next_real) ** 2).mean().item())

        # 3) 空间感知更新: command 滑窗推进观测
        new_window = torch.cat([w[:, 1:, :],
                                command.reshape(1, 1, 6)], dim=1)
        self.last_window = new_window
        self.step_index += 1
        rec = {
            "step": self.step_index - 1,
            "feedback_l1": round(feedback_l1, 6),
            "feedback_l2": round(feedback_l2, 6),
            "reflex_triggered": ctrl_out["reflex_triggered"],
            "winner": ctrl_out["priority_winner"],
        }
        self.feedback_log.append(rec)

        return {
            "command": command,
            "step_index": self.step_index - 1,
            "priority_winner": ctrl_out["priority_winner"],
            "reflex_triggered": ctrl_out["reflex_triggered"],
            "safety_state": ctrl_out["safety_state"],
            "cortex": ctrl_out["cortex"],
            "cerebellum": ctrl_out["cerebellum"],
            "spinal": ctrl_out["spinal"],
            "wm_feedback": {
                "horizon": self.wm_horizon,
                "feedback_l1_to_next": round(feedback_l1, 6),
                "feedback_l2_to_next": round(feedback_l2, 6),
                "imagined_next": [round(float(x), 6) for x in next_real.tolist()],
            },
            "perception": {
                "updated_window_shape": list(new_window.shape),
                "window_slid": True,
            },
            "deterministic": True,
            "zero_gradient": True,
            "analogy_not_reproduction": True,
        }
