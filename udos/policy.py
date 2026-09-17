"""
MPC 式候选动作优选 (v2.7.0, opt-in 推理外挂)
=================================================
在已有 PhysicsPredictor 之上, 构建**从预测到行动的闭环**: 给定当前观测窗口 +
场景参数 + 一组候选动作, 对每个动作做一次自由 rollout 预测, 按下式打分并排序:

    score = objective_reward(pred_rollout, action)
            - lambda_risk  * risk_score            # decision.RiskGrader.grade
            - (0 if safe else safety_penalty)      # decision.safety_boundary 违例

设计纪律 (与全工程一致):
    * 纯前向、**确定性**、**不修改 predictor** (只读外挂; 调用前会 eval()).
    * opt-in: 不调用本模块时, 旧推理路径逐位一致; 本模块不挂任何默认钩子。
    * 引擎**不生成动作空间** —— 候选动作由调用方提供。每个动作是一个 dict, 可选:
        - "scene_param":       [4] 张量/list, 覆盖该次 rollout 的场景隐藏参数;
        - "state_perturbation": [6] 张量/list, 叠加到当前窗口末帧 (初始状态扰动);
        - "candidate_state":   [6] 张量/list, 动作提议的下一目标态, 供 safety_boundary
                               判定是否落在预测区间下界之上 (缺省则用 rollout 首步预测)。
      前二者可同时给出 (先叠加扰动, 再用覆盖后的 scene_params 前向)。
    * objective_reward 为可配置 callable ``fn(pred[1,H,6], action) -> float``;
      默认 None: 若构造时给了 reference[H,6] 则取负 MSE(pred, reference), 否则 0.0
      (此时排序仅由风险/安全项决定, 诚实不伪造奖励)。
    * 空候选集 => no_valid_action=True, best_action=None。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.policy")


from typing import Any, Callable, Dict, List, Optional

import torch

from .decision import RiskGrader, safety_boundary
from .dynamics import RAW_DIM, SCENE_PARAM_DIM


def _as3d(window: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(window, dtype=torch.float32)
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.dim() != 3:
        raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
    return w


def _as2d(sp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if sp is None:
        return None
    s = torch.as_tensor(sp, dtype=torch.float32)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    return s


class MPCActionSelector:
    """对候选动作集做 rollout + 风险/安全打分, 返回最优动作与全排序。"""

    def __init__(self, predictor, horizon: int = 4,
                 lambda_risk: float = 1.0, safety_penalty: float = 10.0,
                 objective_reward: Optional[Callable[[torch.Tensor, Dict[str, Any]], float]] = None,
                 reference: Optional[torch.Tensor] = None,
                 risk_grader: Optional[RiskGrader] = None) -> None:
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1")
        self.predictor = predictor
        self.horizon = int(horizon)
        self.lambda_risk = float(lambda_risk)
        self.safety_penalty = float(safety_penalty)
        self.objective_reward = objective_reward
        self.reference = (None if reference is None
                          else torch.as_tensor(reference, dtype=torch.float32))
        self.risk_grader = risk_grader or RiskGrader()

    # ---------------- 默认目标奖励 ---------------- #
    def _default_reward(self, pred: torch.Tensor) -> float:
        if self.reference is None:
            return 0.0
        ref = self.reference
        if ref.dim() == 2:
            ref = ref.unsqueeze(0)
        return float(-((pred - ref) ** 2).mean().item())

    # ---------------- 动作 -> 有效窗口/场景参数 ---------------- #
    def _materialize(self, base_window: torch.Tensor,
                     base_sp: Optional[torch.Tensor],
                     action: Dict[str, Any]):
        window = base_window.clone()
        if "state_perturbation" in action and action["state_perturbation"] is not None:
            pert = torch.as_tensor(action["state_perturbation"],
                                   dtype=torch.float32).reshape(-1)
            if pert.numel() != RAW_DIM:
                raise ValueError(
                    f"state_perturbation 维数 {pert.numel()} != RAW_DIM {RAW_DIM}")
            window[0, -1, :] = window[0, -1, :] + pert
        sp = base_sp
        if "scene_param" in action and action["scene_param"] is not None:
            sp = torch.as_tensor(action["scene_param"],
                                 dtype=torch.float32).reshape(1, -1)
            if sp.numel() != SCENE_PARAM_DIM:
                raise ValueError(
                    f"scene_param 维数 {sp.numel()} != SCENE_PARAM_DIM {SCENE_PARAM_DIM}")
        return window, sp

    @torch.no_grad()
    def select(self, window: torch.Tensor,
               scene_params: Optional[torch.Tensor] = None,
               candidate_actions: Optional[List[Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
        """
        返回 {best_action, best_score, best_index, ranked_actions, no_valid_action}。
        空候选集 => no_valid_action=True 且 best_action=None。纯前向、确定性。
        """
        base_window = _as3d(window)
        base_sp = _as2d(scene_params)
        self.predictor.eval()

        if not candidate_actions:
            return {
                "best_action": None, "best_score": None, "best_index": None,
                "ranked_actions": [], "no_valid_action": True,
            }

        ranked: List[Dict[str, Any]] = []
        for i, action in enumerate(candidate_actions):
            eff_window, eff_sp = self._materialize(base_window, base_sp, action)
            pred = self.predictor.rollout(eff_window, self.horizon,
                                          scene_params=eff_sp)   # [1,H,RAW]
            risk_info = self.risk_grader.grade(
                self.predictor, eff_window, scene_params=eff_sp,
                horizon=self.horizon)
            risk_score = float(risk_info["risk_score"])

            # 安全候选: 默认用 rollout 首步预测; 动作可显式给出提议目标态
            # ("candidate_state"[RAW]) 供 safety_boundary 判定是否落在区间下界之上。
            nxt = pred[0, 0, :]
            if action.get("candidate_state") is not None:
                nxt = torch.as_tensor(action["candidate_state"],
                                      dtype=torch.float32).reshape(-1)
            sb = safety_boundary(self.predictor, eff_window, [nxt],
                                 scene_params=eff_sp)[0]
            safe = bool(sb["safe"])

            if self.objective_reward is not None:
                reward = float(self.objective_reward(pred, action))
            else:
                reward = self._default_reward(pred)

            penalty = 0.0 if safe else self.safety_penalty
            score = reward - self.lambda_risk * risk_score - penalty
            ranked.append({
                "index": i, "action": action,
                "score": round(score, 8), "risk": round(risk_score, 6),
                "safe": safe, "reward": round(reward, 8),
            })

        ranked.sort(key=lambda r: r["score"], reverse=True)
        best = ranked[0]
        return {
            "best_action": best["action"],
            "best_score": best["score"],
            "best_index": best["index"],
            "ranked_actions": ranked,
            "no_valid_action": False,
        }
