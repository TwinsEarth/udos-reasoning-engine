"""
自规划目标分解 (v4.1.0.dev6)
=====================================
analogy, not reproduction —— 受"长程任务拆解 / 自规划"思想启发的合成低维类比:
给定起始窗口与目标态 goal[6], 自动把长程目标拆成一串可达子目标链,
复用 policy.MPCActionSelector (按目标奖励对候选中间步打分优选)。

GoalDecomposer.decompose(window, goal, n_subgoals):
    1) 起点 = 窗口末帧; goal = 目标态;
    2) 在 start -> goal 间线性插值出 n_subgoals 个候选中间子目标;
    3) 对每段, 用 MPCActionSelector 对"朝该子目标的小步状态扰动"候选动作
       按 rollout 奖励 (-MSE to subgoal) + 风险/安全 打分, 选出最优中间步;
    4) 输出子目标链 + 每段到 goal 的残差距离 (应单调下降)。

设计纪律 (与全工程一致):
    * 纯前向、确定性、零梯度、不改主 predictor 52191 权重 (只读外挂组合)。
    * opt-in: 不构造本模块时, 旧推理路径逐位一致。
    * 空 / 非法输入显式 ValueError。
    * 日志走 logging_config (stderr), 绝不写 stdout / HTTP 体 / metrics。
    * 第二引擎统一称 GPM。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.selfplan")

from typing import Any, Dict, List, Optional

import torch

from .dynamics import RAW_DIM
from .policy import MPCActionSelector


class GoalDecomposer:
    """goal -> 子目标链 自规划分解器 (复用 MPCActionSelector, 纯前向外挂)。"""

    def __init__(self, predictor, horizon: int = 2,
                 lambda_risk: float = 0.5) -> None:
        if horizon < 1:
            raise ValueError("horizon 需 >=1")
        self.predictor = predictor
        self.horizon = int(horizon)
        self.selector = MPCActionSelector(predictor, horizon=horizon,
                                          lambda_risk=lambda_risk)

    @staticmethod
    def _as_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3 or w.size(0) != 1:
            raise ValueError("window 需为 [W,6] 或 [1,W,6] (单任务)")
        return w

    @staticmethod
    def _as_goal(goal: torch.Tensor) -> torch.Tensor:
        g = torch.as_tensor(goal, dtype=torch.float32).reshape(-1)
        if g.numel() != RAW_DIM:
            raise ValueError(f"goal 维数 {g.numel()} != RAW_DIM {RAW_DIM}")
        if not bool(torch.isfinite(g).all()):
            raise ValueError("goal 含 NaN/inf")
        return g

    @torch.no_grad()
    def decompose(self, window: torch.Tensor, goal: torch.Tensor,
                  n_subgoals: int = 4,
                  scene_params: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """把长程目标拆成可达子目标链。

        window: [W,6]|[1,W,6]; goal: [6]; n_subgoals>=1。
        返回: {goal, subgoal_chain[K,6], residuals_to_goal[K],
               monotone_decreasing, chosen_actions}。
        """
        w = self._as_window(window)
        g = self._as_goal(goal)
        if not isinstance(n_subgoals, int) or n_subgoals < 1:
            raise ValueError("n_subgoals 需为 >=1 的整数")
        self.predictor.eval()

        start = w[0, -1, :].clone()
        # 线性插值候选子目标 (不含起点)
        ts = torch.linspace(0.0, 1.0, steps=n_subgoals + 1)[1:]   # [K]
        chain: List[torch.Tensor] = []
        residuals: List[float] = []
        chosen: List[Dict[str, Any]] = []
        cur = w
        for k in range(n_subgoals):
            subgoal = start + ts[k] * (g - start)
            # 候选动作: 朝 subgoal 的 ±小步扰动 (policy 负责优选)
            step = (subgoal - start)
            cand = [
                {"state_perturbation": (step * f).tolist(),
                 "candidate_state": subgoal.tolist()}
                for f in (0.5, 1.0, 1.5)
            ]
            self.selector.reference = subgoal.unsqueeze(0).unsqueeze(0)
            res = self.selector.select(cur, scene_params=scene_params,
                                       candidate_actions=cand)
            # 用选中动作扰动后更新当前窗
            if not res["no_valid_action"]:
                pert = torch.as_tensor(res["best_action"]["state_perturbation"],
                                       dtype=torch.float32)
                nxt = cur[:, -1, :] + pert
                cur = torch.cat([cur[:, 1:, :], nxt.reshape(1, 1, RAW_DIM)],
                                dim=1)
            chain.append(subgoal)
            chosen.append({"score": res["best_score"],
                            "no_valid_action": res["no_valid_action"]})
            residuals.append(round(float((cur[0, -1, :] - g).norm().item()), 6))

        chain_t = torch.stack(chain, dim=0)
        mono = all(residuals[i] >= residuals[i + 1] - 1e-6
                   for i in range(len(residuals) - 1))
        return {
            "goal": g.tolist(),
            "start": start.tolist(),
            "subgoal_chain": [torch.round(c, decimals=6).tolist() for c in chain_t],
            "residuals_to_goal": residuals,
            "monotone_decreasing": bool(mono),
            "n_subgoals": n_subgoals,
            "selected_actions": chosen,
            "main_predictor_untouched": True,
            "zero_gradient": True,
            "analogy_not_reproduction": True,
        }


class StopCorrectController:
    """"何时停止 / 自我纠正"判据 (v4.1.1) —— 复用语义: 置信门控(calibration) +
    收敛停止 + 不确定主动验证(ood/guard)。

    对一条自规划子目标链做三种判据:
        1) 置信门控: 外部给的 confidence_fn(window) 若 < conf_threshold,
           判低置信 => 决策 self_correct (重规划), 不直接采纳;
        2) 收敛停止: 残差序列相邻两步变化 < convergence_tol => 已收敛 => stop;
        3) 不确定主动验证: 若 ood_detector 判当前窗 OOD => 触发主动验证标志
           (交回 SolvabilityVerifier / guard 复核), decision=verify。

    纯前向、确定性、零梯度; 空/非法显式 ValueError。opt-in。
    """

    def __init__(self, decomposer: GoalDecomposer,
                 confidence_fn=None,
                 ood_detector=None,
                 conf_threshold: float = 0.5,
                 convergence_tol: float = 1e-3,
                 max_iter: int = 8) -> None:
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError("conf_threshold 需在 [0,1]")
        if convergence_tol < 0:
            raise ValueError("convergence_tol 需 >=0")
        if max_iter < 1:
            raise ValueError("max_iter 需 >=1")
        self.decomposer = decomposer
        self.confidence_fn = confidence_fn
        self.ood_detector = ood_detector
        self.conf_threshold = float(conf_threshold)
        self.convergence_tol = float(convergence_tol)
        self.max_iter = int(max_iter)

    @torch.no_grad()
    def decide(self, window: torch.Tensor, goal: torch.Tensor,
               n_subgoals: int = 4,
               scene_params: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """对自规划链做停止/纠正判据。返回 decision ∈ {accept, stop,
        self_correct, verify} 及各判据诊断。"""
        plan = self.decomposer.decompose(window, goal,
                                         n_subgoals=n_subgoals,
                                         scene_params=scene_params)
        residuals = plan["residuals_to_goal"]

        # 1) 置信门控
        if self.confidence_fn is not None:
            conf = float(self.confidence_fn(window))
        else:
            conf = 1.0
        low_conf = conf < self.conf_threshold

        # 2) 收敛停止: 残差已平台
        if len(residuals) >= 2:
            plateau = abs(residuals[-1] - residuals[-2]) < self.convergence_tol
        else:
            plateau = False

        # 3) 不确定主动验证: OOD?
        ood = False
        if self.ood_detector is not None:
            w = GoalDecomposer._as_window(window)
            flat = w.reshape(1, -1)
            ood = bool(self.ood_detector.is_ood(flat).any().item())

        # 决策优先级: 低置信 > OOD主动验证 > 收敛停止 > 采纳
        if low_conf:
            decision, reason = "self_correct", f"置信{conf:.3f}<{self.conf_threshold}"
        elif ood:
            decision, reason = "verify", "OOD, 触发主动验证"
        elif plateau:
            decision, reason = "stop", "残差已收敛"
        else:
            decision, reason = "accept", "置信达标且未收敛, 继续规划"

        return {
            "decision": decision, "reason": reason,
            "confidence": round(conf, 6), "low_confidence": low_conf,
            "converged": plateau, "ood": ood,
            "final_residual": residuals[-1],
            "residuals": residuals,
            "main_predictor_untouched": True,
            "zero_gradient": True,
        }
