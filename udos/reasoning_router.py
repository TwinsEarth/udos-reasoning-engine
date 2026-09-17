"""
UDOS v4.5.0.dev1 隐式->显式自适应路由 (Reasoning Router)
================================================================
性质: 规则+信号阈值, 零梯度外挂, 不训练, 不入主 state_dict。

用难度信号决定:
    - 隐式思考的预算 (effort: none/low/high/max)
    - 是否/何时把隐式探索投影为可读显式链
    - 给出可解释的切换理由 (审计/可追溯)

难度信号来源 (全部既有模块, 避让不重造):
    - CTM 同步收敛 (engine.reason 的 final certainty)  -> 越收敛越易
    - ensemble 分歧                                      -> 分歧越大越难
    - calibration / OOD 不确定度                        -> 越 OOD 越难
    - 任务类型复杂度 (调用方 hint)
    - 4.3 自进化停机判据                                -> 越未收敛越需深想

设计纪律: 纯前向、确定性、不修改主权重; 简体中文; analogy not reproduction。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("udos.reasoning_router")


@dataclass
class DifficultySignals:
    """归一化难度信号, 各分量 ∈ [0,1]。越接近 1 越难。"""
    ctm_convergence: float            # CTM 终态 certainty (1-归一化熵); 高=易
    ensemble_disagreement: float = 0.0   # 集成成员分歧; 高=难
    ood_score: float = 0.0               # OOD/漂移分; 高=难
    task_complexity: float = 0.0        # 调用方任务复杂度 hint; 高=难
    needs_deeper: float = 0.0           # 4.3 停机判据: 越未收敛越需深想; 高=难

    def clamped(self) -> "DifficultySignals":
        def _c(x): return max(0.0, min(1.0, float(x)))
        return DifficultySignals(
            ctm_convergence=_c(self.ctm_convergence),
            ensemble_disagreement=_c(self.ensemble_disagreement),
            ood_score=_c(self.ood_score),
            task_complexity=_c(self.task_complexity),
            needs_deeper=_c(self.needs_deeper),
        )


@dataclass
class RoutingDecision:
    effort: str
    difficulty: float                 # 合成难度分 ∈ [0,1]
    externalize: bool
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"recommended_effort": self.effort,
                "difficulty": round(self.difficulty, 4),
                "externalize": self.externalize,
                "rationale": self.rationale}


class ReasoningRouter:
    """难度 -> effort 路由 (规则, 零梯度)。

    权重与阈值均为固定工程常数 (CPU 合成类比), 不学习。
    """

    # 合成难度分的权重 (易信号为 1-ctm_convergence)
    W_CTM = 0.40
    W_ENSEMBLE = 0.20
    W_OOD = 0.20
    W_TASK = 0.10
    W_DEEPER = 0.10

    # effort 门槛 (difficulty 升序)
    THR_NONE = 0.30
    THR_LOW = 0.50
    THR_HIGH = 0.72
    # THR_MAX = 1.0  (上界)

    EXTERNALIZE_AT = 0.50   # difficulty >= 此值才显式化

    def synthesize(self, s: DifficultySignals) -> float:
        """把多信号合成一个难度分 ∈ [0,1]。"""
        s = s.clamped()
        unc = 1.0 - s.ctm_convergence   # 收敛低 -> 不确定 -> 难
        difficulty = (self.W_CTM * unc
                      + self.W_ENSEMBLE * s.ensemble_disagreement
                      + self.W_OOD * s.ood_score
                      + self.W_TASK * s.task_complexity
                      + self.W_DEEPER * s.needs_deeper)
        return max(0.0, min(1.0, difficulty))

    def route(self, s: DifficultySignals) -> RoutingDecision:
        difficulty = self.synthesize(s)
        s = s.clamped()

        # 阈值映射
        if difficulty < self.THR_NONE:
            effort = "none"
        elif difficulty < self.THR_LOW:
            effort = "low"
        elif difficulty < self.THR_HIGH:
            effort = "high"
        else:
            effort = "max"

        externalize = difficulty >= self.EXTERNALIZE_AT

        # 可解释理由
        rationale: List[str] = []
        rationale.append(
            f"CTM 收敛 certainty={round(s.ctm_convergence,3)} "
            f"-> 不确定分量={round(1.0 - s.ctm_convergence,3)}")
        if s.ensemble_disagreement > 0.05:
            rationale.append(
                f"集成分歧={round(s.ensemble_disagreement,3)} 偏高 -> 上调难度")
        if s.ood_score > 0.05:
            rationale.append(
                f"OOD/漂移分={round(s.ood_score,3)} 偏高 -> 上调难度")
        if s.task_complexity > 0.05:
            rationale.append(
                f"任务复杂度 hint={round(s.task_complexity,3)} -> 上调难度")
        if s.needs_deeper > 0.05:
            rationale.append(
                f"停机判据 needs_deeper={round(s.needs_deeper,3)} -> 需更深思考")
        rationale.append(
            f"合成难度={round(difficulty,3)} 落在 effort={effort} 区间; "
            + ("升级为显式可读链" if externalize else "隐式直出, 不产可读步"))

        return RoutingDecision(effort=effort, difficulty=difficulty,
                               externalize=externalize, rationale=rationale)


def quick_signals(scene, engine, task_complexity: float = 0.0,
                  needs_deeper: float = 0.0) -> DifficultySignals:
    """用一次廉价 none 前向读 CTM 收敛, 组装难度信号 (不含 ensemble/OOD,
    后者由调用方在其已挂载时补入)。"""
    r = engine.reason(scene)
    return DifficultySignals(
        ctm_convergence=float(r.convergence()),
        task_complexity=task_complexity,
        needs_deeper=needs_deeper,
    )
