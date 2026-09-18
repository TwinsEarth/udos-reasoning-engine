"""自动化分级门禁（AL0–AL5，框架参照 Epoch AI 的 Automation Levels）。

用途：给系统的每类能力诚实标注“能自主到什么程度”，并在缺少后端（LLM key /
云端 / RSI）时显式抛 GateError，而不是用本地确定性流程冒充 AL4/AL5。

分级（窄任务口径，非通用自治声明）：
- AL0 纯人工；AL1 AI 少量帮助；AL2 AI 辅助完成；AL3 深度人机协作；
- AL4 凭一个高阶指令端到端完成（窄域）；AL5 脱离人类干预的递归自我改进闭环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class AL(IntEnum):
    AL0 = 0
    AL1 = 1
    AL2 = 2
    AL3 = 3
    AL4 = 4
    AL5 = 5


AL_NAMES = {
    AL.AL0: "纯人类操作",
    AL.AL1: "AI 提供少量帮助",
    AL.AL2: "AI 辅助人类完成任务",
    AL.AL3: "深度人机协作",
    AL.AL4: "高阶指令端到端完成（窄域）",
    AL.AL5: "递归自我改进、脱离人类干预的自主闭环",
}


class GateError(RuntimeError):
    """能力所需的后端/授权缺失（如 LLM key、云端、RSI），当前环境不可用。"""


@dataclass
class CapabilityGate:
    """描述某项能力运行所需的外部开关。"""
    llm_keys_configured: bool = False
    cloud_context_configured: bool = False
    arbitrary_code_exec_approved: bool = False
    rsi_loop_present: bool = False

    def missing_for(self, level: AL) -> List[str]:
        missing: List[str] = []
        if level >= AL.AL4:
            if not self.llm_keys_configured:
                missing.append("多供应商 LLM API key（自由任务拆解/写代码）")
            if not self.cloud_context_configured:
                missing.append("云端共享上下文（跨机数千 Agent）")
        if level >= AL.AL5:
            if not self.rsi_loop_present:
                missing.append("可验证的递归自我改进闭环（RSI）")
        return missing

    def require(self, level: AL) -> None:
        missing = self.missing_for(level)
        if missing:
            raise GateError(
                f"该能力声明需要 AL{int(level)}，但当前环境缺少：" + "；".join(missing)
                + "。CPU 沙箱内仅提供 AL0–AL3 的确定性可测实现。")


@dataclass
class CapabilityAssessment:
    feature: str
    al: AL
    evidence: str          # verified / cpu-proto / unverified
    notes: str = ""


def baseline_assessment(gate: Optional[CapabilityGate] = None
                        ) -> List[CapabilityAssessment]:
    """对本仓库实际具备的能力做诚实分级（随功能演进而更新）。"""
    gate = gate or CapabilityGate()
    return [
        CapabilityAssessment("确定性轨迹预测单任务", AL.AL1, "verified",
                             "人发起，模型直接给预测"),
        CapabilityAssessment("多 Agent 预测集成/讨论/投票选优", AL.AL2, "verified",
                             "Agent 辅助，人定义目标与候选"),
        CapabilityAssessment("协调器高阶目标→拆解→并行验证→选优合并（窄域）",
                             AL.AL3, "verified",
                             "目标模板内端到端，边界与异常仍需人审"),
        CapabilityAssessment("受约束代码/配置改进（候选集内自动跑分选优）",
                             AL.AL2, "cpu-proto",
                             "候选策略集封闭，非自由编程"),
        CapabilityAssessment("自由形式云端数千 LLM 子 Agent 写代码/改架构",
                             AL.AL4, "unverified",
                             "需 LLM key + 云上下文，未配置即门禁拦截"),
        CapabilityAssessment("递归自我改进（RSI）自治闭环", AL.AL5, "unverified",
                             "本仓库不提供，亦不声称"),
    ]
