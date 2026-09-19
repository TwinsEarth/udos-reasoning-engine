"""UDOS 推演引擎 v7 —— 统一世界模型内核（重写版）。

单一推理图：观测 → 统一场景通道（显式参数 + 估计器 + 零初始化场景桥在同一
ctx 求和）→ 唯一残差循环世界模型 → 解码 → 按 α 校准的不确定性。

证据分级见 contracts.EvidenceGrade；旧 udos 包为冻结遗留（见 docs7）。
"""
from __future__ import annotations

__version__ = "7.5.0"

from .contracts import (DT, HORIZON, KINDS, SCENE_DIM, SCENE_PARAM_NAMES,
                        STATE_DIM, WINDOW, EvidenceGrade, StateContract)
from .model import WorldModelCore

__all__ = ["__version__", "WorldModelCore", "StateContract",
           "EvidenceGrade", "STATE_DIM", "SCENE_DIM", "SCENE_PARAM_NAMES",
           "KINDS", "WINDOW", "HORIZON", "DT"]
