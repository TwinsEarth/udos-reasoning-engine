"""v7 统一契约：状态/场景张量形状、证据分级、数据划分种子。

设计原则（见 docs7/ARCHITECTURE.md）：
- 唯一事实口径：状态向量 STATE_DIM=6 = [px,py,pz,vx,vy,vz]。
- 证据分级强制：任何性能数字必须能回链到可复现脚本，否则标 UNVERIFIED。
- 数据划分在**轨迹级**完成（不是窗口级），杜绝同一轨迹的相邻窗口跨集合泄漏。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch

STATE_DIM = 6          # [position(3), velocity(3)]
POSITION_SLICE = slice(0, 3)
VELOCITY_SLICE = slice(3, 6)

# 场景隐藏物理参数槽位（标量幅值；方向是可观测量，不进隐藏参数）：
#   v0 初速度幅值 / accel_a 加速度幅值 / spring_omega 角频率 / other_v2 碰撞第二质点速度
SCENE_PARAM_NAMES: Tuple[str, ...] = ("v0", "accel_a", "spring_omega", "other_v2")
SCENE_DIM = len(SCENE_PARAM_NAMES)

KINDS: Tuple[str, ...] = ("uniform", "accel", "spring", "collision")

# 严格三分独立种子（训练/验证/测试轨迹参数域相同但抽样独立，无窗口泄漏）
TRAIN_SEED = 42
VAL_SEED = 1337
TEST_SEED = 2026
CALIB_SEED = 314       # conformal 校准集（从训练轨迹中独立划出，不参与梯度）

DT = 0.5
WINDOW = 6
HORIZON = 4


class EvidenceGrade(str, Enum):
    """证据分级：每个对外结论必须挂其中之一。"""
    VERIFIED = "verified"        # 本仓可复现脚本 + 固定 seed 实测
    CPU_PROTO = "cpu-proto"      # CPU 原型已跑通，未在 GPU/生产规模验证
    UNVERIFIED = "unverified"    # 无 benchmark，仅机制/文献主张，禁止写进性能结论


@dataclass(frozen=True)
class StateContract:
    """状态张量契约。M1/M2 为单体（n_bodies=1）；多体在后续里程碑扩展。"""
    state_dim: int = STATE_DIM
    n_bodies: int = 1
    dt: float = DT

    def check_window(self, x: torch.Tensor) -> None:
        """x: [B, W, state_dim]。"""
        if x.dim() != 3 or x.size(-1) != self.state_dim:
            raise ValueError(
                f"窗口需为 [B,W,{self.state_dim}]，实际 {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("窗口含 NaN/inf，拒绝进入推理图")
