"""
时间记忆机制 (v3.2.0.dev3)
==========================================
analogy, not reproduction —— 受滑动窗口记忆 + 历史摘要思想启发的**轻量化类比实现**。

UDOS predictor 默认只看最近 window(=6) 帧。`TemporalMemory` 在推理外挂层维护:

    1. 滑动窗口记忆: 容量为 C 的环形缓冲, 保存超出当前窗口的历史状态帧;
    2. 历史摘要向量: 对缓冲状态做指数滑动平均 (EMA), 保留长时域统计;
    3. 更新 / 重置 / 查询接口;
    4. 与 predictor 集成: 把缓冲的旧帧前插到当前窗口前, 形成更长上下文
       (委托 ExtendedContextWindow, 不改权重)。

设计纪律: 纯推理外挂、确定性、**不修改模型权重**; 默认不挂时旧路径逐位不变。
第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.temporal_memory")


from typing import List, Optional

import torch

from .dynamics import RAW_DIM


class TemporalMemory:
    """滑动窗口记忆 + EMA 历史摘要 (单条轨迹的有状态外挂)。

    Parameters
    ----------
    capacity:
        环形缓冲容量 C (>0); 超出时淘汰最旧帧。
    ema_alpha:
        摘要向量的 EMA 平滑系数 (0,1]; 越大越偏向最新状态。
    state_dim:
        状态维度 (=RAW_DIM=6)。
    """

    def __init__(self, capacity: int = 32, ema_alpha: float = 0.5,
                 state_dim: int = RAW_DIM) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须 >= 1")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha 需在 (0,1]")
        self.capacity = int(capacity)
        self.ema_alpha = float(ema_alpha)
        self.state_dim = int(state_dim)
        self._buf: List[torch.Tensor] = []
        self._summary: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # 更新 / 重置
    # ------------------------------------------------------------------ #
    def update(self, state: torch.Tensor) -> None:
        """推入一帧 [state_dim]; 缓冲超容量淘汰最旧; 同步更新 EMA 摘要。"""
        s = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
        if s.numel() != self.state_dim:
            raise ValueError(
                f"状态维度 {s.numel()} != memory state_dim {self.state_dim}")
        if not bool(torch.isfinite(s).all()):
            raise ValueError("状态含 NaN/inf")
        self._buf.append(s.clone())
        if len(self._buf) > self.capacity:
            self._buf.pop(0)
        # EMA 摘要
        if self._summary is None:
            self._summary = s.clone()
        else:
            self._summary = self.ema_alpha * s + (1.0 - self.ema_alpha) * self._summary

    def reset(self) -> None:
        """清空缓冲与摘要。"""
        self._buf = []
        self._summary = None

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._buf)

    @property
    def buffered(self) -> int:
        return len(self._buf)

    def summary(self) -> torch.Tensor:
        """EMA 历史摘要向量 [state_dim] (未 update 过时显式报错, 不伪造)。"""
        if self._summary is None:
            raise RuntimeError("记忆为空, 请先 update()")
        return self._summary.clone()

    def history(self, k: int) -> torch.Tensor:
        """返回最近 k 帧 [k', state_dim] (k' = min(k, buffered)); k<=0 显式报错。"""
        if k <= 0:
            raise ValueError("k 必须为正整数")
        k = min(int(k), len(self._buf))
        if k == 0:
            raise RuntimeError("记忆为空, 无可查询历史")
        return torch.stack(self._buf[-k:], dim=0)

    # ------------------------------------------------------------------ #
    # 与 predictor 集成: 把缓冲旧帧前插到当前窗口
    # ------------------------------------------------------------------ #
    def build_extended_window(self, window: torch.Tensor, k: int = 0
                              ) -> torch.Tensor:
        """把缓冲的最近 k 帧前插到当前窗口 [W, state_dim] 前, 返回 [W+k', state_dim]。

        k=0 或缓冲为空时返回原窗口副本 (不改形状, 与旧路径逐位一致)。
        """
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() != 2:
            raise ValueError("window 需为 [W, state_dim]")
        if k <= 0 or len(self._buf) == 0:
            return w.clone()
        old = self.history(k)              # [k', state_dim]
        return torch.cat([old, w], dim=0).contiguous()


__all__ = ["TemporalMemory"]
