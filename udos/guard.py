"""
退化守卫 / 安全护栏 (v2.4.5, opt-in)
====================================
在线推理时预测可能出现 NaN/inf（数值发散）或物理量越界（位置/速度跑出合理区间）。
PredictionGuard 对单步预测做最后一道防线:
  1. NaN/inf -> 回退到上一有效预测; 尚无有效预测时回退零向量;
  2. 有限但越界的元素 -> 截断到 [low, high];
  3. 记录修复/截断次数, 供服务层观测退化。

设计: 无状态负担外的最少状态 (仅 last_valid 与计数); 纯前向、确定性。
PhysicsPredictor.predict_next(guard=True) 调用本守卫, 默认 guard=False 透传旧输出。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.guard")


from typing import Any, Dict, Optional

import torch


class PredictionGuard:
    """NaN/inf 检测 + 物理越界截断 + 回退。"""

    def __init__(self, low: Optional[torch.Tensor] = None,
                 high: Optional[torch.Tensor] = None) -> None:
        # low/high: 与最后一维 (RAW) 对齐的张量或标量; None 表示不检查该侧
        self.low = low
        self.high = high
        self.n_fallbacks: int = 0     # NaN/inf 触发回退次数
        self.n_clips: int = 0         # 越界截断次数
        self._last_valid: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._last_valid = None
        self.n_fallbacks = 0
        self.n_clips = 0

    def _as_tensor(self, bound, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if bound is None:
            return None
        t = torch.as_tensor(bound, dtype=torch.float32)
        return t

    @torch.no_grad()
    def sanitize(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred: [..., R]。返回清洗后的同形张量 (不原地修改输入)。
        NaN/inf 用 last_valid (无则零向量) 替换; 越界元素截断到 [low, high]。
        """
        out = pred.detach().to(torch.float32).clone()
        R = out.shape[-1]

        # 1) 越界截断 (对有限值)
        low = self._as_tensor(self.low, out)
        high = self._as_tensor(self.high, out)
        finite = torch.isfinite(out)
        if low is not None:
            clipped_low = (out < low) & finite
            self.n_clips += int(clipped_low.sum())
            out = torch.where(clipped_low, low, out)
        if high is not None:
            clipped_high = (out > high) & finite
            self.n_clips += int(clipped_high.sum())
            out = torch.where(clipped_high, high, out)

        # 2) NaN/inf 所在整行回退到 last_valid / 零向量 ("回退到上一有效步")
        bad_row = ~torch.isfinite(out).all(dim=-1)     # 任一维非有限 => 整行坏
        if bool(bad_row.any()):
            fallback = (self._last_valid
                        if self._last_valid is not None
                        else torch.zeros(R, dtype=torch.float32))
            self.n_fallbacks += int(bad_row.sum())
            out = torch.where(bad_row.unsqueeze(-1), fallback, out)
            logger.warning(
                "guard fallback triggered n_fallbacks=%d rows=%d",
                self.n_fallbacks, int(bad_row.sum()))

        # 3) 记录最后一条有效预测 (取批内最后一行, 广播用)
        if out.numel() >= R:
            self._last_valid = out.reshape(-1, R)[-1].detach().clone()
        return out

    def stats(self) -> Dict[str, Any]:
        return {"n_fallbacks": self.n_fallbacks, "n_clips": self.n_clips}
