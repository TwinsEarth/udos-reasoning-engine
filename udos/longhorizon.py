"""
长时域 Rollout 与记忆协同 (v3.2.0.dev6)
==========================================
analogy, not reproduction —— 受长时域预测思想启发的**轻量化类比实现**, 非复现。

UDOS predictor.rollout 默认递归自由滚动 H 步 (误差随步累积)。`LongHorizonRollout`:

    1. H>4 长时域预测 (H=8/12/16);
    2. 可选结合 TemporalMemory: 每步把预测状态推入记忆, 并把历史旧帧前插到窗口前,
       用更长上下文逐步累积信息, 缓解纯自回归的误差累积;
    3. **H=4 / use_memory=False 时逐位委托 predictor.rollout** (bit-identical 锚点);
    4. 与现有 HierarchicalRollout 对比接口一致。

设计纪律: 纯推理外挂、确定性、不修改主模型权重; 默认路径逐位不变。
第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.longhorizon")


from typing import Optional

import torch

from .dynamics import RAW_DIM
from .temporal_memory import TemporalMemory


class LongHorizonRollout:
    """长时域多步 rollout, 可选 TemporalMemory 累积上下文。

    Parameters
    ----------
    predictor:
        已训练 PhysicsPredictor (只读外挂)。
    memory:
        可选 TemporalMemory (use_memory=True 时使用); 不传则新建空记忆。
    """

    def __init__(self, predictor, memory: Optional[TemporalMemory] = None) -> None:
        self.predictor = predictor
        self.memory = memory

    @staticmethod
    def _as_window(window: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        if w.dim() != 3:
            raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
        if w.size(1) == 0:
            raise ValueError("window 时间维为空")
        return w

    @torch.no_grad()
    def rollout(self, window: torch.Tensor, horizon: int,
                scene_params: Optional[torch.Tensor] = None,
                use_memory: bool = False) -> torch.Tensor:
        """递归自由滚动 horizon 步 -> [B, horizon, RAW_DIM]。

        use_memory=False (默认) 时逐位委托 predictor.rollout (bit-identical 锚点);
        use_memory=True 时每步把预测状态推入记忆并前插历史帧 (单条 B=1 路径)。
        """
        w = self._as_window(window)
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1")
        self.predictor.eval()
        if not use_memory:
            return self.predictor.rollout(w, horizon, scene_params=scene_params)
        # ---- 记忆协同路径 (B=1) ----
        if w.size(0) != 1:
            raise ValueError("use_memory=True 当前仅支持 batch=1")
        # 注意: TemporalMemory 定义了 __len__, 空缓冲时 bool(mem)=False; 必须用 is None
        mem = self.memory if self.memory is not None else TemporalMemory(capacity=64)
        mem.reset()
        cur = w
        outs = []
        for _ in range(horizon):
            ext = mem.build_extended_window(cur[0], k=4).unsqueeze(0)
            nxt = self.predictor.predict_next(ext, scene_params=scene_params)
            outs.append(nxt)
            mem.update(nxt[0])
            cur = torch.cat([cur[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)


__all__ = ["LongHorizonRollout"]
