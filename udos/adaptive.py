"""
自适应计算 (v2.6.0+dev3) — 自适应 iterations + horizon
==========================================================
两类早退, 均 opt-in, 默认关闭时与旧路径逐位一致:
    * AdaptiveStopper: 纯逻辑/无状态, 判断 certainty 轨迹是否已收敛
      (最后 patience 个 tick 的 certainty 变化均 < threshold)。
    * adaptive_rollout: 逐步 rollout, 若已挂载 conformal 半宽, 相邻步区间宽度
      增长率 > width_growth_threshold 则提前停止 (区间发散 => 外推不可信)。
      未挂载半宽时退化为完整 max_horizon rollout (与旧版一致)。

predict_next_adaptive 用 wrapper 方式: 跑满全部 CTM iterations 后, 依据 stopper
在已解码的逐 tick 输出里选最终 tick (不真截断 CTM 内部循环), 故 stopper=None
时与旧 predict_next 逐位一致。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.adaptive")


from typing import Any, Dict, List, Optional, Sequence

import torch


class AdaptiveStopper:
    """certainty 收敛判断器 (纯逻辑, 无状态)。"""

    def __init__(self, threshold: float = 1e-3, patience: int = 3):
        self.threshold = float(threshold)
        self.patience = int(patience)

    def should_stop(self, certainty_trajectory: Sequence[float]) -> bool:
        """最后 patience 个 tick 的相邻 certainty 变化绝对值均 < threshold 则停。
        点数不足 patience+1 时返回 False (尚未收敛, 继续)。"""
        traj = [float(x) for x in certainty_trajectory]
        if len(traj) < self.patience + 1:
            return False
        recent = traj[-(self.patience + 1):]
        for k in range(self.patience):
            if abs(recent[k + 1] - recent[k]) >= self.threshold:
                return False
        return True


@torch.no_grad()
def adaptive_rollout(predictor, raw_window: torch.Tensor, max_horizon: int,
                     scene_params: Optional[torch.Tensor] = None,
                     width_growth_threshold: float = 2.0) -> Dict[str, Any]:
    """
    逐步 rollout, 按 conformal 半宽增长率早退。
    未挂半宽 (residual_quantiles) => 完整 max_horizon (与旧 rollout 逐位一致)。
    返回 trajectory[B,H_actual,R] / horizon_used / truncated / reason。
    """
    predictor.eval()
    ctx = predictor._resolve_context(None, scene_params)
    window = raw_window
    outs: List[torch.Tensor] = []
    widths: List[float] = []
    truncated = False
    reason = "completed_full_horizon"
    quantiles = getattr(predictor, "residual_quantiles", None)

    for h in range(max_horizon):
        nxt = predictor(window, scene_context=ctx)[2]
        outs.append(nxt)
        window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        if quantiles is not None and len(quantiles) > 0:
            qh = quantiles[min(h, len(quantiles) - 1)].reshape(-1).float()
            w = float(qh.mean())
            widths.append(w)
            if (len(widths) >= 2 and widths[-2] > 1e-12
                    and widths[-1] > width_growth_threshold * widths[-2]):
                truncated = True
                reason = "interval_width_growth"
                break
    return {
        "trajectory": torch.stack(outs, dim=1),
        "horizon_used": len(outs),
        "truncated": truncated,
        "reason": reason,
    }
