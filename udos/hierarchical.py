"""
分层 / 多尺度长时域 rollout (v2.7.0.dev4, opt-in)
====================================================
把长时域自由 rollout 按 coarse_factor 切成粗粒度块: 每块独立调一次 predictor.rollout,
块末作为下一粗粒度锚点重新滑窗。
  * horizon <= coarse_factor: 退化为普通 rollout, **逐位一致**;
  * horizon >  coarse_factor: 块间以粗粒度锚点重新滑窗, 返回 {predictions, coarse_points, truncated}。

诚实说明 (证据纪律):
    本引擎的 predictor 是**单自回归头** —— 没有独立的"多步直接预测"粗粒度头。
    因此分块滑窗在数学上 == 连续单步 rollout (A/B 实测逐步 MSE 与平铺逐位一致,
    见 benchmarks/results/hierarchical_ablation_v2.7.0.json)。本模块是为将来接入独立
    粗粒度头预留的 opt-in 脚手架: 一旦 coarse 分支改为非自回归多步预测, 即可在此基础上
    叠加细粒度修正, 而不改变短 horizon 退化契约。当前不伪造"误差下降"。

纯前向、确定性、不修改 predictor。第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.hierarchical")


from typing import Dict, List, Optional

import torch


class HierarchicalRollout:
    """coarse_factor 可配的分块粗粒度 rollout。"""

    def __init__(self, predictor, coarse_factor: int = 4) -> None:
        if coarse_factor < 1:
            raise ValueError("coarse_factor 必须 >= 1")
        self.predictor = predictor
        self.coarse_factor = int(coarse_factor)

    @torch.no_grad()
    def rollout(self, window: torch.Tensor, horizon: int,
                scene_params: Optional[torch.Tensor] = None
                ) -> Dict[str, object]:
        """
        window: [W,RAW] 或 [B,W,RAW]; 返回 {predictions:[B,H,RAW], coarse_points, truncated}。
        单窗口调用方自行取 predictions[0]。horizon <= coarse_factor 时与
        predictor.rollout 逐位一致。
        """
        w = torch.as_tensor(window, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        self.predictor.eval()

        if horizon <= self.coarse_factor:
            pred = self.predictor.rollout(w, horizon, scene_params=scene_params)
            return {"predictions": pred, "coarse_points": [],
                    "truncated": False}

        coarse = self.coarse_factor
        preds: List[torch.Tensor] = []
        coarse_points: List[int] = []
        cur = w
        t = 0
        while t < horizon:
            block = min(coarse, horizon - t)
            blk = self.predictor.rollout(cur, block,
                                         scene_params=scene_params)  # [1,block,R]
            for s in range(block):
                preds.append(blk[:, s])
            t += block
            if t < horizon:
                coarse_points.append(t)
                cur = torch.cat([cur[:, block:, :], blk], dim=1)
        pred = torch.stack(preds, dim=1)   # [B,H,R]
        return {"predictions": pred, "coarse_points": coarse_points,
                "truncated": horizon % coarse != 0}
