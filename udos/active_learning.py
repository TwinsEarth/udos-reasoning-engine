"""
主动学习 / 不确定性采样选点 (v2.7.0.dev2, opt-in)
=====================================================
对未标注样本池计算信息增益分并选 top-K:

    info_gain[n] = alpha * ensemble_variance_norm[n]
                 + beta  * interval_width_norm
                 + gamma * ood_score_norm[n]

    * ensemble_variance: predictor 是 DeepEnsemble 时取成员预测间方差 (逐样本);
      否则退化为确定性代理 (1 - 校准后置信), 逐样本输入相关。
    * interval_width_norm: conformal 半宽 (残差分位) 归一化, 全局标量 (同口径常量)。
    * ood_score_norm: 马氏距离 / 阈值, 逐样本, clip 到 [0,1]。

设计纪律:
    * 纯前向、确定性、不修改 predictor / 权重;
    * 权重 alpha/beta/gamma 可配置 (默认 0.4/0.3/0.3);
    * 未挂 OOD 检测器 / 未挂半宽时对应分量诚实置 0, 不伪造。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.active_learning")


from typing import Dict, List, Optional, Tuple

import torch

from .ensemble import DeepEnsemble


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _minmax01(v: torch.Tensor) -> torch.Tensor:
    """跨池 min-max 归一化到 [0,1]; 常量向量退化为全 0。"""
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-12:
        return torch.zeros_like(v)
    return (v - lo) / (hi - lo)


class UncertaintySampler:
    """对未标注样本池排序, 选最有信息量的 top-K。"""

    def __init__(self, alpha: float = 0.4, beta: float = 0.3,
                 gamma: float = 0.3, width_low: float = 0.5,
                 width_high: float = 2.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.width_low = float(width_low)
        self.width_high = float(width_high)

    # ---------------- 逐分量 ---------------- #
    def _ensemble_variance(self, predictor, X, P) -> torch.Tensor:
        """[N]。DeepEnsemble => 成员间方差跨维均值; 否则 (1-置信) 代理。"""
        if isinstance(predictor, DeepEnsemble):
            out = predictor.predict_next(X, scene_params=P)
            var = out["variance"].mean(dim=-1)         # [N]
            return _minmax01(var)
        # 确定性代理: 1 - 校准后置信 (逐样本输入相关)
        _, certs, _, _ = predictor(X, scene_params=P)
        conf = certs[:, 1, -1]
        if getattr(predictor, "is_calibrated", False) \
                and getattr(predictor, "calibrator", None) is not None:
            conf = predictor.calibrator.transform(conf.reshape(-1)).reshape(-1)
        return (1.0 - conf.float()).clamp(0.0, 1.0)

    def _interval_width_norm(self, predictor) -> float:
        q = getattr(predictor, "residual_quantiles", None)
        if q is None or len(q) == 0:
            return 0.0
        w = float(torch.cat([t.reshape(-1).float() for t in q]).mean())
        return _clip01((w - self.width_low) / (self.width_high - self.width_low))

    def _ood_norm(self, predictor, X) -> torch.Tensor:
        n = X.size(0)
        if not getattr(predictor, "has_ood_detector", False):
            return torch.zeros(n)
        scores = predictor.ood_score(X)
        thr = float(predictor.ood_detector.threshold_)
        denom = max(thr, 1e-12)
        return (scores / denom).clamp(0.0, 1.0)

    # ---------------- 主接口 ---------------- #
    @torch.no_grad()
    def score_samples(self, predictor, sample_pool: torch.Tensor,
                      scene_params: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """sample_pool: [N,W,RAW] -> 返回 [N] 信息增益分 (越大越值得标注)。"""
        X = torch.as_tensor(sample_pool, dtype=torch.float32)
        P = scene_params
        if P is not None:
            P = torch.as_tensor(P, dtype=torch.float32)
        predictor.eval()
        if X.size(0) == 0:
            return torch.zeros(0)          # 空池守卫
        ens = self._ensemble_variance(predictor, X, P)        # [N]
        wid = self._interval_width_norm(predictor)             # 标量
        ood = self._ood_norm(predictor, X)                     # [N]
        return (self.alpha * ens + self.beta * wid + self.gamma * ood)

    @torch.no_grad()
    def select_top_k(self, predictor, sample_pool: torch.Tensor, k: int,
                     scene_params: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (indices[k], scores[k]) 信息增益最大的 k 个样本索引; k>=N 时返回全部。"""
        scores = self.score_samples(predictor, sample_pool, scene_params)
        k = int(k)
        if k <= 0 or scores.numel() == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)
        k = min(k, scores.numel())
        topv, topi = torch.topk(scores, k, largest=True, sorted=True)
        return topi, topv
