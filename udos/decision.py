"""
不确定性向下游决策传播 (v2.6.0+dev4)
====================================
把 predictor 已有的三类不确定性信号 —— split-conformal 区间宽度、OOD 马氏距离、
校准后置信度 —— 合成为一个 [0,1] 的风险分与三档风险等级, 并据此对候选下一动作
做安全边界过滤。

设计纪律 (与全工程一致):
    * 纯前向、确定性、**不修改 predictor** (只读外挂状态)。
    * 新能力 opt-in: 未挂半宽 / 未挂 OOD / 未挂校准时**诚实退化** —— 对应分量按
      保守中性值处理 (区间宽用 certainty 倒数代理; OOD 记 0; 置信用原始 certainty),
      绝不静默崩溃, 也不伪造区间信息。
    * 不改变任何旧输出; 本模块只是在已有推理之上做后处理聚合。

RiskGrader.grade:
    risk_score = 0.4*width_norm + 0.3*ood_norm + 0.3*(1 - confidence)  ∈ [0,1]
        interval 越宽 + OOD 越远 + 置信越低 => 风险越高
    risk_level: low (<0.33) / medium (0.33-0.66) / high (>0.66)

safety_boundary:
    有半宽时, 对每个候选下一状态, 检验其位置维是否落在预测区间下界之上
    (position dim >= interval_lower)。无半宽时全部 safe=True (无信息不过滤)。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.decision")


from typing import Any, Dict, List, Optional

import torch

from .dynamics import RAW_DIM

_POS_DIM = 3   # [pos(3), vel(3)] 布局: 安全边界只约束位置维


def _as_batch(window: torch.Tensor, scene_params: Optional[torch.Tensor]):
    """[W,R] -> [1,W,R]; scene_params [P] -> [1,P]。返回 (window[3D], sp[2D|None])。"""
    w = torch.as_tensor(window, dtype=torch.float32)
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.dim() != 3:
        raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
    sp = scene_params
    if sp is not None:
        sp = torch.as_tensor(sp, dtype=torch.float32)
        if sp.dim() == 1:
            sp = sp.unsqueeze(0)
    return w, sp


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class RiskGrader:
    """把 conformal 区间宽 / OOD 距离 / 校准置信聚合为风险分 (只读, 确定性)。"""

    def __init__(self, width_low: float = 0.5, width_high: float = 2.0,
                 ood_threshold_factor: float = 1.0):
        if width_high <= width_low:
            raise ValueError("width_high 必须 > width_low")
        self.width_low = float(width_low)
        self.width_high = float(width_high)
        self.ood_threshold_factor = float(ood_threshold_factor)

    def _interval_width(self, predictor, horizon: int) -> float:
        """已挂半宽 => 各步逐维平均半宽; 否则用 certainty 倒数代理 (诚实退化)。"""
        q = getattr(predictor, "residual_quantiles", None)
        if q is not None and len(q) > 0:
            widths = []
            for h in range(horizon):
                qh = q[min(h, len(q) - 1)].reshape(-1).float()
                widths.append(float(qh.mean()))
            return sum(widths) / len(widths)
        return float("nan")   # 哨兵: 调用方用 certainty 倒数代理

    @torch.no_grad()
    def grade(self, predictor, raw_window: torch.Tensor,
              scene_params: Optional[torch.Tensor] = None,
              horizon: int = 1) -> Dict[str, Any]:
        """
        返回 {risk_score, risk_level, components}。纯前向、确定性、不改 predictor。
        horizon 仅用于区间宽的步长平均 (grade 本身单步风险, 默认 1)。
        """
        window, sp = _as_batch(raw_window, scene_params)
        predictor.eval()
        # 一次前向同时拿原始 certainty (供置信与无半宽时的宽度代理)
        _, certs, _, _ = predictor(window, scene_params=sp)
        raw_cert = certs[:, 1, -1].float()          # [B]

        # --- 置信分量: 已挂校准器则经 calibrator.transform --- #
        confidence = raw_cert
        if getattr(predictor, "is_calibrated", False) \
                and getattr(predictor, "calibrator", None) is not None:
            confidence = predictor.calibrator.transform(
                raw_cert.reshape(-1).contiguous()).reshape(-1)
        conf_mean = float(confidence.mean().clamp(0.0, 1.0).item())

        # --- 区间宽度分量: 半宽 or certainty 倒数代理 --- #
        width_raw = self._interval_width(predictor, horizon)
        used_proxy = False
        if width_raw != width_raw:                  # NaN => 无半宽
            # certainty 越低, 等效区间越宽; 映射到与半宽同量纲的代理值
            inv = 1.0 / (conf_mean + 1e-6)
            width_raw = inv
            used_proxy = True
        width_norm = _clip01(
            (width_raw - self.width_low) / (self.width_high - self.width_low))

        # --- OOD 分量: 已挂检测器 => 马氏距离 / (阈值*因子); 否则 0 --- #
        # v2.6.1: OOD score 为 NaN 时降级为 0 并记录 ood_nan_degraded, 不污染 risk_score
        ood_score = 0.0
        ood_norm = 0.0
        ood_nan_degraded = False
        if getattr(predictor, "has_ood_detector", False):
            scores = predictor.ood_detector.score(window)
            ood_score = float(scores.mean().item())
            if ood_score != ood_score:      # NaN 自检 (NaN != NaN)
                ood_score = 0.0
                ood_nan_degraded = True
            thr = float(predictor.ood_detector.threshold_)
            denom = max(thr * self.ood_threshold_factor, 1e-12)
            ood_norm = _clip01(ood_score / denom)

        risk_score = _clip01(0.4 * width_norm + 0.3 * ood_norm
                             + 0.3 * (1.0 - conf_mean))
        if risk_score < 0.33:
            level = "low"
        elif risk_score <= 0.66:
            level = "medium"
        else:
            level = "high"
        return {
            "risk_score": round(risk_score, 6),
            "risk_level": level,
            "components": {
                "interval_width": round(width_raw, 6),
                "interval_width_proxy": used_proxy,
                "interval_width_norm": round(width_norm, 6),
                "ood_score": round(ood_score, 6),
                "ood_norm": round(ood_norm, 6),
                "ood_nan_degraded": ood_nan_degraded,
                "confidence": round(conf_mean, 6),
            },
        }


@torch.no_grad()
def safety_boundary(predictor, raw_window: torch.Tensor,
                    action_candidates: List[torch.Tensor],
                    scene_params: Optional[torch.Tensor] = None,
                    safe_threshold: float = 0.0, alpha: float = 0.1
                    ) -> List[Dict[str, Any]]:
    """
    对每个候选下一状态 (长度 RAW_DIM 的张量) 判定是否安全。
    有 conformal 半宽时: 候选位置维 >= 区间下界 (median - half_width) 即不越下界;
    distance_to_boundary = min_{pos维}(cand - lower) (<0 即越界)。
    无半宽时退化为全部 safe=True (无区间信息, 诚实不过滤)。纯前向、确定性。
    """
    window, sp = _as_batch(raw_window, scene_params)
    predictor.eval()

    q = getattr(predictor, "residual_quantiles", None)
    if q is None or len(q) == 0:
        # 诚实退化: 无区间信息 => 不做任何过滤
        return [{"action_index": i, "safe": True,
                 "distance_to_boundary": None,
                 "reason": "no_conformal_half_width_present"}
                for i in range(len(action_candidates))]

    median = predictor.predict_next(window, scene_params=sp)   # [B,R]
    median0 = median[0, :_POS_DIM].float()
    q0 = q[0].reshape(-1).float()[:_POS_DIM]
    lower_pos = median0 - q0                                  # [3]

    out: List[Dict[str, Any]] = []
    for i, cand in enumerate(action_candidates):
        c = torch.as_tensor(cand, dtype=torch.float32).reshape(-1)
        if c.numel() < _POS_DIM:
            raise ValueError(f"候选 {i} 维数 {c.numel()} < 位置维 {_POS_DIM}")
        pos = c[:_POS_DIM]
        dist = float((pos - lower_pos).min().item())
        safe = dist >= safe_threshold
        reason = ("within_interval_lower_bound" if safe
                  else "below_interval_lower_bound")
        out.append({
            "action_index": i,
            "safe": safe,
            "distance_to_boundary": round(dist, 6),
            "reason": reason,
        })
    return out
