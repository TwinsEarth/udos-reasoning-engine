"""
在线增量适配与漂移触发再校准闭环 (v2.7.0.dev1, opt-in)
=========================================================
在不重训主模型的前提下, 用 `ood.StreamingDriftDetector` 做在线分布漂移监测:
  - observe(window):          把新观测推入流式滑动窗口;
  - check_and_adapt(...):    若窗口整体漂移超过阈值, 触发一次适配:
      * 默认 (enable_finetune=False): 仅用新校准集重跑 PAVA 校准 + conformal 半宽,
        **不修改模型权重** (weights_modified=False);
      * opt-in (enable_finetune=True): 再做少量 epoch 增量微调 (有 epochs 上限, 默认 3)。
  - 无漂移时**完全不触发、不改变任何状态** (校准器/权重均不动)。
  - 每次触发都在 adaptation_log 记录 {timestamp, trigger_reason, drift_score,
    before_ece, after_ece, weights_modified}。

设计纪律 (与全工程一致):
    * 默认只再校准不改权重; 微调为 opt-in 且有 epochs 硬上限, 防止在线灾难性遗忘;
    * 纯统计触发, 无新可学参数; reset() 清空滑动窗口与日志 (保留参考分布)。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.online")


import time
from typing import Any, Dict, List, Optional

import torch

from .ood import StreamingDriftDetector
from .calibration import fit_predictor_calibration
from .training import CTMTrainer, TrainConfig


def _flatten_window(window: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(window, dtype=torch.float32)
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.dim() != 3:
        raise ValueError("window 需为 [W,RAW] 或 [B,W,RAW]")
    return w


class OnlineAdapter:
    """封装流式漂移检测器, 漂移触发时再校准 (默认不改权重)。"""

    def __init__(self, reference_data: torch.Tensor,
                 drift_factor: float = 1.0, drift_window: int = 64,
                 enable_finetune: bool = False, finetune_epochs: int = 3,
                 finetune_lr: float = 1e-3, ridge: float = 1e-3,
                 alpha: float = 0.05) -> None:
        """
        reference_data: [N,W,RAW] 参考 (训练分布) 窗口, 用于拟合 μ/精度矩阵/阈值。
        drift_factor: 触发阈值 = detector.threshold_ * drift_factor (>1 更不敏感)。
        drift_window: 流式滑动窗口长度 (StreamingDriftDetector.window)。
        enable_finetune: True 时触发后做少量增量微调 (opt-in); 默认 False 只再校准。
        finetune_epochs: 微调 epoch 硬上限 (默认 3), 防止在线灾难性遗忘。
        """
        self.detector = StreamingDriftDetector(
            window=drift_window, ridge=ridge, alpha=alpha)
        self.detector.fit(_flatten_window(reference_data))
        self.drift_factor = float(drift_factor)
        self.enable_finetune = bool(enable_finetune)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.adaptation_log: List[Dict[str, Any]] = []

    # ---------------- 在线观测 ---------------- #
    @torch.no_grad()
    def observe(self, window: torch.Tensor) -> None:
        """推入单条 [W,RAW] 或一批 [B,W,RAW] 观测到流式窗口。
        非有限 (NaN/inf) 行直接跳过, 不污染滑动窗口与漂移统计。"""
        w = _flatten_window(window)
        for b in range(w.size(0)):
            row = w[b].reshape(-1)
            if not bool(torch.isfinite(row).all()):
                continue     # NaN/inf 防护: 跳过坏行, 不入窗口
            self.detector.update(row)   # [W,RAW] -> [W*RAW]

    # ---------------- 触发判定 ---------------- #
    def drift_score(self) -> float:
        """当前滑动窗口均值相对参考分布的马氏距离; 空窗口返回 nan。"""
        return self.detector.window_drift_score()

    def is_drifted(self) -> bool:
        """窗口非空且 drift_score > threshold_ * drift_factor。"""
        if self.detector.n_window < 2:
            return False
        s = self.drift_score()
        if s != s:        # nan
            return False
        return s > float(self.detector.threshold_) * self.drift_factor

    # ---------------- ECE 度量 ---------------- #
    @staticmethod
    def _current_ece(predictor, dataset) -> Optional[float]:
        from .evaluation import evaluate_predictor
        if not getattr(predictor, "is_calibrated", False):
            return None
        rep = evaluate_predictor(predictor, dataset)
        cal = rep.get("calibration") or {}
        return (cal.get("calibrated") or {}).get("ece")

    # ---------------- 触发 -> 适配 ---------------- #
    def check_and_adapt(self, predictor, new_calibration_data,
                        ) -> Dict[str, Any]:
        """
        若漂移触发: 用 new_calibration_data (ParametricDynamicsDataset) 重跑 PAVA 校准
        (默认不改权重); opt-in enable_finetune 时再少量增量微调。返回适配结果 dict。
        无漂移时完全不改变 predictor 与日志。
        """
        if not self.is_drifted():
            return {"adapted": False, "reason": "no_drift",
                    "drift_score": self.drift_score()}

        drift = self.drift_score()
        before_ece = self._current_ece(predictor, new_calibration_data)
        logger.warning(
            "online drift triggered recalibration drift_score=%.4f threshold=%.4f",
            float(drift), float(self.detector.threshold_))

        # 默认: 仅重跑 PAVA 校准 + conformal 半宽 (不改权重)
        calibrator, _report, rq = fit_predictor_calibration(
            predictor, new_calibration_data)
        predictor.attach_calibration(calibrator, rq)
        weights_modified = False

        # opt-in: 少量增量微调 (epochs 硬上限)
        if self.enable_finetune:
            tcfg = TrainConfig(epochs=self.finetune_epochs,
                               lr=self.finetune_lr, batch_size=64,
                               patience=None, step_weight_scheme="front",
                               hybrid_weight=0.0)
            CTMTrainer(predictor, tcfg).train(new_calibration_data)
            weights_modified = True

        after_ece = self._current_ece(predictor, new_calibration_data)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_reason": "streaming_window_drift",
            "drift_score": round(float(drift), 6),
            "threshold": round(float(self.detector.threshold_), 6),
            "before_ece": before_ece,
            "after_ece": after_ece,
            "weights_modified": weights_modified,
            "finetune_epochs": self.finetune_epochs if weights_modified else 0,
        }
        self.adaptation_log.append(entry)
        logger.info(
            "online adapt done adapted=True weights_modified=%s "
            "before_ece=%s after_ece=%s",
            weights_modified, before_ece, after_ece)
        return {"adapted": True, "reason": "streaming_window_drift",
                "drift_score": round(float(drift), 6),
                "weights_modified": weights_modified, "log": entry}

    # ---------------- 重置 ---------------- #
    def reset(self) -> None:
        """清空流式滑动窗口与适配日志 (保留参考分布/阈值)。"""
        self.detector.reset()
        self.adaptation_log = []
