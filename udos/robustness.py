"""
综合鲁棒性评估器 (v3.3.0.dev6)
====================================================================
analogy, not reproduction: 在合成参数化数据上综合评估四类鲁棒性, 不声称复现
真实对抗鲁棒性 / 真机扰动。只读评估, 不训练、不改模型权重。

四维:
    1. 噪声鲁棒性: train 干净 / test 注入噪声 sigma 网格下的 MSE 退化;
    2. OOD 检测: 已挂检测器时的命中率 (OOD 被判为 OOD) / 误报率 (ID 被误判);
    3. 外推误差: 输入按 scale 放大后模型的预测 MSE (分布外幅度外推);
    4. 对抗扰动 FGSM 代理: 沿损失对输入梯度符号做一步小扰动后的 MSE 上升。
综合鲁棒性分数 ∈ [0,100] (越高越鲁棒)。与现有 ood / noise_augment 模块集成。
第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.robustness")


import math
from typing import Dict, Optional, Sequence

import torch

from .dynamics import noise_augment


class RobustnessEvaluator:
    """只读综合鲁棒性评估器。"""

    def __init__(self, model,
                 noise_sigmas: Sequence[float] = (0.0, 0.05, 0.15),
                 fgsm_eps: float = 0.05,
                 extrapolation_scale: float = 2.0,
                 seed: int = 0) -> None:
        self.model = model
        self.noise_sigmas = tuple(noise_sigmas)
        self.fgsm_eps = float(fgsm_eps)
        self.extrapolation_scale = float(extrapolation_scale)
        self.seed = seed

    @torch.no_grad()
    def _mse(self, x, p, y) -> float:
        pred = self.model.predict_next(x, scene_params=p)
        return float(((pred - y) ** 2).mean())

    def noise_robustness(self, X, P, Y) -> Dict[str, float]:
        """噪声 sigma 网格下的 MSE; 返回 dict + 干净基准。"""
        clean = self._mse(X, P, Y)
        rows = {}
        for s in self.noise_sigmas:
            torch.manual_seed(self.seed)
            Xn = noise_augment(X, float(s))
            rows[f"test_sigma={s}"] = round(self._mse(Xn, P, Y), 6)
        return {"clean_mse": round(clean, 6), "grid": rows}

    def ood_detection(self, X, X_ood) -> Dict[str, Optional[float]]:
        """OOD 命中率 / ID 误报率 (依赖已挂 OOD 检测器; 未挂则返回 None)。"""
        if not getattr(self.model, "has_ood_detector", False):
            return {"hit_rate": None, "false_alarm_rate": None,
                    "note": "未挂载 OOD 检测器"}
        id_scores = self.model.ood_score(X)
        ood_scores = self.model.ood_score(X_ood)
        thr = float(self.model.ood_detector.threshold_)
        return {
            "hit_rate": round(float((ood_scores > thr).float().mean()), 4),
            "false_alarm_rate": round(float((id_scores > thr).float().mean()), 4),
            "threshold": round(thr, 4),
        }

    def extrapolation(self, X, P, Y) -> Dict[str, float]:
        """输入按 scale 幅度放大后的外推 MSE (相对干净退化倍数)。"""
        clean = self._mse(X, P, Y)
        Xe = X * self.extrapolation_scale
        ext_mse = self._mse(Xe, P, Y * self.extrapolation_scale)
        return {
            "scale": self.extrapolation_scale,
            "extrap_mse": round(ext_mse, 6),
            "degradation_x": round(ext_mse / max(clean, 1e-9), 3),
        }

    def fgsm_proxy(self, X, P, Y, n: int = 32) -> Dict[str, float]:
        """一步 FGSM 代理: x_adv = x + eps * sign(grad loss wrt x)。"""
        self.model.eval()
        xb = X[:n].clone().detach().requires_grad_(True)
        pb = P[:n]
        yb = Y[:n]                       # Y 已是单步目标 [n, R]
        out = self.model(xb, scene_params=pb)[2]
        loss = ((out - yb) ** 2).sum()
        grad = torch.autograd.grad(loss, xb)[0]
        x_adv = (xb + self.fgsm_eps * grad.sign()).detach()
        clean = float(((out.detach() - yb) ** 2).mean())
        adv = self._mse(x_adv, pb, yb)
        return {
            "eps": self.fgsm_eps,
            "clean_mse": round(clean, 6),
            "adv_mse": round(adv, 6),
            "degradation_x": round(adv / max(clean, 1e-9), 3),
        }

    def evaluate(self, ds) -> Dict[str, object]:
        X, P, Y = ds.X, ds.P, ds.Y
        Y1 = Y[:, 0, :]
        noise = self.noise_robustness(X, P, Y1)
        ood = self.ood_detection(X, X * 5.0)
        ext = self.extrapolation(X, P, Y1)
        fgsm = self.fgsm_proxy(X, P, Y1)

        # ---- 综合分数 0-100 (越高越鲁棒) ----
        clean = noise["clean_mse"]
        noisy_vals = [v for k, v in noise["grid"].items() if k != "clean_mse"]
        noise_deg = (sum(noisy_vals) / max(len(noisy_vals), 1)) / max(clean, 1e-9)
        noise_score = 100.0 * math.exp(-max(noise_deg - 1.0, 0.0))
        ext_score = 100.0 * math.exp(-max(ext["degradation_x"] - 1.0, 0.0))
        fgsm_score = 100.0 * math.exp(-max(fgsm["degradation_x"] - 1.0, 0.0))
        if ood.get("hit_rate") is not None:
            ood_score = 100.0 * max(0.0, ood["hit_rate"] - ood["false_alarm_rate"])
        else:
            ood_score = 50.0   # 未挂检测器, 中性占位
        score = 0.3 * noise_score + 0.3 * ood_score + 0.2 * ext_score + 0.2 * fgsm_score
        score = float(max(0.0, min(100.0, score)))

        return {
            "noise": noise, "ood": ood,
            "extrapolation": ext, "fgsm_proxy": fgsm,
            "sub_scores": {
                "noise": round(noise_score, 2),
                "ood": round(ood_score, 2),
                "extrapolation": round(ext_score, 2),
                "fgsm": round(fgsm_score, 2),
            },
            "robustness_score": round(score, 2),
            "analogy_not_reproduction": True,
        }
