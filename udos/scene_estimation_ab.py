"""
场景条件三方 A/B (v5.4.9)
=========================
在同一 held-out 数据集上, 对训练过场景门的 PhysicsPredictor 比较三种条件:

    blind      场景盲: 不喂任何场景参数 (旧行为)
    explicit   显式:   喂数据生成器的真值隐藏参数 P (可达到的上界)
    estimated  估计:   喂 v5.4.8 经典估计器从窗口反演、不可观测槽位置 0 的参数

报告单步与 H 步 rollout 的整体/分运动类型 MSE, 以及"估计相对显式的增益
恢复率":

    recovery = (MSE_blind - MSE_estimated) / (MSE_blind - MSE_explicit)

recovery≈1 表示仅凭观测窗口就恢复了显式场景参数的全部增益; ≈0 表示与场景盲
无异; 可为负 (估计注入了误导量)。这是诚实的能力度量, 不做粉饰。

纯函数、确定性、不写文件; 脚本 scripts/ab_scene_estimation.py 负责加载
checkpoint/数据并落 JSON。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .dynamics import SCENE_PARAM_NAMES, ParametricDynamicsDataset
from .scene_estimator import estimate_scene_params


def _mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(((pred - true) ** 2).mean())


def _recovery(blind: float, explicit: float, estimated: float,
              tol: float = 1e-12) -> Optional[float]:
    denom = blind - explicit
    if abs(denom) < tol:
        return None                      # 显式无增益, 恢复率无定义
    return (blind - estimated) / denom


def _triplet(blind: float, explicit: float, estimated: float) -> Dict[str, float]:
    out = {
        "blind": round(blind, 6),
        "explicit": round(explicit, 6),
        "estimated": round(estimated, 6),
        "est_over_blind": round(estimated / blind, 4) if blind > 1e-12 else None,
    }
    rec = _recovery(blind, explicit, estimated)
    out["recovery"] = round(rec, 4) if rec is not None else None
    return out


@torch.no_grad()
def three_way_scene_ab(predictor, dataset: ParametricDynamicsDataset,
                       dt: float = 0.5, horizon: int = 4) -> Dict:
    """对 dataset 跑盲/显式/估计三方对比, 返回结构化报告 dict。"""
    X, Y, P = dataset.X, dataset.Y, dataset.P
    H = int(horizon)
    estimate = estimate_scene_params(X, dt)
    P_est = estimate.values_filled(0.0)

    blind1 = predictor.predict_next(X)
    explicit1 = predictor.predict_next(X, scene_params=P)
    estimated1 = predictor.predict_next(X, scene_params=P_est)
    blindH = predictor.rollout(X, H)
    explicitH = predictor.rollout(X, H, scene_params=P)
    estimatedH = predictor.rollout(X, H, scene_params=P_est)

    def block(mask=None):
        m = mask if mask is not None else slice(None)
        return {
            "single_step": _triplet(_mse(blind1[m], Y[m, 0]),
                                    _mse(explicit1[m], Y[m, 0]),
                                    _mse(estimated1[m], Y[m, 0])),
            f"rollout_{H}": _triplet(_mse(blindH[m], Y[m]),
                                     _mse(explicitH[m], Y[m]),
                                     _mse(estimatedH[m], Y[m])),
        }

    per_kind = {k: block(dataset.kind_mask(k)) for k in dataset.class_names}

    # 估计器可观测性统计 (真值类别已知, 用于解释 recovery 缺口)
    obs = estimate.observable.float()
    slot_rates = {name: round(float(obs[:, i].mean()), 4)
                  for i, name in enumerate(SCENE_PARAM_NAMES)}
    spring_mask = dataset.kind_mask("spring")
    omega_idx = SCENE_PARAM_NAMES.index("spring_omega")
    omega_rate = float(obs[spring_mask, omega_idx].mean()) if spring_mask.any() else None

    return {
        "config": {
            "horizon": H, "dt": dt, "n_samples": int(X.size(0)),
            "window": int(X.size(1)),
        },
        "overall": block(),
        "per_kind": per_kind,
        "estimator_observability": {
            "slot_observable_rates": slot_rates,
            "spring_omega_detection_rate": round(omega_rate, 4)
            if omega_rate is not None else None,
        },
    }
