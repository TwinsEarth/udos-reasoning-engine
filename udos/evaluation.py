"""
物理推演评估体系 (v2.1.0 建立, v2.2.0 增强)
=========================================
对训练好的 PhysicsPredictor 在 ParametricDynamicsDataset 上统一计算:
  - 单步 MSE (有/无场景条件, 及场景条件增益 condition_gain_x)
  - 分运动类型 MSE
  - 多步自由滚动 rollout 逐步误差曲线 (长时程误差累积) + 累积率 rollout_growth_x
  - 运动学一致性残差 (一阶欧拉, 分类型)
  - v2.2: 置信度分层校准 confidence_stratification (置信越高应误差越低)
全部纯前向、确定性, 供验证报告与服务 /train、/evaluate 返回。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.evaluation")


from typing import Dict, List, Optional

import torch

from .dynamics import kinematic_residual
from .calibration import calibration_report


@torch.no_grad()
def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred - target) ** 2).mean().item()


@torch.no_grad()
def _confidence_stratification(conf: torch.Tensor, y_pred: torch.Tensor,
                               y_true: torch.Tensor, n_bins: int = 3) -> Dict:
    """
    v2.2.0 置信度分层校准诊断: 按最终 tick 置信度等频分 n_bins 档,
    看各档实际单步 MSE。良好校准应呈现"置信越高、误差越低"。
    """
    err = ((y_pred - y_true) ** 2).mean(dim=-1)     # 每样本误差 [B]
    order = torch.argsort(conf)                     # 置信升序
    bins: List[Dict[str, float]] = []
    edges = torch.linspace(0, conf.numel(), n_bins + 1).long()
    monotonic, prev = True, None
    for i in range(n_bins):
        idx = order[edges[i]:max(edges[i + 1], edges[i] + 1)]
        if idx.numel() == 0:
            continue
        mse_i = err[idx].mean().item()
        bins.append({"bin": i, "n": int(idx.numel()),
                     "mean_confidence": round(conf[idx].mean().item(), 4),
                     "mse": round(mse_i, 6)})
        if prev is not None and mse_i > prev + 1e-9:
            monotonic = False  # 期望随置信升高误差递减
        prev = mse_i
    low = bins[0]["mse"] if bins else float("nan")
    high = bins[-1]["mse"] if bins else float("nan")
    ratio = high / low if low and low == low else float("nan")
    return {"bins": bins, "high_over_low_mse_ratio": round(ratio, 4),
            "monotonic_decreasing": bool(monotonic)}


@torch.no_grad()
def _interval_coverage(model, dataset, scene, H: int) -> Dict:
    """v2.3 split-conformal 区间: 每步真值落入区间的覆盖率与区间宽度 (全维平均)。"""
    iv = model.predict_interval(dataset.X, H, scene_params=scene)
    Y = dataset.Y[:, :H, :]
    inside = ((Y >= iv["lower"] - 1e-9) & (Y <= iv["upper"] + 1e-9))
    cover_by_step = [round(inside[:, h, :].float().mean().item(), 4)
                     for h in range(H)]
    width = iv["upper"] - iv["lower"]
    width_by_step = [round(width[:, h, :].mean().item(), 4) for h in range(H)]
    mono = all(width_by_step[i + 1] + 1e-9 >= width_by_step[i]
               for i in range(len(width_by_step) - 1))
    return {"coverage_by_step": cover_by_step,
            "coverage_overall": round(float(inside.float().mean()), 4),
            "width_by_step": width_by_step,
            "width_non_decreasing": bool(mono)}


@torch.no_grad()
def evaluate_predictor(model, dataset, use_scene: bool = True,
                       max_rollout: Optional[int] = None) -> Dict:
    """
    返回结构化指标 dict。model 为 PhysicsPredictor, dataset 为
    ParametricDynamicsDataset (需含 X/Y/P/kinds/dt)。
    """
    model.eval()
    X, P, Y = dataset.X, dataset.P, dataset.Y
    if X.size(0) == 0:
        raise ValueError("evaluate_predictor 需要非空测试集")
    H = Y.size(1) if max_rollout is None else min(max_rollout, Y.size(1))
    scene = P if (use_scene and model.scene_encoder is not None) \
        else torch.zeros_like(P)

    # 1) 单步 (首步) MSE + 最终 tick 置信度 (一次前向同时取, 供校准分层)
    _, certs, single, _ = model(X, scene_params=scene)
    confidence = certs[:, 1, -1]
    single_mse = _mse(single, Y[:, 0, :])

    # 2) 场景条件消融
    ablation: Dict[str, float] = {}
    if model.scene_encoder is not None:
        cond = _mse(model.predict_next(X, scene_params=P), Y[:, 0, :])
        uncond = _mse(model.predict_next(X, scene_params=torch.zeros_like(P)),
                      Y[:, 0, :])
        ablation = {
            "conditioned_mse": round(cond, 6),
            "unconditioned_mse": round(uncond, 6),
            "condition_gain_x": round(uncond / max(cond, 1e-12), 3),
        }

    # 3) 分类型单步 MSE
    per_kind: Dict[str, float] = {}
    for k in dataset.class_names:
        mk = dataset.kind_mask(k)
        if int(mk.sum()) == 0:
            continue
        pr = model.predict_next(X[mk], scene_params=scene[mk])
        per_kind[k] = round(_mse(pr, Y[mk, 0, :]), 6)

    # 4) 多步自由滚动逐步误差
    roll = model.rollout(X, H, scene_params=scene)
    rollout_curve = [round(_mse(roll[:, h, :], Y[:, h, :]), 6)
                     for h in range(H)]

    # 5) 运动学一致性残差 (一阶欧拉, 用前一帧速度), 总体与分类型
    resid = kinematic_residual(single, X[:, -1, :], dataset.dt)
    kin_overall = round(resid.mean().item(), 6)
    kin_kind: Dict[str, float] = {}
    for k in dataset.class_names:
        mk = dataset.kind_mask(k)
        if int(mk.sum()) == 0:
            continue
        pr = model.predict_next(X[mk], scene_params=scene[mk])
        kin_kind[k] = round(kinematic_residual(
            pr, X[mk, -1, :], dataset.dt).mean().item(), 6)

    growth = (rollout_curve[-1] / max(rollout_curve[0], 1e-12)
              if rollout_curve else float("nan"))
    # v2.4.10: 逐步逐维置信矩阵 (跨测试集平均 -> [H, RAW_DIM]), 可解释性段。
    # 纯前向、确定性, 未挂校准器/区间也可计算 (退化为原始 certainty 广播)。
    psc = model.per_step_confidence(X, H, scene_params=scene)   # [N,H,RAW]
    mean_matrix = psc.mean(dim=0)                              # [H,RAW]
    breakdown = {
        "shape": [int(mean_matrix.size(0)), int(mean_matrix.size(1))],
        "matrix": [[round(float(mean_matrix[h, r]), 4)
                    for r in range(mean_matrix.size(1))]
                   for h in range(mean_matrix.size(0))],
        "mean_by_step": [round(float(psc[:, h, :].mean()), 4)
                         for h in range(H)],
        "min": round(float(psc.min()), 4),
        "max": round(float(psc.max()), 4),
    }
    result = {
        "n_samples": len(dataset),
        "horizon": H,
        "single_step_mse": round(single_mse, 6),
        "ablation": ablation,
        "per_kind_mse": per_kind,
        "rollout_mse_curve": rollout_curve,
        "rollout_growth_x": round(growth, 3),
        "confidence_stratification": _confidence_stratification(
            confidence, single, Y[:, 0, :]),
        "confidence_breakdown": breakdown,
        "kinematic_residual": {"overall": kin_overall, "per_kind": kin_kind},
    }
    # v2.3: 仅当模型已外挂校准器时附加校准段 (独立测试集上的泛化校准诊断)
    if getattr(model, "is_calibrated", False):
        sample_err = ((single - Y[:, 0, :]) ** 2).mean(dim=-1)
        result["calibration"] = calibration_report(
            confidence, sample_err, model.calibrator)
        if getattr(model, "residual_quantiles", None) is not None:
            result["interval"] = _interval_coverage(model, dataset, scene, H)
    return result
