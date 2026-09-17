"""v7 指标：整体/分类别/分轴 MSE，oracle 场景 vs 估计场景（盲）对照。"""
from __future__ import annotations

from typing import Dict

import torch

from .contracts import KINDS, VELOCITY_SLICE
from .dynamics import TrajectoryDataset
from .kinematics import kinematic_features

AXES = ("x", "y", "z")


def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float((pred - true).pow(2).mean())


@torch.no_grad()
def evaluate(model, ds: TrajectoryDataset, horizon: int,
             use_explicit: bool, batch: int = 256) -> Dict[str, float]:
    """返回 overall + 分类别 + 分轴（位置/速度）rollout MSE。"""
    model.eval()
    res: Dict[str, float] = {}
    preds, trues = [], []
    for s in range(0, len(ds), batch):
        x = ds.X[s:s + batch]
        exp = ds.P[s:s + batch] if use_explicit else None
        preds.append(model.rollout(x, horizon, explicit=exp))
        trues.append(ds.Y[s:s + batch])
    P = torch.cat(preds, 0)
    T = torch.cat(trues, 0)
    tag = "oracle" if use_explicit else "blind"
    res[f"{tag}_overall"] = mse(P, T)
    for k in KINDS:
        m = ds.kind_mask(k)
        res[f"{tag}_{k}"] = mse(P[m], T[m])
    for i, ax in enumerate(AXES):
        res[f"{tag}_pos_{ax}"] = float((P[..., i] - T[..., i]).pow(2).mean())
        res[f"{tag}_vel_{ax}"] = float((P[..., i + 3] - T[..., i + 3]).pow(2).mean())
    return res


@torch.no_grad()
def estimator_param_error(model, ds: TrajectoryDataset,
                          batch: int = 256) -> Dict[str, Dict[str, float]]:
    """逐隐藏参数评估估计器（盲，从窗口反推）：只在该参数**有效**的类型上计
    MAE 与平均可观测置信；无效槽位（如 uniform 的 ω）不纳入，避免稀释。
    """
    from .contracts import SCENE_PARAM_NAMES
    from .dynamics import active_param_mask
    model.eval()
    amask = active_param_mask(ds.kinds)
    errs = [[] for _ in range(len(SCENE_PARAM_NAMES))]
    obs = [[] for _ in range(len(SCENE_PARAM_NAMES))]
    ptru = [[] for _ in range(len(SCENE_PARAM_NAMES))]
    for s in range(0, len(ds), batch):
        sc = model.scene(ds.X[s:s + batch], None)
        err = (sc.params_hat - ds.P[s:s + batch]).abs()
        m = amask[s:s + batch]
        for j in range(len(SCENE_PARAM_NAMES)):
            sel = m[:, j] > 0
            if int(sel.sum()) > 0:
                errs[j].append(err[sel, j])
                obs[j].append(sc.observability[sel, j])
                ptru[j].append(ds.P[s:s + batch][sel, j])
    out = {}
    for j, name in enumerate(SCENE_PARAM_NAMES):
        if errs[j]:
            mae_est = float(torch.cat(errs[j]).mean())
            true = torch.cat(ptru[j])
            mae_mean = float((true - true.mean()).abs().mean())  # 均值预测器基线
            skill = 1.0 - mae_est / mae_mean if mae_mean > 1e-8 else None
            out[name] = {"mae": round(mae_est, 4),
                         "mae_mean_predictor": round(mae_mean, 4),
                         "identifiability_skill": (
                             round(skill, 3) if skill is not None else None),
                         "mean_observability": round(
                             float(torch.cat(obs[j]).mean()), 3),
                         "n": int(true.numel())}
        else:
            out[name] = {"mae": None, "mae_mean_predictor": None,
                         "identifiability_skill": None,
                         "mean_observability": None, "n": 0}
    return out


def _velocity_slope(vel: torch.Tensor, dt: float) -> torch.Tensor:
    """vel [N,T,3] 对时间最小二乘斜率（恒定加速度向量真值）。"""
    N, T, _ = vel.shape
    t = torch.arange(T, dtype=vel.dtype) * dt
    tc = (t - t.mean()).view(1, T, 1)
    denom = float((tc.view(-1) ** 2).sum())
    return ((tc * (vel - vel.mean(1, keepdim=True))).sum(1) / denom)


@torch.no_grad()
def kinematic_recovery(ds: TrajectoryDataset, batch: int = 256) -> Dict[str, object]:
    """评估**确定性可观测运动学通道**（不依赖训练）：

    - accel：三维恒定加速度向量 a_lin 对真值（X+Y 速度斜率）的向量恢复技能；
      这把 v7.0.1“标量 accel_a 技能为负”重新口径化为“可观测向量是否可辨识”。
    - uniform：|a_lin| 应≈0（零误报加速度）。
    - spring：ω 召回（有效比例）、有效样本 MAE、非弹簧误报率。
    - collision：跳变检出率（仅跨碰撞窗可检）与非碰撞误报率。
    """
    dt = ds.dt
    feats = []
    for s in range(0, len(ds), batch):
        feats.append(kinematic_features(ds.X[s:s + batch], dt))
    K = torch.cat(feats, 0)
    a_lin = K[:, 3:6]
    omega, omega_valid = K[:, 6], K[:, 7]
    jump_valid = K[:, 9]

    out: Dict[str, object] = {}

    # 加速度向量真值：用 X+Y 全部速度帧的斜率（恒定加速度生成器）
    vel_full = torch.cat([ds.X[..., VELOCITY_SLICE],
                          ds.Y[..., VELOCITY_SLICE]], dim=1)
    a_true = _velocity_slope(vel_full, dt)

    m_acc = ds.kind_mask("accel")
    if int(m_acc.sum()) > 0:
        mse_est = float((a_lin[m_acc] - a_true[m_acc]).pow(2).mean())
        mse_mean = float(a_true[m_acc].pow(2).mean())  # 均值预测器≈0 向量
        out["accel_vector"] = {
            "mse_vs_truth": round(mse_est, 6),
            "mse_mean_predictor": round(mse_mean, 6),
            "skill": round(1.0 - mse_est / max(mse_mean, 1e-12), 4),
            "mae_norm": round(float((a_lin[m_acc] - a_true[m_acc])
                                    .norm(dim=1).mean()), 5)}
    m_uni = ds.kind_mask("uniform")
    if int(m_uni.sum()) > 0:
        out["uniform_accel_falsepos_norm"] = round(
            float(a_lin[m_uni].norm(dim=1).mean()), 6)

    m_spr = ds.kind_mask("spring")
    non_spr = ~m_spr
    spr_valid = omega_valid.bool() & m_spr
    out["spring_omega"] = {
        "recall": round(float(spr_valid.float().sum()
                              / max(int(m_spr.sum()), 1)), 4),
        "mae_on_valid": (round(float((omega[spr_valid]
                                      - ds.P[spr_valid, 2]).abs().mean()), 4)
                         if int(spr_valid.sum()) > 0 else None),
        "false_positive_rate_non_spring": round(
            float((omega_valid.bool() & non_spr).float().sum()
                  / max(int(non_spr.sum()), 1)), 4)}

    m_col = ds.kind_mask("collision")
    out["collision_jump"] = {
        "detection_rate": round(
            float((jump_valid.bool() & m_col).float().sum()
                  / max(int(m_col.sum()), 1)), 4),
        "false_positive_rate_non_collision": round(
            float((jump_valid.bool() & ~m_col).float().sum()
                  / max(int((~m_col).sum()), 1)), 4)}
    return out
