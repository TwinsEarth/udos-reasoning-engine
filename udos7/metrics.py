"""v7 指标：整体/分类别/分轴 MSE，oracle 场景 vs 估计场景（盲）对照。"""
from __future__ import annotations

from typing import Dict

import torch

from .contracts import KINDS
from .dynamics import TrajectoryDataset

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
