"""v7 训练闭环：单步 + 自回归多步损失，scene dropout 同时教会 oracle/盲两条路径。

scene dropout：每个 batch 以概率 p 把部分样本的显式场景参数置为 NaN（缺槽哨兵），
强制估计器 + 零初始化场景桥学会从窗口恢复**可观测**参数；碰撞 v2、弹簧 ω 等
不可观测量在盲路径下的残余误差被如实保留（M2 逐参数报告，不掩盖）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .dynamics import TrajectoryDataset, active_param_mask
from .metrics import mse
from .model import WorldModelCore


@dataclass
class TrainConfig:
    epochs: int = 60
    batch: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    rollout_weight: float = 0.7
    scene_dropout: float = 0.5     # 训练时走盲路径的样本比例
    ident_weight: float = 0.3      # 有监督隐藏参数辨识（仅有效槽位）
    obs_weight: float = 0.1        # 可观测性头拟合真实估计误差
    obs_scale: float = 0.5
    patience: int = 12
    seed: int = 0


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_oracle: List[float] = field(default_factory=list)
    val_blind: List[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1


def _apply_scene_dropout(P: torch.Tensor, p: torch.Tensor,
                         gen: torch.Generator) -> torch.Tensor:
    """以概率 p 把整行显式参数置 NaN（=> SceneChannel 逐槽回退估计）。"""
    B = P.size(0)
    keep = torch.rand(B, generator=gen) >= p
    eff = P.clone()
    eff[~keep] = float("nan")
    return eff


def fit(model: WorldModelCore, train: TrajectoryDataset,
        val: TrajectoryDataset, cfg: Optional[TrainConfig] = None,
        log_every: int = 10, verbose: bool = False) -> TrainHistory:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    hist = TrainHistory()
    best_state = None
    stale = 0
    H = train.horizon

    def batches():
        idx = torch.randperm(len(train), generator=gen)
        for s in range(0, len(train), cfg.batch):
            b = idx[s:s + cfg.batch]
            kinds = [train.kinds[i] for i in b.tolist()]
            yield (train.X[b], train.P[b], train.Y[b],
                   active_param_mask(kinds))

    for ep in range(cfg.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for X, P, Y, amask in batches():
            Peff = _apply_scene_dropout(P, cfg.scene_dropout, gen)
            nxt, sc = model(X, explicit=Peff, return_scene=True)
            loss1 = nn.functional.mse_loss(nxt, Y[:, 0])
            traj = model.rollout_grad(X, H, explicit=Peff)
            lossH = nn.functional.mse_loss(traj, Y)
            # 有监督参数辨识：仅在该类型有效槽位上把 p_hat 拉向真值
            perr = (sc.params_hat - P).pow(2) * amask
            denom = amask.sum().clamp_min(1.0)
            loss_id = perr.sum() / denom
            # 可观测性头学习真实（脱梯度）估计误差：误差越小越接近 1
            with torch.no_grad():
                obs_target = torch.exp(
                    (sc.params_hat - P).abs().neg() / cfg.obs_scale)
            loss_obs = ((sc.observability - obs_target).pow(2)
                        * amask).sum() / denom
            loss = (loss1 + cfg.rollout_weight * lossH
                    + cfg.ident_weight * loss_id
                    + cfg.obs_weight * loss_obs)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        sched.step()
        hist.train_loss.append(ep_loss / nb)

        # 验证：固定 oracle 与盲两条路径
        model.eval()
        with torch.no_grad():
            vo = mse(model.rollout(val.X, H, explicit=val.P), val.Y)
            vb = mse(model.rollout(val.X, H, explicit=None), val.Y)
        hist.val_oracle.append(vo)
        hist.val_blind.append(vb)
        # 以 oracle+盲 均值作为档位选择准则（两条路径都要好）
        crit = 0.5 * (vo + vb)
        if crit < hist.best_val - 1e-6:
            hist.best_val, hist.best_epoch = crit, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if verbose and (ep % log_every == 0 or ep == cfg.epochs - 1):
            print(f"ep{ep:03d} train={hist.train_loss[-1]:.4f} "
                  f"val_oracle={vo:.4f} val_blind={vb:.4f}")
        if stale >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
