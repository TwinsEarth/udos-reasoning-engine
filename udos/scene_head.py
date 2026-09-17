"""
学习型场景估计头 (v5.5.0)
=========================
"双引擎名副其实"大版的训练产物。主预测员 PhysicsPredictor 冻结 (52191 参数锚点
checkpoint 不动), 仅训练一个小型 MLP 场景头:

    观测窗口 [B, W, 6]  ->  SceneEstimationHead  ->  4 维隐藏场景参数 P_hat
    P_hat 经主预测员**已训练好的** scene_encoder(4->scene_dim) 进入 CTM,
    与显式真值参数走完全相同的条件通道。

训练目标是端到端 rollout MSE (梯度穿过冻结主预测员, 只更新头), 因此头学到的是
"让冻结预测员推得最准"的**任务最优场景条件**, 不保证逐槽位物理参数精确
(例如弹簧 omega 的名义估计可能有偏, 但下游轨迹最优)。可解释的精确反演仍由
v5.4.8 经典估计器 scene_estimator 提供, 二者互补。

独立 nn.Module / 独立 state_dict / 独立权重文件, 可单独挂载、回滚, 不进
PhysicsPredictor.state_dict, 不改变主锚点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .dynamics import RAW_DIM, SCENE_PARAM_DIM


class SceneEstimationHead(nn.Module):
    """观测窗口 -> 4 维隐藏场景参数 (v0, accel_a, spring_omega, other_v2)。"""

    def __init__(self, window: int = 6, raw_dim: int = RAW_DIM,
                 scene_dim: int = SCENE_PARAM_DIM, hidden: int = 64):
        super().__init__()
        self.window = int(window)
        self.raw_dim = int(raw_dim)
        self.scene_param_dim = int(scene_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * raw_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, scene_dim),
        )
        # 末层零初始化: 挂载未训练头时输出全 0, 从"近似场景盲"平稳起步训练。
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, raw_window: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(raw_window, dtype=torch.float32)
        if x.dim() != 3 or x.size(-1) != self.raw_dim:
            raise ValueError(
                f"raw_window 形状须为 [B,W,{self.raw_dim}], 收到 {tuple(x.shape)}")
        if x.size(1) != self.window:
            raise ValueError(
                f"窗口长度须为 {self.window}, 收到 {x.size(1)}")
        return self.net(x)


def differentiable_rollout(predictor, raw_window: torch.Tensor,
                           scene_params: torch.Tensor,
                           horizon: int) -> torch.Tensor:
    """
    与 PhysicsPredictor.rollout 数值相同的自回归滚动, 但**保留 autograd**,
    使损失能穿过冻结主预测员回到场景头。

    冻结参数作为常数参与前向; 只有 scene_params 的来源 (场景头) 收到梯度。
    必须与 training.PhysicsPredictor.rollout 的算子序列保持一致 (同一 ctx
    解析 + 同一 forward[2] + 同一滑窗拼接), 否则训练目标与推理路径不一致。
    """
    ctx = predictor._resolve_context(None, scene_params, None)
    window, outs = raw_window, []
    for _ in range(int(horizon)):
        nxt = predictor(window, scene_context=ctx)[2]
        outs.append(nxt)
        window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
    return torch.stack(outs, dim=1)


@dataclass
class HeadTrainResult:
    head: SceneEstimationHead
    loss_history: List[float]
    epochs: int
    config: Dict[str, Any]


def freeze_predictor(predictor) -> None:
    """冻结主预测员: eval + 全部参数 requires_grad=False (不改权重/checkpoint)。"""
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)


def train_scene_head(predictor, dataset, *, epochs: int = 40,
                     batch_size: int = 128, lr: float = 2e-3,
                     weight_decay: float = 1e-4, horizon: int = 4,
                     seed: int = 42, verbose: bool = False
                     ) -> HeadTrainResult:
    """
    冻结主预测员, 端到端训练场景头最小化 rollout MSE。
    dataset: build_parametric_dataset 产物 (含 .X/.Y/.window 等)。
    """
    freeze_predictor(predictor)
    g = torch.Generator().manual_seed(seed)
    window = int(dataset.X.shape[1])
    head = SceneEstimationHead(window=window, raw_dim=dataset.X.shape[2],
                               scene_dim=dataset.P.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    X, Y = dataset.X, dataset.Y
    n = X.size(0)
    history: List[float] = []
    for ep in range(int(epochs)):
        head.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            p_hat = head(X[idx])
            pred = differentiable_rollout(predictor, X[idx], p_hat, horizon)
            loss = ((pred - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(idx.numel())
        mean_loss = total / n
        history.append(mean_loss)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"[scene-head] epoch {ep:3d} rollout_mse={mean_loss:.5f}")
    head.eval()
    return HeadTrainResult(
        head=head, loss_history=history, epochs=int(epochs),
        config={"epochs": int(epochs), "batch_size": batch_size, "lr": lr,
                "weight_decay": weight_decay, "horizon": horizon, "seed": seed,
                "window": window, "hidden": head.hidden})


def save_scene_head(path: str, head: SceneEstimationHead,
                    meta: Optional[Dict[str, Any]] = None) -> None:
    torch.save({
        "state_dict": head.state_dict(),
        "config": {"window": head.window, "raw_dim": head.raw_dim,
                   "scene_dim": head.scene_param_dim, "hidden": head.hidden},
        "meta": meta or {},
    }, path)


def load_scene_head(path: str, map_location: str = "cpu"
                    ) -> Tuple[SceneEstimationHead, Dict[str, Any]]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = blob["config"]
    head = SceneEstimationHead(
        window=cfg["window"], raw_dim=cfg["raw_dim"],
        scene_dim=cfg.get("scene_dim", SCENE_PARAM_DIM), hidden=cfg["hidden"])
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head, blob.get("meta", {})
