"""v7 checkpoint 存取：模型权重 + 配置 + 证据元数据（单一事实源）。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .model import WorldModelCore


def save_worldmodel(model: WorldModelCore, path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 从对象反推配置，保证可重建
    cfg = {"window": model.window, "hidden": model.hidden,
           "n_layers": model.gru.num_layers, "scene_dim": model.scene.scene_dim,
           "use_kinematics": bool(getattr(model, "use_kinematics", False))}
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "meta": meta or {}}, path)
    return path


def load_worldmodel(path, map_location="cpu") -> Tuple[WorldModelCore, dict]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    model = WorldModelCore(window=cfg["window"], hidden=cfg["hidden"],
                           scene_dim=cfg.get("scene_dim", 32),
                           n_layers=cfg.get("n_layers", 2),
                           # v7.0.1 checkpoint 无运动学通道 => 默认关闭以兼容旧权重
                           use_kinematics=cfg.get("use_kinematics", False))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
