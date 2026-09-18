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
    # v7.0.2→v7.0.3 迁移：运动学特征 KIN_DIM 10→11（新增 ca_conf）。
    # 旧 scene.kin_encoder 权重为 [scene_dim,10]，在末列补零升到 [scene_dim,11]，
    # 使新特征 ca_conf 对旧模型贡献为 0（旧预测行为不变）；门控小头缺失则零初始化。
    state = dict(ckpt["model_state"])
    padded = []
    ke = "scene.kin_encoder.weight"
    if ke in state:
        old_w = state[ke]
        new_w = getattr(model, "scene").kin_encoder.weight
        if old_w.shape != new_w.shape and old_w.shape[0] == new_w.shape[0] \
                and old_w.shape[1] < new_w.shape[1]:
            pad = torch.zeros(old_w.shape[0],
                              new_w.shape[1] - old_w.shape[1],
                              dtype=old_w.dtype, device=old_w.device)
            state[ke] = torch.cat([old_w, pad], dim=1)
            padded.append(ke)
    # 非严格加载：新增门控小头在旧权重中缺失，保持零初始化（g≡0，等价旧版）。
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"kin_gate.0.weight", "kin_gate.0.bias",
                       "kin_gate.2.weight", "kin_gate.2.bias"}
    bad = [k for k in missing if k not in allowed_missing]
    if bad or unexpected:
        raise RuntimeError(
            f"checkpoint 与模型结构不一致；bad_missing={bad} "
            f"unexpected={unexpected}")
    ckpt.setdefault("meta", {})["load_missing_zero_init"] = missing
    ckpt["meta"]["load_padded_zero_cols"] = padded
    model.eval()
    return model, ckpt
