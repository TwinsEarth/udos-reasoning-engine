"""v5.5.4 报告: LoRA 前向闭环自检 (GPM->LoRA->基座前向->复位)。

对四类运动场景分别在 live 档 (scaler_B=1) 与训练档 (scaler_B=0) 下跑
lora_path_self_test, 记录注入差 ||注入后-注入前|| 与复位误差 ||复位后-注入前||。
CPU 诚实档; 基座为演示用 TinyBaseModel, 不接入冻结物理预测员。
用法: PYTHONPATH=. python3 scripts/lora_forward_selftest.py
"""
from __future__ import annotations

import json
import os

import torch

from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.pce_format import PhysicalToken, PhysicsScene
from udos.reasoning import UDOSReasoningEngine
from udos.dynamics import (
    traj_uniform, traj_accel, traj_spring, traj_collision,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 6


def _engine(live: bool) -> UDOSReasoningEngine:
    base = TinyBaseModel(hidden=32, n_layers=2)
    gpm = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    layer_indices=(0, 1), num_pre_head_layers=1, heads=2,
                    init_scaler_b_zero=not live)
    ctm = CTMConfig(iterations=2, d_model=32, d_input=32, heads=2,
                    n_synch_out=8, n_synch_action=8, memory_length=4,
                    nlm_hidden=8, out_dims=32, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    return UDOSReasoningEngine(ctm, gpm, base_model=base)


def _scene(kind: str) -> PhysicsScene:
    if kind == "uniform":
        traj = traj_uniform(W, 0.5, v0=1.3, x0=0.1)
    elif kind == "accel":
        traj = traj_accel(W, 0.5, v0=0.4, a=1.1)
    elif kind == "spring":
        traj = traj_spring(W, 0.5, amp=1.5, omega=1.2, phi=0.3)
    else:
        traj = traj_collision(W, 0.5, x1=-2.0, v1=2.4, x2=1.2, v2=0.1)
    scene = PhysicsScene(scene_id=kind, duration=W)
    for i, (pos, vel) in enumerate(traj):
        scene.add(PhysicalToken(object_id="obj-a", timestamp=i,
                                position=pos, velocity=vel,
                                attributes={"mass": 1.0}))
    return scene


def main() -> None:
    kinds = ["uniform", "accel", "spring", "collision"]
    out = {"version": "5.5.4", "live": {}, "training_scaler_b_zero": {}}
    for live, key in ((True, "live"), (False, "training_scaler_b_zero")):
        eng = _engine(live=live)
        for k in kinds:
            r = eng.lora_path_self_test(_scene(k))
            out[key][k] = {
                "injection_delta": round(r["injection_delta"], 6),
                "reset_error": round(r["reset_error"], 12),
                "patched_modules": r["patched_modules"],
                "lora_params": r["lora_params"],
            }
    path = os.path.join(HERE, "reports", "v554_lora_forward.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
