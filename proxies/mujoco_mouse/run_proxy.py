#!/usr/bin/env python3
"""最小虚拟小鼠因果代理（cpu-proxy，MuJoCo CPU）。

目的不是复刻 DeepMind virtual rodent（无 RL、无神经对齐），而是验证世界模型的
核心主张——一个**因果模型**：在相同初始状态上对动作/身体参数做干预(do)，
物理仿真给出不同的后续状态，而不只是描述观测序列。

实验：
  A 基线步态（前后腿半周期相位差）
  B 动作干预：同初始状态，仅改变步态相位 -> 轨迹应发散
  C 身体干预：躯干质量加倍 -> 运动学应改变（对应论文 mass-doubling）
输出 reports7/mujoco_mouse_proxy.json。
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    print("mujoco 未安装：pip install mujoco（CPU wheel）")
    sys.exit(2)

REPO = Path(__file__).resolve().parents[2]
XML = REPO / "proxies" / "mujoco_mouse" / "minimal_mouse.xml"
OUT = REPO / "reports7" / "mujoco_mouse_proxy.json"


def make_model(mass_scale=1.0):
    model = mujoco.MjModel.from_xml_path(str(XML))
    if mass_scale != 1.0:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        model.body_mass[tid] *= mass_scale
    return model


def state(data):
    return np.concatenate([np.asarray(data.qpos).ravel(),
                           np.asarray(data.qvel).ravel()])


def rollout(mass_scale=1.0, phase_shift=0.0, n_steps=400, seed=0):
    model = make_model(mass_scale)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    np.random.seed(seed)
    # free joint：保留 body 初始位姿（含单位四元数），仅清零速度
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    s0 = state(data).copy()

    dt = model.opt.timestep
    A, f = 0.5, 2.0
    traj = [state(data).copy()]
    ctrls = []
    for k in range(n_steps):
        t = k * dt
        phase = 2 * np.pi * f * t
        cf = A * np.sin(phase + phase_shift)
        cb = A * np.sin(phase + np.pi + phase_shift)
        data.ctrl[:] = [cf, cb]
        ctrls.append([cf, cb])
        mujoco.mj_step(model, data)
        traj.append(state(data).copy())
    return {"s0": s0, "traj": np.asarray(traj),
            "ctrls": np.asarray(ctrls), "dt": dt}


def main():
    base = rollout(phase_shift=0.0)
    act = rollout(phase_shift=np.pi / 2)       # 仅动作干预
    mass = rollout(mass_scale=2.0)             # 仅身体干预

    # 初始状态必须一致（隔离干预）
    same_init = bool(np.allclose(base["s0"], act["s0"], atol=1e-9) and
                     np.allclose(base["s0"], mass["s0"], atol=1e-9))

    def disp(r):
        # planar qpos: [x, z, pitch, hip_f, hip_b]
        return float(r["traj"][-1, 0] - r["traj"][0, 0])

    def divergence(a, b):
        d = np.linalg.norm(a["traj"] - b["traj"], axis=1)
        return {"max": float(d.max()), "final": float(d[-1]),
                "first_nonzero_step": int(np.argmax(d > 1e-6))}

    div_act = divergence(base, act)
    div_mass = divergence(base, mass)
    causal_action = same_init and div_act["max"] > 1e-3
    causal_mass = same_init and div_mass["max"] > 1e-3

    report = {
        "evidence_grade": "cpu-proxy",
        "mujoco_version": mujoco.__version__,
        "note": "最小平面铰接代理，非 DeepMind virtual rodent；无 RL/神经对齐",
        "n_steps": 400, "same_initial_state": same_init,
        "baseline_net_x_displacement": round(disp(base), 5),
        "action_intervention": {
            "phase_shift_rad": 1.5708,
            "net_x_displacement": round(disp(act), 5),
            "state_divergence": {k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in div_act.items()},
            "causally_changes_future": causal_action},
        "mass_intervention": {
            "torso_mass_scale": 2.0,
            "net_x_displacement": round(disp(mass), 5),
            "state_divergence": {k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in div_mass.items()},
            "causally_changes_future": causal_mass},
        "causal_model_demonstrated": bool(causal_action and causal_mass),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", OUT)
    if not report["causal_model_demonstrated"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
