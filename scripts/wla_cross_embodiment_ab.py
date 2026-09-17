"""
3.9.1 A/B: 统一动作空间跨本体/跨末端迁移 (复用 retargeting)
================================================================
类比 WLA "统一动作空间 -> 跨本体先验迁移 (不同本体/末端不单独训策略)"。
UDOS 用 MorphologyConfig/ActionRetargeter 把**同一统一动作策略**重定向到不同合成
本体 (prime_u_60dof 全身 / arm_7dof 臂 / gripper_4dof 夹爪), 验证:
    * 零样本迁移后动作严格落在目标关节限位内;
    * 迁移前后动作代理能量近似守恒 (梯形积分);
    * 关节限幅违例率诚实报告。
落 benchmarks/results/wla_cross_embodiment_ab.json。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.retargeting import (MorphologyConfig, ActionRetargeter,  # noqa: E402
                              MorphologyLibrary)
from udos.embodied import ActionTriGroup  # noqa: E402

torch.set_num_threads(2)


def main():
    lib = MorphologyLibrary()
    # 统一动作策略: 以 arm_7dof 为源本体的合成动作轨迹
    src = lib.get("arm_7dof")
    g = torch.Generator().manual_seed(0)
    traj = torch.randn(12, src.dof, generator=g) * 0.8   # [T, dof]

    results = {"source": "arm_7dof", "traj_shape": list(traj.shape),
               "transfers": {}}
    for tgt_name in ("gripper_4dof", "prime_u_60dof"):
        tgt = lib.get(tgt_name)
        rt = ActionRetargeter(src, tgt)
        out = rt.zero_shot_transfer(traj)
        energy_src = rt.action_energy(traj, dt=0.02)
        # 重采样到目标频率后再算能量, 近似守恒
        rs = rt.resample(traj, src_freq=src.control_freq,
                         dst_freq=tgt.control_freq)
        energy_tgt = rt.action_energy(rs, dt=1.0 / tgt.control_freq)
        results["transfers"][tgt_name] = {
            "src_dof": src.dof, "tgt_dof": tgt.dof,
            "out_shape": list(out["actions"].shape),
            "within_limits": bool(out["within_limits"]),
            "violation_rate_before_clamp": round(float(out["violation_rate"]), 4),
            "energy_src": round(energy_src, 4),
            "energy_resampled_tgt": round(energy_tgt, 4),
            "zero_shot_adapt": True,
        }

    # 三分组 + 跨本体: 统一动作先切三分组, 再分别重定向 (EEF pose/joints/下肢)
    tg = ActionTriGroup(eef_pose_dim=3, eef_joints_dim=2, lower_body_dim=2)
    unif = torch.randn(1, 7, generator=g)
    groups = tg.split(unif)
    results["tri_group_cross"] = {
        "groups": {k: list(v.shape) for k, v in groups.items()},
        "merge_roundtrip": bool(torch.allclose(tg.merge(groups), unif)),
    }

    results["analogy_not_reproduction"] = True
    results["no_retrained_policy"] = True

    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/wla_cross_embodiment_ab.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
