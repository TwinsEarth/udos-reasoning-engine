"""v3.8.1 多体协同 A/B: 有协调 vs 无协调的碰撞/到达/延迟对比。

扫描智能体数 N in {2,4,8}; 每个 agent 朝各自目标匀速行进 T 步。
    * 无协调 (baseline): 直接按期望速度积分;
    * 有协调: AgentCoordinator 优先级让行/速度调节后再积分。
指标: 全程侵入(碰撞)次数、到达目标率、单次协调开销(合成可测)。
落 benchmarks/results/multi_agent_ab_v3.8.0.json。纯确定性、零梯度。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.multi_agent import AgentCoordinator, MultiAgentScene  # noqa: E402

torch.set_num_threads(2)

T_STEPS = 20
ARRIVE_THRESH = 0.4
RADIUS = 0.3
SPEED = 1.0


def _desired_vel(ag):
    d = ag.goal - ag.pos
    n = d.norm().item()
    if n < 1e-9:
        return torch.zeros(3)
    return d / n * min(SPEED, n)


def run_one(n_agents, use_coordinator, seed):
    g = torch.Generator().manual_seed(seed)
    import math
    sc = MultiAgentScene()
    for k in range(n_agents):
        ang = 2 * math.pi * k / n_agents
        pos = torch.tensor([math.cos(ang), math.sin(ang), 0.0]) * 5.0
        goal = -pos.clone()                       # 朝圆心对穿 -> 必然接近
        sc.add_agent(f"ag{k}", state=torch.cat([pos, torch.zeros(3)]),
                     goal=goal, priority=(k % 3) + 1, radius=RADIUS,
                     max_speed=SPEED)
    coord = AgentCoordinator()
    collisions = 0.0
    t0 = time.perf_counter()
    for _ in range(T_STEPS):
        # 期望速度 = 朝目标
        cmds = {i: [round(float(x), 6) for x in _desired_vel(sc.get(i)).tolist()]
                for i in sc.ids}
        if use_coordinator:
            rep = coord.resolve(sc)
            cmds = rep["commands"]
        sc = coord.apply_commands(sc, cmds, dt=1.0)
        # 统计侵入
        for ii in range(n_agents):
            for jj in range(ii + 1, n_agents):
                a, b = sc.ids[ii], sc.ids[jj]
                dist = float((sc.get(a).pos - sc.get(b).pos).norm())
                if dist < 2 * RADIUS:
                    collisions += 1
    per_call_ms = (time.perf_counter() - t0) / T_STEPS * 1000.0
    arrived = sum(1 for i in sc.ids
                  if float((sc.get(i).pos - sc.get(i).goal).norm()) < ARRIVE_THRESH)
    return {
        "collision_events": collisions,
        "arrival_rate": round(arrived / n_agents, 4),
        "per_call_ms": round(per_call_ms, 4),
    }


def main():
    results = {}
    for n in (2, 4, 8):
        base = run_one(n, use_coordinator=False, seed=100 + n)
        coord = run_one(n, use_coordinator=True, seed=100 + n)
        results[f"n{n}"] = {
            "no_coordinator": base,
            "with_coordinator": coord,
            "collision_reduction_pct": round(
                (1.0 - coord["collision_events"] / max(base["collision_events"], 1e-9))
                * 100.0, 2),
        }
    summary = {
        "feature": "multi_agent_conflict_resolution_ab",
        "sweep_n_agents": [2, 4, 8],
        "t_steps": T_STEPS,
        "routing": "priority yield + proportional slowdown (non-learning)",
        "zero_gradient": True,
        "learned": False,
        "analogy_not_reproduction": True,
        "results": results,
        "note": "协调器降低对穿场景碰撞事件数; 到达率/延迟为合成可测; "
                "收益不稳处保持 opt-in",
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/multi_agent_ab_v3.8.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
