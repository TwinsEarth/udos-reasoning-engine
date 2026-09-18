"""合成第一人称片段：在具身环境上跑混合策略，记录状态/手部轨迹/接触/子任务。

“第一人称视频”在此为**代理量**：末端（手）轨迹 + 接触事件 + 子任务边界 +
环境/任务标签，不包含真实图像。可注入三类真实采集中常见缺陷供质检盲检：
- static：静止/无效片段；- drift：镜头（SLAM 位姿）漂移；- duplicate：重复摆拍。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..embodied.env import EnvTask, PointMassEnv
from ..embodied.hybrid import Candidate, MotionPrior, SemanticCritic

RAW_MINUTES_PER_EPISODE = 3.0     # 一个片段折算 3 分钟“采集时长”
TASK_NAMES = ("reach_free", "ordered_sort", "contact_gate", "disturb_recover")


def make_task(task_name: str, variant: int) -> EnvTask:
    """每个任务有若干参数变体（环境/物体/受力差异），构成状态覆盖空间。"""
    v = variant
    if task_name == "reach_free":
        a = 2 * math.pi * (v % 8) / 8.0
        return EnvTask(name=task_name, start=(-2.0, 0.0, 0.0),
                       waypoints=[(1.6 * math.cos(a), 1.6 * math.sin(a), 0.0)],
                       max_steps=45)
    if task_name == "contact_gate":
        r = 0.50 + 0.05 * (v % 6)                    # 0.50..0.75
        y = 0.55 * ((v % 3) - 1)                     # 障碍横向位置
        return EnvTask(name=task_name, start=(-2.2, 0.0, 0.0),
                       waypoints=[(2.2, 0.0, 0.0)],
                       obstacles=(((0.0, y, 0.0), r),),
                       contact_terminal=True, max_steps=60)
    if task_name == "ordered_sort":
        g = torch.Generator().manual_seed(100 + v)
        off = 0.25 * (v % 4 - 1.5)
        w1 = (-0.4 + off, 1.3, 0.0)
        w2 = (-0.4 - off, -1.3, 0.0)
        w3 = (1.6, 0.3 * ((v % 3) - 1), 0.0)
        return EnvTask(name=task_name, start=(-1.6, -1.3, 0.0),
                       waypoints=[w1, w2, w3], max_steps=70)
    if task_name == "disturb_recover":
        mag = 0.6 + 0.35 * (v % 4)                   # 扰动强度
        side = -1.0 if v % 2 else 1.0
        return EnvTask(name=task_name, start=(-2.0, 0.0, 0.0),
                       waypoints=[(-0.5, 1.0, 0.0), (0.8, -0.8, 0.0),
                                  (2.0, 0.2, 0.0)],
                       disturb_step=18,
                       disturb_delta=(0.0, side * mag, 0.0),
                       max_steps=80)
    raise ValueError(task_name)


@dataclass
class EpisodeRecord:
    episode_id: str
    task_name: str
    variant: int
    seed: int
    success: bool
    states: List[List[float]]            # [x,y,z,vx,vy,vz] 每步
    hand_track: List[List[float]]       # 相机系“手部轨迹”代理（含可能漂移）
    contacts: int
    subtask_events: List[int]
    raw_minutes: float
    defect: Optional[str] = None        # 真值标签，仅供评估，质检不得使用
    params_key: Tuple = ()
    qc: Dict = field(default_factory=dict)
    accepted: bool = False
    density: Dict = field(default_factory=dict)


def _run_hybrid(task: EnvTask, seed: int, K: int, static: bool
                ) -> Tuple[List[torch.Tensor], int, int, bool, List[int]]:
    env = PointMassEnv(task); env.reset()
    prior = MotionPrior(K=K, seed=seed)
    critic = SemanticCritic()
    states, events = [], []
    last_active = 0
    while not env.done:
        tgt = critic._active_target(env)
        if static:
            chosen = Candidate("static", [torch.zeros(3) for _ in range(6)])
        else:
            cands = prior.propose(env, tgt)
            cands.append(Candidate("override", critic.override_program(env, tgt)))
            chosen = critic.decide(env, cands)
        for acc in chosen.accs:
            states.append(env.state().tolist())
            _, info = env.step(acc)
            if info["active"] > last_active:
                events.append(env.steps); last_active = info["active"]
            if env.done:
                break
    return states, env.collisions, env.steps, bool(env.success), events


def collect_episode(task_name: str, variant: int, seed: int, K: int = 16,
                    defect: Optional[str] = None) -> EpisodeRecord:
    task = make_task(task_name, variant)
    states, collisions, steps, success, events = _run_hybrid(
        task, seed, K=K, static=(defect == "static"))
    hand = [s[:3] for s in states]
    if defect == "drift":
        g = torch.Generator().manual_seed(seed + 777)
        walk = torch.zeros(3)
        noisy = []
        for p in hand:
            walk = walk + torch.randn(3, generator=g) * 0.07
            noisy.append((torch.tensor(p) + walk).tolist())
        hand = noisy
    eid = f"{task_name}-v{variant}-s{seed}"
    params_key = (task_name, variant if defect != "duplicate" else
                  _dup_anchor(task_name, seed))
    return EpisodeRecord(
        episode_id=eid + (f"-{defect}" if defect else ""),
        task_name=task_name, variant=variant, seed=seed,
        success=success, states=states, hand_track=hand,
        contacts=collisions, subtask_events=events,
        raw_minutes=RAW_MINUTES_PER_EPISODE, defect=defect,
        params_key=params_key)


def _dup_anchor(task_name: str, seed: int) -> int:
    """重复片段复用同任务的另一个常规变体编号，制造签名碰撞。"""
    return { "reach_free": 0, "contact_gate": 1, "ordered_sort": 0,
             "disturb_recover": 0}[task_name]


def generate_pool(n: int, seed: int = 0, K: int = 16,
                  defect_rate: float = 0.25,
                  task_skew: Optional[Dict[str, float]] = None
                  ) -> List[EpisodeRecord]:
    """生成候选片段池。task_skew 给定时按该偏态抽任务（模拟被动采集偏好）。"""
    g = torch.Generator().manual_seed(seed)
    pool: List[EpisodeRecord] = []
    for i in range(n):
        if task_skew:
            r = torch.rand(1, generator=g).item()
            acc = 0.0
            task = TASK_NAMES[-1]
            for tname, p in task_skew.items():
                acc += p
                if r <= acc:
                    task = tname; break
        else:
            task = TASK_NAMES[int(torch.randint(0, len(TASK_NAMES), (1,),
                                                generator=g))]
        variant = int(torch.randint(0, 12, (1,), generator=g))
        ep_seed = int(torch.randint(0, 10_000, (1,), generator=g))
        defect = None
        if torch.rand(1, generator=g).item() < defect_rate:
            defect = ("static", "drift", "duplicate")[
                int(torch.randint(0, 3, (1,), generator=g))]
        if defect == "duplicate" and pool:
            # 复刻池中已有片段（同任务/变体/seed），制造真实签名碰撞
            src = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
            ep = collect_episode(src.task_name, src.variant, src.seed, K=K)
            ep.defect = "duplicate"
            ep.episode_id = f"{ep.episode_id}-duplicate"
            pool.append(ep)
        else:
            pool.append(collect_episode(task, variant, ep_seed, K=K,
                                        defect=defect))
    return pool
