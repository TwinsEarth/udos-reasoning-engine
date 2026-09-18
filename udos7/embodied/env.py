"""最小具身闭环环境：三维点质量“末端执行器” + 有序目标 + 球形障碍 + 扰动。

状态沿用引擎契约 STATE_DIM=6 = [px,py,pz,vx,vy,vz]；动作是三维指令加速度
（类比末端位姿修正）。这是**已知模型的控制基准**（MPC 意义下的环境模型），
不是学习型 VLA，也不是 MuJoCo 接触动力学——后者属 GPU/仿真器闸门。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from ..contracts import STATE_DIM

Vec3 = Sequence[float]


@dataclass(frozen=True)
class EnvTask:
    name: str
    start: Vec3                                   # 初始位置
    waypoints: List[Vec3]                         # 必须**按顺序**到达的目标
    obstacles: Tuple[Tuple[Vec3, float], ...] = ()  # (球心, 半径)，接触即碰撞
    max_steps: int = 80
    reach: float = 0.28                           # 到达半径
    contact_terminal: bool = False                # 接触敏感任务：碰撞即终止失败
    disturb_step: int = -1                        # >0 时在该步施加位置扰动
    disturb_delta: Vec3 = (0.0, 0.0, 0.0)
    dt: float = 0.2
    amax: float = 3.2
    vmax: float = 2.2
    damping: float = 0.06


class PointMassEnv:
    """点质量双积分环境。pos/vel 用 torch 向量，整段推演可在 clone 上无副作用进行。"""

    def __init__(self, task: EnvTask):
        self.task = task
        self.reset()

    def reset(self) -> torch.Tensor:
        t = self.task
        self.pos = torch.tensor(t.start, dtype=torch.float32)
        self.vel = torch.zeros(3)
        self.active = 0                 # 有序目标索引
        self.visited: List[int] = []
        self.collisions = 0
        self.steps = 0
        self.done = False
        self.success = False
        return self.state()

    def state(self) -> torch.Tensor:
        return torch.cat([self.pos.detach().clone(), self.vel.detach().clone()])

    # ---- 物理（候选推演与真实执行共用同一已知模型）----
    def _integrate(self, pos: torch.Tensor, vel: torch.Tensor,
                   acc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        t = self.task
        acc = torch.clamp(acc, -t.amax, t.amax)
        vel = vel + acc * t.dt
        vel = vel * (1.0 - t.damping)
        vnorm = vel.norm().item()
        if vnorm > t.vmax:
            vel = vel * (t.vmax / vnorm)
        pos = pos + vel * t.dt
        hit = False
        for c, r in t.obstacles:
            if (pos - torch.as_tensor(c, dtype=torch.float32)).norm().item() < r:
                hit = True
        return pos, vel, hit

    def step(self, acc: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        t = self.task
        if self.done:
            raise RuntimeError("episode 已结束，请先 reset()")
        if t.disturb_step >= 0 and self.steps == t.disturb_step:
            self.pos = self.pos + torch.as_tensor(t.disturb_delta, dtype=torch.float32)
        self.pos, self.vel, hit = self._integrate(self.pos, self.vel, acc)
        self.steps += 1
        if hit:
            self.collisions += 1
        self._update_waypoints()
        terminal_contact = t.contact_terminal and self.collisions > 0
        self.success = self.active >= len(t.waypoints) and self.collisions == 0
        self.done = bool(self.success or terminal_contact or self.steps >= t.max_steps)
        info = {"collision": hit, "active": self.active,
                "success": self.success, "done": self.done}
        return self.state(), info

    def _update_waypoints(self) -> None:
        """严格有序：只认当前 active 目标，提前碰到后面的目标不计入。"""
        t = self.task
        while self.active < len(t.waypoints):
            tgt = torch.as_tensor(t.waypoints[self.active], dtype=torch.float32)
            if (self.pos - tgt).norm().item() <= t.reach:
                self.visited.append(self.active)
                self.active += 1
            else:
                break

    # ---- 供候选动作段无副作用推演 ----
    def clone(self) -> "PointMassEnv":
        e = PointMassEnv.__new__(PointMassEnv)
        e.task = self.task
        e.pos = self.pos.clone()
        e.vel = self.vel.clone()
        e.active = self.active
        e.visited = list(self.visited)
        e.collisions = self.collisions
        e.steps = self.steps
        e.done = self.done
        e.success = self.success
        return e

    def rollout_program(self, accs: Sequence[torch.Tensor]
                        ) -> dict:
        """在克隆环境上执行一段加速度程序，返回评分所需轨迹量（不改真实环境）。"""
        e = self.clone()
        min_margin = math.inf
        effort = 0.0
        reached = 0
        collide = False
        positions = [e.pos.clone()]
        for acc in accs:
            p, v, hit = e._integrate(e.pos, e.vel, acc)
            e.pos, e.vel = p, v
            if hit:
                collide = True
            effort += float(acc.norm()) ** 2
            for c, r in e.task.obstacles:
                m = (e.pos - torch.as_tensor(c, dtype=torch.float32)).norm().item() - r
                min_margin = min(min_margin, m)
            before = e.active
            e._update_waypoints()
            reached += e.active - before
            positions.append(e.pos.clone())
            if e.active >= len(e.task.waypoints):
                break
        return {"min_margin": min_margin, "effort": effort, "reached": reached,
                "collide": collide, "end_active": e.active,
                "end_pos": e.pos.clone(), "positions": positions}


def standard_suite() -> List[EnvTask]:
    """四类任务，覆盖文章揭示的“语义 vs 接触”互补面。"""
    return [
        EnvTask(name="reach_free",
                start=(-2.0, 0.0, 0.0),
                waypoints=[(2.0, 0.0, 0.0)],
                max_steps=45),
        EnvTask(name="ordered_sort",
                start=(-1.6, -1.3, 0.0),
                # 起点离 B 最近，但必须先 A 后 B 再 C：考任务语义/顺序
                waypoints=[(-0.4, 1.3, 0.0), (-0.4, -1.3, 0.0), (1.6, 0.0, 0.0)],
                max_steps=70),
        EnvTask(name="contact_gate",
                start=(-2.0, 0.0, 0.0),
                waypoints=[(2.0, 0.0, 0.0)],
                obstacles=(((0.0, 0.0, 0.0), 0.70),),
                contact_terminal=True, max_steps=60),
        EnvTask(name="disturb_recover",
                start=(-2.0, 0.0, 0.0),
                waypoints=[(-0.5, 1.0, 0.0), (0.8, -0.8, 0.0), (2.0, 0.2, 0.0)],
                disturb_step=18, disturb_delta=(0.0, 1.4, 0.0),
                max_steps=80),
    ]
