"""合成数字孪生场景 —— DigitalTwinScene (v3.8.0.dev3, 全域调度线)
======================================================================
analogy, not reproduction —— **合成**多体+多物体+障碍的参数化场景生成器; 非真机数字孪生。
可配置智能体数 / 障碍数 / 场景尺寸 / 随机种子; 支持场景快照与回放。

组合既有模块:
    * MultiAgentScene (3.8.0) 承载 N 智能体状态;
    * AgentCoordinator 做一步冲突消解 + 积分 (step_dynamics);
    * 障碍为合成几何点 (供脊髓反射/碰撞检测复用)。

设计纪律 (与全工程一致):
    * 纯确定性生成 (torch.Generator(seed)), 同 seed 同布局逐位可复现。
    * 零梯度、无可训参数; opt-in。
    * 非法配置显式 ValueError。

第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from .multi_agent import AgentCoordinator, MultiAgentScene

logger = logging.getLogger("udos.digital_twin")


class DigitalTwinScene:
    """参数化合成数字孪生场景: N 智能体 + M 障碍, 确定性生成 / 快照 / 回放。"""

    def __init__(self, n_agents: int = 4, n_obstacles: int = 6,
                 seed: int = 0, bounds: float = 10.0,
                 agent_radius: float = 0.3, agent_max_speed: float = 1.0) -> None:
        if n_agents < 1:
            raise ValueError("n_agents 必须 >= 1")
        if n_obstacles < 0:
            raise ValueError("n_obstacles 必须 >= 0")
        if bounds <= 0:
            raise ValueError("bounds 必须 > 0")
        self.n_agents = int(n_agents)
        self.n_obstacles = int(n_obstacles)
        self.seed = int(seed)
        self.bounds = float(bounds)
        self.agent_radius = float(agent_radius)
        self.agent_max_speed = float(agent_max_speed)
        self.agents = MultiAgentScene()
        self.obstacles: List[torch.Tensor] = []
        self.scene_params = torch.zeros(4)
        self.coordinator = AgentCoordinator()
        self.step_index = 0
        self._build()

    # -- 确定性生成 --------------------------------------------------- #
    def _build(self) -> None:
        g = torch.Generator().manual_seed(self.seed)
        b = self.bounds
        # 智能体: 位置/速度/优先级确定性采样
        for k in range(self.n_agents):
            pos = (torch.rand(3, generator=g) * 2 - 1) * b
            vel = (torch.rand(3, generator=g) * 2 - 1) * self.agent_max_speed
            goal = (torch.rand(3, generator=g) * 2 - 1) * b
            self.agents.add_agent(
                f"ag{k}", state=torch.cat([pos, vel], dim=0), goal=goal,
                priority=(k % 5) + 1, radius=self.agent_radius,
                max_speed=self.agent_max_speed)
        # 障碍: 随机点 (避开与智能体重叠由碰撞检测负责)
        self.obstacles = [
            (torch.rand(3, generator=g) * 2 - 1) * b
            for _ in range(self.n_obstacles)]
        # 场景参数 [4]: 与主 predictor scene_param_dim 对齐的合成占位
        self.scene_params = torch.tensor([0.0, 0.0, 0.0, 0.0])

    # -- 一步推进 ----------------------------------------------------- #
    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """冲突消解 + 积分一步, 原地推进 (供回放/仿真)。"""
        rep = self.coordinator.resolve(self.agents)
        self.agents = self.coordinator.apply_commands(
            self.agents, rep["commands"], dt=dt)
        self.step_index += 1
        return {"step": self.step_index - 1,
                "n_conflicts": rep["n_conflicts"],
                "yielded": rep["yielded_agents"]}

    # -- 快照 / 回放 -------------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        """可序列化快照 (含智能体 + 障碍 + 元信息)。"""
        return {
            "kind": "digital_twin_scene",
            "config": {"n_agents": self.n_agents,
                       "n_obstacles": self.n_obstacles,
                       "seed": self.seed, "bounds": self.bounds},
            "step_index": self.step_index,
            "agents": self.agents.snapshot()["agents"],
            "obstacles": [[round(float(x), 6) for x in o.tolist()]
                          for o in self.obstacles],
            "scene_params": [round(float(x), 6) for x in
                             self.scene_params.tolist()],
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any]) -> "DigitalTwinScene":
        """从快照回放重建场景 (不重新随机采样, 精确恢复当时状态)。"""
        if snap.get("kind") != "digital_twin_scene":
            raise ValueError("快照类型非法: 需 kind=='digital_twin_scene'")
        cfg = snap["config"]
        obj = cls.__new__(cls)
        obj.n_agents = int(cfg["n_agents"])
        obj.n_obstacles = int(cfg["n_obstacles"])
        obj.seed = int(cfg["seed"])
        obj.bounds = float(cfg["bounds"])
        obj.agents = MultiAgentScene.from_snapshot(
            {"agents": snap["agents"]})
        obj.obstacles = [torch.as_tensor(o, dtype=torch.float32)
                         for o in snap["obstacles"]]
        obj.scene_params = torch.as_tensor(snap["scene_params"],
                                           dtype=torch.float32)
        obj.coordinator = AgentCoordinator()
        obj.step_index = int(snap["step_index"])
        return obj

    def summary(self) -> Dict[str, Any]:
        return {
            "n_agents": self.agents.n_agents,
            "n_obstacles": self.n_obstacles,
            "seed": self.seed, "bounds": self.bounds,
            "step_index": self.step_index,
            "agent_ids": self.agents.ids,
        }
