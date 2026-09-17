"""多体协同核心 —— MultiAgentScene / AgentCoordinator (v3.8.0, 全域调度线)
======================================================================
analogy, not reproduction —— 多体为**合成参数化代理**, 非真机多机器人; 每个智能体携带
一个与主 predictor 一致的 6 维状态布局 [pos(3), vel(3)], 用于在同一合成场景内考察
N 体协同、冲突消解与 (后续节点的) 世界模型预算调度。

本模块分工 (纯推理外挂, 零梯度, 确定性, opt-in):
    * MultiAgentScene  —— N 个智能体的状态容器: 每个智能体有独立 id / 状态 / 目标 /
                          优先级 / 形态半径 / 限速; 支持快照/回放 (供数字孪生节点复用)。
    * AgentCoordinator —— **非学习**冲突消解: 优先级让行 + 速度调节。
                          检测两两接近/侵入 => 高优先级智能体保持原速, 低优先级智能体
                          按侵入深度成比例减速 (接触时减速最少, 深入时趋零)。
                          被让行者/被否决候选显式记录, 不静默过滤。

设计纪律 (与全工程一致):
    * 纯前向、**确定性**、**不修改主 predictor 52191 参数** (本模块甚至不持有模型)。
    * 无任何可训练参数 (纯算法): 冲突消解为确定性几何规则。
    * opt-in: 不构造 MultiAgentScene 时, 旧推理路径逐位一致; 本模块不挂任何默认钩子。
    * 空 / 非法输入显式 ValueError, 不静默 inf 传播。
    * 日志沿用 logging_config (默认 WARNING, 仅 stderr)。

第二引擎一律称 GPM。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, OrderedDict, Sequence

import torch

logger = logging.getLogger("udos.multi_agent")

# 与主 predictor 状态布局一致: [pos(3), vel(3)]
POS_DIM = 3
RAW_DIM = 6


def _as_state6(v: Any, name: str) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
    if t.numel() != RAW_DIM:
        raise ValueError(f"{name} 应为 {RAW_DIM} 维 [pos(3)+vel(3)], 实际 {t.numel()}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} 含 NaN/inf 非有限值")
    return t


def _as_vec3(v: Any, name: str) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
    if t.numel() != POS_DIM:
        raise ValueError(f"{name} 应为 {POS_DIM} 维, 实际 {t.numel()}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} 含 NaN/inf 非有限值")
    return t


@dataclass
class AgentState:
    """单个合成智能体的状态容器。state 布局 [pos(3), vel(3)]。"""

    agent_id: str
    state: torch.Tensor
    goal: Optional[torch.Tensor] = None     # 目标位置 (3 维)
    priority: int = 5                        # 越小越优先
    radius: float = 0.3                      # 形态半径 (几何形态, 非真机)
    max_speed: float = 1.0                   # 限速 (合成参考)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise ValueError("agent_id 须为非空字符串")
        self.state = _as_state6(self.state, f"agent[{self.agent_id}].state")
        if self.goal is not None:
            self.goal = _as_vec3(self.goal, f"agent[{self.agent_id}].goal")
        if self.radius <= 0:
            raise ValueError("radius 必须 > 0")
        if self.max_speed <= 0:
            raise ValueError("max_speed 必须 > 0")

    @property
    def pos(self) -> torch.Tensor:
        return self.state[:POS_DIM]

    @property
    def vel(self) -> torch.Tensor:
        return self.state[POS_DIM:]


class MultiAgentScene:
    """N 个智能体的状态容器 (有序; 插入序稳定, 供确定性回放)。

    空场景守卫: 任何需要智能体的操作在 n==0 时显式 ValueError。
    """

    def __init__(self) -> None:
        self._agents: "OrderedDict[str, AgentState]" = OrderedDict()

    # -- 构造 --------------------------------------------------------- #
    def add_agent(self, agent_id: str, state: Any,
                  goal: Optional[Any] = None, priority: int = 5,
                  radius: float = 0.3, max_speed: float = 1.0) -> AgentState:
        if agent_id in self._agents:
            raise ValueError(f"agent_id 重复: {agent_id}")
        ag = AgentState(agent_id=agent_id, state=state, goal=goal,
                        priority=int(priority), radius=float(radius),
                        max_speed=float(max_speed))
        self._agents[agent_id] = ag
        return ag

    # -- 查询 --------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    @property
    def n_agents(self) -> int:
        return len(self._agents)

    @property
    def ids(self) -> List[str]:
        return list(self._agents.keys())

    def get(self, agent_id: str) -> AgentState:
        if agent_id not in self._agents:
            raise ValueError(f"未知 agent_id: {agent_id}")
        return self._agents[agent_id]

    def require_agents(self) -> None:
        if self.n_agents == 0:
            raise ValueError("空场景: 至少需要 1 个智能体")

    def states(self) -> torch.Tensor:
        """返回 [N, 6] 堆叠状态。"""
        self.require_agents()
        return torch.stack([a.state for a in self._agents.values()], dim=0)

    def positions(self) -> torch.Tensor:
        """返回 [N, 3] 位置。"""
        self.require_agents()
        return torch.stack([a.pos for a in self._agents.values()], dim=0)

    def set_goal(self, agent_id: str, goal: Any) -> None:
        ag = self.get(agent_id)
        ag.goal = _as_vec3(goal, f"agent[{agent_id}].goal")

    # -- 快照 / 回放 -------------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        """可序列化快照 (供数字孪生回放 / HTTP 响应)。"""
        return {
            "n_agents": self.n_agents,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "state": [round(float(x), 6) for x in a.state.tolist()],
                    "goal": (None if a.goal is None
                             else [round(float(x), 6) for x in a.goal.tolist()]),
                    "priority": int(a.priority),
                    "radius": float(a.radius),
                    "max_speed": float(a.max_speed),
                }
                for a in self._agents.values()
            ],
        }

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any]) -> "MultiAgentScene":
        """从 snapshot() 重建场景 (回放入口)。"""
        if not isinstance(snap, dict) or "agents" not in snap:
            raise ValueError("snapshot 格式非法: 需含 'agents'")
        sc = cls()
        for rec in snap["agents"]:
            sc.add_agent(agent_id=rec["agent_id"], state=rec["state"],
                         goal=rec.get("goal"), priority=rec.get("priority", 5),
                         radius=rec.get("radius", 0.3),
                         max_speed=rec.get("max_speed", 1.0))
        return sc


class AgentCoordinator:
    """非学习多体冲突消解: 优先级让行 + 速度调节 (确定性几何规则)。

    规则:
        * 两两检测: 若 ||pos_i - pos_j|| < r_i + r_j + buffer => 冲突 (侵入/接近)。
        * 决胜: 优先级数字小者优先 (保持原速); 同级时按 agent_id 字典序小者优先。
        * 让行: 败者 (低优先级) 速度按侵入深度成比例衰减:
                yield = clamp(1 - penetration / (r_i+r_j+buffer), 0, 1)
                接触瞬间 penetration~0 => yield~1 (不减速); 越深入越减速直至 0。
        * 无冲突智能体: 速度保持不变 (单体退化时恒等)。
    被让行者/冲突对显式记录 (诚实不静默过滤); 无可训练参数。
    """

    def __init__(self, buffer: float = 0.0) -> None:
        if buffer < 0:
            raise ValueError("buffer 必须 >= 0")
        self.buffer = float(buffer)
        self.last_report: Dict[str, Any] = {}

    @staticmethod
    def _winner(pri_i: int, id_i: str, pri_j: int, id_j: str):
        """返回 (winner_id, loser_id)。优先级小者胜; 同级按字典序。"""
        if pri_i != pri_j:
            return (id_i, id_j) if pri_i < pri_j else (id_j, id_i)
        return (id_i, id_j) if id_i < id_j else (id_j, id_i)

    def resolve(self, scene: MultiAgentScene) -> Dict[str, Any]:
        """对场景做一次冲突消解, 返回每个智能体的建议速度 [3] 与诊断。

        返回:
            commands: {agent_id: List[3]} 建议速度 (让行后的)
            conflicts: 冲突对明细
            n_conflicts / yielded / kept / winner_ids
            deterministic: True
        """
        scene.require_agents()
        ids = scene.ids
        n = scene.n_agents
        agents = [scene.get(i) for i in ids]
        # 默认建议速度 = 当前速度 (无冲突则恒等)
        commands: Dict[str, List[float]] = {
            a.agent_id: [round(float(x), 6) for x in a.vel.tolist()]
            for a in agents
        }
        conflicts: List[Dict[str, Any]] = []
        yielded: List[str] = []
        winner_ids: List[str] = []

        for ii in range(n):
            for jj in range(ii + 1, n):
                a, b = agents[ii], agents[jj]
                dist = float((a.pos - b.pos).norm().item())
                safe = a.radius + b.radius + self.buffer
                if dist >= safe:
                    continue
                penetration = safe - dist
                yield_factor = max(0.0, min(1.0,
                                             1.0 - penetration / max(safe, 1e-12)))
                wid, lid = self._winner(a.priority, a.agent_id,
                                        b.priority, b.agent_id)
                loser = scene.get(lid)
                slowed = (loser.vel * yield_factor)
                commands[lid] = [round(float(x), 6) for x in slowed.tolist()]
                if lid not in yielded:
                    yielded.append(lid)
                if wid not in winner_ids:
                    winner_ids.append(wid)
                conflicts.append({
                    "pair": [a.agent_id, b.agent_id],
                    "dist": round(dist, 6),
                    "safe_distance": round(safe, 6),
                    "penetration": round(penetration, 6),
                    "winner": wid, "loser": lid,
                    "yield_factor": round(yield_factor, 6),
                })

        rep = {
            "n_agents": n,
            "commands": commands,
            "conflicts": conflicts,
            "n_conflicts": len(conflicts),
            "yielded_agents": yielded,
            "winner_agents": winner_ids,
            "kept_agents": [i for i in ids
                            if i not in yielded and i not in winner_ids],
            "deterministic": True,
            "learned": False,
        }
        self.last_report = rep
        return rep

    def apply_commands(self, scene: MultiAgentScene,
                       commands: Dict[str, List[float]], dt: float = 1.0
                       ) -> MultiAgentScene:
        """按建议速度积分一步 (pos += vel*dt), 返回**新**场景 (不改原场景)。

        越速: 任一智能体合成速度 > max_speed 时按比例缩到限速内 (与脊髓纪律一致)。
        """
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        scene.require_agents()
        out = MultiAgentScene()
        for a in (scene._agents.values()):  # 只读原场景
            vel = _as_vec3(commands[a.agent_id], f"cmd[{a.agent_id}]")
            speed = float(vel.norm().item())
            if speed > a.max_speed:
                vel = vel * (a.max_speed / max(speed, 1e-12))
            new_pos = a.pos + vel * float(dt)
            new_state = torch.cat([new_pos, vel], dim=0)
            out.add_agent(agent_id=a.agent_id, state=new_state, goal=a.goal,
                          priority=a.priority, radius=a.radius,
                          max_speed=a.max_speed)
        return out
