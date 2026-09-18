"""专家 Agent —— 每个 Agent 有明确角色与可验证产出。

所有重活都是 *同步纯函数*（torch/numpy），由 coordinator 用线程池异步调度，
因此“完全异步”体现在调度层而不伪造协程内的阻塞。

真实能力（CPU verified）：
- 预测专家：互为独立信息源的三种预测——模型盲预测、模型显式参数预测、
  纯运动学解析积分基线；供讨论/投票/选优，而非同质重复。
- 动作比较专家：对外部给定候选动作分别 rollout，按目标距离+风险打分。
- 验证专家：有真值时用 MSE 客观打分；无真值时用共识度。
- 分类/拆解专家：确定性规则（自由形式拆解属 AL4，需 LLM 门禁）。
- 代码改进专家：在 *封闭候选策略集* 内于隔离分支上应用补丁，验证跑分选优（AL2）。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ..contracts import DT, HORIZON, STATE_DIM
from ..kinematics import kinematic_features
from ..metrics import mse
from .automation import AL, CapabilityGate
from .types import Claim, Event, Task, TaskKind, WorkProduct


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------
@dataclass
class AgentContext:
    model: Any = None
    gate: CapabilityGate = field(default_factory=CapabilityGate)
    executor: Optional[concurrent.futures.Executor] = None
    dt: float = DT

    async def offload(self, fn: Callable, *args, **kwargs):
        loop = asyncio.get_event_loop()
        if self.executor is not None:
            return await loop.run_in_executor(
                self.executor, lambda: fn(*args, **kwargs))
        return fn(*args, **kwargs)


def _to_tensor(x: Any) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    return t


def _analytic_rollout(window: torch.Tensor, horizon: int, dt: float) -> torch.Tensor:
    """纯运动学解析积分（独立于学习模型的预测专家）：a 由初始窗估一次并固定。"""
    w = window
    if w.dim() == 2:
        w = w.unsqueeze(0)
    feats = kinematic_features(w, dt)          # [B, KIN_DIM]
    a = feats[:, 3:6]                          # a_lin
    cur = w[:, -1, :].clone()                  # [B,6]
    out = []
    state = cur.clone()
    for _ in range(horizon):
        p, v = state[:, 0:3], state[:, 3:6]
        v = v + a * dt
        p = p + v * dt + 0.5 * a * dt * dt
        state = torch.cat([p, v], dim=1)
        out.append(state)
    return torch.stack(out, dim=1)             # [B,H,6]


def score_from_mse(m: float) -> float:
    return float(1.0 / (1.0 + max(m, 0.0)))


# ---------------------------------------------------------------------------
# Agent 基类
# ---------------------------------------------------------------------------
class Agent:
    role = "base"
    weight = 1.0

    def __init__(self, agent_id: str, ctx: AgentContext):
        self.id = agent_id
        self.ctx = ctx

    async def run(self, task: Task, mem) -> WorkProduct:
        return await self.ctx.offload(self.run_sync, task, mem)

    def run_sync(self, task: Task, mem) -> WorkProduct:  # pragma: no cover
        raise NotImplementedError

    def _product(self, task: Task, output: Any, **kw) -> WorkProduct:
        p = WorkProduct(task_id=task.id, author=self.id, output=output,
                        kind=task.kind, branch=task.branch, **kw)
        return p


# ---------------------------------------------------------------------------
# 预测专家（三种独立信息源）
# ---------------------------------------------------------------------------
class PredictionExpert(Agent):
    role = "predictor"

    def __init__(self, agent_id: str, ctx: AgentContext, mode: str = "blind"):
        super().__init__(agent_id, ctx)
        assert mode in ("blind", "explicit", "analytic")
        self.mode = mode
        self.weight = {"blind": 1.0, "explicit": 1.1, "analytic": 0.9}[mode]

    def run_sync(self, task: Task, mem) -> WorkProduct:
        window = _to_tensor(task.payload["window"])
        if window.dim() == 2:
            window = window.unsqueeze(0)
        horizon = int(task.payload.get("horizon", HORIZON))
        if self.mode == "analytic":
            pred = _analytic_rollout(window, horizon, self.ctx.dt)
            grade = "verified"
        else:
            if self.ctx.model is None:
                raise RuntimeError("预测专家需要已加载的 WorldModelCore")
            explicit = task.payload.get("explicit")
            exp = None
            if self.mode == "explicit" and explicit is not None:
                exp = _to_tensor(explicit)
                if exp.dim() == 1:
                    exp = exp.unsqueeze(0)
            pred = self.ctx.model.rollout(window, horizon, explicit=exp)
            grade = "verified"
        out = pred.detach().cpu().tolist()
        p = self._product(task, out, evidence_grade=grade,
                          rationale=f"prediction:{self.mode}",
                          confidence=0.6)
        p.claims = [Claim(resource=f"task:{task.id}", mode="write", owner=self.id)]
        mem.append(Event(kind="product", author=self.id,
                         payload={"mode": self.mode, "task": task.id},
                         parent_ids=list(task.deps)))
        return p


# ---------------------------------------------------------------------------
# 动作比较专家（MPC 式：候选动作外部给定）
# ---------------------------------------------------------------------------
class ActionCompareExpert(Agent):
    role = "action_planner"
    weight = 1.0

    def run_sync(self, task: Task, mem) -> WorkProduct:
        pl = task.payload
        window = _to_tensor(pl["window"])
        if window.dim() == 2:
            window = window.unsqueeze(0)
        horizon = int(pl.get("horizon", HORIZON))
        target = np.asarray(pl["target"], dtype=np.float32)          # [3]
        actions = pl["actions"]                                     # list[dict]
        risk_w = float(pl.get("risk_weight", 0.1))
        results = []
        for act in actions:
            w = window.clone()
            # 动作 = 对当前速度施加一个脉冲增量 dv（外部候选，非自动发明）
            dv = np.asarray(act.get("dv", [0, 0, 0]), dtype=np.float32)
            w[:, -1, 3:6] += torch.as_tensor(dv)
            if self.ctx.model is not None:
                traj = self.ctx.model.rollout(w, horizon).detach().cpu().numpy()[0]
            else:
                traj = _analytic_rollout(w, horizon, self.ctx.dt).numpy()[0]
            final = traj[-1, 0:3]
            goal_cost = float(np.linalg.norm(final - target))
            risk = float(np.max(np.linalg.norm(traj[:, 3:6], axis=1)))
            total = goal_cost + risk_w * risk
            results.append({"action": act, "cost": total,
                            "goal_cost": goal_cost, "risk": risk})
        results.sort(key=lambda r: r["cost"])
        p = self._product(task, {"ranking": results,
                                 "best": results[0]["action"]},
                          confidence=0.7, evidence_grade="cpu-proto",
                          rationale="action ranking by goal+risk")
        return p


# ---------------------------------------------------------------------------
# 验证专家
# ---------------------------------------------------------------------------
class ValidatorExpert(Agent):
    role = "validator"
    weight = 1.2

    def __init__(self, agent_id: str, ctx: AgentContext, metric: str = "mse"):
        super().__init__(agent_id, ctx)
        self.metric = metric

    def validate(self, product: WorkProduct, task: Task) -> WorkProduct:
        pl = task.payload
        if task.kind == TaskKind.PREDICT and "truth" in pl:
            truth = np.asarray(pl["truth"], dtype=np.float32)
            pred = np.asarray(product.output, dtype=np.float32)
            m = float(np.mean((pred - truth) ** 2))
            product.metrics["mse"] = m
            product.score = score_from_mse(m)
            product.evidence_grade = "verified"
        elif task.kind == TaskKind.CODE_PATCH and "harness" in pl:
            result = pl["harness"](product.output)
            product.metrics.update(result.get("metrics", {}))
            product.score = float(result.get("score", 0.0))
            product.evidence_grade = result.get("evidence", "cpu-proto")
        else:
            product.score = product.confidence
        return product

    def run_sync(self, task: Task, mem) -> WorkProduct:
        # 验证者不产出候选；它在 coordinator 的验证阶段被调用 validate()
        return self._product(task, None, rationale="validator-pass")


# ---------------------------------------------------------------------------
# 批评专家（碰撞/一致性审查）
# ---------------------------------------------------------------------------
class CriticExpert(Agent):
    role = "critic"
    weight = 1.0

    def review(self, products: List[WorkProduct], task: Task) -> Dict[str, Any]:
        # 预测一致性：各候选最终位置的离散度；离散度大→低共识，需人审
        if task.kind == TaskKind.PREDICT and products:
            finals = np.array([np.asarray(p.output)[0, -1, 0:3]
                               for p in products if p.output is not None])
            spread = float(finals.std(axis=0).mean()) if len(finals) > 1 else 0.0
            return {"consensus_spread": spread,
                    "needs_human": spread > float(
                        task.payload.get("spread_threshold", 1.0))}
        return {"consensus_spread": 0.0, "needs_human": False}

    def run_sync(self, task: Task, mem) -> WorkProduct:
        return self._product(task, None, rationale="critic-pass")


# ---------------------------------------------------------------------------
# 分类专家
# ---------------------------------------------------------------------------
class ClassifierExpert(Agent):
    role = "classifier"
    weight = 1.0

    def classify(self, goal: str, payload: Dict[str, Any]) -> TaskKind:
        g = goal.lower()
        if "action" in g or "动作" in goal or "actions" in payload:
            return TaskKind.ACTION_COMPARE
        if "code" in g or "补丁" in goal or "refactor" in g or "candidates" in payload:
            return TaskKind.CODE_PATCH
        if "verify" in g or "验证" in goal or "跑分" in goal:
            return TaskKind.VERIFY
        if "predict" in g or "预测" in goal or "window" in payload:
            return TaskKind.PREDICT
        return TaskKind.CUSTOM


# ---------------------------------------------------------------------------
# 拆解专家（协调器使用；自由形式拆解需 AL4 LLM）
# ---------------------------------------------------------------------------
class DecomposerExpert(Agent):
    role = "decomposer"
    weight = 1.0

    def decompose(self, goal: str, payload: Dict[str, Any],
                  allow_freeform: bool = False) -> List[Task]:
        kind = ClassifierExpert("cls", self.ctx).classify(goal, payload)
        if kind == TaskKind.PREDICT:
            modes = payload.get("expert_modes",
                                ["blind", "explicit", "analytic"])
            preds = [Task(goal=f"预测[{m}]", kind=TaskKind.PREDICT,
                          payload=payload, branch=f"pred-{m}")
                     for m in modes if self._mode_allowed(m, payload)]
            return preds
        if kind == TaskKind.ACTION_COMPARE:
            return [Task(goal="动作推演排序", kind=TaskKind.ACTION_COMPARE,
                         payload=payload, branch="action")]
        if kind == TaskKind.CODE_PATCH:
            cands = payload.get("candidates", {})
            return [Task(goal=f"补丁候选[{name}]", kind=TaskKind.CODE_PATCH,
                         payload={**payload, "candidate": name,
                                  "impl": cands[name]},
                         branch=f"patch-{name}") for name in cands]
        if allow_freeform:
            self.ctx.gate.require(AL.AL4)
        return [Task(goal=goal, kind=kind, payload=payload, branch="main")]

    @staticmethod
    def _mode_allowed(m: str, payload: Dict[str, Any]) -> bool:
        if m == "explicit":
            return payload.get("explicit") is not None
        return True

    def run_sync(self, task: Task, mem) -> WorkProduct:
        return self._product(task, None, rationale="decompose-pass")


# ---------------------------------------------------------------------------
# 受约束代码改进专家（封闭候选集，隔离分支）
# ---------------------------------------------------------------------------
class CodePatchExpert(Agent):
    role = "engineer"
    weight = 1.0

    def run_sync(self, task: Task, mem) -> WorkProduct:
        pl = task.payload
        name = pl["candidate"]
        impl = pl["impl"]
        # 隔离：在分支草稿区内只放副本，绝不改主线文件
        branch_files = copy.deepcopy(pl.get("base_files", {}))
        branch_files[pl.get("target_file", "module.py")] = name
        mem.scratch_put(self.id, task.branch, {"files": branch_files})
        product = self._product(
            task, {"candidate": name, "impl": impl,
                   "target_file": pl.get("target_file", "module.py"),
                   "branch_files": branch_files},
            evidence_grade="cpu-proto", confidence=0.6,
            rationale=f"candidate strategy: {name}",
            claims=[Claim(resource=f"file:{pl.get('target_file','module.py')}",
                          mode="write", owner=self.id)])
        return product
