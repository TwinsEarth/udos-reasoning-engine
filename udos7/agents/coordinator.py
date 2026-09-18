"""CoordinatorAgent —— 不亲自写核心代码的协调者 / 项目经理 / 架构师。

职责（对应 Claude Code Projects 式协调器的本地可测内核）：
1. 接收高阶目标，分类并拆解为带依赖 DAG 的子任务；
2. 在 *隔离工作区*（独立 branch / 草稿区）中完全异步派发专家 Agent；
3. 跟踪依赖、收集产物，组织验证/讨论/提名/投票，选出当时最优；
4. 代码类任务仅合并获胜分支（PR 语义），败者分支丢弃；
5. 技术决策与产物实时写入共享记忆，并经 Transport 同步项目上下文
   （本地 LocalTransport 即真同步；云端 CloudTransport 未配置则门禁拦截）。

人在回路中的角色：拍板方向、审核 Decision、处理 needs_human 异常。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..persistence import load_worldmodel
from .automation import CapabilityGate
from .cloud import LocalTransport, Transport
from .memory import SharedMemory
from .protocols import (merge_winning_patch, select_best, weighted_ensemble)
from .types import Event, Task, TaskKind, TaskStatus, WorkProduct
from .workers import (ActionCompareExpert, AgentContext, ClassifierExpert,
                      CodePatchExpert, CriticExpert, DecomposerExpert,
                      PredictionExpert, ValidatorExpert)


@dataclass
class GoalReport:
    goal: str
    kind: str
    winner: Optional[dict] = None
    ensemble: Optional[list] = None
    decision: Optional[dict] = None
    products: List[dict] = field(default_factory=list)
    needs_human: bool = False
    note: str = ""
    main_files: Dict[str, str] = field(default_factory=dict)


class CoordinatorAgent:
    def __init__(self, project: str = "udos",
                 model=None, checkpoint: Optional[str] = None,
                 memory: Optional[SharedMemory] = None,
                 transport: Optional[Transport] = None,
                 gate: Optional[CapabilityGate] = None,
                 concurrency: int = 8,
                 executor: Optional[concurrent.futures.Executor] = None):
        self.project = project
        self.mem = memory or SharedMemory(namespace=project)
        self.transport = transport or LocalTransport()
        self.gate = gate or CapabilityGate()
        self._owns_executor = executor is None
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency)
        if model is not None:
            self.model = model
        elif checkpoint:
            self.model, _ = load_worldmodel(checkpoint)
        else:
            self.model = None
        self.ctx = AgentContext(model=self.model, gate=self.gate,
                                executor=self.executor)
        self.classifier = ClassifierExpert("classifier", self.ctx)
        self.decomposer = DecomposerExpert("decomposer", self.ctx)
        self.validator = ValidatorExpert("validator", self.ctx)
        self.critic = CriticExpert("critic", self.ctx)
        self.sem = asyncio.Semaphore(concurrency)

    async def aclose(self):
        if self._owns_executor:
            # wait=True 确定性回收工作线程，避免非守护线程拖住进程退出
            self.executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    def _sync_pull(self):
        snap = self.transport.pull(self.project)
        if snap:
            self.mem.merge(snap)

    def _sync_push(self):
        self.transport.push(self.project, self.mem.snapshot())

    # ------------------------------------------------------------------
    def _agent_for(self, task: Task):
        if task.kind == TaskKind.PREDICT:
            mode = task.branch.split("pred-", 1)[1] if task.branch and "pred-" in task.branch else "blind"
            return PredictionExpert(f"pred-{mode}", self.ctx, mode=mode)
        if task.kind == TaskKind.ACTION_COMPARE:
            return ActionCompareExpert("action-planner", self.ctx)
        if task.kind == TaskKind.CODE_PATCH:
            return CodePatchExpert(f"eng-{task.branch}", self.ctx)
        return None

    async def _run_task(self, task: Task,
                        products: Dict[str, WorkProduct]):
        async with self.sem:
            task.status = TaskStatus.RUNNING
            agent = self._agent_for(task)
            if agent is None:
                task.status = TaskStatus.BLOCKED
                return
            try:
                prod = await agent.run(task, self.mem)
                prod.event_id = self.mem.append(Event(
                    kind="work_product", author=agent.id,
                    payload={"task": task.id, "branch": task.branch},
                    parent_ids=list(task.deps))).id
                products[task.id] = prod
                task.assignee = agent.id
                task.status = TaskStatus.DONE
            except Exception as e:  # 失败显式落账，不静默吞
                task.status = TaskStatus.FAILED
                self.mem.append(Event(kind="task_failed", author=agent.id,
                                      payload={"task": task.id,
                                               "error": repr(e)}))

    async def _execute_dag(self, tasks: List[Task]) -> Dict[str, WorkProduct]:
        products: Dict[str, WorkProduct] = {}
        pending = {t.id: t for t in tasks}
        done: set = set()
        # 简单可靠的依赖调度：每轮派发所有就绪任务，直到收敛
        while pending:
            ready = [t for t in pending.values()
                     if t.is_ready(done) and t.status != TaskStatus.RUNNING]
            if not ready:
                # 依赖无法满足（失败/环）→ 剩余标记 blocked
                for t in pending.values():
                    t.status = TaskStatus.BLOCKED
                break
            await asyncio.gather(*(self._run_task(t, products) for t in ready))
            for t in ready:
                pending.pop(t.id, None)
                if t.status == TaskStatus.DONE:
                    done.add(t.id)
        return products

    # ------------------------------------------------------------------
    async def run(self, goal: str, payload: Optional[Dict[str, Any]] = None,
                  votes: Optional[list] = None) -> GoalReport:
        payload = dict(payload or {})
        self._sync_pull()
        kind = self.classifier.classify(goal, payload)
        self.mem.append(Event(kind="goal", author="human",
                              payload={"goal": goal, "kind": kind.value}))

        tasks = self.decomposer.decompose(goal, payload)
        for t in tasks:
            t.status = TaskStatus.READY
        products = await self._execute_dag(tasks)

        # 用一个“代表任务”承载验证/选优（同类子任务共享目标语义）
        rep = tasks[0] if tasks else Task(goal=goal, kind=kind, payload=payload)
        cand = list(products.values())
        winner, decision = select_best(
            rep, cand, self.mem, self.validator, self.critic, votes=votes)

        report = GoalReport(goal=goal, kind=kind.value,
                            decision={
                                "winner_id": decision.winner_id,
                                "method": decision.method,
                                "tally": decision.tally,
                                "collisions": [c.__dict__ for c in
                                               decision.collisions],
                                "note": decision.note})
        report.needs_human = bool(decision.note)
        report.note = decision.note
        report.products = [{
            "id": p.id, "author": p.author, "score": p.score,
            "metrics": p.metrics, "rationale": p.rationale,
            "evidence": p.evidence_grade, "branch": p.branch}
            for p in cand]
        if winner is not None:
            report.winner = {
                "author": winner.author, "score": winner.score,
                "metrics": winner.metrics, "rationale": winner.rationale,
                "branch": winner.branch}

        # 预测任务额外给出加权集成（无真值时的生产路径）
        if kind == TaskKind.PREDICT:
            weights = {p.author: self._weight_of(p.author) for p in cand}
            ens = weighted_ensemble(cand, weights)
            if ens is not None:
                report.ensemble = ens.tolist()

        # 代码任务：仅合并获胜分支到主线文件
        if kind == TaskKind.CODE_PATCH and winner is not None:
            main_files = self.mem.get("workspace", "files", {}) or {}
            main_files = merge_winning_patch(main_files, winner)
            self.mem.put("workspace", "files", main_files, owner="coordinator")
            report.main_files = main_files

        self.mem.append(Event(kind="goal_complete", author="coordinator",
                              payload={"goal": goal,
                                       "winner": decision.winner_id}))
        self._sync_push()
        return report

    @staticmethod
    def _weight_of(author: str) -> float:
        if author.startswith("pred-explicit"):
            return 1.1
        if author.startswith("pred-analytic"):
            return 0.9
        return 1.0
