"""Agent 内部市场与贡献结算（v7.4.9）。

任务定价 → 投标 → 授标 → 验收 → 结算。规则：
- 仅对**验收通过**的唯一完成者付费；重复劳动（同任务重复提交）不付费；
- 被 QA 拒绝/拜占庭结果不付费，并按 slashing 扣减保证金；
- 守恒：总支出 ≤ 总预算；余额变动之和 = 发行 − 罚没。
确定性 CPU 模拟，价格为内部记账单位，证据 cpu-proto。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Bid:
    agent: str
    cost: float
    quality: float            # 自评/历史质量，0..1
    load: int = 0


@dataclass
class TaskAuction:
    task_id: str
    reward: float
    quality_floor: float = 0.0
    winner: Optional[str] = None
    settled: bool = False

    def award(self, bids: List[Bid]) -> Optional[str]:
        eligible = [b for b in bids if b.quality >= self.quality_floor]
        if not eligible:
            return None
        # 性价比：质量/成本，负载作次序兜底
        best = max(eligible, key=lambda b: (b.quality / max(b.cost, 1e-9),
                                            -b.load))
        self.winner = best.agent
        return best.agent


@dataclass
class ContributionLedger:
    budgets: Dict[str, float] = field(default_factory=dict)
    balances: Dict[str, float] = field(default_factory=dict)
    paid_tasks: Dict[str, str] = field(default_factory=dict)   # task -> agent
    rejected: List[Dict] = field(default_factory=list)
    slashed: float = 0.0
    total_paid: float = 0.0

    def deposit(self, task_id: str, amount: float):
        self.budgets[task_id] = amount

    def settle(self, task_id: str, agent: str, accepted: bool,
               duplicate: bool = False, slash: float = 0.0) -> Dict:
        """验收结算。返回结算明细；不满足条件一律不产生支付。"""
        if task_id in self.paid_tasks:
            return {"task": task_id, "agent": agent, "paid": 0.0,
                    "reason": "already_paid"}
        if duplicate:
            self.rejected.append({"task": task_id, "agent": agent,
                                  "reason": "duplicate_work"})
            return {"task": task_id, "agent": agent, "paid": 0.0,
                    "reason": "duplicate_work"}
        if not accepted:
            self.rejected.append({"task": task_id, "agent": agent,
                                  "reason": "rejected"})
            if slash:
                self.balances[agent] = self.balances.get(agent, 0.0) - slash
                self.slashed += slash
            return {"task": task_id, "agent": agent, "paid": 0.0,
                    "reason": "rejected", "slashed": slash}
        reward = self.budgets.get(task_id, 0.0)
        self.balances[agent] = self.balances.get(agent, 0.0) + reward
        self.total_paid += reward
        self.paid_tasks[task_id] = agent
        return {"task": task_id, "agent": agent, "paid": reward,
                "reason": "accepted"}

    def conservation_check(self) -> Dict:
        total_budget = sum(self.budgets.values())
        balance_sum = sum(self.balances.values())
        # 未支付预算仍在池子里：已付 + 罚没（负余额）与余额和对应
        return {
            "total_budget": total_budget,
            "total_paid": self.total_paid,
            "slashed": self.slashed,
            "balance_sum": balance_sum,
            "holds": total_budget - self.total_paid,
            "conserved": self.total_paid <= total_budget + 1e-9
                         and abs(balance_sum -
                                 (self.total_paid - self.slashed)) < 1e-9,
        }
