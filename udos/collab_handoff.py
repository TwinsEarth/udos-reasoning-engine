"""链式 Handoff —— collab_handoff.py (dev3, v4.4.0)
======================================================================
对应蓝图图7: Handoff = **转移责任不是转发消息**。流程 Triage->Specialist->Return:
    * Triage:    判断任务归哪个专业(路由表), 不解决问题, 只决定下一棒。
    * Specialist: 带 Transfer Bundle(五要素) 接手, 调对应能力。
    * Return:    把结果返回 triage/或继续交接; 失败可**回退/补救**到上一棒。

每次交接都: validate TransferBundle(缺关键要素拒交接) + owner 责任转移留痕 +
trace 事件链追加。StopGuard 限跳数防无限链式转接。

设计纪律: 纯算法、确定性、零梯度、opt-in; 空/非法显式 ValueError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .collab_agents import CapabilityRegistry
from .collab_governance import OwnerLedger, StopGuard, TraceChain
from .transfer_bundle import TransferBundle

logger = logging.getLogger("udos.collab_handoff")


class ChainHandoff:
    """Triage -> Specialist -> Return 链式交接。"""

    def __init__(self, registry: CapabilityRegistry,
                 routes: Dict[str, str], max_hops: int = 8) -> None:
        """routes: {专业键 -> 已注册能力名}。 如 {"billing": "wla_action"}。"""
        if not isinstance(registry, CapabilityRegistry):
            raise ValueError("registry 须为 CapabilityRegistry")
        if not isinstance(routes, dict) or not routes:
            raise ValueError("routes 须为非空 dict {specialty: tool}")
        for spec, tool in routes.items():
            if not registry.has(tool):
                raise ValueError(f"路由 {spec} 指向未注册能力 {tool}")
        self.reg = registry
        self.routes = dict(routes)
        self.owners = OwnerLedger()
        self.traces = TraceChain()
        self.stop = StopGuard(max_hops=max_hops)

    def _triage(self, specialty: str) -> str:
        if specialty not in self.routes:
            raise ValueError(f"triage 未知专业: {specialty} (已注册 {list(self.routes)})")
        return self.routes[specialty]

    def run(self, task_id: str, goal: str,
            specialty_chain: List[str], payload: Dict[str, Any],
            owner: str = "triage", has_stop: bool = True,
            context: str = "") -> Dict[str, Any]:
        """沿 specialty_chain 逐棒交接执行。

        Args:
            specialty_chain: 专业键序列, 如 ["inquiry","billing","return"];
                             最后一棒返回 triage 收口。
        """
        self.stop.require_stop_defined(has_stop)
        if not specialty_chain:
            raise ValueError("chain 至少需要 1 个交接棒")
        self.owners.assign(task_id, owner, reason="start")
        trace_id = f"trace-{task_id}"
        bundle = TransferBundle(
            goal=goal, trace={"trace_id": trace_id, "owner": owner},
            context=context)

        chain_log: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        failed_spec: Optional[str] = None

        prev_event = self.traces.append(trace_id, owner, "triage",
                                        detail={"goal": goal})
        for hop, spec in enumerate(specialty_chain):
            # 跳数/循环硬门
            gate = self.stop.tick(task_id, state_fingerprint=(spec, hop))
            if gate["stop"]:
                # 补救/回退: 强制收口, 不在链上无限打转
                self.traces.append(trace_id, self.owners.current(task_id),
                                   "stop", parent_event=prev_event,
                                   detail={"reason": "max_hops/loop",
                                           "hops": gate["hops"]})
                failed_spec = spec
                break

            tool = self._triage(spec)
            # 责任转移: 这一棒 owner 变成 specialist
            self.owners.assign(task_id, f"specialist:{spec}", reason=f"handoff->{spec}")
            bundle.reassign_owner(f"specialist:{spec}", reason=f"handoff@{hop}")
            # 交接硬校验: 缺关键要素拒交接(状态丢失)
            bundle.validate()
            ev = self.traces.append(
                trace_id, f"specialist:{spec}", "handoff",
                parent_event=prev_event, detail={"tool": tool})
            res = self.reg.call(tool, payload)
            bundle.record_done(f"{spec}->{tool}:{res['error_type']}")
            chain_log.append({"hop": hop, "specialty": spec, "tool": tool,
                              "error_type": res["error_type"]})
            results.append({"hop": hop, "specialty": spec, "tool": tool,
                            "output": res["output"],
                            "confidence": res["confidence"],
                            "error_type": res["error_type"]})
            prev_event = ev
            if res["error_type"] != "none":
                # 补救路径: 该棒失败 -> 回退 triage 收口, 不盲目继续转
                failed_spec = spec
                break

        # Return: 回到 triage 收口 (唯一收口人)
        closer = owner
        self.owners.assign(task_id, closer, reason="return")
        bundle.reassign_owner(closer, reason="return")
        self.traces.append(trace_id, closer, "return", parent_event=prev_event,
                           detail={"recovered": failed_spec is None,
                                   "failed_specialty": failed_spec})
        integ = self.traces.integrity_report(trace_id)
        return {
            "task_id": task_id, "topology": "chain", "goal": goal,
            "chain": chain_log, "results": results,
            "failed_specialty": failed_spec,
            "recovered": failed_spec is None,
            "final_owner": closer,
            "owner_transfers": self.owners.history(task_id),
            "bundle": bundle.to_dict(),
            "trace_integrity": integ,
            "main_params_untouched": True,
        }
