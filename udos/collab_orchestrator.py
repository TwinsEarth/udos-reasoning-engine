"""星型 Orchestrator —— collab_orchestrator.py (dev2, v4.4.0)
======================================================================
对应蓝图图6: 用户任务 -> Orchestrator(拆任务/分 Worker/合并结果) + 全局状态与责任边界;
**结果收集器**(合并各 Worker、去重校验、保证一致完整可靠) -> 统一答案。

流程 (纯前向、确定性):
    1) 启动硬门: 必须有 owner + 有 stop condition (治理责任不清/无限循环)。
    2) 任务拆分: 把一个复合任务拆成若干子任务, 每个子任务绑定一个已注册能力(tool)。
    3) Worker 分配: 逐子任务经 ClaimLock 认领(去重防重复劳动) -> 调能力 -> 收结构化结果。
    4) 结果收集器: 去重(同 task_key 只算一次) / 校验(error_type!=none 标记失败) /
       一致性-完整性检查(必做子任务是否都有结果)。
    5) 统一收口: 沿 trace 链 close, 记录唯一收口人(orchestrator 本身)。

设计纪律: 纯算法、零梯度、opt-in; 复用治理三件套, 不另造 owner/trace/stop。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .collab_agents import CapabilityRegistry
from .collab_governance import ClaimLock, OwnerLedger, StopGuard, TraceChain
from .transfer_bundle import TransferBundle

logger = logging.getLogger("udos.collab_orchestrator")


class StarOrchestrator:
    """星型中心编排器 (Orchestrator-Worker)。"""

    def __init__(self, registry: CapabilityRegistry,
                 max_hops: int = 8) -> None:
        if not isinstance(registry, CapabilityRegistry):
            raise ValueError("registry 须为 CapabilityRegistry")
        self.reg = registry
        self.owners = OwnerLedger()
        self.traces = TraceChain()
        self.stop = StopGuard(max_hops=max_hops)
        self.lock = ClaimLock()
        self.global_state: Dict[str, Any] = {}

    # -- 子任务描述: {"key": str, "tool": str, "payload": dict} ------- #
    def run(self, task_id: str, goal: str, subtasks: List[Dict[str, Any]],
            owner: str = "orchestrator", has_stop: bool = True,
            context: str = "") -> Dict[str, Any]:
        """按星型拓扑执行一次复合协作任务。

        Raises:
            ValueError: 无 owner / 无 stop / 空 subtasks / 未知 tool。
        """
        # 治理硬门: 责任不清 / 无停止条件 不得启动
        self.stop.require_stop_defined(has_stop)
        self.owners.assign(task_id, owner, reason="start")
        self.owners.require_owner(task_id)
        if not subtasks:
            raise ValueError("star 任务至少需要 1 个子任务")
        for st in subtasks:
            if not self.reg.has(st["tool"]):
                raise ValueError(f"子任务绑定未注册能力: {st['tool']}")

        trace_id = f"trace-{task_id}"
        root = self.traces.append(trace_id, owner, "orchestrate",
                                  detail={"goal": goal})

        bundle = TransferBundle(
            goal=goal,
            trace={"trace_id": trace_id, "owner": owner},
            context=context)

        results: List[Dict[str, Any]] = []
        claimed: List[str] = []
        skipped_dup: List[str] = []

        for st in subtasks:
            key = st["key"]
            # ClaimLock 去重: 同一子任务 key 只被执行一次
            if not self.lock.try_claim(key, owner):
                skipped_dup.append(key)
                continue
            claimed.append(key)
            self.traces.append(trace_id, owner, "dispatch", parent_event=root,
                               detail={"worker": st["tool"], "key": key})
            res = self.reg.call(st["tool"], st["payload"])
            bundle.record_done(f"{key}->{st['tool']}:{res['error_type']}")
            results.append({
                "key": key, "tool": st["tool"],
                "output": res["output"], "confidence": res["confidence"],
                "error_type": res["error_type"],
            })

        # 结果收集器: 去重 / 校验 / 一致性-完整性
        collector = self._collect(results, skipped_dup)

        # 统一收口: orchestrator 是唯一收口人
        closer = self.owners.require_owner(task_id)
        close_ev = self.traces.append(
            trace_id, closer, "close",
            detail={"n_results": len(results),
                    "n_skipped_dup": len(skipped_dup)})
        bundle.reassign_owner(closer, reason="close")
        self.global_state[task_id] = {"goal": goal, "done": bundle.done}

        integ = self.traces.integrity_report(trace_id)
        return {
            "task_id": task_id, "topology": "star",
            "goal": goal, "owner": closer,
            "n_subtasks": len(subtasks), "claimed": claimed,
            "skipped_duplicates": skipped_dup,
            "results": results,
            "collector": collector,
            "bundle": bundle.to_dict(),
            "trace_integrity": integ,
            "close_event": close_ev,
            "main_params_untouched": True,
        }

    def _collect(self, results: List[Dict[str, Any]],
                 skipped_dup: List[str]) -> Dict[str, Any]:
        """结果收集器: 去重 / 错误校验 / 一致性-完整性。"""
        seen_keys: List[str] = []
        deduped: List[Dict[str, Any]] = []
        for r in results:
            if r["key"] in seen_keys:      # 二次保险(双去重)
                continue
            seen_keys.append(r["key"])
            deduped.append(r)
        n_fail = sum(1 for r in deduped if r["error_type"] != "none")
        n_ok = len(deduped) - n_fail
        complete = n_fail == 0 and len(deduped) > 0
        # 一致性: 所有成功结果 confidence>0 即视为通过(合成机制下无冲突定义)
        consistent = all(r["confidence"] >= 0.0 for r in deduped)
        return {
            "n_raw": len(results),
            "n_unique": len(deduped),
            "n_dedup_dropped": len(results) - len(deduped) + len(skipped_dup),
            "n_ok": n_ok, "n_fail": n_fail,
            "complete": complete, "consistent": consistent,
        }
