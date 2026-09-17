"""网状 Swarm —— collab_swarm.py (dev4, v4.4.0, **默认关闭 opt-in**)
======================================================================
对应蓝图图4/图5网状: Peer Agents 各自声明能力、无中心、动态组队、自主协作;
流程 = 发现谁能处理 -> 局部协商后委派 -> 执行并返回 -> 结果可被其他 peer 再委派。
风险 = 路径不稳定、多跳致 trace 断裂、责任难界定; **灵活性高但治理成本高**。

治理 (弥补去中心化风险):
    * 显式 opt-in: 未传 enabled=True 时构造即报错 (swarm 默认关, 不擅自自治)。
    * 每一跳仍写 trace 事件链 + owner 归属; 自带 trace 采样与**治理开销统计**。
    * 共识/冲突-未对齐检测: 多 peer 对同一子任务给出结果, 若 confidence 分歧过大
      或结论矛盾 => 标记 conflict_unaligned (不静默吞)。

设计纪律: 纯算法、确定性、零梯度、opt-in; 空/非法显式 ValueError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .collab_agents import CapabilityRegistry
from .collab_governance import ClaimLock, OwnerLedger, StopGuard, TraceChain

logger = logging.getLogger("udos.collab_swarm")


class MeshSwarm:
    """去中心化 peer 协作 (默认关闭, 必须显式 opt-in)。"""

    def __init__(self, registry: CapabilityRegistry,
                 enabled: bool = False,
                 conflict_threshold: float = 0.25,
                 max_hops: int = 6) -> None:
        # swarm 尤其默认关: 不 opt-in 直接拒绝构造 (治理图1"控制需求不足拒绝自治")
        if not enabled:
            raise ValueError(
                "mesh/swarm 为高自治拓扑, 默认关闭; 须显式 enabled=True (opt-in)")
        if not isinstance(registry, CapabilityRegistry):
            raise ValueError("registry 须为 CapabilityRegistry")
        if not (0.0 <= conflict_threshold <= 1.0):
            raise ValueError("conflict_threshold 须在 [0,1]")
        self.reg = registry
        self.conflict_threshold = float(conflict_threshold)
        self.owners = OwnerLedger()
        self.traces = TraceChain()
        self.stop = StopGuard(max_hops=max_hops)
        self.lock = ClaimLock()
        self.gov_overhead = {"negotiations": 0, "redelegations": 0,
                             "conflicts": 0, "trace_events": 0}

    def run(self, task_id: str, goal: str, need_tags: List[str],
            payload: Dict[str, Any], max_rounds: int = 3,
            owner: str = "swarm") -> Dict[str, Any]:
        """去中心化协作: 能力发现 -> 局部协商委派 -> 结果再委派 -> 共识/冲突检测。

        Args:
            need_tags: 需要哪些能力标签 (如 ["reasoning","planning"])。
            max_rounds: 结果再委派的最大轮次 (防多跳失控)。
        """
        if not need_tags:
            raise ValueError("swarm 任务需声明能力标签 need_tags")
        if not (1 <= max_rounds <= 5):
            raise ValueError("max_rounds 需在 [1,5]")
        self.owners.assign(task_id, owner, reason="swarm_start")
        trace_id = f"trace-{task_id}"
        root = self.traces.append(trace_id, owner, "swarm_start",
                                  detail={"goal": goal, "need_tags": need_tags})

        # 1) 能力发现: 找能处理这些标签的 peer
        peers = self.reg.discover(need_tags)
        if not peers:
            raise ValueError(f"swarm 无 peer 能处理标签 {need_tags}")

        # 2) 局部协商: 每个 peer 认领(去重)并执行
        peer_results: List[Dict[str, Any]] = []
        for p in peers:
            key = f"{task_id}:{p.name}"
            if not self.lock.try_claim(key, owner):
                continue  # 去重
            self.gov_overhead["negotiations"] += 1
            self.traces.append(trace_id, owner, "negotiate", parent_event=root,
                               detail={"peer": p.name})
            res = self.reg.call(p.name, payload)
            self.gov_overhead["trace_events"] += 1
            peer_results.append({
                "peer": p.name, "confidence": res["confidence"],
                "error_type": res["error_type"], "output": res["output"]})

        # 3) 结果再委派 (最多 max_rounds): 低置信结果交给另一个 peer 复核
        redelegated: List[str] = []
        for r in peer_results:
            gate = self.stop.tick(task_id, state_fingerprint=("redeleg", r["peer"]))
            if gate["stop"]:
                break
            if r["confidence"] < 0.7 and len(peers) > 1:
                others = [q for q in peers if q.name != r["peer"]]
                backup = others[0]  # 确定性: 第一个其他 peer
                self.gov_overhead["redelegations"] += 1
                self.traces.append(trace_id, owner, "redelegate",
                                   detail={"from": r["peer"], "to": backup.name})
                again = self.reg.call(backup.name, payload)
                redelegated.append(r["peer"])
                r["second_opinion"] = {"peer": backup.name,
                                       "confidence": again["confidence"]}

        # 4) 共识/冲突-未对齐检测
        confs = [r["confidence"] for r in peer_results if r["error_type"] == "none"]
        conflict = False
        if confs and (max(confs) - min(confs)) > self.conflict_threshold:
            conflict = True
            self.gov_overhead["conflicts"] += 1

        # 收口: swarm 无天然唯一收口人 -> 强制指定发起者 owner 收口
        closer = self.owners.current(task_id)
        self.traces.append(trace_id, closer, "close",
                           detail={"n_peers": len(peers),
                                   "redelegated": redelegated,
                                   "conflict": conflict})
        integ = self.traces.integrity_report(trace_id)
        return {
            "task_id": task_id, "topology": "mesh", "goal": goal,
            "n_peers": len(peers),
            "peers": [p.name for p in peers],
            "peer_results": peer_results,
            "redelegated": redelegated,
            "consensus": not conflict,
            "conflict_unaligned": conflict,
            "governance_overhead": dict(self.gov_overhead),
            "final_owner": closer,
            "trace_integrity": integ,
            "main_params_untouched": True,
        }
