"""协作治理三件套 —— collab_governance.py (v4.4.0, 多智能体协作线)
======================================================================
对应蓝图图2 三失败修复:
    * Owner   唯一负责人: 每任务有且仅有一个 active owner; 归属转移留痕。
    * Trace   事件链: (trace_id, 父事件/委派边, 责任人) 追加式事件, 可重建完整责任链,
              并能检测断裂 (某事件指向不存在的父 / 终态无收口人)。
    * Stop    停止条件: 清晰完成标准 + 最大跳数 + 循环检测, 防无限循环与重复劳动。
    * ClaimLock 任务认领锁: 同一 (task_key) 只被首个认领者执行, 去重防重复劳动。

设计纪律 (与全工程一致):
    * 纯算法、确定性、无可训练参数、零梯度; opt-in (不构造则旧路径逐位一致)。
    * 空 / 非法输入显式 ValueError, 不静默 inf 传播。
    * 日志沿用 logging_config (默认 WARNING, stderr)。
analogy, not reproduction —— 责任链为内存事件列表类比, 非分布式追踪系统。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("udos.collab_governance")


class OwnerLedger:
    """唯一负责人账本: 每 task 一个 active owner, 转移留痕。"""

    def __init__(self) -> None:
        self._current: Dict[str, str] = {}
        self._history: Dict[str, List[Dict[str, str]]] = {}

    def assign(self, task_id: str, owner: str, reason: str = "init") -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id 须为非空字符串")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner 须为非空字符串")
        prev = self._current.get(task_id)
        self._current[task_id] = owner
        self._history.setdefault(task_id, []).append(
            {"from": prev if prev is not None else "", "to": owner, "reason": reason})

    def current(self, task_id: str) -> Optional[str]:
        return self._current.get(task_id)

    def require_owner(self, task_id: str) -> str:
        """无 owner 不得启动 (治理责任不清失败)。"""
        o = self._current.get(task_id)
        if o is None:
            raise ValueError(f"任务 {task_id} 无 owner —— 拒绝启动 (责任不清)")
        return o

    def history(self, task_id: str) -> List[Dict[str, str]]:
        return list(self._history.get(task_id, []))


class TraceChain:
    """追加式 trace 事件链: 可重建责任链并检测断裂。"""

    def __init__(self) -> None:
        # event_id -> event dict
        self._events: Dict[str, Dict[str, Any]] = {}
        # trace_id -> [event_id] (插入序稳定)
        self._order: Dict[str, List[str]] = {}
        self._counter = 0

    def append(self, trace_id: str, actor: str, action: str,
               parent_event: Optional[str] = None,
               detail: Optional[Dict[str, Any]] = None) -> str:
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("trace_id 须为非空字符串")
        if not isinstance(actor, str) or not actor:
            raise ValueError("actor 须为非空字符串")
        self._counter += 1
        eid = f"{trace_id}:e{self._counter}"
        if parent_event is not None and parent_event not in self._events:
            raise ValueError(
                f"trace 断裂: 事件 {eid} 指向不存在的父事件 {parent_event}")
        self._events[eid] = {
            "event_id": eid, "trace_id": trace_id, "actor": actor,
            "action": action, "parent": parent_event,
            "detail": dict(detail or {}), "ts": time.time(),
        }
        self._order.setdefault(trace_id, []).append(eid)
        return eid

    def chain(self, trace_id: str) -> List[Dict[str, Any]]:
        """返回该 trace 的完整事件序列 (插入序)。"""
        return [self._events[e] for e in self._order.get(trace_id, [])]

    def responsible_actors(self, trace_id: str) -> List[str]:
        """沿事件链出现过的责任人 (去重, 保序)。"""
        seen: List[str] = []
        for e in self.chain(trace_id):
            if e["actor"] not in seen:
                seen.append(e["actor"])
        return seen

    def find_closer(self, trace_id: str) -> Optional[str]:
        """找收口人: 最后一个 action 为 'close'/'return' 的事件责任人; 否则 None。"""
        closer = None
        for e in self.chain(trace_id):
            if e["action"] in ("close", "return", "finalize"):
                closer = e["actor"]
        return closer

    def integrity_report(self, trace_id: str) -> Dict[str, Any]:
        """责任链完整性自检: 有事件、无悬空父、终态有唯一收口人。"""
        evs = self.chain(trace_id)
        n = len(evs)
        dangling = [e["event_id"] for e in evs
                    if e["parent"] is not None and e["parent"] not in self._events]
        closer = self.find_closer(trace_id)
        return {
            "trace_id": trace_id,
            "n_events": n,
            "has_events": n > 0,
            "dangling_parents": dangling,
            "chain_intact": n > 0 and not dangling,
            "closer": closer,
            "closed": closer is not None,
            "responsible_actors": self.responsible_actors(trace_id),
        }


class StopGuard:
    """停止条件: 最大跳数 + 循环检测 + 显式 stop 谓词。"""

    def __init__(self, max_hops: int = 8) -> None:
        if not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_hops 须为 >=1 整数")
        self.max_hops = int(max_hops)
        self._hops: Dict[str, int] = {}
        # 循环检测: task_key -> 已访问的状态指纹集合
        self._visited: Dict[str, List[Any]] = {}

    def require_stop_defined(self, has_stop: bool) -> None:
        if not has_stop:
            raise ValueError("任务无 stop condition —— 拒绝启动 (责任不清/无限循环风险)")

    def tick(self, task_key: str, state_fingerprint: Any = None) -> Dict[str, Any]:
        """记录一跳; 超 max_hops 或状态指纹重复 => 停止。"""
        self._hops[task_key] = self._hops.get(task_key, 0) + 1
        hops = self._hops[task_key]
        looped = False
        if state_fingerprint is not None:
            seen = self._visited.setdefault(task_key, [])
            if state_fingerprint in seen:
                looped = True
            else:
                seen.append(state_fingerprint)
        over_hops = hops > self.max_hops
        return {"hops": hops, "max_hops": self.max_hops,
                "over_hops": over_hops, "loop_detected": looped,
                "stop": over_hops or looped}

    def reset(self, task_key: str) -> None:
        self._hops.pop(task_key, None)
        self._visited.pop(task_key, None)


class ClaimLock:
    """任务认领锁: 同一 task_key 只被首个调用方认领执行 (去重防重复劳动)。"""

    def __init__(self) -> None:
        self._claimed: Dict[str, str] = {}

    def try_claim(self, task_key: str, worker: str) -> bool:
        if not isinstance(task_key, str) or not task_key:
            raise ValueError("task_key 须为非空字符串")
        if task_key in self._claimed:
            return False  # 已被别人认领, 去重
        self._claimed[task_key] = worker
        return True

    def owner(self, task_key: str) -> Optional[str]:
        return self._claimed.get(task_key)

    def release(self, task_key: str) -> None:
        self._claimed.pop(task_key, None)

    def claimed_keys(self) -> List[str]:
        return list(self._claimed.keys())
