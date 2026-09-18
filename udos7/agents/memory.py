"""共享记忆（Shared Memory）—— v7.1 多 Agent 协同的唯一事实源。

能力：
- 命名空间 KV，带单调版本号与 CAS（compare-and-swap），用于无锁冲突检测；
- 只追加事件日志（Event Log），每个产物/决策可回溯父事件（血缘）；
- 资源声明表（Claim Registry），读写冲突即“碰撞”；
- 快照 / 合并，供跨协调器或云端共享上下文同步（传输层见 cloud.py）。

线程安全：Agent 的重活在线程池执行，故用 threading.RLock；asyncio 单线程内也安全。
不依赖网络；云端点未配置时同步走 LocalTransport 或显式报门禁错误。
"""
from __future__ import annotations

import copy
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .types import Claim, Collision, Decision, Event


class CollisionError(RuntimeError):
    """两个 Agent 对同一资源提交不兼容写声明。"""


class SharedMemory:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self._lock = threading.RLock()
        self._kv: Dict[str, Dict[str, Any]] = {}
        self._ver: Dict[str, int] = defaultdict(int)
        self._events: List[Event] = []
        self._claims: Dict[str, List[Claim]] = defaultdict(list)
        self._decisions: List[Decision] = []
        # 每个 agent 的私有草稿区
        self._scratch: Dict[str, Dict[str, Any]] = defaultdict(dict)

    # ---------- KV + CAS ----------
    def put(self, ns: str, key: str, value: Any, owner: str = "") -> int:
        with self._lock:
            full = f"{ns}/{key}"
            self._kv[full] = copy.deepcopy(value)
            self._ver[full] += 1
            self.append(Event(kind="kv.put", author=owner,
                              payload={"ns": ns, "key": key,
                                       "version": self._ver[full]}))
            return self._ver[full]

    def get(self, ns: str, key: str, default: Any = None) -> Any:
        with self._lock:
            full = f"{ns}/{key}"
            return copy.deepcopy(self._kv.get(full, default))

    def version(self, ns: str, key: str) -> int:
        with self._lock:
            return self._ver.get(f"{ns}/{key}", 0)

    def cas(self, ns: str, key: str, expected_version: int,
            value: Any, owner: str = "") -> Tuple[bool, int]:
        """仅当当前版本==expected_version 时写入；返回 (是否成功, 新版本)。"""
        with self._lock:
            full = f"{ns}/{key}"
            cur = self._ver.get(full, 0)
            if cur != expected_version:
                return False, cur
            self._kv[full] = copy.deepcopy(value)
            self._ver[full] = cur + 1
            self.append(Event(kind="kv.cas", author=owner,
                              payload={"ns": ns, "key": key,
                                       "version": cur + 1}))
            return True, cur + 1

    # ---------- 私有草稿（隔离工作区）----------
    def scratch_put(self, agent: str, key: str, value: Any) -> None:
        with self._lock:
            self._scratch[agent][key] = copy.deepcopy(value)

    def scratch_get_all(self, agent: str) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._scratch.get(agent, {}))

    def drop_scratch(self, agent: str) -> None:
        with self._lock:
            self._scratch.pop(agent, None)

    # ---------- 事件日志 ----------
    def append(self, event: Event) -> Event:
        with self._lock:
            self._events.append(event)
            return event

    def events(self, kind: Optional[str] = None) -> List[Event]:
        with self._lock:
            return [e for e in self._events if kind is None or e.kind == kind]

    def event_by_id(self, eid: str) -> Optional[Event]:
        with self._lock:
            return next((e for e in self._events if e.id == eid), None)

    # ---------- 资源声明 / 碰撞 ----------
    def declare(self, claim: Claim, raise_on_collision: bool = True
                ) -> List[Collision]:
        with self._lock:
            collisions: List[Collision] = []
            for existing in self._claims[claim.resource]:
                if existing.owner == claim.owner:
                    continue
                if claim.conflicts(existing):
                    c = Collision(resource=claim.resource,
                                  agents=sorted({claim.owner, existing.owner}),
                                  kind="write_write")
                    collisions.append(c)
            if collisions and raise_on_collision:
                raise CollisionError(
                    f"{claim.owner} 与 {collisions[0].agents} 在 "
                    f"{claim.resource} 上发生写碰撞")
            if not collisions:
                self._claims[claim.resource].append(claim)
            return collisions

    def claims_for(self, resource: str) -> List[Claim]:
        with self._lock:
            return list(self._claims.get(resource, []))

    def release(self, owner: str, resource: Optional[str] = None) -> None:
        with self._lock:
            resources = [resource] if resource else list(self._claims.keys())
            for r in resources:
                self._claims[r] = [c for c in self._claims[r]
                                   if not (c.owner == owner
                                           and (resource is None or c.resource == r))]

    # ---------- 决策 ----------
    def record_decision(self, d: Decision) -> Decision:
        with self._lock:
            self._decisions.append(d)
            self.append(Event(kind="decision", author="coordinator",
                              payload={"task": d.task_id, "winner": d.winner_id,
                                       "method": d.method, "id": d.id}))
            return d

    def decisions(self) -> List[Decision]:
        with self._lock:
            return list(self._decisions)

    # ---------- 快照 / 合并（云共享上下文基础）----------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "namespace": self.namespace,
                "kv": copy.deepcopy(self._kv),
                "ver": dict(self._ver),
                "events": [e.__dict__.copy() for e in self._events],
                "claims": [[c.__dict__.copy() for c in lst]
                           for lst in self._claims.values()],
                "claim_resources": list(self._claims.keys()),
                "decisions": [d.__dict__.copy() for d in self._decisions],
            }

    def merge(self, snap: Dict[str, Any], policy: str = "higher_version") -> int:
        """把对端快照并入；KV 按版本号取新，事件按 id 去重后按时间排序。

        返回新并入事件数。冲突 KV 在 higher_version 策略下以版本高者为准
        （真实云端应配合向量时钟；此处为单主聚合，足够本地多协调器与云代理使用）。
        """
        with self._lock:
            n_new = 0
            for full, val in snap.get("kv", {}).items():
                v = int(snap.get("ver", {}).get(full, 0))
                if v >= self._ver.get(full, 0):
                    self._kv[full] = copy.deepcopy(val)
                    self._ver[full] = v
            have = {e.id for e in self._events}
            for ed in snap.get("events", []):
                if ed["id"] not in have:
                    self._events.append(Event(**{
                        k: ed[k] for k in
                        ("kind", "author", "payload", "parent_ids", "id", "ts")
                        if k in ed}))
                    n_new += 1
            self._events.sort(key=lambda e: e.ts)
            return n_new
