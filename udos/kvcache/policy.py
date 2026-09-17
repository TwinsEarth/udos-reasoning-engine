"""冷热策略 LRU/LFU/TTL: pin/evict/offload/prefetch/promote/demote。"""
from __future__ import annotations
import time
from typing import Dict, List, Set


class EvictPolicy:
    def __init__(self, kind="lru"):
        self.kind = kind
        self.accesses: Dict[str, int] = {}
        self.last: Dict[str, float] = {}
        self.ttl: Dict[str, float] = {}
        self.pinned: Set[str] = set()

    def record(self, bid: str, now=None):
        self.accesses[bid] = self.accesses.get(bid, 0) + 1
        self.last[bid] = now if now is not None else time.time()

    def pin(self, bid): self.pinned.add(bid)
    def unpin(self, bid): self.pinned.discard(bid)

    def evict_candidates(self, exclude_hbm: Set[str]) -> List[str]:
        """按策略选可驱逐(非 pinned)。"""
        cands = [b for b in self.last if b not in self.pinned
                 and b not in exclude_hbm]
        if self.kind == "lfu":
            return sorted(cands, key=lambda b: self.accesses.get(b, 0))
        return sorted(cands, key=lambda b: self.last.get(b, 0))  # lru: 最久未用

    def should_promote_hot(self, bid: str, hot_thresh: int = 5) -> bool:
        return self.accesses.get(bid, 0) >= hot_thresh

    def should_demote_cold(self, bid: str, cold_age: float = 100.0,
                           now=None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.last.get(bid, 0)) > cold_age
