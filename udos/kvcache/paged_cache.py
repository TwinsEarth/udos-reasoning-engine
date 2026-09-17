"""PagedAttention 风格块表: 逻辑→物理映射 + 引用计数 + 前缀共享。"""
from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Optional


class PagedCache:
    def __init__(self):
        self.refcount: Dict[str, int] = {}
        self.physical: Dict[str, List[str]] = {}   # seq_id -> [block_id]
        self._lru: "OrderedDict[str, None]" = OrderedDict()

    def allocate(self, seq_id: str, blocks: List[str]) -> None:
        self.physical.setdefault(seq_id, [])
        for b in blocks:
            if b not in self.physical[seq_id]:
                self.physical[seq_id].append(b)
                self.refcount[b] = self.refcount.get(b, 0) + 1

    def share_prefix(self, seq_id: str, prefix_seq: str) -> int:
        """前缀共享: 复用 prefix_seq 的物理块, 返回共享块数。"""
        shared = 0
        if prefix_seq in self.physical:
            for b in self.physical[prefix_seq]:
                if b not in self.physical.get(seq_id, []):
                    self.allocate(seq_id, [b])
                    shared += 1
        return shared

    def access(self, block_id: str) -> None:
        if block_id in self._lru:
            self._lru.move_to_end(block_id)
        else:
            self._lru[block_id] = None

    def evict_lru(self, pinned: set) -> Optional[str]:
        for b in list(self._lru):
            if self.refcount.get(b, 0) <= 1 and b not in pinned:
                self._lru.pop(b)
                return b
        return None

    def free(self, block_id: str) -> None:
        if block_id in self.refcount:
            self.refcount[block_id] -= 1
