"""分页 KV 块 + 多级后端存储。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Block:
    block_id: str
    token_start: int
    token_end: int
    layer: int
    bytes_n: int
    precision: str = "bf16"
    pinned: bool = False

    @property
    def ntokens(self):
        return self.token_end - self.token_start


class BlockStore:
    """多级后端 {tier: 容量}; 每级块集合, 记录字节占用。"""

    def __init__(self, capacities_mb: Optional[Dict[str, float]] = None):
        caps = capacities_mb or {"hbm": 256.0, "ddr": 2048.0,
                                  "ssd": 8192.0, "remote": 1e9}
        self.capacity = {k: v * 1e6 for k, v in caps.items()}
        self.used = {k: 0.0 for k in caps}
        self.blocks: Dict[str, Block] = {}
        self.loc: Dict[str, str] = {}

    def put(self, block: Block, tier: str) -> bool:
        if self.used[tier] + block.bytes_n > self.capacity[tier]:
            return False
        self.blocks[block.block_id] = block
        self.loc[block.block_id] = tier
        self.used[tier] += block.bytes_n
        return True

    def move(self, bid: str, to_tier: str) -> bool:
        if bid not in self.loc:
            return False
        frm = self.loc[bid]
        blk = self.blocks[bid]
        if self.used[to_tier] + blk.bytes_n > self.capacity[to_tier]:
            return False
        self.used[frm] -= blk.bytes_n
        self.used[to_tier] += blk.bytes_n
        self.loc[bid] = to_tier
        return True

    def evict(self, bid: str) -> bool:
        if bid not in self.loc:
            return False
        blk = self.blocks[bid]
        self.used[self.loc[bid]] -= blk.bytes_n
        del self.blocks[bid]
        del self.loc[bid]
        return True

    def tier_of(self, bid: str) -> Optional[str]:
        return self.loc.get(bid)
