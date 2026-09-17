"""KV Infinity: 长任务 working-set 窗口 + 滑动预取 + 远端 miss 回源。"""
from __future__ import annotations
from collections import deque


class InfinityWindow:
    def __init__(self, window: int = 8):
        self.window = window
        self._q: deque = deque()
        self.hits = 0
        self.misses = 0

    def access(self, token: int) -> bool:
        if token in self._q:
            self.hits += 1
            return True
        self.misses += 1
        self._q.append(token)
        if len(self._q) > self.window:
            self._q.popleft()      # 滑出窗口 -> 远端回源
        return False

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return round(self.hits / t, 4) if t else 0.0


def infinity_run(trace: list, window: int = 64, hbm_full: int = 512) -> dict:
    """按需加载/预取命中率 + HBM 常驻下降比例(相对全常驻)。"""
    if not trace:
        raise ValueError("需要 trace")
    w = InfinityWindow(window)
    for t in trace:
        w.access(t)
    resident = min(hbm_full, len(set(trace)))
    return {
        "hit_rate": w.hit_rate,
        "prefetch_hits": w.hits,
        "remote_misses": w.misses,
        "hbm_resident_blocks": resident,
        "hbm_drop_vs_full": round(1.0 - resident / hbm_full, 4),
        "disclaimer": "analogy, not reproduction",
    }
