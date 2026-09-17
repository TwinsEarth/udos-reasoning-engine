"""
UDOS 分级 Debug 面板
====================
调试默认关闭, UDOS_DEBUG=1 显式开启, 无需任何口令。
    L0: 静默
    L1: 形状 / 参数量 / 时延 (概览)
    L2: + 同步分数、certainty 轨迹、LoRA 注入清单
    L3: + 张量数值统计 (均值/方差/范数/极值)

用法:
    import os; os.environ["UDOS_DEBUG"] = "1"
    from udos.debug import DebugPanel
    dbg = DebugPanel(level=2)
    with dbg.section("CTM forward"):
        dbg.shape("physical_tokens", x)

本模块不挂载任何网络路由, 不接受口令解锁; 输出走 stderr。
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch

logger = logging.getLogger("udos.debug")


class DebugPanel:
    _global: Optional["DebugPanel"] = None

    def __init__(self, level: int = 1, enabled: Optional[bool] = None):
        # enabled 显式参数优先; 缺省读 UDOS_DEBUG (默认 0=关闭)。无口令。
        if enabled is None:
            enabled = os.environ.get("UDOS_DEBUG", "0") not in ("0", "", "off")
        self.level = max(0, min(3, int(level)))
        self.enabled = bool(enabled)
        self._records: List[str] = []
        DebugPanel._global = self

    @classmethod
    def global_panel(cls) -> Optional["DebugPanel"]:
        return cls._global

    # ---------- 基础输出 (走 stderr, 不写 stdout/响应) ----------
    def _emit(self, msg: str, need_level: int) -> None:
        if self.enabled and self.level >= need_level:
            line = f"[UDOS-DBG L{need_level}] {msg}"
            self._records.append(line)
            logger.info("%s", line)

    @contextmanager
    def section(self, name: str):
        if not (self.enabled and self.level >= 1):
            yield
            return
        t0 = time.perf_counter()
        self._emit(f"▶ {name} 开始", 1)
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000
            self._emit(f"■ {name} 完成, 耗时 {dt:.2f} ms", 1)

    # ---------- L1: 形状/参数量 ----------
    def shape(self, name: str, tensor: torch.Tensor) -> None:
        self._emit(f"{name}.shape = {tuple(tensor.shape)} dtype={tensor.dtype}", 1)

    def num_params(self, name: str, module: torch.nn.Module) -> int:
        n = sum(p.numel() for p in module.parameters())
        self._emit(f"{name} 参数量 = {n:,} ({n * 4 / 1024:.1f} KB fp32)", 1)
        return n

    # ---------- L2: 轨迹/同步 ----------
    def trace(self, name: str, values: Sequence[float] | torch.Tensor) -> None:
        if isinstance(values, torch.Tensor):
            values = values.detach().float().flatten().tolist()
        values = list(values)
        if not values:
            return
        fmt = " -> ".join(f"{v:.3f}" for v in values[:12])
        more = "" if len(values) <= 12 else f" ... (共{len(values)}点)"
        self._emit(f"{name} 轨迹: {fmt}{more}", 2)

    def kv(self, name: str, value, level: int = 2) -> None:
        self._emit(f"{name} = {value}", level)

    # ---------- L3: 张量统计 ----------
    def stats(self, name: str, tensor: torch.Tensor) -> Dict[str, float]:
        t = tensor.detach().float()
        stat = {
            "mean": t.mean().item(),
            "std": t.std().item() if t.numel() > 1 else 0.0,
            "norm": t.norm().item(),
            "min": t.min().item(),
            "max": t.max().item(),
        }
        self._emit(
            f"{name} 统计 mean={stat['mean']:+.4f} std={stat['std']:.4f} "
            f"norm={stat['norm']:.4f} min={stat['min']:+.4f} max={stat['max']:+.4f}",
            3,
        )
        return stat

    def dump(self) -> List[str]:
        return list(self._records)
