"""udos.kvcache — v5.1.0 推理记忆分层与调度机制原型(analogy, not reproduction)。

CPU-only 纯逻辑/零梯度/零训练。用合成长会话 trace + 显式参数化设备模型,
测量多级缓存/卸载/冷热调度/压缩/保留vs重算账本。opt-in: UDOS_KVCACHE=on。
不下载权重、不伪造 GPU/QAT 硬件数字。
"""
from .device import DeviceBackend, cpu_backend, gpu_available
from .block_store import Block, BlockStore
from .paged_cache import PagedCache
from .policy import EvictPolicy
from .compression import compress_block, decompress_block, qat_status
from .cost_ledger import decide_action
from .fuse import fuse_overlap
from .cascade import CascadeRouter
from .infinity import InfinityWindow
from .sim import run_simulation

__all__ = [
    "DeviceBackend", "cpu_backend", "gpu_available",
    "Block", "BlockStore", "PagedCache", "EvictPolicy",
    "compress_block", "decompress_block", "qat_status",
    "decide_action", "fuse_overlap", "CascadeRouter", "InfinityWindow",
    "run_simulation",
]
