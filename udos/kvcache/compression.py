"""软件无损压缩(zlib; zstd 可用则用)。QAT 仅接口契约, 无硬件 503/ENV_BLOCKED。

绝不返回伪造的"约2倍"加速。
"""
from __future__ import annotations
import zlib

try:
    import zstandard as _zstd
    ZSTD_AVAILABLE = True
except Exception:
    _zstd = None
    ZSTD_AVAILABLE = False


def qat_status() -> dict:
    """QAT 硬件状态。本沙箱无 QAT -> ENV_BLOCKED, 不伪造加速。"""
    return {"qat": "ENV_BLOCKED",
            "message": "无 QAT 硬件; 仅软件 zlib/zstd 路径, 不返回伪造加速"}


def compress_block(data: bytes) -> dict:
    """无损压缩; 返回 compressed 字节 + ratio + 耗时代理。"""
    if ZSTD_AVAILABLE:
        c = _zstd.ZstdCompressor().compress(data)
        method = "zstd"
    else:
        c = zlib.compress(data, 6)
        method = "zlib"
    ratio = len(data) / len(c) if c else 1.0
    return {"compressed": c, "method": method, "ratio": round(ratio, 3),
            "orig": len(data), "comp": len(c)}


def decompress_block(c: bytes, method: str = "zlib") -> bytes:
    if method == "zstd" and ZSTD_AVAILABLE:
        return _zstd.ZstdDecompressor().decompress(c)
    return zlib.decompress(c)


def layout_reorder(data: bytes) -> bytes:
    """存储格式重排(提升可压缩性的可测变换); 这里做确定性字节分块重排。"""
    n = len(data)
    if n == 0:
        return data
    half = n // 2
    return data[half:] + data[:half]
