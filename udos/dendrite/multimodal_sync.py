"""多模态时间戳对齐与重合检测(合成时间戳, 微秒级属生物/硬件口径, 标 analogy)。"""
from __future__ import annotations


def align_channels(channels: dict, tol_us: float = 1000.0) -> dict:
    """channels={name:[timestamps]}; 对齐到共同时间基准, 报告各通道 jitter。
    合成时间戳; '微秒级' 为生物/硬件口径, 引擎只做相对对齐。"""
    if not channels:
        raise ValueError("需要通道")
    ref = next(iter(channels.values()))
    ref_mean = sum(ref) / len(ref) if ref else 0.0
    out = {}
    for name, ts in channels.items():
        mean = sum(ts) / len(ts) if ts else 0.0
        jitter = mean - ref_mean
        aligned = [t - jitter for t in ts]
        out[name] = {"aligned": aligned, "jitter_us": round(jitter, 3)}
    return {"channels": out, "tol_us": tol_us,
            "disclaimer": "合成时间戳, analogy; 微秒级为生物/硬件口径"}
