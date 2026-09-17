"""模拟与基准: 合成多 Agent 长会话 trace, 测命中率/重算/TTFT 代理/压缩率。"""
from __future__ import annotations
import random
from .policy import EvictPolicy
from .infinity import InfinityWindow
from .compression import compress_block, decompress_block, qat_status


def _zipf_trace(n: int, vocab: int, seed: int, zipf_s: float = 1.2):
    """Zipf 可配的局部性 trace。"""
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(vocab)]
    tot = sum(weights)
    weights = [w / tot for w in weights]
    cum = []
    acc = 0.0
    for w in weights:
        acc += w; cum.append(acc)
    out = []
    for _ in range(n):
        r = rng.random(); idx = 0
        while cum[idx] < r:
            idx += 1
        out.append(idx)
    return out


def _pos_int(v, name):
    try:
        iv = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 须为正整数")
    if iv <= 0:
        raise ValueError(f"{name} 须为正整数")
    return iv


def _int(v, name):
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 须为整数")


def run_simulation(config: dict | None = None) -> dict:
    if config is not None and not isinstance(config, dict):
        raise ValueError("配置须为 JSON 对象")
    cfg = config or {}
    n = _pos_int(cfg.get("n_tokens"), "n_tokens")
    vocab = _pos_int(cfg.get("vocab", 200), "vocab")
    seed = _int(cfg.get("seed", 7), "seed")
    window = _pos_int(cfg.get("window", 64), "window")
    trace = _zipf_trace(n, vocab, seed)

    win = InfinityWindow(window=window)
    for t in trace:
        win.access(t)
    hit = win.hit_rate
    recompute_misses = win.misses          # miss -> prefill 重算代理
    ttft_proxy = round(win.misses * 0.5 + win.hits * 0.01, 3)

    # 压缩率: 合成固定种子块
    rng = random.Random(seed)
    blob = bytes(rng.getrandbits(8) for _ in range(4096))
    cm = compress_block(blob)
    ok = decompress_block(cm["compressed"], cm["method"]) == blob

    return {
        "n_tokens": n, "vocab": vocab, "seed": seed, "window": window,
        "hit_rate": hit,
        "recompute_units": recompute_misses,
        "ttft_proxy_ms": ttft_proxy,
        "compression_ratio": cm["ratio"],
        "compression_method": cm["method"],
        "lossless_roundtrip": ok,
        "qat": qat_status(),
        "disclaimer": "analogy, not reproduction; 数字为本引擎合成 trace 自测, 非厂商数字",
    }
