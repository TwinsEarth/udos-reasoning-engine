#!/usr/bin/env python3
"""v4.5.6 资源注册表性能 A/B 基线 (CPU-only, 真实计时)。

测量: 注册表装配、probe 缓存命中、performance vs full 列表、L1 转换热路径。
不宣称未测项的提速; 结果落 benchmarks/results/resource_registry_v456.json。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.connectors import build_default_registry  # noqa: E402


def timeit(fn, n=20):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0   # ms/op


def main():
    out = {"version": "4.5.6", "env": "cpu-only", "unit": "ms/op"}

    # 1. 装配 79 条 (含 catalog 编译)
    out["build_registry_ms"] = round(timeit(lambda: build_default_registry("full"), 3), 3)

    reg = build_default_registry("full")

    # 2. probe 未缓存 vs 缓存 (单条)
    def fresh_probe():
        c = reg.get("diffusion_policy")
        c._report = None
        return c.probe()
    out["probe_uncached_ms"] = round(timeit(fresh_probe, 3), 4)
    reg.probe("diffusion_policy")  # 热
    out["probe_cached_ms"] = round(timeit(lambda: reg.probe("diffusion_policy"), 50), 4)

    # 3. 列表: performance vs full
    out["list_performance_ms"] = round(timeit(lambda: reg.list(), 20), 4)
    out["list_full_autoprobe_ms"] = round(timeit(lambda: reg.list(profile="full"), 3), 4)

    # 4. L1 转换热路径 (action chunk convert)
    payload = {"chunk": [[0.1, 0.2, 0.3]] * 8}
    out["convert_action_chunk_ms"] = round(timeit(
        lambda: reg.invoke("diffusion_policy", "convert", {"payload": payload}), 100), 5)

    # 5. profile 切换开销
    out["apply_profile_ms"] = round(timeit(
        lambda: reg.apply_profile("performance") or reg.apply_profile("full"), 10), 4)

    # 6. 状态分布 (真实探测)
    summ = reg.summary("full")
    out["summary_full"] = summ

    dest = ROOT / "benchmarks" / "results" / "resource_registry_v456.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
