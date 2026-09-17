"""
UDOS 性能基准 + 回归守卫 (无测量, 不结论)
========================================
固定 workload / seed, warmup 后多次采样, 报告 wall-time 的
min/median/mean/max、参数量与进程峰值 RSS; 结果同时落 JSON。

用法:
    python -m benchmarks.benchmark                 # 跑基准并打印
    python -m benchmarks.benchmark --guard         # 与门槛比较, 超限退出码=1
    python -m benchmarks.benchmark --json out.json
门槛为工程自定的 CPU 宽松上界 (基线中位数的约 2 倍), 非外部标准,
见 GUARD_MS; 仅用于捕获显著性能回归, 不用作性能宣传。
"""

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udos.ctm_engine import CTMConfig, CTMPhysicsEngine
from udos.gpm_engine import (
    GPMConfig, PhysicsHypernetwork, LoRAInjector, TinyBaseModel,
    infer_dims_from_model,
)
from udos.pce_format import PhysicsSceneEncoder
from demos.scene_factory import build_factory_scene

# 工程自定回归门槛 (毫秒, CPU), 超过即视为显著回归
GUARD_MS = {
    "ctm_small_forward": 300.0,
    "ctm_medium_forward": 1500.0,
    "gpm_small_internalize": 1000.0,
    "e2e_small": 2000.0,
}


def _time_it(fn, repeats: int = 20, warmup: int = 3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "min_ms": round(min(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "max_ms": round(max(samples), 3),
        "samples": repeats,
    }


def _peak_rss_mb():
    # ru_maxrss: Linux 单位 KB
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _num_params(m):
    return sum(p.numel() for p in m.parameters())


def bench_ctm(size: str, seq_tokens: int, repeats: int):
    cfgs = {
        "small": CTMConfig(iterations=16, d_model=128, d_input=64, heads=4,
                           n_synch_out=32, n_synch_action=32, memory_length=12,
                           out_dims=64, certainty_threshold=0.0),
        "medium": CTMConfig(iterations=32, d_model=256, d_input=128, heads=4,
                            n_synch_out=64, n_synch_action=32, memory_length=16,
                            out_dims=128, certainty_threshold=0.0),
    }
    torch.manual_seed(0)
    eng = CTMPhysicsEngine(cfgs[size]).eval()
    x = torch.randn(1, seq_tokens, cfgs[size].d_input)
    with torch.no_grad():
        stat = _time_it(lambda: eng(x), repeats=repeats)
    stat["params"] = _num_params(eng)
    return stat


def bench_gpm(size: str, repeats: int):
    scene = build_factory_scene(n_steps=12)
    cfgs = {
        "small": dict(width=64, n_layers=2, rank=4),
        "medium": dict(width=128, n_layers=4, rank=8),
    }[size]
    torch.manual_seed(0)
    base = TinyBaseModel(cfgs["width"], cfgs["n_layers"])
    targets = ("down_proj", "gate_proj", "up_proj")
    cfg = GPMConfig(feature_dim=cfgs["width"], latent_size=cfgs["width"],
                    n_latents=16, lora_rank=cfgs["rank"], target_modules=targets,
                    layer_indices=tuple(range(cfgs["n_layers"])),
                    dims=infer_dims_from_model(base, targets),
                    init_scaler_b_zero=False)
    hyper = PhysicsHypernetwork(cfg).eval()
    with torch.no_grad():
        stat = _time_it(lambda: hyper(scene), repeats=repeats)
    lora = hyper(scene)
    stat["hyper_params"] = _num_params(hyper)
    stat["lora_params"] = lora.num_params()
    stat["lora_kb_fp32"] = round(lora.num_bytes_fp32() / 1024, 2)
    return stat


def bench_e2e(repeats: int):
    from udos.reasoning import UDOSReasoningEngine
    scene = build_factory_scene(n_steps=12)
    torch.manual_seed(0)
    base = TinyBaseModel(64, 2)
    eng = UDOSReasoningEngine(
        CTMConfig(iterations=16, d_model=128, d_input=64, heads=4,
                  n_synch_out=32, n_synch_action=32, memory_length=12,
                  out_dims=64, certainty_threshold=0.0),
        GPMConfig(feature_dim=64, latent_size=64, n_latents=16, lora_rank=4,
                  layer_indices=(0, 1), init_scaler_b_zero=False),
        base_model=base).eval()
    eng.internalize_scene(scene)

    def _run():
        with torch.no_grad():
            eng.reason(scene, query="bench")
    stat = _time_it(_run, repeats=repeats)
    return stat


def run_all(repeats: int):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = {
        "runtime": {"torch": torch.__version__,
                    "threads": torch.get_num_threads(),
                    "peak_rss_mb": None},
        "ctm_small_forward": bench_ctm("small", 36, repeats),
        "ctm_medium_forward": bench_ctm("medium", 36, repeats),
        "gpm_small_internalize": bench_gpm("small", repeats),
        "gpm_medium_internalize": bench_gpm("medium", repeats),
        "e2e_small": bench_e2e(repeats),
    }
    results["runtime"]["peak_rss_mb"] = _peak_rss_mb()

    # 上游真实 CTM 对照 (依赖可用时)
    try:
        from udos.adapters import SakanaCTMAdapter
        if SakanaCTMAdapter.available():
            ad = SakanaCTMAdapter(iterations=16, d_model=128, d_input=64,
                                  heads=4, n_synch_out=32, n_synch_action=32,
                                  memory_length=12, out_dims=64)
            ad.load()
            seq = torch.randn(1, 36, 64)
            with torch.no_grad():
                results["upstream_ctm_forward"] = _time_it(
                    lambda: ad.forward(seq), repeats=repeats)
                results["upstream_ctm_forward"]["params"] = ad.num_params()
    except Exception as e:  # 对照失败不影响主基准
        results["upstream_ctm_forward"] = {"skipped": str(e)}
    return results


def print_table(results):
    print(f"\n{'workload':<26}{'median(ms)':>12}{'min':>10}{'max':>10}"
          f"{'mean':>10}  notes")
    print("-" * 86)
    for k, v in results.items():
        if k == "runtime" or "median_ms" not in v:
            continue
        note = v.get("lora_kb_fp32", v.get("params", ""))
        print(f"{k:<26}{v['median_ms']:>12.3f}{v['min_ms']:>10.3f}"
              f"{v['max_ms']:>10.3f}{v['mean_ms']:>10.3f}  {note}")
    rt = results["runtime"]
    print("-" * 86)
    print(f"torch={rt['torch']} threads={rt['threads']} "
          f"peak_rss={rt['peak_rss_mb']}MB")


def guard(results):
    failed = []
    for name, limit in GUARD_MS.items():
        if name in results and results[name]["median_ms"] > limit:
            failed.append((name, results[name]["median_ms"], limit))
    if failed:
        print("\n!! 性能回归守卫未通过:")
        for n, got, lim in failed:
            print(f"   {n}: median {got}ms > 门槛 {lim}ms")
        return 1
    print("\n>> 性能回归守卫通过 (所有 workload median 低于门槛)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = run_all(args.repeats)
    print_table(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n结果已写入 {args.json}")
    if args.guard:
        sys.exit(guard(results))


if __name__ == "__main__":
    main()
