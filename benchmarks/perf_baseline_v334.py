#!/usr/bin/env python3
"""UDOS v3.3.3 -> v3.3.4 性能基线基准脚本 (只读诊断, 不修改生产代码)。

测量并落盘 benchmarks/results/perf_baseline_v334.json:
  1. 单次推理延迟 (predict_next, 多采样 p50/p95/p99)
  2. 批量推理延迟 (BatchPredictor, 不同 batch_size)
  3. 各算法热点 (ctm/gpm 前向, calibration, counterfactual, identify, risk)
  4. HTTP 端到端 /predict 延迟
  5. RSS 内存
  6. 缓存命中率
  7. physical_loop 单步耗时

输出含原始样本数组, 非仅汇总。
"""
import json
import os
import platform
import resource
import sys
import threading
import time
import urllib.request

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from udos.persistence import load_predictor  # noqa: E402
from udos.batch import BatchPredictor  # noqa: E402
from udos.cache import InferenceCache  # noqa: E402
from udos.counterfactual import CounterfactualEngine  # noqa: E402
from udos.identification import SceneParameterIdentifier  # noqa: E402
from udos.decision import RiskGrader  # noqa: E402
from udos.physical_loop import PhysicalLoopRunner  # noqa: E402
from udos.server import create_server  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "predictor_v3.3.3.pt")


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]))


def summarize(samples):
    s = sorted(samples)
    return {
        "n": len(s),
        "min": round(s[0], 6) if s else 0,
        "p50": round(pct(s, 0.50), 6) if s else 0,
        "p95": round(pct(s, 0.95), 6) if s else 0,
        "p99": round(pct(s, 0.99), 6) if s else 0,
        "max": round(s[-1], 6) if s else 0,
        "mean": round(sum(s) / len(s), 6) if s else 0,
    }


def rss_mb():
    """RSS 内存峰值 (ru_maxrss, KB on Linux)。"""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main():
    print("=== 加载 v3.3.3 checkpoint ===")
    predictor, meta = load_predictor(CKPT, map_location="cpu")
    predictor.eval()
    raw_dim = predictor.raw_dim
    sp_dim = predictor.scene_param_dim
    print(f"  raw_dim={raw_dim}  scene_param_dim={sp_dim}  "
          f"params={sum(p.numel() for p in predictor.parameters())}")

    # 固定输入
    W = 6
    g = torch.Generator().manual_seed(123)
    window1 = torch.randn(1, W, raw_dim, generator=g)
    sp1 = torch.randn(1, sp_dim, generator=g) if sp_dim else None

    results = {"env": {}, "single_predict": {}, "batch": {},
               "algo_hotspots": {}, "http": {}, "memory": {},
               "cache": {}, "loop": {}}

    # ---- 环境信息 ----
    results["env"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "nproc": os.cpu_count(),
        "platform": platform.platform(),
        "checkpoint_md5": "f993bcbdd476473c28dd4604bbbe11d6",
    }

    # ---- 1. 单次推理延迟 ----
    print("=== 1. 单次推理延迟 (predict_next) ===")
    # warmup
    with torch.no_grad():
        for _ in range(5):
            predictor.predict_next(window1, scene_params=sp1)
    samples = []
    with torch.no_grad():
        for _ in range(60):
            t0 = time.perf_counter()
            predictor.predict_next(window1, scene_params=sp1)
            samples.append(time.perf_counter() - t0)
    results["single_predict"] = {
        "summary_s": summarize(samples),
        "raw_samples_s": [round(x, 6) for x in samples],
    }
    print(f"  p50={results['single_predict']['summary_s']['p50']*1000:.2f}ms  "
          f"p95={results['single_predict']['summary_s']['p95']*1000:.2f}ms")

    # ---- 2. 批量推理延迟 ----
    print("=== 2. 批量推理延迟 (BatchPredictor) ===")
    bp = BatchPredictor(predictor)  # v3.3.4 起默认 max_shard=64 (C2)
    batch_results = {}
    for bs in [1, 4, 8, 16, 32, 64]:
        batch = torch.randn(bs, W, raw_dim, generator=g)
        sp_batch = torch.randn(bs, sp_dim, generator=g) if sp_dim else None
        # warmup
        with torch.no_grad():
            for _ in range(3):
                bp.predict(batch, scene_params=sp_batch)
        samps = []
        with torch.no_grad():
            for _ in range(30):
                t0 = time.perf_counter()
                bp.predict(batch, scene_params=sp_batch)
                samps.append(time.perf_counter() - t0)
        batch_results[f"bs_{bs}"] = {
            "summary_s": summarize(samps),
            "raw_samples_s": [round(x, 6) for x in samps],
        }
        print(f"  bs={bs:>3}: p50={summarize(samps)['p50']*1000:.2f}ms")
    results["batch"] = batch_results

    # ---- 3. 各算法热点 ----
    print("=== 3. 算法热点 ===")
    # ctm 前向
    from udos.reasoning import UDOSReasoningEngine
    from udos.ctm_engine import CTMConfig
    from udos.gpm_engine import GPMConfig, TinyBaseModel
    ctm = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    gpm_cfg = GPMConfig(feature_dim=64, latent_size=64, n_latents=16,
                        lora_rank=4, target_modules=("down_proj",),
                        layer_indices=(0,), init_scaler_b_zero=False)
    base = TinyBaseModel(hidden=64, n_layers=2)
    engine = UDOSReasoningEngine(ctm, gpm_cfg, base_model=base, lora_scaling=0.1)
    engine.eval()

    # ctm+gpm 端到端前向 (internalize + reason, 用合法 demo scene)
    from demos.scene_factory import build_factory_scene
    from udos.pce_format import PCEParser
    demo_scene = build_factory_scene(n_steps=12)
    samps = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            engine.internalize_scene(demo_scene)
            engine.reason(demo_scene, query="bench", horizon=2)
            engine.reset_scene(demo_scene.scene_id)
            samps.append(time.perf_counter() - t0)
    results["algo_hotspots"]["gpm_ctm_reason_e2e"] = summarize(samps)

    # calibration (fit)
    from udos.dynamics import build_parametric_dataset
    from udos.calibration import fit_predictor_calibration
    cal_ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                      horizon=4, dt=0.5, seed=2031)
    samps = []
    for _ in range(5):
        t0 = time.perf_counter()
        fit_predictor_calibration(predictor, cal_ds)
        samps.append(time.perf_counter() - t0)
    results["algo_hotspots"]["calibration_fit"] = summarize(samps)

    # counterfactual
    samps = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            CounterfactualEngine(predictor).counterfactual(window1, 2,
                                                           scene_params=sp1)
            samps.append(time.perf_counter() - t0)
    results["algo_hotspots"]["counterfactual"] = summarize(samps)

    # identify
    samps = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            SceneParameterIdentifier(predictor, grid_size=3).identify(window1, horizon=2)
            samps.append(time.perf_counter() - t0)
    results["algo_hotspots"]["identify"] = summarize(samps)

    # risk
    samps = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            RiskGrader().grade(predictor, window1, scene_params=sp1, horizon=1)
            samps.append(time.perf_counter() - t0)
    results["algo_hotspots"]["risk"] = summarize(samps)

    print(f"  gpm_ctm_reason_e2e p50={results['algo_hotspots']['gpm_ctm_reason_e2e']['p50']*1000:.2f}ms")

    # ---- 4. HTTP 端到端 /predict ----
    print("=== 4. HTTP 端到端延迟 ===")
    httpd = create_server("127.0.0.1", 0, "small", checkpoint=CKPT)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    host, port = httpd.server_address
    # wait
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
            break
        except Exception:
            time.sleep(0.1)

    body = json.dumps({"window": window1[0].tolist(),
                       "scene_params": sp1[0].tolist()}).encode()
    # warmup
    for _ in range(5):
        req = urllib.request.Request(f"http://{host}:{port}/predict",
                                    data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    samps = []
    for _ in range(40):
        t0 = time.perf_counter()
        req = urllib.request.Request(f"http://{host}:{port}/predict",
                                    data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10):
            pass
        samps.append(time.perf_counter() - t0)
    results["http"]["predict_e2e"] = summarize(samps)
    results["http"]["predict_e2e_raw"] = [round(x, 6) for x in samps]

    # /metrics 延迟
    samps = []
    for _ in range(20):
        t0 = time.perf_counter()
        urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=5)
        samps.append(time.perf_counter() - t0)
    results["http"]["metrics"] = summarize(samps)

    # /health 延迟
    samps = []
    for _ in range(20):
        t0 = time.perf_counter()
        urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5)
        samps.append(time.perf_counter() - t0)
    results["http"]["health"] = summarize(samps)

    httpd.shutdown(); httpd.server_close()
    print(f"  HTTP /predict e2e p50={results['http']['predict_e2e']['p50']*1000:.2f}ms")

    # ---- 5. RSS 内存 ----
    results["memory"]["rss_max_mb"] = rss_mb()

    # ---- 6. 缓存命中率 ----
    print("=== 6. 缓存命中率 ===")
    cache = InferenceCache(predictor, maxsize=64)
    cache.enabled = True
    bp_c = BatchPredictor(predictor)
    # 第一轮: 全 miss
    with torch.no_grad():
        for _ in range(10):
            bp_c.predict(window1, scene_params=sp1, cache=cache)
    hit1 = cache.hits
    miss1 = cache.misses
    # 第二轮: 全 hit (相同输入)
    with torch.no_grad():
        for _ in range(10):
            bp_c.predict(window1, scene_params=sp1, cache=cache)
    hit2 = cache.hits - hit1
    miss2 = cache.misses - miss1
    results["cache"] = {
        "round1_misses": miss1, "round1_hits": hit1,
        "round2_hits": hit2, "round2_misses": miss2,
        "hit_rate_total": round(cache.hit_rate, 4),
    }
    print(f"  round1: {hit1}H/{miss1}M  round2: {hit2}H/{miss2}M")

    # ---- 7. physical_loop 单步 ----
    print("=== 7. physical_loop 单步 ===")
    pl = PhysicalLoopRunner(predictor, horizon=4)
    samps = []
    with torch.no_grad():
        for _ in range(15):
            t0 = time.perf_counter()
            pl.run(window1, scene_params=sp1)
            samps.append(time.perf_counter() - t0)
    results["loop"]["physical_loop_step"] = summarize(samps)
    print(f"  loop step p50={results['loop']['physical_loop_step']['p50']*1000:.2f}ms")

    # ---- 落盘 ----
    out_dir = os.path.join(ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perf_baseline_v334_post.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== 落盘 {out_path} ===")
    print(f"  RSS peak: {results['memory']['rss_max_mb']} MB")


if __name__ == "__main__":
    main()
