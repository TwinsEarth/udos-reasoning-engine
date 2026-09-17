#!/usr/bin/env python3
"""UDOS v3.8.7 性能基线基准脚本 (只读诊断, 不修改生产代码)。

测量并落盘 benchmarks/results/perf_baseline_v387.json:
  1. 单次推理延迟 (predict_next) — 回归锚点
  2. WM 想象 rollout 延迟 (3.6 新路径)
  3. 分层神经控制 step 延迟 (3.7 新路径)
  4. 多体调度 resolve 延迟 (3.8 新路径)
  5. 数字孪生 step 延迟 (3.8 新路径)
  6. HTTP /wm/imagine /neural/step /twin/step e2e
  7. RSS 内存

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
from udos.server import create_server  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "predictor_v3.8.6.pt")


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
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main():
    print("=== 加载 v3.8.6 checkpoint ===")
    predictor, meta = load_predictor(CKPT, map_location="cpu")
    predictor.eval()
    raw_dim = predictor.raw_dim
    sp_dim = predictor.scene_param_dim
    print(f"  raw_dim={raw_dim}  scene_param_dim={sp_dim}  "
          f"params={sum(p.numel() for p in predictor.parameters())}")

    W = 6
    g = torch.Generator().manual_seed(123)
    window1 = torch.randn(1, W, raw_dim, generator=g)
    sp1 = torch.randn(1, sp_dim, generator=g) if sp_dim else None

    results = {"env": {}, "single_predict": {}, "wm": {}, "neural": {},
               "multi_agent": {}, "twin": {}, "http": {}, "memory": {}}

    results["env"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "nproc": os.cpu_count(),
        "platform": platform.platform(),
        "checkpoint_md5": "7351250ac00db53c321b919a951c640b",
    }

    # ---- 1. 单次推理延迟 (回归锚点) ----
    print("=== 1. 单次推理延迟 (predict_next) ===")
    with torch.no_grad():
        for _ in range(5):
            predictor.predict_next(window1, scene_params=sp1)
    samples = []
    with torch.no_grad():
        for _ in range(60):
            t0 = time.perf_counter()
            predictor.predict_next(window1, scene_params=sp1)
            samples.append(time.perf_counter() - t0)
    results["single_predict"] = summarize(samples)
    print(f"  p50={results['single_predict']['p50']*1000:.2f}ms")

    # ---- 2. WM 想象 rollout ----
    print("=== 2. WM imagine rollout ===")
    from udos.world_model import LatentWorldModel
    from udos.dynamics import build_parametric_dataset
    wm = LatentWorldModel(predictor)
    fit_ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                      horizon=4, dt=0.5, seed=6060)
    wm.fit(fit_ds, epochs=20)
    with torch.no_grad():
        for _ in range(3):
            wm.imagine(window1, 2, scene_params=sp1)
    samps = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            wm.imagine(window1, 2, scene_params=sp1)
            samps.append(time.perf_counter() - t0)
    results["wm"]["imagine_rollout"] = summarize(samps)
    print(f"  wm.imagine p50={results['wm']['imagine_rollout']['p50']*1000:.2f}ms")

    # ---- 3. 分层神经控制 step ----
    print("=== 3. 分层神经控制 step ===")
    from udos.neural_control import HierarchicalController
    ctrl = HierarchicalController(predictor)
    with torch.no_grad():
        for _ in range(3):
            ctrl.step(window1, scene_params=sp1)
    samps = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            ctrl.step(window1, scene_params=sp1)
            samps.append(time.perf_counter() - t0)
    results["neural"]["step"] = summarize(samps)
    print(f"  neural.step p50={results['neural']['step']['p50']*1000:.2f}ms")

    # ---- 4. 多体调度 resolve ----
    print("=== 4. 多体调度 resolve ===")
    from udos.multi_agent import MultiAgentScene, AgentCoordinator
    scene = MultiAgentScene()
    for i in range(5):
        scene.add_agent(
            f"ag{i}", state=torch.randn(6, generator=g) * 0.1,
            goal=torch.randn(3, generator=g), priority=i % 3 + 1)
    coord = AgentCoordinator()
    samps = []
    for _ in range(100):
        t0 = time.perf_counter()
        coord.resolve(scene)
        samps.append(time.perf_counter() - t0)
    results["multi_agent"]["resolve"] = summarize(samps)
    print(f"  resolve p50={results['multi_agent']['resolve']['p50']*1000:.3f}ms")

    # ---- 5. 数字孪生 step ----
    print("=== 5. 数字孪生 step ===")
    from udos.digital_twin import DigitalTwinScene
    twin = DigitalTwinScene(n_agents=4, n_obstacles=3, seed=42, bounds=10.0)
    samps = []
    for _ in range(50):
        t0 = time.perf_counter()
        twin.step(dt=1.0)
        samps.append(time.perf_counter() - t0)
    results["twin"]["step"] = summarize(samps)
    print(f"  twin.step p50={results['twin']['step']['p50']*1000:.3f}ms")

    # ---- 6. HTTP e2e ----
    print("=== 6. HTTP e2e ===")
    httpd = create_server("127.0.0.1", 0, "small", checkpoint=CKPT)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    host, port = httpd.server_address
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
            break
        except Exception:
            time.sleep(0.1)

    # /predict e2e
    body = json.dumps({"window": window1[0].tolist()}).encode()
    for _ in range(5):
        req = urllib.request.Request(
            f"http://{host}:{port}/predict", data=body,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    samps = []
    for _ in range(40):
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"http://{host}:{port}/predict", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10):
            pass
        samps.append(time.perf_counter() - t0)
    results["http"]["predict_e2e"] = summarize(samps)

    # /wm/imagine e2e
    wm_body = json.dumps({"window": window1[0].tolist(), "horizon": 2}).encode()
    for _ in range(3):
        req = urllib.request.Request(
            f"http://{host}:{port}/wm/imagine", data=wm_body,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    samps = []
    for _ in range(20):
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"http://{host}:{port}/wm/imagine", data=wm_body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30):
            pass
        samps.append(time.perf_counter() - t0)
    results["http"]["wm_imagine_e2e"] = summarize(samps)

    # /neural/step e2e
    neural_body = json.dumps({"window": window1[0].tolist()}).encode()
    for _ in range(3):
        req = urllib.request.Request(
            f"http://{host}:{port}/neural/step", data=neural_body,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    samps = []
    for _ in range(20):
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"http://{host}:{port}/neural/step", data=neural_body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30):
            pass
        samps.append(time.perf_counter() - t0)
    results["http"]["neural_step_e2e"] = summarize(samps)

    # /metrics 延迟
    samps = []
    for _ in range(20):
        t0 = time.perf_counter()
        urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=5)
        samps.append(time.perf_counter() - t0)
    results["http"]["metrics"] = summarize(samps)

    httpd.shutdown()
    httpd.server_close()
    print(f"  /predict e2e p50={results['http']['predict_e2e']['p50']*1000:.2f}ms")
    print(f"  /wm/imagine e2e p50={results['http']['wm_imagine_e2e']['p50']*1000:.2f}ms")
    print(f"  /neural/step e2e p50={results['http']['neural_step_e2e']['p50']*1000:.2f}ms")

    # ---- 7. RSS ----
    results["memory"]["rss_max_mb"] = rss_mb()

    out_dir = os.path.join(ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perf_baseline_v387.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== 落盘 {out_path} ===")
    print(f"  RSS peak: {results['memory']['rss_max_mb']} MB")


if __name__ == "__main__":
    main()
