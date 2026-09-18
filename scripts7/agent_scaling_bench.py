"""Agent Scaling Law 实测基准（v7.2）。

用法：
  PYTHONPATH=. python3 scripts7/agent_scaling_bench.py \
      --ckpt checkpoints7/worldmodel_v7.0.3.pt --tasks 64 \
      --workers 1,2,4,8,16 --out reports7/agent_scaling.json

两条曲线均为 *本机实测*：
- throughput：固定 M 个独立预测任务，变并发 Agent 数，记录墙钟/tasks·s⁻¹/speedup；
- quality：k 个预测者集成的 held-out MSE（前 D 个独立信息源 verified，
  其后近相关副本 cpu-proto，刻画边际递减/饱和）。
不输出任何未经测量的加速倍数。
"""
import argparse
import json
import os
import platform
import time

import torch

from udos7.agents.legion import (fit_quality_saturation, io_bound_curve,
                                 quality_curve, throughput_curve)
from udos7.persistence import load_worldmodel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints7/worldmodel_v7.0.3.pt")
    ap.add_argument("--tasks", type=int, default=64)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--workers", default="1,2,4,8,16")
    ap.add_argument("--k", default="1,2,3,5,9,17")
    ap.add_argument("--out", default="reports7/agent_scaling.json")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    model, ckpt_meta = load_worldmodel(args.ckpt)
    workers = tuple(int(x) for x in args.workers.split(","))
    ks = tuple(int(x) for x in args.k.split(","))

    t0 = time.time()
    tp = throughput_curve(model, n_tasks=args.tasks, workers_list=workers)
    io = io_bound_curve(n_tasks=args.tasks, latency_s=0.05,
                        concurrency_list=workers)
    ql = quality_curve(model, n_samples=args.samples, k_list=ks)
    fit = fit_quality_saturation(ql)
    wall = time.time() - t0

    cpu_peak = max(tp, key=lambda r: r["tasks_per_s"])
    report = {
        "schema": "udos7.agent_scaling/v1",
        "evidence_grade": "cpu-proto",
        "headline": {
            "cpu_bound_peak": cpu_peak,
            "io_bound_note": "远程Agent(I/O等待)区间并发近线性扩展，见 io_curve(simulation)",
            "quality_note": "增益来自多样性+校准加权；同质化扩招饱和，见 quality_curve",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "ckpt": args.ckpt,
            "ckpt_meta_keys": sorted(list(ckpt_meta.keys()))
            if isinstance(ckpt_meta, dict) else str(type(ckpt_meta)),
        },
        "workload": {"independent_tasks": args.tasks,
                     "quality_samples": args.samples},
        "throughput_curve_cpu_bound": tp,
        "io_curve_remote_agent_simulation": io,
        "quality_curve": ql,
        "quality_saturation_fit": fit,
        "bench_wall_s": round(wall, 3),
        "notes": [
            "throughput_curve_cpu_bound：CPU 密集型本地推理，受物理核/GIL/torch 线程约束，",
            "  并发超过并行度后可能不升反降（实测如实记录，不外推）。",
            "io_curve_remote_agent_simulation：用 asyncio.sleep 模拟远程 LLM/工具等待，",
            "  这是云端数千子 Agent 的真实受益区间，近线性直到并发上限(simulation 时延)。",
            "quality_curve：按独立校准集 MSE 做 softmax 加权（不偷看 test）；k<=独立源数",
            "  为 verified，更大 k 为最强专家的近相关副本(cpu-proto)，刻画同质化扩招饱和；",
            "  并给 best_single/naive_equal 对照——等权平均差专家会拖差结果。",
            "亿级在线 LLM 子 Agent 需 LLM key+云预算(AL4 门禁)，不在本基准范围。",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"throughput": tp, "quality": ql, "fit": fit},
                     ensure_ascii=False, indent=2))
    print("written:", args.out)


if __name__ == "__main__":
    main()
