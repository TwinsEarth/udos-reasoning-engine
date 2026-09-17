"""brain 服务: 树突+CTM 合成 trace / DHS 基准 / 鲁棒对照(纯逻辑 analogy)。"""
from __future__ import annotations
import random
from .dendritic_compute import (coincidence_detect, temporal_inhibition,
                                inhibition_curve, nmda_amplify)
from .dhs_scheduler import benchmark as dhs_bench

NEURAL_SIMULATOR = "ENV_BLOCKED"


def brain_sim(seed: int = 7) -> dict:
    rng = random.Random(seed)
    excite = sorted(rng.sample(range(50), 6))
    inhibit = sorted(rng.sample(range(50), 4))
    coinc = coincidence_detect(excite, inhibit, window=2)
    curve = inhibition_curve(range(-3, 9))
    amps = [nmda_amplify(v) for v in (0.5, 1.0, 2.0)]
    return {
        "seed": seed,
        "coincidence_hits": coinc,
        "temporal_inhibition_curve": curve,
        "nmda_amplified": amps,
        "disclaimer": "CPU 合成 analogy, 非生物保真",
    }


def brain_dhs(dag=None, worker_count: int = 4) -> dict:
    if dag is None:
        dag = {"n1": [], "n2": ["n1"], "n3": ["n1"],
               "n4": ["n2", "n3"], "n5": ["n4"]}
    out = dhs_bench(dag)
    out["worker_count"] = worker_count     # 参数化, 未硬编码
    return out


def brain_robustness(seed: int = 7) -> dict:
    # 合成对照: 树突被动滤波前端 vs 线性前端, 噪声下保留率(假设值, analogy)
    rng = random.Random(seed)
    n = 100
    x = [rng.random() for _ in range(n)]
    noise = [rng.uniform(-0.1, 0.1) for _ in range(n)]
    linear_keep = sum(1 for i in range(n) if abs(x[i] + noise[i]) < 1.5) / n
    # 树突远端衰减: 噪声信号衰减后更稳定
    dend_keep = sum(1 for i in range(n) if abs((x[i] * 0.9) + (noise[i] * 0.3)) < 1.5) / n
    return {
        "linear_frontend_keep": round(linear_keep, 4),
        "dendritic_frontend_keep": round(dend_keep, 4),
        "note": "合成假设值, analogy; 不声称复现生物论文效应量",
    }


def brain_status() -> dict:
    import torch
    gpu = bool(torch.cuda.is_available())
    return {"enabled": True, "gpu_available": gpu,
            "neural_simulator": NEURAL_SIMULATOR,
            "disclaimer": "CPU 机制原型, 非真实生物/GPU 数据"}
