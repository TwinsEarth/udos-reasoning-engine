"""v3.9.3 WLA 线性能基准: ER 头 / change-mask / RVQ / flow 外挂延迟 (CPU 2 线程)。

落 benchmarks/results/feature_latency_v3.9.2.json。纯外挂零梯度, 主参 52191 不变。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import load_predictor  # noqa: E402
from udos.embodied import EmbodiedReasoningHead  # noqa: E402
from udos.wla import ChangeMask, RVQActionTokenizer, FlowMatchingDecoder  # noqa: E402

torch.set_num_threads(2)


def _time(fn, repeats=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return round((time.perf_counter() - t0) / repeats * 1000, 3)


def main():
    model, _ = load_predictor("checkpoints/predictor_v3.9.0.pt")
    n_params = sum(p.numel() for p in model.parameters())
    w = torch.randn(4, 6, 6)

    head = EmbodiedReasoningHead(model, enable=True)
    cm = ChangeMask(0.05)
    tok = RVQActionTokenizer(8, 2, seed=1)
    tok.fit(torch.randn(200, 6))
    z = torch.randn(16, 32)
    dec = FlowMatchingDecoder(32, 6, 16, 3)
    dec.fit(z, torch.randn(16, 6), epochs=3)

    lat = {
        "baseline_predict_next_ms": _time(lambda: model.predict_next(w)),
        "er_head_ms": _time(lambda: head(w)),
        "change_mask_sparsity_ms": _time(lambda: cm.sparsity(w)),
        "rvq_encode_ms": _time(lambda: tok.encode(torch.randn(8, 6))),
        "flow_decode_ms": _time(lambda: dec.decode(z, seed=0)),
    }
    out = {
        "version": "3.9.2", "main_params": n_params,
        "latency_ms": lat,
        "all_external_zero_grad": True,
        "analogy_not_reproduction": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v3.9.2.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(lat, ensure_ascii=False))


if __name__ == "__main__":
    main()
