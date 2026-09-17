"""
ICM 特性性能基准 (v3.4.2) — feature_latency_v3.4.0.json
================================================================
测量各 ICM 组件单次推理延迟 (ms), 落 benchmarks/results/feature_latency_v3.4.0.json:
    * 纯 0-shot predict_next 延迟;
    * ICM k=3 检索聚合延迟;
    * 事件切分 EventSegmenter.detect 延迟;
    * 跨本体归一化注册单条延迟;
    * PCE 提示词包解析延迟。
analogy, not reproduction。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)

from udos import load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator  # noqa: E402
from udos.icm_events import EventSegmenter, ThreeStreamAligner  # noqa: E402
from udos.pce_format import DemonstrationPrompt, PCEPromptParser  # noqa: E402


def _bench(fn, n=50):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return round((time.perf_counter() - t0) / n * 1000.0, 4)


def main():
    m, _ = load_predictor("checkpoints/predictor_v3.4.0.pt")
    te = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=555)
    mem = DemonstrationMemory()
    agg = ICMAggregator(m)
    for i in range(len(te)):
        ep = DemonstrationEpisode(te.X[i], te.Y[i, 0], kind=te.kinds[i],
                                 scene_params=te.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=te.P[i])
    w = te.X[0:1]
    sp = te.P[0:1]

    lat = {
        "zeroshot_predict_next_ms": _bench(
            lambda: m.predict_next(w, scene_params=sp)),
        "icm_k3_predict_ms": _bench(
            lambda: agg.predict(te.X[0], memory=mem, k=3, scene_params=sp)),
        "event_detect_ms": _bench(
            lambda: EventSegmenter().detect(te.X[0])),
        "three_stream_split_ms": _bench(
            lambda: ThreeStreamAligner().split(te.X[0])),
    }
    # PCE 解析
    p = DemonstrationPrompt("lat", kind="x")
    for _ in range(5):
        p.add_block(torch.randn(6, 6), torch.randn(6), torch.randn(6))
    raw = json.loads(p.dumps())
    lat["pce_prompt_parse_ms"] = _bench(
        lambda: PCEPromptParser.load_episodes(
            PCEPromptParser.from_dict(raw)))

    out = {"feature": "icm_component_latency",
           "analogy_not_reproduction": True,
           "threads": 2, "device": "cpu",
           "latency_ms": lat}
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v3.4.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(lat, ensure_ascii=False, indent=2))
    print("saved benchmarks/results/feature_latency_v3.4.0.json")


if __name__ == "__main__":
    main()
