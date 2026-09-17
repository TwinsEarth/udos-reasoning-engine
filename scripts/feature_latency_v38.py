"""v3.8.3 性能基准: 3.8 线新特性单次延迟 (合成可测)。

测量: 多体冲突消解 / WM 调度分配 / 闭环一步 / 数字孪生一步 的单次延迟,
对照主 predictor baseline predict_next。落 benchmarks/results/feature_latency_v3.8.0.json。
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
from udos.multi_agent import AgentCoordinator, MultiAgentScene  # noqa: E402
from udos.wm_scheduler import WMScheduler  # noqa: E402
from udos.closed_loop import ClosedLoopOrchestrator  # noqa: E402
from udos.digital_twin import DigitalTwinScene  # noqa: E402

torch.set_num_threads(2)
CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def _ms(fn, iters=50):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return round((time.perf_counter() - t0) / iters * 1000.0, 4)


def main():
    model, _ = load_predictor(CKPT)
    n_params = sum(p.numel() for p in model.parameters())
    w = torch.randn(1, 6, 6)

    # 多体场景
    sc = MultiAgentScene()
    for k in range(8):
        sc.add_agent(f"a{k}", [float(k), 0, 0, 0, 0, 0], priority=(k % 3) + 1)
    coord = AgentCoordinator()

    sched = WMScheduler(total_imagination_budget=16, base_horizon=1,
                        max_horizon=4)
    sched.allocate(sc)

    orch = ClosedLoopOrchestrator(model, wm_horizon=2)
    twin = DigitalTwinScene(n_agents=8, n_obstacles=6, seed=0)

    latency = {
        "baseline_predict_next_ms": _ms(lambda: model.predict_next(w)),
        "multi_agent_resolve_ms": _ms(lambda: coord.resolve(sc)),
        "wm_scheduler_allocate_ms": _ms(lambda: sched.allocate(sc)),
        "closed_loop_step_ms": _ms(lambda: orch.step(w)),
        "digital_twin_step_ms": _ms(lambda: twin.step()),
    }
    summary = {
        "feature": "v38_multi_agent_wm_closed_loop_twin_latency",
        "main_params": n_params,
        "n_agents_tested": 8,
        "latency_ms": latency,
        "budget": {"cpu_threads": 2, "method": "perf_counter 实测, 非真机总线"},
        "analogy_not_reproduction": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v3.8.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(latency, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
