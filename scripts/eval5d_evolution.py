"""
四代正式件五维评测演化 (v3.0.0.dev6)
==========================================
**UDOS 内部基准, 与 PhysBrain 外部数字严格分开。**

对 v2.7.3 / v2.8.0 / v2.9.0 / v3.0.0 四代正式 checkpoint 跑 FiveDimensionEvaluator,
落 benchmarks/results/five_dim_evolution_v3.0.0.json。

诚实: 不强制分数单调提升 (不同代外挂特性不同, 评测口径一致但模型演化有取舍)。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__  # noqa: E402
from udos.persistence import load_predictor  # noqa: E402
from udos.eval_suite import FiveDimensionEvaluator  # noqa: E402

torch.set_num_threads(2)

CKPTS = [
    ("v2.7.3", "predictor_v2.7.3.pt"),
    ("v2.8.0", "predictor_v2.8.0.pt"),
    ("v2.9.0", "predictor_v2.9.0.pt"),
    ("v3.0.0", "predictor_v3.0.0.pt"),
]
OUT = ROOT / "benchmarks" / "results" / "five_dim_evolution_v3.0.0.json"


def main():
    rows = []
    for ver, name in CKPTS:
        path = ROOT / "checkpoints" / name
        model, meta = load_predictor(str(path))
        ev = FiveDimensionEvaluator(model, seed=2025, n_per_kind=16,
                                    grid_size=3)
        scores = ev.evaluate()
        composite = ev.composite_score(scores)
        rows.append({
            "version": ver,
            "checkpoint": name,
            "udos_version_meta": meta.get("udos_version"),
            **{k: round(v, 2) for k, v in scores.items()},
            "composite": round(composite, 2),
        })
        print(ver, json.dumps(rows[-1], ensure_ascii=False))

    # 单调合理性检查 (不强制提升, 仅记录)
    comp = [r["composite"] for r in rows]
    monotonic = all(comp[i] <= comp[i + 1] + 1e-6 for i in range(len(comp) - 1))

    summary = {
        "version": __version__,
        "is_internal_benchmark": True,
        "not_physbrain_leaderboard": True,
        "note": "UDOS 内部五维演化, 与 PhysBrain 外部数字严格分开",
        "generations": rows,
        "composite_sequence": comp,
        "monotonic_non_decreasing": bool(monotonic),
    }
    os.makedirs(OUT.parent, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
