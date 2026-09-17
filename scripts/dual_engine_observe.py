"""v5.5.3 报告: 双引擎可观测性 —— 分类型场景门贡献 / 路由置信 / 盲-感知误差。

held-out seed=2026, 条件来源为 v5.5.0 学习型场景头。CPU 诚实档。
用法: PYTHONPATH=. python3 scripts/dual_engine_observe.py
"""
from __future__ import annotations

import json
import os

import torch

from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.scene_head import load_scene_head
from udos.dynamics_router import CLASS_NAMES
from udos.dual_engine_observe import observe_conditioning

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PER_KIND = 16
H, DT = 4, 0.5


def main() -> None:
    pred, _ = load_predictor(os.path.join(HERE, "checkpoints",
                                          "predictor_v4.3.9.pt"))
    pred.eval()
    head, _ = load_scene_head(os.path.join(HERE, "checkpoints",
                                           "scene_head_v5.5.0.pt"))
    ds = build_parametric_dataset(n_per_kind=N_PER_KIND, seed=2026)
    true = torch.tensor([CLASS_NAMES.index(k) for k in ds.kinds])

    per_kind = {k: {"gate_mean": [], "gate_final": [], "route_conf": [],
                    "route_correct": [], "blind_mse": [], "cond_mse": []}
                for k in CLASS_NAMES}

    with torch.no_grad():
        p_hat = head(ds.X)

    for i in range(ds.X.size(0)):
        kind = ds.kinds[i]
        obs = observe_conditioning(
            pred, ds.X[i:i + 1], conditioned_params=p_hat[i:i + 1],
            source="learned_head", horizon=H, dt=DT)
        Y = ds.Y[i:i + 1]
        per_kind[kind]["gate_mean"].append(obs.gate_contribution)
        per_kind[kind]["gate_final"].append(obs.gate_final_delta)
        per_kind[kind]["route_conf"].append(obs.route_confidence)
        per_kind[kind]["route_correct"].append(
            1.0 if obs.route_name == kind else 0.0)
        per_kind[kind]["blind_mse"].append(
            float(((obs.blind_trajectory - Y) ** 2).mean()))
        per_kind[kind]["cond_mse"].append(
            float(((obs.conditioned_trajectory - Y) ** 2).mean()))

    def agg(vals):
        t = torch.tensor(vals, dtype=torch.float32)
        return round(float(t.mean()), 4)

    summary = {}
    for k in CLASS_NAMES:
        d = per_kind[k]
        summary[k] = {
            "n": len(d["gate_mean"]),
            "gate_contribution_mean": agg(d["gate_mean"]),
            "gate_final_delta_mean": agg(d["gate_final"]),
            "route_accuracy": agg(d["route_correct"]),
            "route_confidence_mean": agg(d["route_conf"]),
            "blind_rollout4_mse": agg(d["blind_mse"]),
            "conditioned_rollout4_mse": agg(d["cond_mse"]),
        }

    report = {"version": "5.5.3", "heldout_seed": 2026,
              "conditioning_source": "learned_head",
              "horizon": H, "per_kind": summary}
    out = os.path.join(HERE, "reports", "v553_gate_observation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
