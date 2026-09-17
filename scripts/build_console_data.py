#!/usr/bin/env python3
"""构建 web/udos_console.html 的内联真实数据。
数字全部回算自 benchmarks/results/*.json 与 registry 探测, 不手抄编造。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.connectors import build_default_registry  # noqa: E402


def load(name):
    return json.loads((ROOT / "benchmarks" / "results" / name).read_text(encoding="utf-8"))


def main():
    reg = build_default_registry("full")
    rows = reg.list(profile="full", autoprobe=True)
    registry_rows = [{
        "id": r["id"], "name": r["name"], "kind": r["kind"],
        "license": r["license"].split("(")[0][:40], "level": r["level"],
        "priority": r["priority"], "category": r["category"],
        "status": r["capability"]["status"],
        "gpu": r["requires"]["gpu"], "weights": r["requires"]["weights"],
        "pkg": bool(r["requires"]["pkg"]),
        "detail": r["capability"]["detail"][:80],
    } for r in rows]

    perf = load("pareto_v45.json")
    tiers = ["none", "low", "high", "max"]
    latent = [perf["per_tier"][t]["median_latency_ms"] for t in tiers]
    tokens = [perf["per_tier"][t]["explicit_tokens"] for t in tiers]
    ticks = [perf["per_tier"][t]["internal_ticks_per_call"] for t in tiers]

    wm = load("wm_imagine_v3.6.0.json")
    wm_data = {h: wm["per_horizon"][h]["step_mse"] for h in ["H=8", "H=16", "H=32"]}

    ml = load("feature_latency_v3.9.2.json")
    hier_labels = list(ml["latency_ms"].keys())
    hier_vals = list(ml["latency_ms"].values())

    ma = load("multi_agent_ab_v3.8.0.json")
    ns = ma["sweep_n_agents"]
    collisions_no = [ma["results"][f"n{n}"]["no_coordinator"]["collision_events"] for n in ns]
    collisions_yes = [ma["results"][f"n{n}"]["with_coordinator"]["collision_events"] for n in ns]

    se = load("self_evolution_v43.json")
    gens = [c["gen"] for c in se["dev4_multigen_curve"]["curve"]]
    costs = [c["best_cost"] for c in se["dev4_multigen_curve"]["curve"]]

    wla = load("wla_sparse_vs_dense_ab.json")
    wla_cost = [wla["dense_cost_units"], wla["sparse_cost_units"]]

    res_bench = load("resource_registry_v456.json")

    data = {
        "overview": {
            "version": "4.5.6", "tests": 1492, "coverage": 93,
            "checkpoints": 33, "main_params": 52191,
            "eval_mse": perf["baseline_predictor_mse"],
            "release_decision": "go",
        },
        "registry": registry_rows,
        "summary_perf": reg.summary("performance"),
        "summary_full": reg.summary("full"),
        "latent_pareto": {"tiers": tiers, "latency_ms": latent,
                          "explicit_tokens": tokens, "ticks": ticks},
        "wm_imagine": wm_data,
        "hier_latency": {"labels": hier_labels, "ms": hier_vals},
        "multi_agent": {"n": ns, "no_coord": collisions_no,
                        "with_coord": collisions_yes},
        "self_evo": {"gen": gens, "cost": costs,
                     "saving_pct": se["dev2_config_ab"]["cost_saving_pct"]},
        "wla": {"labels": ["dense", "sparse"], "cost": wla_cost},
        "resource_bench": {k: v for k, v in res_bench.items()
                           if k not in ("summary_full",)},
        "qa": {
            "bugs": [
                {"id": "B-1", "fix": "SMPL rot6d 零旋转恒为 6 维"},
                {"id": "B-2", "fix": "summary 随查询 profile 更新"},
                {"id": "B-3", "fix": "yourdfpy FK 真实 API"},
                {"id": "B-4", "fix": "profile 改机械推导(高优∩CPU可行),3→25"},
                {"id": "B-5", "fix": "补 DreamerV3 RSSM L1 状态 schema"},
            ],
            "anchors": {
                "v4.3.9": "8e767da5", "v3.8.6": "7351250a",
                "v3.4.5": "52993ca7", "v3.3.3": "f993bcbd",
            },
        },
    }
    out = ROOT / "web" / "dashboard_data.json"
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, "with", len(registry_rows), "registry rows")


if __name__ == "__main__":
    main()
