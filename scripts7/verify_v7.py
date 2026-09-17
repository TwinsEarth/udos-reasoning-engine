#!/usr/bin/env python3
"""v7 综合验证：对已落盘 checkpoint 重算指标 / 覆盖率 / 扇形 / 延迟（可复现门禁）。

输出 reports7/v7_verification.json。全部 CPU 固定 seed，证据分级 verified。
任何一项覆盖率不达标会在 gating 段标 FAIL（供 CI / eval 门禁读取）。
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from udos7 import __version__
from udos7.dynamics import three_way_splits
from udos7.persistence import load_worldmodel
from udos7.metrics import evaluate, estimator_param_error
from udos7.uncertainty import (ConformalCalibrator, MonteCarloParamFan,
                               empirical_coverage)

CKPT = REPO / "checkpoints7" / "worldmodel_v7.0.1.pt"
OUT = REPO / "reports7" / "v7_verification.json"
ALPHAS = (0.2, 0.1, 0.05)
NOMINAL = {0.2: 0.80, 0.1: 0.90, 0.05: 0.95}
COV_TOL = 0.05


def main():
    torch.set_num_threads(2)
    model, ckpt = load_worldmodel(CKPT)
    splits = three_way_splits(n_traj_per_kind=64)
    H = splits["test"].horizon

    report = {
        "version": __version__, "evidence_grade": "verified",
        "torch_version": torch.__version__, "device": "cpu", "threads": 2,
        "params": model.num_parameters(),
        "checkpoint_config": ckpt["config"],
        "dataset": {k: {"windows": len(v), "trajectories": v.trajectories()}
                    for k, v in splits.items()},
        "test_metrics": {}, "coverage": {}, "param_fan": {},
        "latency_ms": {}, "gating": {},
    }

    # ---- 指标（oracle / blind，整体/分类别/分轴）----
    report["test_metrics"]["oracle"] = evaluate(model, splits["test"], H, True)
    report["test_metrics"]["blind"] = evaluate(model, splits["test"], H, False)
    report["estimator_param_error"] = estimator_param_error(
        model, splits["test"])

    # ---- conformal 覆盖率（oracle 与 blind 各自校准）----
    gates = []
    for mode, use_exp in (("oracle", True), ("blind", False)):
        cal = ConformalCalibrator(H).fit(
            model, splits["calib"], use_explicit=use_exp, alphas=ALPHAS)
        for a in ALPHAS:
            iv = cal.predict_interval(model, splits["test"].X, H, a,
                                      explicit=splits["test"].P if use_exp else None)
            cov = empirical_coverage(iv["lower"], iv["upper"], splits["test"].Y)
            ok = abs(cov["marginal"] - NOMINAL[a]) <= COV_TOL
            gates.append(ok)
            report["coverage"][f"{mode}_alpha_{a}"] = {
                "nominal": NOMINAL[a], **cov,
                "bandwidth_marginal": round(cal.bands[a].q_marginal, 4),
                "pass": ok}

    # ---- 参数蒙特卡洛扇形（blind，膨胀到名义 80%）----
    fan = MonteCarloParamFan(draws=64).fit(model, splits["calib"])
    fan.calibrate_inflation(model, splits["calib"], H, nominal=0.8, max_n=640)
    n = min(256, len(splits["test"]))
    f = fan.sample(model, splits["test"].X[:n], H)
    truth = splits["test"].Y[:n]
    fc = ((truth >= f.p10) & (truth <= f.p90)).float().mean().item()
    report["param_fan"] = {"nominal": 0.8, "empirical_coverage": round(fc, 4),
                           "pass": abs(fc - 0.8) <= 0.10}
    gates.append(report["param_fan"]["pass"])

    # ---- 延迟 batch=1 ----
    x1, p1 = splits["test"].X[:1], splits["test"].P[:1]
    for _ in range(20):
        model(x1, explicit=p1)
    t1, t4 = [], []
    for _ in range(200):
        t0 = time.perf_counter(); model(x1, explicit=p1)
        t1.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); model.rollout(x1, H, explicit=p1)
        t4.append((time.perf_counter() - t0) * 1000)
    report["latency_ms"] = {
        "predict_next_batch1_median": round(statistics.median(t1), 4),
        "rollout4_batch1_median": round(statistics.median(t4), 4)}

    report["gating"] = {"all_pass": all(gates),
                        "checked": len(gates),
                        "note": "覆盖率容差 ±0.05（扇形 ±0.10），CPU 固定 seed"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("params", "test_metrics", "coverage", "param_fan",
                       "latency_ms", "gating")}, ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
