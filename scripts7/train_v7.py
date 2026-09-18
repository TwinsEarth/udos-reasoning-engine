#!/usr/bin/env python3
"""v7 M2：按 held-out 收敛曲线选择模型档位（不预设参数量、不吹倍数）。

对若干 hidden 档位各自独立训练（轨迹级三分），记录参数量、val(oracle/blind)、
test 分类别/分轴指标与估计器逐参数 MAE；选在 val 准则 3% 容差内的**最小**档位
（奥卡姆），保存该档 checkpoint 与收敛曲线。全部 CPU、固定 seed，可复现。

证据分级：verified（本脚本固定 seed 实测，CPU）。
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from udos7 import __version__
from udos7.dynamics import three_way_splits
from udos7.model import WorldModelCore
from udos7.train import TrainConfig, fit
from udos7.metrics import evaluate, estimator_param_error, kinematic_recovery

SIZES = [64, 128, 256]
N_TRAIN = 64
TOL = 0.03
CKPT = REPO / "checkpoints7" / "worldmodel_v7.0.3.pt"
REPORT = REPO / "reports7" / "model_size_convergence.json"


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    splits = three_way_splits(n_traj_per_kind=N_TRAIN)
    H = splits["train"].horizon
    print({k: (len(v), v.trajectories()) for k, v in splits.items()})

    runs = []
    for hidden in SIZES:
        t0 = time.time()
        torch.manual_seed(0)
        model = WorldModelCore(window=6, hidden=hidden, n_layers=2)
        npar = model.num_parameters()
        cfg = TrainConfig(epochs=80, batch=128, lr=2e-3, patience=20,
                          scene_dropout=0.5, seed=0)
        hist = fit(model, splits["train"], splits["val"], cfg, verbose=True)
        test_oracle = evaluate(model, splits["test"], H, use_explicit=True)
        test_blind = evaluate(model, splits["test"], H, use_explicit=False)
        param_err = estimator_param_error(model, splits["test"])
        kin = kinematic_recovery(splits["test"])
        run = {
            "hidden": hidden, "n_layers": 2, "params": npar,
            "use_kinematics": True,
            "epochs_run": len(hist.train_loss),
            "best_epoch": hist.best_epoch,
            "best_val_criterion": round(hist.best_val, 6),
            "val_oracle_curve": [round(x, 5) for x in hist.val_oracle],
            "val_blind_curve": [round(x, 5) for x in hist.val_blind],
            "train_loss_curve": [round(x, 5) for x in hist.train_loss],
            "test_oracle": {k: round(v, 5) for k, v in test_oracle.items()},
            "test_blind": {k: round(v, 5) for k, v in test_blind.items()},
            "estimator_param_error": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in param_err.items()},
            "kinematic_recovery": kin,
            "train_sec": round(time.time() - t0, 1),
        }
        runs.append(run)
        print(f"[hidden={hidden}] params={npar} best_val={hist.best_val:.5f} "
              f"test_oracle={test_oracle['oracle_overall']:.5f} "
              f"test_blind={test_blind['blind_overall']:.5f} "
              f"({run['train_sec']}s)")

    # 档位选择：val 准则在最优值 3% 内的最小档
    best = min(r["best_val_criterion"] for r in runs)
    chosen = next(r for r in runs
                  if r["best_val_criterion"] <= best * (1 + TOL))
    print("CHOSEN hidden =", chosen["hidden"], "params =", chosen["params"])

    # 重训选中档（fit 已含 best_state，这里直接重建并再拟合一次确保落盘干净）
    torch.manual_seed(0)
    model = WorldModelCore(window=6, hidden=chosen["hidden"], n_layers=2)
    fit(model, splits["train"], splits["val"],
        TrainConfig(epochs=80, batch=128, lr=2e-3, patience=20,
                    scene_dropout=0.5, seed=0))
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {"window": 6, "hidden": chosen["hidden"], "n_layers": 2,
                   "scene_dim": 32, "use_kinematics": True},
        "version": __version__,
        "evidence_grade": "verified",
        "runtime": {"torch": torch.__version__, "threads": 2, "device": "cpu"},
        "splits": {"train_seed": 42, "val_seed": 1337, "test_seed": 2026,
                   "n_traj_per_kind_train": N_TRAIN},
        "test_metrics": {"oracle": chosen["test_oracle"],
                         "blind": chosen["test_blind"]},
    }, CKPT)

    report = {
        "version": __version__, "evidence_grade": "verified",
        "torch_version": torch.__version__, "threads": 2,
        "dataset": {k: {"windows": len(v), "trajectories": v.trajectories(),
                        "horizon": v.horizon} for k, v in splits.items()},
        "selection_rule": f"smallest hidden with val_criterion <= best*(1+{TOL})",
        "chosen_hidden": chosen["hidden"], "chosen_params": chosen["params"],
        "runs": runs,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("saved", CKPT)
    print("saved", REPORT)


if __name__ == "__main__":
    main()
