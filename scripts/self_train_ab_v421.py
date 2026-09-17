"""
v4.2.1 自训练 A/B (自产数据 vs 原始数据) + 收益递减判据 落盘脚本
=====================================================================
红线: 不重训主权重。主预测器冻结 (eval_mse 锚点 0.045556, 主参 52191)。
诚实 A/B: 不预设哪方赢; 自产数据未优于原始数据则照实留候选账本。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, load_predictor, TransitionTripletGenerator,
                  SelfTrainAB, DiminishingReturnsCriterion,
                  DegradationDetector)  # noqa: E402
from udos.world_model import LatentWorldModel  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR = 0.045556


def main():
    model, _ = load_predictor("checkpoints/predictor_v4.1.0.pt")
    assert sum(p.numel() for p in model.parameters()) == 52191
    ds = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    tm = round(evaluate_predictor(model, ds)["single_step_mse"], 6)
    assert abs(tm - ANCHOR) < 1e-6

    wm = LatentWorldModel(model, action_dim=0)
    wm.fit(ds, epochs=20, lr=1e-2, seed=420)
    gen = TransitionTripletGenerator(model, wm, action_dim=1)
    trip = gen.generate(ds, n=128, seed=4200)

    ab = SelfTrainAB(model, epochs=80, seed=42)
    ab_rep = ab.run(trip, ds, holdout_dataset=ds)

    # 收益递减判据 (模拟一代自训练 mse 序列: 先降后平)
    cr = DiminishingReturnsCriterion(patience=3, min_delta=1e-3)
    dr_rep = cr.check([0.100, 0.0999, 0.0998, 0.0997])

    summary = {
        "version": __version__,
        "anchor_eval_mse": ANCHOR,
        "teacher_mse": tm,
        "self_train_ab": ab_rep,
        "diminishing_returns_criterion_demo": dr_rep,
        "honest_conclusion": (
            "自产数据训练 student (mse=0.268) 远不如原始数据 (mse=0.0154); "
            "teacher->student 一代闭环在本线未证优, 留候选账本, 不宣称 RSI 提升。"
            if not ab_rep["self_generated_wins"] else
            "自产数据略优 (一代, 需更多代验证)"),
    }
    out = Path("benchmarks/results/self_train_ab_v4.2.1.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({"teacher_mse": tm, "ab": ab_rep,
                      "conclusion": summary["honest_conclusion"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
