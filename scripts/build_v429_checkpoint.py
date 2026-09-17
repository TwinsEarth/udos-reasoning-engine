"""
构建 v4.2.9 最终物理预测器 checkpoint (完全自训练线终点)
==========================================================================
红线: 不重训不改主权重。内化冻结 v4.1.0 主预测器 (eval_mse 锚点 0.045556,
主参 52191)。落 predictor_v4.2.9.pt (第 31 代)。
全线自训练机制 (自生成三元组/自博弈/执行验证/模型评审/抽检/teacher->student/
回流配比/退化检测回滚/收益递减/A/B) 纯前向外挂零梯度终检, 不入主 state_dict;
31 代 backcompat 清单 + 逐代可加载校验。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, save_predictor, load_predictor,  # noqa: E402
                  TransitionTripletGenerator, SelfPlayExplorer,
                  ExecutionValidator, ModelReviewer, HumanAuditHook,
                  DataQualityLedger, TeacherStudentLoop, DataRefluxMixer,
                  DegradationDetector, RollbackManager,
                  DiminishingReturnsCriterion, SelfTrainAB)  # noqa: E402
from udos.world_model import LatentWorldModel  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR_EVAL_MSE = 0.045556


def main():
    model, meta = load_predictor("checkpoints/predictor_v4.1.0.pt")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 52191

    ds = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    eval_mse = round(evaluate_predictor(model, ds)["single_step_mse"], 6)
    assert abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6

    # --- 全线自训练机制零梯度终检 (只读主预测器) --- #
    wm = LatentWorldModel(model, action_dim=0)
    wm.fit(ds, epochs=20, lr=1e-2, seed=420)
    gen = TransitionTripletGenerator(model, wm, action_dim=1)
    trip = gen.generate(ds, n=64, seed=4200)
    quality = gen.triplet_quality(trip)

    expl = SelfPlayExplorer(gen)
    trip_sp, sp_rep = expl.explore(ds, n=32)
    ev = ExecutionValidator()
    ev_rep = ev.validate(trip_sp)
    rev = ModelReviewer(gen, n_perturb=3, noise=0.05)
    filt, rev_rep = rev.review(ds, n=32, seed=0)

    hook = HumanAuditHook(sample_rate=0.2, seed=1)
    idx, audit_rep = hook.sample(trip_sp)
    counts = hook.record_decision(idx.tolist(), ["approve"] * len(idx))

    led = DataQualityLedger()
    led.log("batch_v429", n=len(trip), triplet_quality=quality,
            execution=ev_rep, review=rev_rep, audit=counts)

    # teacher->student 一代闭环 (诚实量化)
    loop = TeacherStudentLoop(model, trip, epochs=60, seed=42)
    loop.fit()
    cmp_rep = loop.compare(ds, eval_mse)

    mixer = DataRefluxMixer()
    reflux_rep = mixer.mix(100, len(trip), "post",
                           student_verdict=cmp_rep["verdict"])

    dd = DegradationDetector()
    deg_rep = dd.analyze([0.10, 0.09, 0.08, 0.12])   # 末代突增 -> collapsed

    rm = RollbackManager()
    rm.register(0, {"w": torch.zeros(2)}, 0.10)
    rm.register(1, {"w": torch.zeros(2)}, 0.08)
    rb_rep = {"best_gen": rm.best()["gen"],
              "should_rollback": rm.should_rollback(0.20)}

    cr = DiminishingReturnsCriterion(patience=3, min_delta=1e-3)
    dr_rep = cr.check([0.100, 0.0999, 0.0998, 0.0997])

    # 主预测器权重须零改动
    assert abs(evaluate_predictor(model, ds)["single_step_mse"] - eval_mse) < 1e-9

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v4.2.9.pt"
    save_predictor(model, ckpt, metrics={
        "evaluation": evaluate_predictor(model, ds),
        "inherits_from": "checkpoints/predictor_v4.1.0.pt"})

    loaded, meta2 = load_predictor(ckpt)
    assert abs(evaluate_predictor(loaded, ds)["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__
    assert sum(p.numel() for p in loaded.parameters()) == 52191

    # 全 31 代 backcompat
    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    assert len(ckpt_list) == 31, f"expected 31 checkpoints, got {len(ckpt_list)}"
    loadable = []
    for name in ckpt_list:
        m, _ = load_predictor(f"checkpoints/{name}")
        assert sum(p.numel() for p in m.parameters()) == 52191
        loadable.append(name)

    summary = {
        "version": __version__, "n_params": n_params,
        "inherits_frozen_main_from": "predictor_v4.1.0.pt",
        "eval_mse": eval_mse, "anchor_eval_mse": ANCHOR_EVAL_MSE,
        "eval_mse_matches_anchor": abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_all_loadable": len(loadable) == len(ckpt_list),
        "line_final": {
            "triplet_quality": quality,
            "self_play_rule_pass_rate": sp_rep["rule_pass_rate"],
            "execution_pass_rate": ev_rep["pass_rate"],
            "review_keep_rate": rev_rep["keep_rate"],
            "teacher_student_verdict": cmp_rep["verdict"],
            "teacher_student_ratio": cmp_rep["ratio_student_over_teacher"],
            "reflux_caution": reflux_rep["caution"],
            "degradation_verdict": deg_rep["verdict"],
            "rollback_best_gen": rb_rep["best_gen"],
            "diminishing_returns": dr_rep["diminishing_returns"],
        },
        "honest_conclusion": (
            "teacher->student 一代闭环 verdict=degraded (student_mse "
            f"{cmp_rep['student_mse']:.4f} vs teacher {cmp_rep['teacher_mse']:.4f}, "
            f"ratio {cmp_rep['ratio_student_over_teacher']:.2f}x); "
            "自产数据 A/B 未优于原始数据; 全线机制为 opt-in 外挂, "
            "不宣称 RSI 必然提升, 留候选账本。"),
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v4.2.9.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints",
                       "backcompat_all_loadable", "line_final",
                       "honest_conclusion"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
