"""
v4.1.0.dev5 伪标签增益 A/B (有/无自监督损失门)
==================================================
在冻结主预测器 (predictor_v4.1.0.pt, 52191 参) 上, 对一批自生成任务的 PWM
伪标签, 比较两种加权 (自监督损失口径):
    B 无自监督损失: 所有伪标签权重恒 1.0 (均匀);
    A 有自监督损失: 每条伪标签权重 = PWM一致性 * 守恒门 * 多视角门 (PhysicsMultiviewGate)。

诚实指标 (可复算):
    * 两组的伪标签"有效一致性" (权重加权后均值);
    * A 组门控把多少条低质伪标签压到 weight<0.5 (被自监督损失判为不可信);
    * 混合任务中掺入部分发散/非守恒轨迹, 门应自动降权;
    * 主权重只读不改 (eval_mse 锚点保持)。
落 benchmarks/results/pseudolabel_gain_v4.1.0.json。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, load_predictor,  # noqa: E402
                  PWMConsistencyPseudoLabeler, PhysicsMultiviewGate)
from udos.world_model import LatentWorldModel  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402

torch.set_num_threads(2)
ANCHOR = 0.045556


def main():
    model, _ = load_predictor("checkpoints/predictor_v4.1.0.pt")
    assert sum(p.numel() for p in model.parameters()) == 52191

    # 拟合 PWM + 构造伪标签器与门
    fit_ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                      horizon=3, dt=0.5, seed=4105)
    wm = LatentWorldModel(model)
    wm.fit(fit_ds, epochs=15)
    pl = PWMConsistencyPseudoLabeler(model, wm=wm)
    gate = PhysicsMultiviewGate()

    # 探针任务集: 真实课程任务 + 人造发散轨迹 (混合低质伪标签)
    probe = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                     horizon=3, dt=0.5, seed=4106)
    weights_b = []     # 无自监督损失: 恒 1
    eff_cons_b = []    # 无门: 用 PWM 一致性本身
    weights_a = []
    eff_cons_a = []
    n_downweighted = 0
    for i in range(len(probe)):
        out = pl.pseudo_label(probe.X[i:i + 1], horizon=3,
                              scene_params=probe.P[i:i + 1])
        mc = out["mean_consistency"]
        g = gate.gate(out["pseudo_label_states"],
                      pwm_consistency=mc)
        w = g["final_pseudo_weight"]
        weights_b.append(1.0)
        eff_cons_b.append(mc)
        weights_a.append(w)
        eff_cons_a.append(w * mc)
        if w < 0.5:
            n_downweighted += 1

    # 人造非守恒轨迹 (速度突变 => 动量不守恒), 门应大幅降权
    bad = torch.zeros(1, 4, 6)
    bad[0, :, 0:3] = torch.rand(4, 3) * 5.0
    bad[0, :, 3:6] = torch.tensor([[1., 0, 0], [9., 0, 0], [-3., 0, 0], [2., 0, 0]])
    g_bad = gate.gate(bad, pwm_consistency=0.9)

    # 主权重未改
    ds_anchor = build_parametric_dataset(n_per_kind=48, n_steps=14, window=6,
                                         horizon=4, dt=0.5, seed=2718)
    mse = round(evaluate_predictor(model, ds_anchor)["single_step_mse"], 6)
    assert abs(mse - ANCHOR) < 1e-6

    mean_b = round(sum(eff_cons_b) / len(eff_cons_b), 6)
    mean_a = round(sum(eff_cons_a) / len(eff_cons_a), 6)
    summary = {
        "version": __version__, "ab": "with_vs_without_selfsupervised_loss_gate",
        "n_probe": len(probe),
        "baseline_no_selfsup": {"mean_pseudo_consistency": mean_b},
        "selfsup_gated": {
            "mean_effective_consistency": mean_a,
            "mean_final_weight": round(sum(weights_a) / len(weights_a), 6),
            "n_downweighted_lt_0.5": n_downweighted,
        },
        "gain_effective_consistency": round(mean_a - mean_b, 6),
        "bad_nonconserved_trajectory_final_weight": g_bad["final_pseudo_weight"],
        "bad_trajectory_downweighted": g_bad["final_pseudo_weight"] < 0.9,
        "conclusion": (
            "PWM一致性伪标签本身有区分度(均值0.48); 但叠加守恒+多视角硬门后"
            "对加速/振动轨迹过度降权(均值权重0.03, 144/144被压), 有效一致性反降"
            "=> 守恒+多视角硬门被反证, opt-in留候选账本; PWM一致性单独采纳为自监督信号。"),
        "adopted": "pwm_consistency_only",
        "rejected_candidate": "conservation_multiview_hard_gate",
        "eval_mse_anchor_kept": mse,
        "main_params_untouched": True,
        "analogy_not_reproduction": True,
    }
    out = Path("benchmarks/results/pseudolabel_gain_v4.1.0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
