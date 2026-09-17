"""v5.5.5 收尾: 汇总 5.4.5-5.5.4 双引擎增量线的关键 A/B 与可信度证据。

读取 reports/v549..v554, 输出单一 reports/v555_regression_summary.json,
并现场复核冻结锚点 (主预测员 52191 参数、场景头 6788 参数)。
用法: PYTHONPATH=. python3 scripts/v555_regression_summary.py
"""
from __future__ import annotations

import json
import os

import torch

from udos.persistence import load_predictor
from udos.scene_head import load_scene_head

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(HERE, "reports")
CKPT = os.path.join(HERE, "checkpoints")


def _load(name):
    with open(os.path.join(REPORTS, name), encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    r549 = _load("v549_scene_ab.json")
    r550 = _load("v550_head_ab.json")
    r551 = _load("v551_fan_coverage.json")
    r552 = _load("v552_kind_fan.json")
    r553 = _load("v553_gate_observation.json")
    r554 = _load("v554_lora_forward.json")

    # 现场复核冻结锚点
    pred, _ = load_predictor(os.path.join(CKPT, "predictor_v4.3.9.pt"))
    n_pred = sum(p.numel() for p in pred.parameters() if p.requires_grad)
    n_buf = len(list(pred.buffers()))
    head, _ = load_scene_head(os.path.join(CKPT, "scene_head_v5.5.0.pt"))
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)

    def ov(r, key, *fields):
        cur = r["overall"][key]
        return {f: round(cur[f], 4) for f in fields}

    kind_fan = r552["fan_coverage"]["per_kind"]
    summary = {
        "version": "5.5.5",
        "increment_line": "v5.4.5 -> v5.5.5 (双引擎名副其实: CTM 主预测 + GPM 场景记忆真正影响主预测)",
        "frozen_anchors": {
            "predictor_params": n_pred,
            "predictor_buffers": n_buf,
            "scene_head_params": n_head,
            "predictor_unchanged": n_pred == 52191,
            "head_unchanged": n_head == 6788,
        },
        "scene_gate_is_lifeline_549": {
            "single_step": ov(r549, "single_step",
                              "blind", "explicit", "estimated", "recovery"),
            "rollout_4": ov(r549, "rollout_4",
                            "blind", "explicit", "estimated", "recovery"),
        },
        "learned_head_550": {
            "head_trainable_params": r550["head_trainable_params"],
            "single_step": ov(r550, "single_step",
                              "blind", "explicit", "classical", "learned",
                              "learned_recovery"),
            "rollout_4": ov(r550, "rollout_4",
                            "blind", "explicit", "classical", "learned",
                            "learned_recovery"),
        },
        "global_conformal_fan_551": {
            "nominal": r551["nominal_coverage"],
            "n_samples": r551["n_samples"],
            "inflation": r551["conformal_inflation"],
            "raw_mc_coverage": r551["heldout"]["raw_mc_coverage"],
            "calibrated_coverage": r551["heldout"]["calibrated_coverage"],
            "calibrated_per_step": r551["heldout"]["calibrated_per_step"],
            "known_gap": "accel 类全局高斯欠覆盖, 交 5.5.2 类型条件化",
        },
        "kind_conditional_fan_552": {
            "nominal": r552["nominal_coverage"],
            "raw_pooled": r552["fan_coverage"]["raw_pooled"],
            "calibrated_pooled": r552["fan_coverage"]["calibrated_pooled"],
            "per_kind_calibrated": {
                k: round(v["calibrated"], 4) for k, v in kind_fan.items()},
            "per_kind_inflation": {
                k: round(v["inflation"], 3) for k, v in kind_fan.items()},
            "routed_rollout4_recovery": {
                k: round(v["recovery"], 4)
                for k, v in r552["routed_classical_ab_rollout4"].items()},
            "honest_note": "accel 膨胀因子 8.25 = 很宽带, 暴露高斯参数误差不覆盖"
                           "加速 rollout 结构误差; 留 v7 多水平/残差建模根治",
        },
        "gate_observation_553": {
            "source": r553["conditioning_source"],
            "per_kind": {
                k: {"route_accuracy": v["route_accuracy"],
                    "blind_rollout4_mse": v["blind_rollout4_mse"],
                    "conditioned_rollout4_mse": v["conditioned_rollout4_mse"],
                    "gate_final_delta_mean": v["gate_final_delta_mean"]}
                for k, v in r553["per_kind"].items()},
            "note": "route_confidence 为判别果断度启发式, 非概率; 干净合成数据饱和 1.0",
        },
        "lora_forward_path_554": {
            "live_injection_delta": {
                k: round(v["injection_delta"], 4)
                for k, v in r554["live"].items()},
            "live_reset_error_max": max(
                v["reset_error"] for v in r554["live"].values()),
            "training_scaler_b_zero_injection_delta_max": max(
                v["injection_delta"]
                for v in r554["training_scaler_b_zero"].values()),
            "patched_modules": r554["live"]["uniform"]["patched_modules"],
            "lora_params": r554["live"]["uniform"]["lora_params"],
            "scope": "演示基座 GPM->LoRA->前向闭环; 不接入冻结物理预测员",
        },
        "deferred_to_v7": [
            "conformal 多水平在 v4.3.9 失效 (名义 80/90/95 经验覆盖相同), 需残差/多水平重写",
            "加速类结构误差的非高斯建模",
            "GPU 栈 (vLLM KV offload / NEURON DHS / MuJoCo MIMIC / 0.5B 端到端) 需云预算",
        ],
    }

    path = os.path.join(REPORTS, "v555_regression_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
