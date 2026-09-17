"""
dev6 A/B: flow-matching 少步动作解码器 vs 直接回归 (对标 MMDiT flow)
======================================================================
在冻结 predictor 的 latent 上, 同口径比较两种连续动作解码:
    A) flow:   FlowMatchingDecoder.decode —— n_steps 步 Euler 去噪;
    B) regress: FlowMatchingDecoder.decode_regress —— z 直接线性映射动作。
两者共用同一外挂解码器实例与同一拟合损失 (flow + 回归联合训练)。
落 benchmarks/results/wla_flow_vs_regress_ab.json。
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig  # noqa: E402
from udos.wla import FlowMatchingDecoder  # noqa: E402

torch.set_num_threads(2)


def small_cfg():
    return CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                     n_synch_out=16, n_synch_action=8, memory_length=8,
                     nlm_hidden=16, out_dims=32, certainty_threshold=0.0)


def main():
    torch.manual_seed(42)
    n_per_kind = 32
    train = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                     window=6, horizon=4, dt=0.5, seed=42)
    tr, te_es = train.split(0.8)
    model = PhysicsPredictor(small_cfg(), scene_param_dim=4)
    CTMTrainer(model, TrainConfig(epochs=20, lr=3e-3, batch_size=64,
                                  patience=6, step_weight_scheme="front",
                                  hybrid_weight=0.0)).train(tr, te_es)

    # 用 predictor  latent 作 z, 合成"动作"目标 (状态末帧前 6 维代理)
    ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=2718)
    with torch.no_grad():
        z = model.obs_encoder(ds.X)[:, -1, :]          # [N,32]
    a = ds.Y[:, 0, :6]                                 # [N,6] 下一状态前 6 维代理动作

    n = z.size(0)
    n_tr = int(n * 0.8)
    dec = FlowMatchingDecoder(32, 6, hidden=16, n_steps=3)
    rep = dec.fit(z[:n_tr], a[:n_tr], epochs=40)

    za, aa = z[n_tr:], a[n_tr:]
    flow_mse = float(((dec.decode(za, seed=0) - aa) ** 2).mean())
    reg_mse = float(((dec.decode_regress(za) - aa) ** 2).mean())

    verdict = {
        "backbone": "frozen_predictor_latent",
        "flow_decoder_params": dec.n_params,
        "n_steps": dec.n_steps,
        "fit": rep,
        "flow_mse": round(flow_mse, 6),
        "regress_mse": round(reg_mse, 6),
        "flow_better": bool(flow_mse <= reg_mse),
        "note": "flow 为少步去噪类比; 在合成小数据上收益不稳时默认 opt-in 并留候选账本",
        "frozen_backbone_zero_grad": True,
        "analogy_not_reproduction": True,
        "not_reproduce_6b_mmdit": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/wla_flow_vs_regress_ab.json", "w",
              encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
