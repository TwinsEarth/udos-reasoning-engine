"""
Demo 5 (v2.1.0): 多步滚动推演 + 场景条件联合训练 + 物理一致性
=============================================================
第二代的场景门控在 v2.0 从未被训练 (恒为 0); 本演示:
  1. 构造"观测窗口 + 隐藏物理参数 + 多步未来"的参数化动力学数据集
  2. 训练带 scene_encoder 的物理预测器 (teacher-forcing 多步)
  3. 消融对照: 给/不给隐藏参数, 证明场景条件确实提升预测
  4. 自由 rollout: 看误差随推演步数累积
  5. 运动学一致性诊断 (一阶欧拉残差, 分运动类型)
运行: python demos/demo5_multistep_reasoning.py   (CPU 约 20-30 秒)
"""
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (build_parametric_dataset, evaluate_predictor,
                  save_predictor, load_predictor, SCENE_PARAM_NAMES)
from udos.ctm_engine import CTMConfig
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig


def main():
    torch.manual_seed(0)
    print("=" * 66)
    print(" UDOS v2.1.0  多步滚动推演 / 场景条件 / 物理一致性")
    print("=" * 66)

    # 1) 参数化数据: 仅凭 6 帧观测窗口无法唯一确定未来, 需要隐藏物理参数 P
    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5)
    tr, te = ds.split(0.8)
    print(f"[数据] 样本 {len(ds)}  窗口 {ds.X.shape[1]} 帧  "
          f"多步目标 {ds.horizon} 步  隐藏参数 {ds.P.shape[1]} 维 "
          f"{list(SCENE_PARAM_NAMES)}")
    print("       运动类型:", ", ".join(ds.class_names))

    # 2) 带场景编码器的预测器
    cfg = CTMConfig(iterations=6, d_model=48, d_input=24, heads=2,
                    n_synch_out=12, n_synch_action=8, memory_length=6,
                    nlm_hidden=12, out_dims=24, certainty_threshold=0.0)
    model = PhysicsPredictor(cfg, scene_param_dim=ds.P.size(1))
    print(f"[模型] 参数 {sum(p.numel() for p in model.parameters()):,}  "
          f"初始 scene_gate={float(model.ctm.scene_gate.detach()):.3f} "
          f"(零门控=初始等价无条件, 向后兼容)")

    before = evaluate_predictor(model, te)["ablation"]
    # 3) teacher-forcing 多步训练
    CTMTrainer(model, TrainConfig(epochs=30, lr=3e-3)).train(tr)
    rep = evaluate_predictor(model, te)
    abl = rep["ablation"]
    print("\n[消融] 训练前: conditioned=%.3f unconditioned=%.3f"
          % (before["conditioned_mse"], before["unconditioned_mse"]))
    print("[消融] 训练后: conditioned=%.3f unconditioned=%.3f  "
          "场景条件增益 %.2fx"
          % (abl["conditioned_mse"], abl["unconditioned_mse"],
             abl["condition_gain_x"]))
    print("       scene_gate 学到 %.3f (不再恒为 0)"
          % float(model.ctm.scene_gate.detach()))

    # 4) 自由 rollout 误差累积
    print("\n[多步] 自由滚动逐步 MSE:",
          " -> ".join(f"{x:.3f}" for x in rep["rollout_mse_curve"]))
    per_kind = rep["per_kind_mse"]
    print("[分类] 单步 MSE:",
          ", ".join(f"{k}={v:.3f}" for k, v in per_kind.items()))

    # 5) 物理一致性 (一阶欧拉, 匀速段应最准)
    kin = rep["kinematic_residual"]
    print("\n[一致性] 运动学残差 overall=%.3f" % kin["overall"])
    print("         分类型:",
            ", ".join(f"{k}={v:.3f}" for k, v in kin["per_kind"].items()))
    print("         (匀速段一阶欧拉最准; 加速/振动段真值固有 O(a*dt^2) 偏离)")

    # 6) 存/载一致
    with tempfile.TemporaryDirectory() as d:
        p = save_predictor(model, Path(d) / "m.pt", metrics=rep)
        m2, meta = load_predictor(p)
        x, pp = te.X[:4], te.P[:4]
        same = torch.allclose(model.predict_next(x, scene_params=pp),
                              m2.predict_next(x, scene_params=pp), atol=1e-6)
        print("\n[持久化] 带场景编码器 checkpoint 存/载逐元素一致:", same,
              " scene_param_dim =", meta["scene_param_dim"])
    print("=" * 66)


if __name__ == "__main__":
    main()
