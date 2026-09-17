"""
v2.1.0 可复现实验: 训练场景条件多步物理预测器
  - 训练前/后对照 (单步 MSE、场景条件消融、分类型、rollout 曲线、运动学一致性)
  - 结果落 benchmarks/results/evaluation_v2.1.0.json
  - 模型落 checkpoints/predictor_v2.1.0.pt (供服务 --checkpoint 开箱预加载)
CPU 可跑, 约 1 分钟。用法: python scripts/build_v21_checkpoint.py
"""
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import (__version__, build_parametric_dataset, evaluate_predictor,
                  save_predictor)
from udos.ctm_engine import CTMConfig
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig


def main():
    torch.manual_seed(42)
    t_start = time.time()
    ds = build_parametric_dataset(n_per_kind=32, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=42)
    tr, te = ds.split(0.8)
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    model = PhysicsPredictor(cfg, scene_param_dim=ds.P.size(1))
    n_params = sum(p.numel() for p in model.parameters())
    before = evaluate_predictor(model, te)
    t0 = time.time()
    hist = CTMTrainer(model, TrainConfig(epochs=45, lr=3e-3)).train(tr, te)
    train_s = time.time() - t0
    after = evaluate_predictor(model, te)

    record = {
        "udos_version": __version__,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": {"python": platform.python_version(),
                    "torch": torch.__version__, "device": "cpu",
                    "train_seconds": round(train_s, 2),
                    "total_seconds": round(time.time() - t_start, 2)},
        "config": {"n_per_kind": 32, "n_steps": 14, "window": 6, "horizon": 4,
                    "dt": 0.5, "epochs": 45, "lr": 3e-3,
                    "predictor_params": n_params,
                    "n_train": len(tr), "n_test": len(te)},
        "before_training": before,
        "after_training": after,
        "final_train_loss": round(hist.train_loss[-1], 6),
        "scene_gate_final": round(float(model.ctm.scene_gate.detach()), 4),
    }
    out_json = ROOT / "benchmarks" / "results" / "evaluation_v2.1.0.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    ckpt = ROOT / "checkpoints" / "predictor_v2.1.0.pt"
    save_predictor(model, ckpt, metrics=after)

    a = after["ablation"]
    print("== v2.1.0 评估完成 ==")
    print(f"参数 {n_params}  训练 {train_s:.1f}s")
    print(f"单步 MSE: 训练前 {before['single_step_mse']:.4f} -> "
          f"训练后 {after['single_step_mse']:.4f}")
    print(f"场景条件消融: conditioned={a['conditioned_mse']:.4f} "
          f"unconditioned={a['unconditioned_mse']:.4f} "
          f"gain={a['condition_gain_x']}x  scene_gate={record['scene_gate_final']}")
    print(f"rollout 曲线: {after['rollout_mse_curve']}")
    print(f"运动学残差: {after['kinematic_residual']}")
    print(f"写入 {out_json.relative_to(ROOT)} 与 {ckpt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
