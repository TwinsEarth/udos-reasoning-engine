"""
Demo 4 (v2.0.0 第二代): 从"机制骨架"到"能学习" —— 物理预测训练闭环
=====================================================================
1) 合成四类物理动力学 (匀速/匀加速/弹簧/弹性碰撞)
2) 对照: 未训练网络 vs "状态不变"朴素基线
3) 训练 PhysicsPredictor (观测编码 + CTM 内部时间轴 + 物理解码头)
4) checkpoint 保存/载入, 逐元素一致
5) 把训练好的预测器挂到 GPM+CTM 协同引擎, 输出可解释的下一时刻 pos/vel

运行:
    python -m demos.demo4_learn_to_predict [epochs]
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udos import (
    CTMConfig, GPMConfig, UDOSReasoningEngine,
    build_dynamics_dataset, naive_baseline_mse,
    PhysicsPredictor, CTMTrainer, TrainConfig,
    save_predictor, load_predictor,
    PhysicalToken, PhysicsScene,
)


def main(epochs: int = 40):
    torch.manual_seed(0)
    print("=" * 68)
    print("Demo4  v2.0.0 学习闭环: 合成动力学 -> 训练 CTM 物理预测器")
    print("=" * 68)

    # 1) 数据
    ds = build_dynamics_dataset(n_per_kind=32, n_steps=12, window=6, dt=0.5)
    tr, te = ds.split(0.8)
    print(f"[数据] 总样本 {len(ds)} (训练 {len(tr)} / 测试 {len(te)}), "
          f"运动类型 {ds.class_names}")

    # 2) 模型与对照基线
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, scene_dim=None)
    model = PhysicsPredictor(cfg)
    print(f"[模型] PhysicsPredictor 参数 "
          f"{sum(p.numel() for p in model.parameters()):,}")
    naive = naive_baseline_mse(te)
    untrained = CTMTrainer(model, TrainConfig(epochs=1)).evaluate(te)
    print(f"[对照] 未训练网络 test MSE = {untrained:.4f}")
    print(f"[对照] '状态不变'朴素基线   = {naive:.4f}")

    # 3) 训练
    trainer = CTMTrainer(model, TrainConfig(epochs=epochs, lr=3e-3))
    hist = trainer.train(tr, te, verbose=True)
    trained = trainer.evaluate(te)
    print(f"[结果] 训练后 test MSE = {trained:.4f}  "
          f"(较未训练 {untrained/max(trained,1e-9):.1f}x, "
          f"较朴素 {naive/max(trained,1e-9):.2f}x), "
          f"certainty {hist.cert[0]:.3f}->{hist.cert[-1]:.3f}")

    # 4) 保存 / 载入
    ckpt = Path(tempfile.gettempdir()) / "udos_predictor_v2.pt"
    save_predictor(model, ckpt, metrics={"test_mse": round(trained, 5)})
    loaded, meta = load_predictor(ckpt)
    xb, _ = next(te.batches(4, shuffle=False))
    same = torch.allclose(model.predict_next(xb), loaded.predict_next(xb),
                          atol=1e-7)
    print(f"[持久化] 保存 {ckpt.name} (version={meta['udos_version']}), "
          f"载入前后逐元素一致 = {same}")

    # 5) 挂到协同引擎, 输出可解释下一时刻物理量
    eng = UDOSReasoningEngine(
        CTMConfig(iterations=6, d_model=64, d_input=32, n_synch_out=16,
                  n_synch_action=8, memory_length=8, out_dims=32,
                  certainty_threshold=0.0),
        GPMConfig(feature_dim=32, latent_size=32, n_latents=8,
                  init_scaler_b_zero=False))
    eng.attach_predictor(loaded)
    scene = PhysicsScene(scene_id="conveyor")
    for t in range(8):  # 近似匀速 0.2/step
        scene.add(PhysicalToken("box", t, position=[0.2 * t, 0, 0],
                                velocity=[0.2, 0, 0],
                                attributes={"mass": 1.0}))
    eng.internalize_scene(scene)
    res = eng.reason(scene, query="下一时刻位置/速度?")
    s = res.summary()
    print(f"[协同] GPM 场景条件化 = {s['scene_conditioned']}, "
          f"ticks = {s['ticks_used']}, certainty = {s['final_certainty']}")
    print(f"[协同] 预测下一时刻 {s['predicted_next_state']}")
    print("=" * 68)


if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(ep)
