"""
Demo 1: CTM 物理时序推演 (机制对齐 SakanaAI CTM, 替代初版示意 Demo)
================================================================
运行: python -m demos.demo1_ctm_physics
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udos.ctm_engine import CTMConfig, CTMPhysicsEngine
from udos.pce_format import PhysicsSceneEncoder
from udos.debug import DebugPanel
from demos.scene_factory import build_factory_scene


def main():
    torch.manual_seed(0)
    dbg = DebugPanel(level=2)

    scene = build_factory_scene(n_steps=12)

    # 物理 Token -> 定长嵌入序列
    enc = PhysicsSceneEncoder(128)
    enc.bind_scene(scene)
    x = enc(scene.tokens).unsqueeze(0)  # [1, seq, d_input]

    cfg = CTMConfig(
        iterations=32, d_model=256, d_input=128, heads=4,
        n_synch_out=64, n_synch_action=32, memory_length=16,
        out_dims=128, certainty_threshold=0.9,
    )
    engine = CTMPhysicsEngine(cfg)
    dbg.num_params("CTM物理推演引擎", engine)
    dbg.shape("physical_tokens", x)

    with dbg.section("CTM 内部时间轴推演"):
        preds, certs, sync_out, info = engine(x, track=True)

    dbg.shape("predictions(每tick一版)", preds)
    dbg.shape("certainties", certs)
    dbg.shape("sync_out(同步表示)", sync_out)
    dbg.kv("实际使用内部 tick 数", info["ticks_used"])
    dbg.trace("certainty(1-归一化熵)", certs[0, 1])

    print("\n=== Demo1 结论 ===")
    if info["ticks_used"] < cfg.iterations:
        print(f"内部时间轴在第 {info['ticks_used']} tick 达到置信阈值, 自适应早停")
    else:
        print(f"内部时间轴跑满 {cfg.iterations} tick 上限 "
              "(未训练权重 certainty 接近 0 属正常; 训练后会在阈值处早停)")
    print(f"同步表示维度 = {tuple(sync_out.shape)} (random-pairing 神经元对)")
    print(f"预测轨迹 = {tuple(preds.shape)}: 每个 tick 都产出一版物理预测")


if __name__ == "__main__":
    main()
