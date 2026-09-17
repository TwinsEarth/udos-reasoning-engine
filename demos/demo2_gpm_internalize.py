"""
Demo 2: GPM 物理场景内化 (机制对齐 SakanaAI Doc-to-LoRA)
========================================================
- 单次前向、零反向传播把物理场景生成 LoRA A/B
- 分块 + 平均聚合
- 前向补丁式注入 / reset 零误差还原 (相对初版改权重 Demo 的关键修复)

运行: python -m demos.demo2_gpm_internalize
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udos.gpm_engine import (
    GPMConfig, PhysicsHypernetwork, LoRAInjector,
    TinyBaseModel, infer_dims_from_model, aggregate_loras,
)
from udos.debug import DebugPanel
from demos.scene_factory import build_scene_chunks


def main():
    torch.manual_seed(0)
    dbg = DebugPanel(level=2)

    full, chunks = build_scene_chunks(n_chunks=3, step_per_chunk=4)

    # 1) 轻量基座 + 自动探测 LoRA 维度
    base = TinyBaseModel(hidden=128, n_layers=4)
    cfg = GPMConfig(
        feature_dim=128, latent_size=128, n_latents=16, lora_rank=8,
        target_modules=("down_proj", "gate_proj", "up_proj"),
        layer_indices=(0, 1, 2, 3),
        init_scaler_b_zero=False,  # 未训练演示: 让生成的 LoRA 非零
    )
    cfg.dims = infer_dims_from_model(base, cfg.target_modules)
    hyper = PhysicsHypernetwork(cfg)
    dbg.num_params("GPM超网络", hyper)
    dbg.kv("自动探测目标层维度", cfg.dims)

    # 2) 单次前向生成 LoRA (无梯度)
    with dbg.section("单场景 -> LoRA"):
        lora = hyper(full)
    for m in lora.module_names():
        A, B = lora.AB[m]
        dbg.shape(f"{m}.A", A), dbg.shape(f"{m}.B", B)
    dbg.kv("LoRA 参数总量", f"{lora.num_params():,} ({lora.num_bytes_fp32()/1024:.1f} KB)")

    # 3) 分块聚合
    with dbg.section("3 分块 -> 平均聚合 LoRA"):
        chunked = hyper.forward_chunked(chunks)
    dbg.kv("分块聚合 LoRA 参数", f"{chunked.num_params():,}")

    # 4) 前向补丁注入 + 无损 reset 验证
    injector = LoRAInjector(base, scaling=0.1)
    probe = torch.randn(2, 5, 128)
    with torch.no_grad():
        y_before = base(probe).clone()
        injector.inject(lora)
        y_injected = base(probe).clone()
        injector.reset()
        y_after_reset = base(probe).clone()

    drift_inject = (y_injected - y_before).abs().mean().item()
    drift_reset = (y_after_reset - y_before).abs().max().item()
    print("\n=== Demo2 结论 ===")
    print(f"注入后输出平均改变量 = {drift_inject:.6f} (场景已被参数化记忆)")
    print(f"reset 后最大还原误差 = {drift_reset:.3e} (前向补丁零误差, 不改原始权重)")
    assert drift_reset == 0.0, "前向补丁必须可无损还原"


if __name__ == "__main__":
    main()
