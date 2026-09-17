"""
Demo 3: CTM + GPM 双引擎协同 + 与 SakanaAI 上游真实 CTM 交叉验证
================================================================
流程: internalize_scene (GPM) -> reason (CTM) -> reset_scene
并加载 third_party/ctm 的真实 ContinuousThoughtMachine 做同构对照。

运行: python -m demos.demo3_dual_engine
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.reasoning import UDOSReasoningEngine
from udos.debug import DebugPanel
from udos.adapters import SakanaCTMAdapter
from demos.scene_factory import build_factory_scene, build_scene_chunks


def run_dual_engine(dbg):
    torch.manual_seed(0)
    scene = build_factory_scene(n_steps=12)
    _, chunks = build_scene_chunks(n_chunks=3, step_per_chunk=4)

    base = TinyBaseModel(hidden=128, n_layers=4)
    gpm_cfg = GPMConfig(
        feature_dim=128, latent_size=128, n_latents=16, lora_rank=8,
        target_modules=("down_proj", "gate_proj", "up_proj"),
        layer_indices=(0, 1, 2, 3),
        init_scaler_b_zero=False)  # 未训练演示: 让生成的 LoRA 非零
    ctm_cfg = CTMConfig(
        iterations=32, d_model=256, d_input=128, heads=4,
        n_synch_out=64, n_synch_action=32, memory_length=16,
        out_dims=128, certainty_threshold=0.9)

    engine = UDOSReasoningEngine(ctm_cfg, gpm_cfg, base_model=base)

    # 步骤 1: GPM 分块内化
    msg = engine.internalize_scene(scene, chunks=chunks)
    print("[1] internalize:", msg)

    # 步骤 2: CTM 时序推演
    with dbg.section("双引擎 reason"):
        result = engine.reason(
            scene, query="机械臂何时与传送带同步并完成抓取?", track=True)
    print("[2] reason 摘要:", result.summary())
    dbg.trace("certainty 轨迹", result.certainty_trajectory[1])
    print("    因果链(前6条):")
    for edge in result.causal_chain[:6]:
        print(f"      {edge['from']} -> {edge['to']} [{edge['source']}]")

    # 步骤 3: 无损移除场景记忆
    print("[3]", engine.reset_scene(scene.scene_id))
    return result


def cross_validate_upstream(dbg):
    print("\n--- 与 SakanaAI 上游真实 CTM 交叉验证 ---")
    if not SakanaCTMAdapter.available():
        print("上游运行依赖缺失 (huggingface_hub/numpy 或 third_party/ctm), 跳过")
        return
    adapter = SakanaCTMAdapter(iterations=8, d_model=128, d_input=128,
                               n_synch_out=32, n_synch_action=32,
                               memory_length=8, out_dims=64)
    with dbg.section("加载上游 ContinuousThoughtMachine"):
        adapter.load()
    dbg.kv("上游 CTM 参数量", f"{adapter.num_params():,}")

    seq = torch.randn(1, 10, 128)
    preds, certs, sync = adapter.forward(seq)
    dbg.shape("上游 predictions [B,out,ticks]", preds)
    dbg.shape("上游 certainties", certs)
    dbg.shape("上游 sync_out", sync)
    print("上游真实 CTM 前向成功: 内部时间轴 ticks =", preds.shape[-1])


def main():
    dbg = DebugPanel(level=2)
    run_dual_engine(dbg)
    cross_validate_upstream(dbg)


if __name__ == "__main__":
    main()
