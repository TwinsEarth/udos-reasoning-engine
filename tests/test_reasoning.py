import torch

from udos.ctm_engine import CTMConfig
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.reasoning import UDOSReasoningEngine


def _engine():
    base = TinyBaseModel(hidden=32, n_layers=2)
    gpm = GPMConfig(feature_dim=32, latent_size=32, n_latents=8, lora_rank=4,
                    layer_indices=(0, 1), num_pre_head_layers=1, heads=2)
    ctm = CTMConfig(iterations=12, d_model=64, d_input=32, heads=2,
                    n_synch_out=16, n_synch_action=16, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    return UDOSReasoningEngine(ctm, gpm, base_model=base)


def test_full_pipeline(tiny_scene, torch_seed):
    eng = _engine()
    msg = eng.internalize_scene(tiny_scene)
    assert "已内化" in msg
    assert tiny_scene.scene_id in eng.scene_memory

    result = eng.reason(tiny_scene, query="test-query")
    assert result.prediction.shape == (32,)
    assert result.prediction_trajectory.shape[0] == 32
    assert result.ticks_used == 12
    assert 0.0 <= result.convergence() <= 1.0
    assert result.lora_params > 0
    # 因果链: 显式 + 时间邻接
    sources = {e["source"] for e in result.causal_chain}
    assert "pce-explicit" in sources
    assert "temporal-adjacent" in sources


def test_reset_clears_memory(tiny_scene, torch_seed):
    eng = _engine()
    eng.internalize_scene(tiny_scene)
    assert eng.injector.active is not None
    eng.reset_scene(tiny_scene.scene_id)
    assert eng.injector.active is None
    assert tiny_scene.scene_id not in eng.scene_memory


def test_reason_without_internalize(tiny_scene, torch_seed):
    eng = _engine()
    # 不内化也可纯 CTM 推演, lora_params=0
    result = eng.reason(tiny_scene)
    assert result.lora_params == 0
    assert result.ticks_used == 12
