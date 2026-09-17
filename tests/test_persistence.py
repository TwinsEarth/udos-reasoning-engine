"""v2.0.0 持久化测试: 存/载逐元素一致、元数据完整、通用模块存载。"""

import torch

from udos import (UDOSReasoningEngine, CTMConfig, GPMConfig, __version__,
                  PhysicsPredictor, save_predictor, load_predictor,
                  save_module, load_module_state)


def _predictor():
    return PhysicsPredictor(CTMConfig(
        iterations=4, d_model=32, d_input=16, n_synch_out=8,
        n_synch_action=6, memory_length=4, out_dims=16,
        certainty_threshold=0.0))


def test_predictor_save_load_elementwise_equal(tmp_path):
    m = _predictor(); m.eval()
    x = torch.randn(3, 5, 6)
    before = m.predict_next(x)

    path = save_predictor(m, tmp_path / "pred.pt",
                          metrics={"eval_mse": 0.04})
    m2, meta = load_predictor(path)
    after = m2.predict_next(x)

    assert torch.allclose(before, after, atol=1e-7)
    assert meta["kind"] == "PhysicsPredictor"
    assert meta["udos_version"] == __version__ == "5.5.5"
    assert meta["metrics"] == {"eval_mse": 0.04}
    assert meta["raw_dim"] == 6


def test_save_load_preserves_scene_dim(tmp_path):
    # 带场景条件化的配置也要能完整重建
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, n_synch_out=8,
                    n_synch_action=6, memory_length=4, out_dims=16,
                    scene_dim=16, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg)
    _, meta = load_predictor(save_predictor(m, tmp_path / "c.pt"))
    assert meta["ctm_config"]["scene_dim"] == 16


def test_generic_module_roundtrip(tmp_path):
    eng = UDOSReasoningEngine(
        CTMConfig(iterations=3, d_model=32, d_input=16, n_synch_out=8,
                  n_synch_action=6, memory_length=4, out_dims=16,
                  certainty_threshold=0.0),
        GPMConfig(feature_dim=16, latent_size=16, n_latents=4,
                  init_scaler_b_zero=False))
    eng.eval()
    path = save_module(eng, tmp_path / "eng.pt", kind="UDOSReasoningEngine")
    eng2 = UDOSReasoningEngine(
        CTMConfig(iterations=3, d_model=32, d_input=16, n_synch_out=8,
                  n_synch_action=6, memory_length=4, out_dims=16,
                  certainty_threshold=0.0),
        GPMConfig(feature_dim=16, latent_size=16, n_latents=4,
                  init_scaler_b_zero=False))
    meta = load_module_state(eng2, path)
    assert meta["kind"] == "UDOSReasoningEngine"
    # 同名参数载入后逐元素一致
    for (n1, p1), (n2, p2) in zip(eng.named_parameters(),
                                  eng2.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2, atol=1e-8)
