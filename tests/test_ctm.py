import torch

from udos.ctm_engine import CTMConfig, CTMPhysicsEngine, normalized_entropy


def _engine(**kw):
    defaults = dict(iterations=16, d_model=64, d_input=32, heads=2,
                    n_synch_out=16, n_synch_action=16, memory_length=8,
                    nlm_hidden=16, out_dims=24, certainty_threshold=0.0,
                    n_random_pairing_self=2)
    defaults.update(kw)
    return CTMPhysicsEngine(CTMConfig(**defaults))


def test_ctm_output_shapes(torch_seed):
    eng = _engine()
    x = torch.randn(3, 7, 32)
    preds, certs, sync, info = eng(x, track=True)
    assert preds.shape == (3, 24, 16)
    assert certs.shape == (3, 2, 16)
    assert sync.shape == (3, 16)
    assert info["ticks_used"] == 16
    assert info["sync_history"].shape[:2] == (3, 16)


def test_certainty_bounds(torch_seed):
    eng = _engine()
    _, certs, _, _ = eng(torch.randn(1, 5, 32))
    assert torch.all(certs >= 0) and torch.all(certs <= 1)
    # 两路 (熵, 1-熵) 互补
    assert torch.allclose(certs[:, 0] + certs[:, 1], torch.ones_like(certs[:, 0]),
                          atol=1e-5)


def test_adaptive_early_stop(torch_seed):
    # 阈值关闭 -> 跑满; 给一个极易"自信"的高阈值时 ticks <= iterations
    eng_full = _engine(certainty_threshold=0.0)
    _, _, _, info_full = eng_full(torch.randn(1, 5, 32))
    assert info_full["ticks_used"] == 16

    eng_stop = _engine(certainty_threshold=-1.0)  # 负值关闭
    _, _, _, info_stop = eng_stop(torch.randn(1, 5, 32))
    assert info_stop["ticks_used"] == 16


def test_early_stop_triggers_on_high_certainty(torch_seed):
    # 把输出头偏置置为尖锐 one-hot 倾向 -> certainty≈1 -> 第 1 tick 即早停
    eng = _engine(certainty_threshold=0.99, out_dims=8)
    with torch.no_grad():
        eng.output_projector.bias.zero_()
        eng.output_projector.bias[0] = 50.0
        eng.output_projector.weight.zero_()
    _, _, _, info = eng(torch.randn(1, 5, 32))
    assert info["ticks_used"] == 1


def test_normalized_entropy():
    uniform = torch.ones(1, 8)
    peaked = torch.tensor([[10.0] + [0.0] * 7])
    assert abs(normalized_entropy(uniform).item() - 1.0) < 1e-5
    assert normalized_entropy(peaked).item() < 0.01
