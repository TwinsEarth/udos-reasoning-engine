"""与 third_party/ctm 真实上游实现的冒烟对照 (依赖缺失时 skip)。"""

import pytest
import torch

from udos.adapters import SakanaCTMAdapter


@pytest.fixture
def adapter_or_skip():
    if not SakanaCTMAdapter.available():
        pytest.skip("上游 CTM 运行依赖缺失 (huggingface_hub) 或代码库未克隆")
    ad = SakanaCTMAdapter(
        iterations=6, d_model=64, d_input=32, heads=2,
        n_synch_out=16, n_synch_action=16, memory_length=8, out_dims=24)
    ad.load()
    return ad


def test_upstream_forward(adapter_or_skip, torch_seed):
    ad = adapter_or_skip
    seq = torch.randn(2, 9, 32)
    preds, certs, sync = ad.forward(seq)
    assert preds.shape == (2, 24, 6)
    assert certs.shape == (2, 2, 6)
    assert sync.shape == (2, 16)


def test_upstream_and_internal_shapes_consistent(adapter_or_skip, torch_seed):
    """上游与内部机制对齐版输出结构一致: [B,out,T] / [B,2,T] / [B,n_synch]。"""
    from udos.ctm_engine import CTMConfig, CTMPhysicsEngine
    ad = adapter_or_skip
    seq = torch.randn(1, 9, 32)
    p_up, c_up, s_up = ad.forward(seq)

    internal = CTMPhysicsEngine(CTMConfig(
        iterations=6, d_model=64, d_input=32, heads=2, n_synch_out=16,
        n_synch_action=16, memory_length=8, nlm_hidden=16, out_dims=24,
        certainty_threshold=0.0, n_random_pairing_self=2))
    p_in, c_in, s_in, _ = internal(seq)
    assert p_up.shape == p_in.shape
    assert c_up.shape == c_in.shape
    assert s_up.shape == s_in.shape
