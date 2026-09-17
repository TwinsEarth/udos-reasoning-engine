"""
v3.1.0.dev1 节点32: ActionPieceCodec 多尺度粗+细两级码本
================================================================
锚点纪律:
    * 两级量化往返误差 <= 粗级单独误差 (残差细化必不更差);
    * 粗/细利用率 > 0, 困惑度在 (1, n_vocab] 内;
    * kmeans++ / random / uniform_grid 三种初始化均可拟合;
    * encode/decode 形状有限; state_dict / save / load 往返一致;
    * 空/未拟合守卫。
analogy, not reproduction —— 合成动作, 非真机/VLM 复现。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import ActionPieceCodec, token_perplexity  # noqa: E402


def make_actions(n_per_cluster=80, n_clusters=4, dim=6, seed=11):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, dim, generator=g) * 3.0
    chunks = [c + 0.2 * torch.randn(n_per_cluster, dim, generator=g) for c in centers]
    return torch.cat(chunks, dim=0)


def test_two_level_error_le_coarse_error():
    x = make_actions()
    codec = ActionPieceCodec(action_dim=6, coarse_size=8, fine_size=8,
                             seed=42).fit(x)
    q = codec.quality(x)
    assert q["two_level_mse"] <= q["coarse_mse"] + 1e-9
    # 细残差量化应真正带来增益
    assert q["gain_vs_coarse"] >= 0.0
    assert q["two_level_mse"] < q["coarse_mse"]


def test_utilization_and_perplexity():
    x = make_actions()
    codec = ActionPieceCodec(action_dim=6, coarse_size=8, fine_size=8,
                             seed=42).fit(x)
    q = codec.quality(x)
    assert 0.0 < q["coarse_utilization"] <= 1.0
    assert 0.0 < q["fine_utilization"] <= 1.0
    # 困惑度落在 [1, n_vocab]
    assert 1.0 <= q["coarse_perplexity"] <= 8.0 + 1e-6
    assert 1.0 <= q["fine_perplexity"] <= 8.0 + 1e-6


def test_encode_decode_shapes():
    x = make_actions()
    codec = ActionPieceCodec(action_dim=6, coarse_size=4, fine_size=4,
                             seed=42).fit(x)
    c, f = codec.encode(x)
    assert c.shape == (x.size(0),) and f.shape == (x.size(0),)
    recon = codec.decode(c, f)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_init_strategies_all_valid():
    x = make_actions()
    for strat in ("kmeans++", "random", "uniform_grid"):
        codec = ActionPieceCodec(action_dim=6, coarse_size=4, fine_size=4,
                                 init=strat, seed=42).fit(x)
        q = codec.quality(x)
        assert q["two_level_mse"] >= 0.0
        assert q["coarse_utilization"] > 0.0


def test_perplexity_extremes():
    # 全同一 token -> 困惑度 1
    assert abs(token_perplexity(torch.zeros(50, dtype=torch.long), 8) - 1.0) < 1e-6
    # 均匀分布 -> 困惑度接近 n_vocab
    t = torch.arange(8).repeat(20)
    pp = token_perplexity(t, 8)
    assert abs(pp - 8.0) < 1e-3
    with pytest.raises(ValueError):
        token_perplexity(torch.empty(0, dtype=torch.long), 8)


def test_save_load_roundtrip(tmp_path):
    x = make_actions()
    codec = ActionPieceCodec(action_dim=6, coarse_size=8, fine_size=8,
                             seed=42).fit(x)
    c, f = codec.encode(x)
    p = tmp_path / "codec.json"
    codec.save(str(p))
    reloaded = ActionPieceCodec.load(str(p))
    assert reloaded.coarse_size == 8 and reloaded.fine_size == 8
    c2, f2 = reloaded.encode(x)
    assert torch.equal(c, c2)
    assert torch.allclose(reloaded.decode(c2, f2), codec.decode(c, f), atol=1e-6)


def test_unfitted_and_length_guard():
    codec = ActionPieceCodec(action_dim=6, coarse_size=4, fine_size=4)
    with pytest.raises(RuntimeError):
        codec.encode(torch.randn(5, 6))
    x = make_actions()
    codec.fit(x)
    with pytest.raises(ValueError):
        codec.decode(torch.tensor([0, 1]), torch.tensor([0]))   # 长度不齐


def test_bad_sizes_guard():
    with pytest.raises(ValueError):
        ActionPieceCodec(action_dim=6, coarse_size=0, fine_size=4)
    with pytest.raises(ValueError):
        ActionPieceCodec(action_dim=6, coarse_size=4, fine_size=0)
