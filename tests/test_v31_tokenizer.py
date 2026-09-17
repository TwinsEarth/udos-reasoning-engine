"""
v3.1.0 节点31: ActionPieceTokenizer (连续动作 -> 离散 token)
================================================================
锚点纪律:
    * 码本学习在合成 6D 动作上收敛 (inertia 单调不增);
    * encode/decode 往返误差有界且小;
    * token 覆盖率非平凡 (多 token 被使用, 非全 0/全 1);
    * 空动作 / 未拟合 / 越界 token 守卫;
    * state_dict / save / load 往返一致;
    * 确定性: 同 seed 两次 fit 逐位一致。
analogy, not reproduction —— 合成动作, 非真机/VLM 复现。
"""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import ActionPieceTokenizer  # noqa: E402


def make_synthetic_actions(n_per_cluster: int = 64, n_clusters: int = 4,
                           action_dim: int = 6, seed: int = 7):
    """合成动作 = 已知中心点的高斯混合 (6D), 供 k-means 收敛验证。"""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, action_dim, generator=g) * 3.0
    chunks = []
    for c in centers:
        chunks.append(c + 0.3 * torch.randn(n_per_cluster, action_dim,
                                            generator=g))
    return torch.cat(chunks, dim=0), centers


def test_fit_converges():
    x, _ = make_synthetic_actions()
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(x)
    hist = tok.inertia_history_
    assert len(hist) >= 2
    # inertia 单调不增 (k-means 必收敛)
    for a, b in zip(hist[:-1], hist[1:]):
        assert b <= a + 1e-9
    # 末端 inertia 应显著小于首项
    assert hist[-1] < hist[0]
    assert tok.fitted_ is True


def test_encode_decode_roundtrip_bounded():
    x, _ = make_synthetic_actions()
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(x)
    ids = tok.encode(x)
    assert ids.shape == (x.size(0),)
    assert ids.dtype == torch.long
    assert int(ids.min()) >= 0 and int(ids.max()) < 8
    recon = tok.decode(ids)
    assert recon.shape == x.shape
    err = tok.roundtrip_error(x)
    assert err < 0.5, f"往返误差 {err} 过大"
    assert torch.isfinite(recon).all()


def test_token_coverage_nontrivial():
    x, _ = make_synthetic_actions()
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(x)
    ids = tok.encode(x)
    cov = tok.token_coverage(ids)
    assert len(cov) == 8
    # 覆盖率和为 1
    assert abs(sum(cov.values()) - 1.0) < 1e-5
    # 4 个真实簇 -> 至少 3 个 token 非空
    used = sum(1 for v in cov.values() if v > 0.0)
    assert used >= 3
    # 不应全部集中到单一 token
    assert max(cov.values()) < 0.6
    assert 0.0 <= tok.utilization() <= 1.0


def test_deterministic_seed():
    x, _ = make_synthetic_actions()
    t1 = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(x)
    t2 = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(x)
    assert torch.allclose(t1.centroids_, t2.centroids_, atol=1e-6)
    ids1 = t1.encode(x)
    ids2 = t2.encode(x)
    assert torch.equal(ids1, ids2)


def test_empty_action_guard():
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=4).fit(
        torch.randn(20, 6))
    with pytest.raises(ValueError):
        tok.encode(torch.empty(0, 6))
    with pytest.raises(ValueError):
        tok.decode(torch.empty(0, dtype=torch.long))


def test_unfitted_guard():
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=4)
    with pytest.raises(RuntimeError):
        tok.encode(torch.randn(5, 6))
    with pytest.raises(RuntimeError):
        tok.decode(torch.tensor([0, 1]))


def test_out_of_range_token_guard():
    x, _ = make_synthetic_actions()
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=4, seed=1).fit(x)
    with pytest.raises(ValueError):
        tok.decode(torch.tensor([0, 4]))   # 越上界
    with pytest.raises(ValueError):
        tok.decode(torch.tensor([-1]))     # 越下界


def test_bad_init_and_dim_guard():
    with pytest.raises(ValueError):
        ActionPieceTokenizer(action_dim=6, codebook_size=4, init="bad")
    with pytest.raises(ValueError):
        ActionPieceTokenizer(action_dim=0, codebook_size=4)
    with pytest.raises(ValueError):
        ActionPieceTokenizer(action_dim=6, codebook_size=0)
    x, _ = make_synthetic_actions()
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=4).fit(x)
    with pytest.raises(ValueError):
        tok.encode(torch.randn(5, 3))     # 维度不符


def test_state_dict_save_load_roundtrip(tmp_path):
    x, _ = make_synthetic_actions()
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(x)
    ids = tok.encode(x)
    p = tmp_path / "cb.json"
    tok.save(str(p))
    reloaded = ActionPieceTokenizer.load(str(p))
    assert reloaded.codebook_size == 8
    assert reloaded.action_dim == 6
    assert torch.allclose(reloaded.centroids_, tok.centroids_, atol=1e-7)
    # 加载后 encode/decode 与原 tokenizer 一致
    assert torch.equal(reloaded.encode(x), ids)
    assert torch.allclose(reloaded.decode(ids), tok.decode(ids), atol=1e-7)


def test_n_greater_than_samples_guard():
    x = torch.randn(5, 6)
    with pytest.raises(ValueError):
        ActionPieceTokenizer(action_dim=6, codebook_size=100).fit(x)
