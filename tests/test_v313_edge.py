"""
v3.1.3 节点40: ActionPiece 边界精修
================================================================
覆盖:
    * 码本大小=1 (退化单点);
    * 空 token 序列;
    * 极端动作值 (大有限值);
    * 服务未训练态 409;
    * in-context 示例超长;
    * codec 粗级=1 退化。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__  # noqa: E402
from udos.action_piece import (  # noqa: E402
    ActionPieceTokenizer, ActionPieceCodec, TokenizedActionPredictor,
    InContextActionPrompter, markov_token_sequences)
from udos.server import UDOSService  # noqa: E402


def test_version():
    assert __version__ == "5.5.5"


def test_codebook_size_one():
    x = torch.randn(50, 6, generator=torch.Generator().manual_seed(0))
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=1, seed=42).fit(x)
    ids = tok.encode(x)
    assert set(ids.tolist()) == {0}
    recon = tok.decode(ids)
    assert torch.isfinite(recon).all()
    assert tok.utilization() == 1.0


def test_empty_token_sequence_guard():
    tok = ActionPieceTokenizer(6, 4, seed=1).fit(torch.randn(30, 6))
    with pytest.raises(ValueError):
        tok.decode(torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError):
        tok.encode(torch.empty(0, 6))


def test_extreme_action_values():
    # 大有限值不应产生 NaN/inf
    x = 1e6 * torch.randn(200, 6)
    tok = ActionPieceTokenizer(6, 8, seed=42).fit(x)
    ids = tok.encode(x)
    recon = tok.decode(ids)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(tok.centroids_).all()


def test_codec_coarse_size_one():
    x = torch.randn(100, 6, generator=torch.Generator().manual_seed(2))
    codec = ActionPieceCodec(action_dim=6, coarse_size=1, fine_size=4,
                             seed=42).fit(x)
    c, f = codec.encode(x)
    assert set(c.tolist()) == {0}
    recon = codec.decode(c, f)
    assert torch.isfinite(recon).all()


def test_service_untrained_409():
    svc = UDOSService(preset="small")
    with pytest.raises(Exception):
        svc.action_tokenize({"actions": [[0.0] * 6]})
    with pytest.raises(Exception):
        svc.action_detokenize({"tokens": [0]})


def test_incontext_overly_long_examples():
    prompter = InContextActionPrompter(vocab_size=8, order=2)
    # 超长示例 (20 条长序列) 不应崩溃
    examples = markov_token_sequences(n_seqs=20, seq_len=40, vocab_size=8,
                                      stickiness=0.9, seed=42)
    nxt = prompter.predict_next(torch.tensor([3, 4, 5]), examples)
    assert 0 <= nxt < 8


def test_predictor_oob_history_guard():
    seqs = markov_token_sequences(60, 10, 8, seed=1)
    pred = TokenizedActionPredictor(8, order=2).fit(seqs)
    with pytest.raises(ValueError):
        pred.predict_logits(torch.tensor([0, 99]))   # 越界 token


def test_decode_single_token_shape():
    tok = ActionPieceTokenizer(6, 4, seed=1).fit(torch.randn(30, 6))
    out = tok.decode(torch.tensor([2]))
    assert out.shape == (1, 6)
