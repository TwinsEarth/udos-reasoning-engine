"""
v3.1.0.dev2 节点33: TokenizedActionPredictor 自回归 next-token
================================================================
锚点纪律:
    * n-gram 在合成 Markov token 序列上 held-out 精度 >> 随机 (1/vocab);
    * predict_logits 形状 [vocab]、非负; generate 长度正确、token 在范围内;
    * token -> 连续动作解码形状与 tokenizer 接口一致;
    * 空历史 / 未拟合 / 越界 token / 空序列 守卫。
analogy, not reproduction —— 合成 token 序列, 推理外挂, 不替换主 CTM。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import (  # noqa: E402
    ActionPieceTokenizer, TokenizedActionPredictor, markov_token_sequences)


def setup_predictor(vocab=8, order=2, seed=42):
    seqs = markov_token_sequences(n_seqs=300, seq_len=14, vocab_size=vocab,
                                  step=1, stickiness=0.9, seed=seed)
    pred = TokenizedActionPredictor(vocab_size=vocab, order=order).fit(seqs)
    return pred, seqs


def test_next_token_accuracy_beats_random():
    pred, seqs = setup_predictor()
    acc = pred.heldout_accuracy(seqs)
    # Markov 结构强 (0.9 stickiness) -> 精度应远高于 1/8=0.125
    assert acc > 0.7, f"held-out 精度 {acc} 过低"


def test_predict_logits_shape_nonneg():
    pred, seqs = setup_predictor()
    hist = seqs[0][:3]
    logits = pred.predict_logits(hist)
    assert logits.shape == (8,)
    assert bool((logits >= 0).all())
    nxt = pred.predict_next(hist)
    assert 0 <= nxt < 8


def test_generate_length_and_range():
    pred, seqs = setup_predictor()
    seed_toks = seqs[0][:2]
    gen = pred.generate(seed_toks, n_steps=5)
    assert gen.shape == (7,)          # 2 seed + 5 generated
    assert int(gen.min()) >= 0 and int(gen.max()) < 8
    # 前 2 个为种子逐位保留
    assert torch.equal(gen[:2], seed_toks)


def test_token_to_continuous_decode():
    # 先在合成动作上拟合 tokenizer, 再用其 token id 喂 predictor
    g = torch.Generator().manual_seed(3)
    actions = torch.randn(200, 6, generator=g)
    tok = ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(actions)
    # 把若干动作编码成 token 序列
    seq_actions = actions[:20]
    toks = tok.encode(seq_actions)
    out_actions = TokenizedActionPredictor.decode_to_actions(toks, tok)
    assert out_actions.shape == (20, 6)
    assert torch.isfinite(out_actions).all()


def test_backoff_to_lower_order():
    # 历史短于 order=3 -> 应回退到低阶而非 KeyError
    pred, _ = setup_predictor(order=3)
    hist = torch.tensor([3])          # 只有 1 个 token
    logits = pred.predict_logits(hist)
    assert logits.shape == (8,)


def test_empty_history_guard():
    pred, _ = setup_predictor()
    with pytest.raises(ValueError):
        pred.predict_logits(torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError):
        pred.generate(torch.empty(0, dtype=torch.long), 5)


def test_unfitted_guard():
    pred = TokenizedActionPredictor(vocab_size=8)
    with pytest.raises(RuntimeError):
        pred.predict_logits(torch.tensor([1, 2]))


def test_fit_empty_and_oob_guard():
    with pytest.raises(ValueError):
        TokenizedActionPredictor(vocab_size=8).fit([])
    with pytest.raises(ValueError):
        TokenizedActionPredictor(vocab_size=8).fit([torch.tensor([0, 9])])  # 越界


def test_bad_params_guard():
    with pytest.raises(ValueError):
        TokenizedActionPredictor(vocab_size=0)
    with pytest.raises(ValueError):
        TokenizedActionPredictor(vocab_size=8, order=0)
