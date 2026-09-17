"""
v3.1.1 节点38: ActionPiece 与 Physical Loop 集成 (opt-in)
================================================================
锚点纪律:
    * 默认路径 (use_tokenized=False) 与 predictor.predict_next 逐位一致;
    * use_tokenized=True 时 tokenized 路径可跑、输出有限、不与 loop 冲突;
    * 与 2.8 loop / 2.9 retarget / 3.0 multimodal 特性组合互不污染。
analogy, not reproduction —— 合成动作 token, 非真机控制。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__  # noqa: E402
from udos.persistence import load_predictor  # noqa: E402
from udos.physical_loop import PhysicalLoopRunner  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.action_piece import (  # noqa: E402
    ActionPieceTokenizer, TokenizedActionPredictor, markov_token_sequences)

CKPT = ROOT / "checkpoints" / "predictor_v3.1.0.pt"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(str(CKPT))
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_default_path_bit_identical(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=717)
    w, p = ds.X[:1], ds.P[:1]
    loop = PhysicalLoopRunner(predictor, horizon=2)   # 默认 use_tokenized=False
    out = loop.run(w, scene_params=p, candidate_actions=[{}])
    expected = predictor.predict_next(w, scene_params=p)
    assert torch.equal(out["prediction"], expected), "默认路径必须逐位一致"


def test_tokenized_path_runs(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=727)
    w, p = ds.X[:1], ds.P[:1]
    # 合成动作上拟合 tokenizer + token_predictor
    g = torch.Generator().manual_seed(1)
    actions = torch.randn(200, 6, generator=g)
    tok = ActionPieceTokenizer(6, codebook_size=8, seed=42).fit(actions)
    seqs = markov_token_sequences(n_seqs=100, seq_len=12, vocab_size=8, seed=42)
    tpred = TokenizedActionPredictor(8, order=2).fit(seqs)

    loop = PhysicalLoopRunner(predictor, horizon=2, use_tokenized=True,
                              tokenizer=tok, token_predictor=tpred)
    out = loop.run(w, scene_params=p)
    assert torch.isfinite(out["prediction"]).all()
    pa = out["loop_state"]["outputs"]["predict_action"]
    assert pa["tokenized"] is True
    assert pa["best_action"]["state_perturbation"] is not None


def test_tokenized_requires_parts(predictor):
    with pytest.raises(ValueError):
        loop = PhysicalLoopRunner(predictor, horizon=2, use_tokenized=True)
        loop.run(torch.randn(1, 6, 6), scene_params=None)


def test_composition_no_conflict(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=737)
    w, p = ds.X[:1], ds.P[:1]
    before = predictor.predict_next(w, scene_params=p)

    g = torch.Generator().manual_seed(2)
    tok = ActionPieceTokenizer(6, codebook_size=8, seed=42).fit(torch.randn(200, 6, generator=g))
    seqs = markov_token_sequences(n_seqs=60, seq_len=10, vocab_size=8, seed=42)
    tpred = TokenizedActionPredictor(8, order=2).fit(seqs)
    tl = PhysicalLoopRunner(predictor, horizon=2, use_tokenized=True,
                            tokenizer=tok, token_predictor=tpred)
    dl = PhysicalLoopRunner(predictor, horizon=2)
    tl.run(w, scene_params=p)
    dl.run(w, scene_params=p, candidate_actions=[{}])
    tl.run(w, scene_params=p)
    after = predictor.predict_next(w, scene_params=p)
    assert torch.equal(before, after), "tokenized 外挂组合不得改主路径"


def test_tokenized_history_accumulates(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=747)
    w, p = ds.X[:1], ds.P[:1]
    g = torch.Generator().manual_seed(3)
    tok = ActionPieceTokenizer(6, codebook_size=8, seed=42).fit(torch.randn(200, 6, generator=g))
    seqs = markov_token_sequences(n_seqs=60, seq_len=10, vocab_size=8, seed=42)
    tpred = TokenizedActionPredictor(8, order=2).fit(seqs)
    loop = PhysicalLoopRunner(predictor, horizon=2, use_tokenized=True,
                              tokenizer=tok, token_predictor=tpred)
    loop.run(w, scene_params=p)
    loop.run(w, scene_params=p)
    # 两次 run 后 token_history 应累积 (种子 1 + 每步 1)
    assert len(loop.loop_state["token_history"]) >= 3
