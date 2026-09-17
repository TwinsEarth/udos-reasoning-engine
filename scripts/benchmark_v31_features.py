"""
v3.1 ActionPiece 推理延迟基准 (tokenize / detokenize / next-token)
================================================================
对加载好的 predictor_v3.1.0.pt, 在合成动作上测量:
    * predict_next          (基线)
    * ActionPieceTokenizer.encode (连续 -> token)
    * ActionPieceTokenizer.decode (token -> 连续)
    * TokenizedActionPredictor.predict_next (自回归下一个 token)
落 benchmarks/results/feature_latency_v3.1.0.json (p50/p95, 毫秒)。
analogy, not reproduction —— 合成动作 token, 非真机。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.persistence import load_predictor  # noqa: E402
from udos.action_piece import (  # noqa: E402
    ActionPieceTokenizer, TokenizedActionPredictor, markov_token_sequences)
from udos.dynamics import build_parametric_dataset  # noqa: E402

torch.set_num_threads(2)


def pct(xs, p):
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return float(s[f] + (k - f) * (s[c] - s[f]))


def main():
    predictor, _ = load_predictor("checkpoints/predictor_v3.1.0.pt")
    ds = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=9393)
    wb, pb = ds.X[:1], ds.P[:1]

    g = torch.Generator().manual_seed(1)
    actions = torch.randn(400, 6, generator=g)
    tok = ActionPieceTokenizer(6, codebook_size=16, seed=42).fit(actions)
    toks = tok.encode(actions)
    seqs = markov_token_sequences(n_seqs=120, seq_len=12, vocab_size=16, seed=42)
    tpred = TokenizedActionPredictor(16, order=2).fit(seqs)
    hist = torch.tensor(toks[:8].tolist(), dtype=torch.long)

    for _ in range(5):
        predictor.predict_next(wb, scene_params=pb)
        tok.encode(actions[:32]); tok.decode(toks[:32])
        tpred.predict_next(hist)

    N = 50

    def bench(fn):
        ts = []
        for _ in range(N):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return {"p50_ms": round(pct(ts, 0.5), 4),
                "p95_ms": round(pct(ts, 0.95), 4),
                "mean_ms": round(sum(ts) / len(ts), 4)}

    out = {
        "version": "3.1.2",
        "n_repeats": N,
        "predict_next": bench(lambda: predictor.predict_next(wb, scene_params=pb)),
        "action_tokenize": bench(lambda: tok.encode(actions[:32])),
        "action_detokenize": bench(lambda: tok.decode(toks[:32])),
        "token_next_token": bench(lambda: tpred.predict_next(hist)),
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/feature_latency_v3.1.0.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
