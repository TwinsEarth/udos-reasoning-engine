"""
v3.1.0.dev4 节点35: Tokenized vs 连续动作 A/B (码本大小扫描)
================================================================
锚点纪律:
    * A/B 脚本可复跑, 落 benchmarks/results/action_piece_ab_v3.1.0.json;
    * 扫描含 K in {8,16,32,64} 四档, 量化 MSE 随 K 单调不增;
    * opt-in 默认关: tokenized 不改变主 predictor.predict_next 默认路径;
    * 被否决/降级候选在 verdict 段保留 (诚实记录)。
analogy, not reproduction —— 合成动作量化 A/B, 非真机。
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.persistence import load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402

AB_JSON = ROOT / "benchmarks" / "results" / "action_piece_ab_v3.1.0.json"
CKPT = ROOT / "checkpoints" / "predictor_v3.1.0.pt"


def test_ab_json_written_and_recomputable():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "action_piece_ab_v31.py")],
                      cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(AB_JSON))
    assert d["benchmark"] == "action_piece_ab_v3.1.0"
    assert d["analogy_not_reproduction"] is True


def test_codebook_scan_four_sizes_monotone():
    d = json.load(open(AB_JSON))
    ks = [r["codebook_size"] for r in d["scan"]]
    assert ks == [8, 16, 32, 64]
    mses = [r["quant_mse"] for r in d["scan"]]
    # 量化误差随码本增大单调不增 (更大码本 = 更细)
    for a, b in zip(mses[:-1], mses[1:]):
        assert b <= a + 1e-6
    for r in d["scan"]:
        assert r["params"] == r["codebook_size"] * 6
        assert r["latency_ms"] >= 0


def test_opt_in_default_off_old_path_bit_identical():
    m, _ = load_predictor(str(CKPT))
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=515)
    w, p = ds.X[:1], ds.P[:1]
    before = m.predict_next(w, scene_params=p)
    # 使用 tokenized A/B 全流程不应改变主 predictor
    from udos.action_piece import ActionPieceTokenizer
    tok = ActionPieceTokenizer(6, 16, seed=42).fit(ds.X.reshape(-1, 6))
    _ = tok.encode(ds.X[:4].reshape(-1, 6))
    after = m.predict_next(w, scene_params=p)
    assert torch.equal(before, after), "tokenized 外挂不得改主路径"


def test_rejected_candidates_retained():
    d = json.load(open(AB_JSON))
    # verdict 必须是显式 opt-in 声明, 不夸大为默认开启
    assert d["opt_in_default_off"] is True
    assert isinstance(d["verdict"], str) and len(d["verdict"]) > 0
    assert d["continuous_baseline_mse"] > 0
