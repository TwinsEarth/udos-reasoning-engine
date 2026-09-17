"""v3.4.5 最终验证：20 代 checkpoint 兼容 + save/load 逐位一致。"""
import hashlib

import torch

from udos import load_predictor, save_predictor, __version__
from udos.persistence import load_predictor as lp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


def test_20_checkpoints_load():
    ckpts = sorted(CKPT_DIR.glob("predictor_v*.pt"))
    # v3.5 线起 backcompat 只增不减 (v3.4.5 为 20 件, v3.5.0 起 21 件)
    assert len(ckpts) >= 20, f"backcompat 应 >= 20 件 checkpoint, 实际 {len(ckpts)}"
    w = torch.randn(1, 6, 6)
    for p in ckpts:
        m, meta = lp(p)
        out = m.predict_next(w)
        assert bool(torch.isfinite(out).all()), p.name


def test_save_load_bitwise_consistent():
    """v3.4.5 正式件 save/load 后 state_dict 逐位一致。"""
    src = CKPT_DIR / "predictor_v3.4.5.pt"
    m1, _ = load_predictor(src)
    def md5(m):
        h = hashlib.md5()
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "rt.pt")
        save_predictor(m1, out, metrics={"reload": True})
        m2, _ = load_predictor(out)
    assert md5(m1) == md5(m2)


def test_version():
    assert __version__ == "5.5.5"
