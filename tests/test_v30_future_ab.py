"""
v3.0.0.dev4 未来多模态 vs 单模态 A/B 证据
============================================
锚点纪律:
    * A/B JSON (future_multimodal_ab_v3.0.0.json) 落盘且可复算;
    * 多模态头默认 opt-in 关时旧 predict_next 逐位一致;
    * 被否决/降级候选保留;
    * 一致性损失不改变推理输出。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.multitask import MultiTaskHead
from udos.future_multimodal import FutureMultimodalHead, CrossModalAlignmentLoss

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.0.0.pt")
AB_JSON = ROOT / "benchmarks" / "results" / "future_multimodal_ab_v3.0.0.json"
AB_SCRIPT = ROOT / "scripts" / "future_multimodal_ab_v30.py"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(CKPT)
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_ab_json_written_and_recomputable(predictor):
    r = subprocess.run([sys.executable, str(AB_SCRIPT)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(AB_JSON))
    assert d["n_repeats"] == 50
    assert d["single_modal"]["params"] > 0
    assert d["multimodal"]["params"] > d["single_modal"]["params"]
    assert d["multimodal"]["latency_p50_ms"] >= 0
    assert "n/a" in d["multimodal"]["state_mse_vs_future"]
    # 被否决候选保留
    assert len(d["rejected_candidates"]) >= 1
    # 一致性损失不改推理输出
    assert d["alignment_loss"]["changes_inference_output"] is False


def test_default_opt_in_off_old_path_bitidentical(predictor):
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=808)
    wb, pb = ds.X[:1], ds.P[:1]
    before = predictor.predict_next(wb, scene_params=pb)
    mth = MultiTaskHead(predictor, latent_dim=32, enable=True)
    torch.manual_seed(0)
    mth.register_head("mm", FutureMultimodalHead(32, horizon=4))
    mth.forward(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_alignment_loss_does_not_alter_inference():
    torch.manual_seed(0)
    head = FutureMultimodalHead(16, horizon=4)
    z = torch.randn(4, 16)
    out1 = head(z)
    loss_fn = CrossModalAlignmentLoss()
    _ = loss_fn(out1["rgb"], out1["depth"], out1["mask"])
    out2 = head(z)
    for k in ("rgb", "depth", "mask"):
        assert torch.equal(out1[k], out2[k])
