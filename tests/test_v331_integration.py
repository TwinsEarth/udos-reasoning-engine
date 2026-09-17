"""
v3.3.1 节点58: 全特性集成 + 综合评测 + backcompat 17 件
====================================================================
锚点纪律:
    * loop+multitask+retarget+affordance+multimodal+eval5d+action_piece+
      ego_data+moe+robustness 组合不冲突, 全部有限;
    * 默认路径逐位一致 (无外挂时 predict_next 不变);
    * 五维评测 + 鲁棒性评测综合报告落 JSON;
    * backcompat v2.1.0..v3.3.0 共 17 件全量加载 predict。
analogy, not reproduction。
"""
import json
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.physical_loop import PhysicalLoopRunner
from udos.retargeting import MorphologyConfig, ActionRetargeter
from udos.affordance import AffordanceScorer
from udos.multitask import MultiTaskHead
from udos.future_multimodal import FutureMultimodalHead
from udos.eval_suite import FiveDimensionEvaluator
from udos.action_piece import ActionPieceTokenizer
from udos.ego_data import SyntheticEgoAugmenter
from udos.moe import LightweightMoE
from udos.robustness import RobustnessEvaluator

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
REPORT = ROOT / "benchmarks" / "results" / "integration_report_v3.3.1.json"

# v2.1.0..v3.3.3 共 18 件
ALL_CKPTS = [
    "predictor_v2.1.0.pt", "predictor_v2.2.1.pt", "predictor_v2.3.1.pt",
    "predictor_v2.4.0.pt", "predictor_v2.5.0.pt", "predictor_v2.5.2.pt",
    "predictor_v2.6.0.pt", "predictor_v2.6.2.pt", "predictor_v2.7.0.pt",
    "predictor_v2.7.3.pt", "predictor_v2.8.0.pt", "predictor_v2.9.0.pt",
    "predictor_v3.0.0.pt", "predictor_v3.0.3.pt", "predictor_v3.1.0.pt",
    "predictor_v3.2.0.pt", "predictor_v3.3.0.pt", "predictor_v3.3.3.pt",
]


def test_backcompat_17_checkpoints():
    ds = build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                  horizon=2, dt=0.5, seed=101)
    assert len(ALL_CKPTS) == 18
    for name in ALL_CKPTS:
        path = CKPT_DIR / name
        assert path.exists(), f"{name} 缺失"
        m, _ = load_predictor(str(path))
        m.eval()
        out = m.predict_next(ds.X[:1], scene_params=ds.P[:1])
        assert out.shape == (1, 6) and torch.isfinite(out).all()


@pytest.fixture(scope="module")
def pred():
    m, _ = load_predictor(str(CKPT_DIR / "predictor_v3.3.0.pt"))
    m.eval()
    return m


@pytest.fixture(scope="module")
def ds():
    return build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=77)


def test_default_path_bitwise(pred, ds):
    loop = PhysicalLoopRunner(pred, horizon=4)
    out = loop.run(ds.X[:1], scene_params=ds.P[:1])
    ref = pred.predict_next(ds.X[:1], scene_params=ds.P[:1])
    assert torch.equal(out["prediction"], ref)


def test_full_feature_combo(pred, ds):
    """跨 2.8-3.3 全部 opt-in 特性组合不冲突。"""
    w1, p1 = ds.X[:1], ds.P[:1]
    ref = pred.predict_next(w1, scene_params=p1)
    # loop
    loop_out = PhysicalLoopRunner(pred, horizon=2).run(w1, scene_params=p1)
    assert torch.isfinite(loop_out["prediction"]).all()
    # retarget
    rt = ActionRetargeter(MorphologyConfig(dof=6, control_freq=60.0,
                        joint_limits=[[-2.0, 2.0]] * 6, name="p"),
                        MorphologyConfig(dof=4, control_freq=120.0,
                        joint_limits=[[-0.5, 0.5]] * 4, name="g"))
    ta = rt.retarget(ds.X[:2, -1, :])
    assert torch.isfinite(ta).all()
    # affordance
    aff = AffordanceScorer()
    sc = aff.score(ds.X[:2, -1, :6],
                   torch.randn(2, 3, 6))
    assert "scores" in sc
    # multimodal
    torch.manual_seed(0)
    mth = MultiTaskHead(pred, latent_dim=32, enable=True)
    mth.register_head("future_mm", FutureMultimodalHead(32, horizon=4))
    mm = mth.forward(w1, scene_params=p1)["future_mm"]
    assert all(torch.isfinite(v).all().item() for v in mm.values())
    # action_piece
    deltas = ds.X[:, 1:, :] - ds.X[:, :-1, :]
    ap = ActionPieceTokenizer(action_dim=6, codebook_size=16,
                             init="kmeans++", seed=0).fit(
        deltas.reshape(-1, deltas.size(-1)))
    assert 0.0 <= ap.utilization() <= 1.0
    # ego augment
    Xa, Ya = SyntheticEgoAugmenter(noise_sigma=0.01).augment_pair(ds.X[:4], ds.Y[:4])
    assert Xa.shape == ds.X[:4].shape
    # moe (外挂)
    moe = LightweightMoE(32, 16, num_experts=4, top_k=2)
    moe_out = moe(torch.randn(2, 32))
    assert torch.isfinite(moe_out).all()
    # 组合后默认预测仍逐位一致 (上述均为外挂)
    after = pred.predict_next(w1, scene_params=p1)
    assert torch.equal(after, ref)


def test_comprehensive_report_json(pred, ds):
    ev = FiveDimensionEvaluator(pred, seed=2025, n_per_kind=8, grid_size=3)
    s5 = ev.evaluate()
    rob = RobustnessEvaluator(pred, seed=0).evaluate(ds)
    report = {
        "version": __version__,
        "five_dim": {k: round(v, 3) for k, v in s5.items()},
        "robustness": rob,
        "all_features_composable": True,
        "analogy_not_reproduction": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    loaded = json.loads(REPORT.read_text(encoding="utf-8"))
    assert "five_dim" in loaded and "robustness" in loaded
    assert 0.0 <= loaded["robustness"]["robustness_score"] <= 100.0


def test_version():
    assert __version__ == "5.5.5"
