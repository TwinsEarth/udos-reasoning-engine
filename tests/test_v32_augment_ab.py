"""
v3.2.0.dev4 节点45: 数据增强 A/B + 消融
==========================================
锚点纪律:
    * benchmarks/results/ego_augment_ab_v3.2.0.json 落盘且结构可复算;
    * 增强类型消融四项 (view/perturb/noise/time_scale) 齐全;
    * opt-in 默认关;
    * 被否决候选保留 (收益不稳如实记录)。
"""
import json
from pathlib import Path

from udos import __version__
from udos.ego_data import SyntheticEgoAugmenter

ROOT = Path(__file__).resolve().parents[1]
AB_JSON = ROOT / "benchmarks" / "results" / "ego_augment_ab_v3.2.0.json"


def test_ab_json_exists_and_structured():
    d = json.loads(AB_JSON.read_text(encoding="utf-8"))
    assert {"raw", "augmented_full", "ablations"} <= set(d.keys())
    for k in ("eval_mse", "noise_robust_mse", "ood_mse"):
        assert k in d["raw"] and k in d["augmented_full"]


def test_ablation_four_types():
    d = json.loads(AB_JSON.read_text(encoding="utf-8"))
    assert set(d["ablations"].keys()) == {"view", "perturb", "noise",
                                          "time_scale"}


def test_opt_in_default_off():
    d = json.loads(AB_JSON.read_text(encoding="utf-8"))
    assert d["opt_in_default_off"] is True
    assert d["analogy_not_reproduction"] is True


def test_recommendation_honest():
    d = json.loads(AB_JSON.read_text(encoding="utf-8"))
    # 全量几何增强在 in-distribution 上不优于原始 (如实记录), 不伪造提升
    assert d["augment_mse_gain"] <= 1e-6
    assert isinstance(d["recommendation"], str) and d["recommendation"]


def test_augmenter_reproducible():
    """同种子增强应可复现 (确定性)。"""
    from udos.dynamics import build_parametric_dataset
    ds = build_parametric_dataset(n_per_kind=2, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=3)
    a1 = SyntheticEgoAugmenter(traj_perturb=0.02, noise_sigma=0.01, seed=9)
    a2 = SyntheticEgoAugmenter(traj_perturb=0.02, noise_sigma=0.01, seed=9)
    X1, Y1 = a1.augment_pair(ds.X, ds.Y)
    X2, Y2 = a2.augment_pair(ds.X, ds.Y)
    assert (X1 - X2).abs().max() < 1e-7


def test_version():
    assert __version__ == "5.5.5"
