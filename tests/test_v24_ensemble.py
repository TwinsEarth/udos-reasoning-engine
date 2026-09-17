"""v2.4.1 回归: 深度集成不确定性 (N=1 等价单模型 / N>1 方差>0 / 存载)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
from udos.ensemble import DeepEnsemble  # noqa: E402


def _factory():
    return lambda: CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0)


def _window(seed=3, n=4):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=seed)
    return ds.X[:n], ds.P[:n]


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_n1_equivalent_to_single_model():
    ens = DeepEnsemble.create(_factory(), n=1, base_seed=0,
                              scene_param_dim=4)
    ens.eval()
    x, p = _window()
    out = ens.predict_next(x, scene_params=p)
    single = ens.members[0].predict_next(x, scene_params=p)
    assert torch.allclose(out["mean"], single, atol=1e-7)
    # N=1 方差恒 0
    assert torch.count_nonzero(out["variance"]) == 0
    assert out["members"].shape == (1, *single.shape)


def test_n_gt1_variance_positive_and_shapes():
    ens = DeepEnsemble.create(_factory(), n=5, base_seed=0,
                              scene_param_dim=4)
    ens.eval()
    x, p = _window()
    out = ens.predict_next(x, scene_params=p)
    B, R = x.size(0), 6
    assert out["mean"].shape == (B, R)
    assert out["variance"].shape == (B, R)
    assert out["members"].shape == (5, B, R)
    # 多种子随机权重 => 成员分歧 => 方差严格 > 0
    assert float(out["variance"].sum()) > 0.0
    # 均值确实是逐成员平均
    manual = out["members"].mean(dim=0)
    assert torch.allclose(out["mean"], manual, atol=1e-6)


def test_rollout_aggregation():
    ens = DeepEnsemble.create(_factory(), n=3, base_seed=1,
                              scene_param_dim=4)
    ens.eval()
    x, p = _window()
    out = ens.rollout(x, horizon=3, scene_params=p)
    B = x.size(0)
    assert out["mean"].shape == (B, 3, 6)
    assert out["members"].shape == (3, B, 3, 6)
    assert float(out["variance"].sum()) > 0.0


def test_create_rejects_bad_n():
    try:
        DeepEnsemble.create(_factory(), n=0); assert False
    except ValueError:
        pass
    try:
        DeepEnsemble([]); assert False
    except ValueError:
        pass


def test_save_load_roundtrip(tmp_path):
    ens = DeepEnsemble.create(_factory(), n=3, base_seed=2,
                              scene_param_dim=4)
    ens.eval()
    x, p = _window()
    before = ens.predict_next(x, scene_params=p)["mean"]
    path = tmp_path / "ens.pt"
    ens.save(str(path))
    loaded = DeepEnsemble.load(str(path))
    assert loaded.n_members == 3
    after = loaded.predict_next(x, scene_params=p)["mean"]
    assert torch.allclose(before, after, atol=1e-6)
