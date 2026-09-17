"""v2.4.13 回归: save_ensemble/load_ensemble 逐位一致、成员数正确、旧单模型加载不受影响。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.ensemble import DeepEnsemble  # noqa: E402
from udos.persistence import save_ensemble, load_ensemble, save_predictor, load_predictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _factory():
    def f():
        return CTMConfig(iterations=3, d_model=24, d_input=16, heads=2,
                         n_synch_out=8, n_synch_action=6, memory_length=4,
                         nlm_hidden=8, out_dims=12, certainty_threshold=0.0)
    return f


def test_ensemble_save_load_bitexact(tmp_path):
    torch.manual_seed(0)
    ens = DeepEnsemble.create(_factory(), n=3, base_seed=1, scene_param_dim=4)
    ens.eval()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=71)
    before = ens.predict_next(ds.X[:4], scene_params=ds.P[:4])
    path = tmp_path / "ens.pt"
    save_ensemble(ens, path, metrics={"n": 3})
    ens2, meta = load_ensemble(path)
    assert ens2.n_members == 3
    assert meta["n_members"] == 3
    after = ens2.predict_next(ds.X[:4], scene_params=ds.P[:4])
    assert torch.allclose(before["mean"], after["mean"], atol=1e-6)
    assert torch.allclose(before["variance"], after["variance"], atol=1e-6)
    # 逐成员 state_dict 逐位一致
    for m, m2 in zip(ens.members, ens2.members):
        for k, v in m.state_dict().items():
            assert torch.equal(v, m2.state_dict()[k])


def test_single_model_path_unchanged(tmp_path):
    """旧 save_predictor/load_predictor 不受新增 ensemble 函数影响。"""
    from udos.training import PhysicsPredictor
    torch.manual_seed(1)
    m = PhysicsPredictor(_factory()(), scene_param_dim=4)
    m.eval()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=72)
    before = m.predict_next(ds.X[:4], scene_params=ds.P[:4])
    path = tmp_path / "single.pt"
    save_predictor(m, path, metrics={"x": 1})
    m2, meta = load_predictor(path)
    after = m2.predict_next(ds.X[:4], scene_params=ds.P[:4])
    assert torch.allclose(before, after, atol=1e-6)
    assert meta["udos_version"] == __version__


def test_load_ensemble_rejects_non_ensemble(tmp_path):
    from udos.training import PhysicsPredictor
    m = PhysicsPredictor(_factory()(), scene_param_dim=4)
    path = tmp_path / "single.pt"
    save_predictor(m, path)
    try:
        load_ensemble(path); assert False
    except ValueError:
        pass
