"""M4 探针：版本化 /api/v7 服务层（不依赖已发布大 checkpoint，自训小模型）。"""
import torch

from udos7 import WorldModelCore, __version__
from udos7.dynamics import three_way_splits, build_split
from udos7.train import TrainConfig, fit
from udos7.persistence import save_worldmodel
from udos7.server import V7Service
from udos7.contracts import TEST_SEED


def _tiny_checkpoint(path):
    sp = three_way_splits(n_traj_per_kind=8)
    torch.manual_seed(0)
    m = WorldModelCore(window=6, hidden=32, n_layers=1)
    fit(m, sp["train"], sp["val"],
        TrainConfig(epochs=12, batch=64, lr=3e-3, patience=12, seed=0))
    save_worldmodel(m, path, {"evidence_grade": "cpu-proto"})
    return path


def test_health_predict_interval(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "tiny.pt")
    svc = V7Service(checkpoint=ckpt, n_traj_per_kind=8).load()
    h = svc.health()
    assert h["version"].startswith("7.") and h["api"] == "v7"
    assert h["model_loaded"] is True

    ds = build_split(TEST_SEED, n_traj_per_kind=2)
    payload = {"window": ds.X[0].tolist(), "explicit": ds.P[0].tolist(),
               "horizon": 4}
    out = svc.predict(payload)
    assert len(out["trajectory"][0]) == 4 and len(
        out["trajectory"][0][0]) == 6

    widths = {}
    for a in (0.2, 0.1, 0.05):
        iv = svc.interval({**payload, "alpha": a})
        assert iv["nominal_coverage"] == round(1 - a, 2)
        up = torch.tensor(iv["upper"]); lo = torch.tensor(iv["lower"])
        widths[a] = float((up - lo).mean())
        assert bool(torch.isfinite(up).all())
    assert widths[0.2] < widths[0.1] < widths[0.05]


def test_window_2d_auto_batched_and_blind(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "tiny2.pt")
    svc = V7Service(checkpoint=ckpt, n_traj_per_kind=8).load()
    ds = build_split(TEST_SEED, n_traj_per_kind=1)
    out = svc.predict({"window": ds.X[0].tolist(), "horizon": 2})  # 无 explicit
    assert len(out["trajectory"]) == 1 and len(out["trajectory"][0]) == 2


def test_missing_checkpoint_503(tmp_path):
    svc = V7Service(checkpoint=tmp_path / "nope.pt", n_traj_per_kind=4)
    try:
        svc.predict({"window": [[[0.0] * 6] * 6], "horizon": 1})
        assert False, "应抛 FileNotFoundError"
    except FileNotFoundError:
        pass
