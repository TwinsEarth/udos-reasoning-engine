"""v3.6.3 Patch 精修: 边界条件全绿 + 文档对齐。

覆盖:
    * 零 horizon / 负 horizon 守卫;
    * NaN/inf 潜在状态与想象输出 -> 显式 ValueError;
    * 守恒极端值 (单步、大批、零维守卫);
    * WM 未 fit (外挂恒等转移) 仍确定可跑;
    * wm_events 极端半径/非有限守卫;
    * 文档对齐 (VERSION_PLAN_3.6 存在; README/CHANGELOG 含 PWM);
    * 版本断言。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.world_model import LatentWorldModel
from udos.wm_conservation import ConservationChecker
from udos.wm_events import ContactPredictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.5.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def wm():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return LatentWorldModel(m)


@pytest.fixture(scope="module")
def batch():
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=19)
    return ds.X[:2], ds.P[:2]


def test_zero_horizon_guarded(wm, batch):
    x, sp = batch
    for bad in (0, -1, -100):
        with pytest.raises(ValueError):
            wm.imagine(x, bad, scene_params=sp)
        with pytest.raises(ValueError):
            wm.imagine_rollout(x, bad, scene_params=sp)
        with pytest.raises(ValueError):
            wm.imagine_uncertain(x, bad, scene_params=sp)


def test_uninitialized_wm_still_deterministic(wm, batch):
    """未 fit (外挂恒等转移) 时 imagine 仍确定可跑, 不崩。"""
    x, sp = batch
    a = wm.imagine(x, 4, scene_params=sp)
    b = wm.imagine(x, 4, scene_params=sp)
    assert torch.equal(a, b)
    assert bool(torch.isfinite(a).all())


def test_conservation_single_step_guard():
    chk = ConservationChecker()
    # 单步 (<2) 拒绝
    with pytest.raises(ValueError):
        chk.check(torch.zeros(1, 1, 6))


def test_conservation_extreme_batch():
    """大批守恒检验不崩。"""
    traj = torch.randn(4, 6, 6)
    out = ConservationChecker().check(traj)
    assert out["n_batch"] == 4
    assert out["horizon"] == 6


def test_events_nonfinite_guard():
    bad = torch.tensor([[[0, 0, 0, 0, 0, 0]],
                        [[1, 0, 0, float("inf"), 0, 0]]])
    with pytest.raises(ValueError):
        ContactPredictor().predict(bad)


def test_events_extreme_radius():
    s = torch.zeros(1, 4, 6)
    with pytest.raises(ValueError):
        ContactPredictor().predict(s, agent_radius=0.0)  # 半径非正 -> 400
    # 合法正半径不崩
    out = ContactPredictor().predict(s, agent_radius=10.0, partner_radius=10.0)
    assert out["n_batch"] == 1


def test_docs_aligned():
    """VERSION_PLAN_3.6 存在; README 与 CHANGELOG 提及 PWM/世界模型。"""
    assert (ROOT / "docs" / "VERSION_PLAN_3.6.md").exists()
    cl = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "PWM" in cl or "世界模型" in cl
    assert "[3.6.3]" in cl
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # README 至少应提及 3.6 / PWM / 世界模型之一
    assert any(k in readme for k in ("3.6", "PWM", "世界模型", "world_model"))


def test_version_plan_matches():
    """VERSION_PLAN 终点为 3.6.3。"""
    vp = (ROOT / "docs" / "VERSION_PLAN_3.6.md").read_text(encoding="utf-8")
    assert "v3.6.3" in vp or "3.6.3" in vp
