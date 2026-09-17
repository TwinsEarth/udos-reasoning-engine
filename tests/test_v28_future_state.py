"""
v2.8.0.dev3 future_state 阶段单元测试
=========================================
锚点纪律:
    * rollout 输出 [B, H, 6] 且有限;
    * 偏差信号 = 首步轨迹 vs understand.target_state; 无动作扰动时二者同为
      predict_next => 偏差逐位为 0;
    * horizon=1 退化为 [B, 1, 6];
    * 与旧 predictor.rollout 逐位一致 (无动作扰动时 trajectory == rollout(window,H));
    * 挂载 conformal 半宽时给出 lower/upper 有限区间。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.physical_loop import PhysicalLoopRunner
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.8.0.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=999)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_trajectory_shape_finite(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=3)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    fs = out["loop_state"]["outputs"]["future_state"]
    assert fs["trajectory"].shape == (1, 3, 6)
    assert torch.isfinite(fs["trajectory"]).all()
    # 2.8.0 件挂载了 conformal 半宽 => 给出区间
    assert fs["has_interval"] is True
    assert torch.isfinite(fs["lower"]).all()
    assert torch.isfinite(fs["upper"]).all()


def test_deviation_zero_when_no_perturbation(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=3)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    fs = out["loop_state"]["outputs"]["future_state"]
    und = out["loop_state"]["outputs"]["understand"]
    # 无扰动: 首步 rollout == predict_next == understand.target_state
    assert torch.allclose(fs["trajectory"][:, 0, :], und["target_state"],
                          atol=1e-6)
    assert torch.allclose(fs["deviation"], torch.zeros(1, 6), atol=1e-6)


def test_horizon_one_degenerate(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=1)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    fs = out["loop_state"]["outputs"]["future_state"]
    assert fs["trajectory"].shape == (1, 1, 6)


def test_bit_identical_to_predictor_rollout(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=4)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    fs = out["loop_state"]["outputs"]["future_state"]
    direct = predictor.rollout(wb, 4, scene_params=pb)
    assert torch.equal(fs["trajectory"], direct)


def test_action_perturbation_changes_trajectory(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    # 选一个强扰动动作, 其轨迹应与无操作轨迹不同
    out = loop.run(wb, scene_params=pb,
                   candidate_actions=[{"state_perturbation": [5.0] * 6}])
    fs = out["loop_state"]["outputs"]["future_state"]
    flat = predictor.rollout(wb, 2, scene_params=pb)
    assert not torch.allclose(fs["trajectory"], flat, atol=1e-4)
