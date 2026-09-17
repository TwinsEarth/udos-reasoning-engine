"""
v2.8.0.dev4 feedback / correction 阶段单元测试
=================================================
锚点纪律:
    * 无偏差/无漂移 => 不触发 (triggered=False);
    * 偏差超阈 => correction 记录 (reason/deviation_norm);
    * 默认 enable_recalibration=False => 即使触发也不改权重/校准 (weights_modified=False);
    * opt-in enable_recalibration=True 且挂载 online => 触发 PAVA 再校准;
    * correction 信号回写 loop_state["correction"];
    * 与旧 online.Adapter 接口一致 (observe/is_drifted/check_and_adapt)。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.physical_loop import PhysicalLoopRunner
from udos.persistence import load_predictor
from udos.online import OnlineAdapter
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
                                  horizon=4, dt=0.5, seed=1010)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_no_deviation_no_trigger(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    out = loop.run(wb, scene_params=pb, candidate_actions=[{}])
    fb = out["loop_state"]["outputs"]["feedback"]
    assert fb["triggered"] is False
    assert fb["weights_modified"] is False
    assert loop.loop_state["correction"]["triggered"] is False


def test_deviation_over_threshold_recorded(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    loop.dev_threshold = 0.0     # 任意非零偏差即触发
    # 强扰动动作 => future_state 首步轨迹偏离 understand 目标态
    out = loop.run(wb, scene_params=pb,
                   candidate_actions=[{"state_perturbation": [3.0] * 6}])
    fb = out["loop_state"]["outputs"]["feedback"]
    assert fb["triggered"] is True
    assert "deviation" in fb["reason"]
    assert fb["deviation_norm"] > 0.0
    # 默认不开启再校准 => 不改权重
    assert fb["weights_modified"] is False


def test_default_no_weight_change_on_trigger(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    loop.dev_threshold = 0.0
    before = predictor.predict_next(wb, scene_params=pb).clone()
    loop.run(wb, scene_params=pb,
             candidate_actions=[{"state_perturbation": [3.0] * 6}])
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after), "默认 feedback 不得改 predictor 权重/校准"


def test_optin_recalibration_on_drift(predictor, window_batch):
    wb, pb = window_batch
    ref = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                   horizon=4, dt=0.5, seed=90210).X
    online = OnlineAdapter(ref, drift_factor=0.3, drift_window=8)
    loop = PhysicalLoopRunner(predictor, horizon=2)
    loop.online = online
    loop.enable_recalibration = True
    loop.recalibration_data = build_parametric_dataset(
        n_per_kind=16, n_steps=14, window=6, horizon=4, dt=0.5, seed=5123)
    # 用 OOD 大扰动窗口灌入流式检测器, 造漂移
    ood = wb * 5.0
    fb_last = None
    for _ in range(10):
        out = loop.run(ood, scene_params=pb, candidate_actions=[{}])
        fb_last = out["loop_state"]["outputs"]["feedback"]
    assert fb_last["drifted"] is True
    # opt-in 再校准: recalibrated=True (PAVA 重拟合, 主权重仍不改)
    assert fb_last["recalibrated"] is True
    assert fb_last["weights_modified"] is False


def test_correction_written_to_loop_state(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    loop.dev_threshold = 0.0
    loop.run(wb, scene_params=pb,
             candidate_actions=[{"state_perturbation": [3.0] * 6}])
    assert loop.loop_state["correction"] is not None
    assert loop.loop_state["correction"]["triggered"] is True
