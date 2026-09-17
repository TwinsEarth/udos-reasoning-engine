"""v3.7.0.dev4 三回路多频率分层调度测试。

覆盖:
    * 多频率调度: 大脑每 cortex_every 步规划一次, 小脑/脊髓每步;
    * 频率比可配: 显式 cortex_every 与频率比自动推算;
    * 大脑不每步调用: run_counts.cortex < total;
    * 脊髓每步检查: run_counts.spinal == total;
    * scheduler_summary 字段正确;
    * 空守卫 (cortex_every<1)。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from udos.neural_control import HierarchicalController

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.7.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


@pytest.fixture(scope="module")
def window():
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=44)
    return ds.X[:1], ds.P[:1]


def test_multi_frequency_scheduling(predictor, window):
    w, sp = window
    every = 4
    ctrl = HierarchicalController(predictor, cortex_every=every)
    n = 10
    for _ in range(n):
        ctrl.step(w, scene_params=sp)
    # 小脑/脊髓每步都跑
    assert ctrl.run_counts["cerebellum"] == n
    assert ctrl.run_counts["spinal"] == n
    # 大脑: step 0,4,8 规划 => ceil((n)/every)=3
    assert ctrl.run_counts["cortex"] == 3
    # 大脑不每步
    assert ctrl.run_counts["cortex"] < n


def test_auto_ratio_from_hz(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor, cortex_hz=2.0,
                                  cerebellum_hz=20.0)
    assert ctrl.cortex_every == 10
    ctrl.step(w, scene_params=sp)
    s = ctrl.scheduler_summary()
    assert s["cortex_every"] == 10
    assert s["ran_cortex"] == 1
    assert s["ran_spinal"] == 1


def test_scheduler_summary_fields(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor, cortex_every=3)
    for _ in range(6):
        ctrl.step(w, scene_params=sp)
    s = ctrl.scheduler_summary()
    assert s["total_steps"] == 6
    assert s["ran_cortex"] == 2       # step 0,3
    assert s["ran_cerebellum"] == 6
    assert s["ran_spinal"] == 6
    assert 0.0 < s["cortex_ratio"] < 1.0


def test_empty_guard(predictor):
    with pytest.raises(ValueError):
        HierarchicalController(predictor, cortex_every=0)
