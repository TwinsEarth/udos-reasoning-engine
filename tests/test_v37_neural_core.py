"""v3.7.0 分层神经控制架构核心测试 (大脑/小脑/脊髓)。

覆盖:
    * 三层结构: HierarchicalController 持有 cortex/cerebellum/spinal, 均为 ControlLayer 子类;
    * step 统一接口: 返回 command[6] / step_index / priority_winner / 各层诊断;
    * 频率配置: cortex_every 由频率比自动推算; 各层 frequency_hz / latency_budget_ms / priority_rank;
    * 空守卫: 空 window / NaN / 错误最后一维 / cortex_every<1 显式 ValueError;
    * 与 policy/loop 接口一致: 首步大脑目标 == predictor.predict_next 逐位一致;
    * 优先级常量: 脊髓(0) < 小脑(1) < 大脑(2);
    * 确定性 + 零梯度: 同输入两次 command 逐位一致; step 不改动主 predictor 权重 md5。
analogy, not reproduction —— 合成延迟预算类比, 非真机 whole-body control 复现。
"""
import hashlib
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset, RAW_DIM
from udos.neural_control import (
    ControlLayer,
    CortexPlanner,
    CerebellumTracker,
    SpinalReflex,
    HierarchicalController,
)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.6.0.pt"


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
                                  horizon=4, dt=0.5, seed=11)
    return ds.X[:1], ds.P[:1]


def _md5(model):
    h = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_priority_ordering():
    """优先级常量: 脊髓最高 (数值最小)。"""
    assert ControlLayer.PRIORITY["spinal"] < ControlLayer.PRIORITY["cerebellum"]
    assert ControlLayer.PRIORITY["cerebellum"] < ControlLayer.PRIORITY["cortex"]


def test_three_layer_structure(predictor):
    ctrl = HierarchicalController(predictor)
    assert isinstance(ctrl.cortex, CortexPlanner)
    assert isinstance(ctrl.cerebellum, CerebellumTracker)
    assert isinstance(ctrl.spinal, SpinalReflex)
    assert isinstance(ctrl.cortex, ControlLayer)
    assert ctrl.cortex.priority_rank == ControlLayer.PRIORITY["cortex"]
    assert ctrl.cerebellum.priority_rank == ControlLayer.PRIORITY["cerebellum"]
    assert ctrl.spinal.priority_rank == ControlLayer.PRIORITY["spinal"]
    # 频率 / 延迟预算配置
    assert ctrl.cortex.frequency_hz > 0
    assert ctrl.cerebellum.frequency_hz > 0
    assert ctrl.spinal.frequency_hz > 0
    assert ctrl.cortex.latency_budget_ms > ctrl.spinal.latency_budget_ms


def test_frequency_ratio_default(predictor):
    """cortex_hz=2, cerebellum_hz=20 => cortex_every=10。"""
    ctrl = HierarchicalController(predictor, cortex_hz=2.0, cerebellum_hz=20.0)
    assert ctrl.cortex_every == 10
    # 显式覆盖
    ctrl2 = HierarchicalController(predictor, cortex_every=4)
    assert ctrl2.cortex_every == 4


def test_step_interface_and_shapes(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor)
    out = ctrl.step(w, scene_params=sp)
    assert out["command"].shape == (RAW_DIM,)
    assert out["step_index"] == 0
    assert out["priority_winner"] in ("spinal", "cerebellum", "cortex")
    assert "cortex" in out and "cerebellum" in out and "spinal" in out
    assert out["cortex"]["ran"] is True   # 首步必规划
    # 各层诊断含实测延迟
    assert out["cortex"]["elapsed_ms"] >= 0
    assert out["cerebellum"]["elapsed_ms"] >= 0
    assert out["spinal"]["elapsed_ms"] >= 0


def test_cortex_target_matches_predict_next(predictor, window):
    """首步大脑目标与 predictor.predict_next 逐位一致 (接口一致)。"""
    w, sp = window
    ctrl = HierarchicalController(predictor)
    out = ctrl.step(w, scene_params=sp)
    with torch.no_grad():
        ref = predictor.predict_next(w, scene_params=sp)[0]
    assert torch.allclose(out["cortex"]["target"], ref, atol=1e-6)


def test_deterministic(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor)
    a = ctrl.step(w, scene_params=sp)["command"]
    ctrl.reset()
    b = ctrl.step(w, scene_params=sp)["command"]
    assert torch.equal(a, b)


def test_zero_grad_no_weight_mutation(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor)
    before = _md5(predictor)
    ctrl.step(w, scene_params=sp)
    ctrl.step(w, scene_params=sp)
    assert _md5(predictor) == before


def test_empty_and_invalid_guards(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor)
    with pytest.raises(ValueError):
        ctrl.step(torch.zeros(0, RAW_DIM))
    with pytest.raises(ValueError):
        bad = w.clone(); bad[0, -1, 0] = float("nan")
        ctrl.step(bad)
    with pytest.raises(ValueError):
        ctrl.step(torch.zeros(3, RAW_DIM + 1))
    with pytest.raises(ValueError):
        HierarchicalController(predictor, cortex_every=0)


def test_reset(predictor, window):
    w, sp = window
    ctrl = HierarchicalController(predictor)
    ctrl.step(w, scene_params=sp)
    ctrl.step(w, scene_params=sp)
    assert ctrl.step_index == 2
    ctrl.reset()
    assert ctrl.step_index == 0
