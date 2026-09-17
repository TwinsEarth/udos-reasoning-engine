"""
v3.2.3 节点50: 3.2 线边界精修
==================================
覆盖极端/非法输入:
    * 增强参数极端值 (极大噪声/扰动/时间缩放);
    * 长窗口 W=0;
    * 记忆溢出 (超 capacity);
    * ICL 示例维度不匹配;
    * 服务未训练态 (409);
    * long horizon H=0。
"""
import pytest
import torch

from udos import __version__
from udos.ego_data import SyntheticEgoAugmenter, time_warp
from udos.extended_context import ExtendedContextWindow
from udos.temporal_memory import TemporalMemory
from udos.incontext import InContextLearner
from udos.longhorizon import LongHorizonRollout
from udos.server import UDOSService, ServiceNotReady
from udos.persistence import load_predictor
from udos.dynamics import build_parametric_dataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.2.0.pt"


@pytest.fixture(scope="module")
def pred():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


@pytest.fixture(scope="module")
def ds():
    return build_parametric_dataset(n_per_kind=2, n_steps=12, window=6,
                                    horizon=2, dt=0.5, seed=333)


def test_augment_extreme_params(ds):
    """极端增强参数不崩溃、输出有限。"""
    huge = SyntheticEgoAugmenter(view_rotate_deg=179.0, traj_perturb=10.0,
                                noise_sigma=5.0, time_scale=3.0, seed=1)
    Xa, Ya = huge.augment_pair(ds.X[:4], ds.Y[:4])
    assert torch.isfinite(Xa).all() and torch.isfinite(Ya).all()
    # 时间缩放极值仍形状保持
    assert Xa.shape == ds.X[:4].shape


def test_time_warp_clamps(ds):
    out = time_warp(ds.X[0], 10.0)
    assert out.shape == ds.X[0].shape
    assert torch.isfinite(out).all()


def test_bad_augment_constructor():
    try:
        SyntheticEgoAugmenter(time_scale=0.0)
        assert False
    except ValueError:
        pass
    try:
        SyntheticEgoAugmenter(view_translate_xyz=(1.0, 2.0))  # 长度 2
        assert False
    except ValueError:
        pass


def test_window_w0_guard(pred, ds):
    ecw = ExtendedContextWindow(pred, max_len=24)
    try:
        ecw.predict_next(torch.zeros(1, 0, 6))
        assert False
    except ValueError:
        pass


def test_memory_overflow():
    mem = TemporalMemory(capacity=4)
    for i in range(20):
        mem.update(torch.tensor([float(i)] * 6))
    assert mem.buffered == 4
    # 保留最近 4 帧 [16,17,18,19]
    assert float(mem.history(4)[-1, 0]) == 19.0


def test_memory_summary_after_reset_empty():
    mem = TemporalMemory(capacity=4)
    mem.update(torch.ones(6))
    mem.reset()
    try:
        mem.summary()
        assert False
    except RuntimeError:
        pass


def test_icl_dim_mismatch(pred, ds):
    icl = InContextLearner(pred)
    bad = torch.randn(6, 6)  # W=6 ok, 但改成 RAW 错
    bad = torch.randn(6, 5)  # RAW=5 != 6
    try:
        icl.build_context(ds.X[0], [bad])
        assert False
    except ValueError:
        pass


def test_service_untrained_augment_ok_but_icl_409(ds):
    svc = UDOSService(preset="small")  # 未加载 checkpoint
    # augment 无状态 => 可用
    out = svc.augment_generate({"window": ds.X[0].tolist()})
    assert out["status"] == "ok"
    # icl 需模型 => 409
    try:
        svc.icl_predict({"window": ds.X[0].tolist()})
        assert False
    except ServiceNotReady:
        pass


def test_long_horizon_h0_guard(pred, ds):
    lh = LongHorizonRollout(pred)
    try:
        lh.rollout(ds.X[:1], 0, scene_params=ds.P[:1])
        assert False
    except ValueError:
        pass


def test_version():
    assert __version__ == "5.5.5"
