"""v3.4.0.dev2 跨本体演示归一化 ICL 测试。"""
import pytest
import torch

from udos import load_predictor, __version__
from udos.dynamics import build_parametric_dataset, RAW_DIM
from udos.retargeting import MorphologyConfig
from udos.icm_cross import CrossEmbodimentICM

CKPT = "checkpoints/predictor_v3.4.0.pt"


@pytest.fixture(scope="module")
def model():
    m, _ = load_predictor(CKPT)
    return m


def _src_morph(dof):
    lim = [[-1.0, 1.0]] * dof
    return MorphologyConfig(dof=dof, control_freq=20.0, joint_limits=lim,
                            name=f"src{dof}")


def _tgt_morph():
    lim = [[-1.0, 1.0]] * RAW_DIM
    return MorphologyConfig(dof=RAW_DIM, control_freq=20.0, joint_limits=lim,
                            name="tgt6")


def test_cross_embodiment_normalizes(model):
    """源本体 dof=4 轨迹 -> 归一化到 dof=6 -> 注册进 ICM。"""
    src = _src_morph(4)
    tgt = _tgt_morph()
    xemb = CrossEmbodimentICM(model, src, tgt)
    # 源本体 4 维轨迹 (12 帧), 模拟不同 DOF 本体采集
    src_traj = torch.randn(12, 4) * 0.5
    n = xemb.register_trajectory(src_traj, window=6)
    assert n > 0
    assert len(xemb) == n
    # 查询窗口必须是目标本体 6 维
    q = torch.randn(6, 6)
    out = xemb.predict(q, k=2)
    assert out.shape == (RAW_DIM,)
    assert bool(torch.isfinite(out).all())


def test_target_dof_must_be_6(model):
    src = _src_morph(4)
    bad_tgt = MorphologyConfig(dof=5, control_freq=20.0,
                               joint_limits=[[-1, 1]] * 5, name="bad")
    with pytest.raises(ValueError):
        CrossEmbodimentICM(model, src, bad_tgt)


def test_source_dof_mismatch_guard(model):
    src = _src_morph(4)
    xemb = CrossEmbodimentICM(model, src, _tgt_morph())
    with pytest.raises(ValueError):
        # 传入 8 维轨迹 != 源 dof 4
        xemb.register_trajectory(torch.randn(12, 8), window=6)


def test_dof0_source_unredirectable(model):
    src = MorphologyConfig(dof=0, control_freq=20.0, name="dof0")
    with pytest.raises(ValueError):
        CrossEmbodimentICM(model, src, _tgt_morph())


def test_time_resample_path(model):
    src = _src_morph(4)
    xemb = CrossEmbodimentICM(model, src, _tgt_morph())
    src_traj = torch.randn(10, 4) * 0.5
    n = xemb.register_trajectory(src_traj, window=6,
                                 src_freq=10.0, dst_freq=20.0)
    assert n > 0


def test_query_shapes(model):
    src = _src_morph(4)
    xemb = CrossEmbodimentICM(model, src, _tgt_morph())
    xemb.register_trajectory(torch.randn(12, 4) * 0.5, window=6)
    # 空记忆已注册 => 不应退化; 再测 k=0 走 0-shot
    out = xemb.predict(torch.randn(6, 6), k=0)
    assert out.shape == (RAW_DIM,)


def test_version():
    assert __version__ == "5.5.5"
