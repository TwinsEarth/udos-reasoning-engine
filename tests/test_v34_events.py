"""v3.4.0.dev1 事件级切分与三流对齐测试。"""
import pytest
import torch

from udos import __version__
from udos.dynamics import (build_parametric_dataset, traj_collision,
                          traj_uniform, RAW_DIM)
from udos.icm_events import EventSegmenter, ThreeStreamAligner
from udos.icm import DemonstrationEpisode, DemonstrationMemory


def _raw_from_traj(traj):
    return torch.tensor([list(p) + list(v) for p, v in traj],
                        dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 1. 变点检测
# --------------------------------------------------------------------------- #
def test_uniform_no_change_point():
    seg = EventSegmenter(fixed_thresh=0.05)
    traj = _raw_from_traj(traj_uniform(20, 0.5, v0=1.0))
    # 匀速: 速度差恒为 0, 不应检出边界
    assert seg.detect(traj) == []


def test_collision_detects_boundary():
    seg = EventSegmenter(fixed_thresh=0.1)
    traj = _raw_from_traj(traj_collision(30, 0.5, x1=-2.0, v1=2.0,
                                         x2=1.0, v2=-0.3))
    bounds = seg.detect(traj)
    # 碰撞瞬间速度交换 => 至少检出一个边界
    assert len(bounds) >= 1
    assert bounds == sorted(bounds)


def test_short_traj_guard():
    seg = EventSegmenter()
    assert seg.detect(torch.zeros(2, 6)) == []
    with pytest.raises(ValueError):
        seg.detect(torch.zeros(6, 5))   # 末维错


# --------------------------------------------------------------------------- #
# 2. 三流对齐
# --------------------------------------------------------------------------- #
def test_three_stream_aligned_lengths():
    al = ThreeStreamAligner()
    traj = _raw_from_traj(traj_collision(30, 0.5, x1=-2.0, v1=2.0,
                                         x2=1.0, v2=-0.3))
    segs = al.split(traj)
    assert len(segs) >= 1
    for seg in segs:
        n = seg["states"].size(0)
        assert seg["actions"].size(0) == n - 1
        assert seg["results"].size(0) == n - 1
        assert seg["actions"].size(1) == RAW_DIM


def test_align_episodes_registerable():
    """对齐产出的 (window, result) 对可直接注册为 ICM episode。"""
    al = ThreeStreamAligner()
    traj = _raw_from_traj(traj_collision(24, 0.5, x1=-2.0, v1=2.0,
                                         x2=1.0, v2=-0.3))
    pairs = al.align_episodes(traj)
    assert len(pairs) >= 1
    mem = DemonstrationMemory()
    for win, res, _st in pairs:
        ep = DemonstrationEpisode(win, res, kind="collision")
        mem.register(ep)
    assert mem.size == len(pairs)


def test_empty_state_guard():
    al = ThreeStreamAligner()
    with pytest.raises(ValueError):
        al.derive_streams(torch.zeros(1, 6))
    with pytest.raises(ValueError):
        al.split(torch.zeros(1, 6))


def test_no_boundary_degrades_to_whole():
    al = ThreeStreamAligner(EventSegmenter(fixed_thresh=1e9))  # 永不触发
    traj = _raw_from_traj(traj_uniform(16, 0.5, v0=1.5))
    segs = al.split(traj)
    assert len(segs) == 1
    assert segs[0]["kind"] == "event" or segs[0]["kind"] == "whole"


def test_change_signal_shape():
    seg = EventSegmenter()
    traj = _raw_from_traj(traj_uniform(10, 0.5, v0=1.0))
    sig = seg.change_signal(traj)
    assert sig.shape == (9,)
    assert torch.allclose(sig, torch.zeros(9), atol=1e-6)


def test_version():
    assert __version__ == "5.5.5"
