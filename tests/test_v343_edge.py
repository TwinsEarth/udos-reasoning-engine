"""v3.4.3 边界测试 (ICM 全特性边界退化路径)。"""
import pytest
import torch

from udos import load_predictor, __version__
from udos.dynamics import build_parametric_dataset, traj_uniform, RAW_DIM
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator
from udos.icm_events import EventSegmenter, ThreeStreamAligner
from udos.icm_budget import ContextBudgetManager
from udos.pce_format import DemonstrationPrompt, PCEPromptParser

CKPT = "checkpoints/predictor_v3.4.0.pt"


@pytest.fixture(scope="module")
def model():
    m, _ = load_predictor(CKPT)
    return m


# --------------------------------------------------------------------------- #
# 空记忆 / 零预算
# --------------------------------------------------------------------------- #
def test_empty_memory_predict_zeroshot(model):
    agg = ICMAggregator(model)
    te = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=3)
    out = agg.predict(te.X[0], memory=DemonstrationMemory(), k=5,
                      scene_params=te.P[0:1])
    p0 = model.predict_next(te.X[0:1], scene_params=te.P[0:1])[0]
    assert torch.allclose(out, p0, atol=1e-6)


def test_zero_budget_cap(model):
    bm = ContextBudgetManager(budget=1)
    assert bm.cap_k(100, 100) == 1


def test_nonexistent_residual_key(model):
    agg = ICMAggregator(model)
    fake = DemonstrationEpisode(torch.randn(6, 6), torch.randn(6))
    with pytest.raises(KeyError):
        agg.get_cached_residual(fake)


# --------------------------------------------------------------------------- #
# 事件边界退化
# --------------------------------------------------------------------------- #
def test_event_boundary_degenerate_uniform():
    seg = EventSegmenter(fixed_thresh=0.01)
    traj = torch.tensor([list(p) + list(v) for p, v in
                         [( [i*0.1,0,0],[1.0,0,0]) for i in range(12)]],
                        dtype=torch.float32)
    assert seg.detect(traj) == []   # 匀速无边界


def test_three_stream_short_trajectory_guard():
    al = ThreeStreamAligner()
    with pytest.raises(ValueError):
        al.split(torch.zeros(1, 6))   # 仅 1 帧 < 最小 2 帧


# --------------------------------------------------------------------------- #
# ICM 未注册演示 / 服务未训练态 (纯对象层)
# --------------------------------------------------------------------------- #
def test_predict_no_memory_arg(model):
    agg = ICMAggregator(model)
    te = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4)
    out = agg.predict(te.X[0], memory=None, k=9, scene_params=te.P[0:1])
    assert out.shape == (6,)


# --------------------------------------------------------------------------- #
# PCE 空包 / 坏包
# --------------------------------------------------------------------------- #
def test_pce_empty_prompt_roundtrip():
    p = DemonstrationPrompt("edge-empty")
    d = p.to_dict()
    p2 = PCEPromptParser.from_dict(d)
    assert PCEPromptParser.load_episodes(p2) == []


def test_pce_bad_version():
    with pytest.raises(ValueError):
        PCEPromptParser.from_dict({"packet": "bad"})


# --------------------------------------------------------------------------- #
# 跨本体 DOF=0 (已在 cross 测试覆盖, 这里补 episode 动作校验)
# --------------------------------------------------------------------------- #
def test_episode_action_override_mismatch(model):
    with pytest.raises(ValueError):
        DemonstrationEpisode(torch.randn(6, 6), torch.randn(6),
                             action=torch.randn(5))


def test_version():
    assert __version__ == "5.5.5"
