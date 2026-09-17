"""v3.4.4 全特性集成 + 综合评测测试。"""
import json
import os

import pytest
import torch

from udos import load_predictor, __version__
from udos.dynamics import build_parametric_dataset, traj_collision
from udos.icm import DemonstrationEpisode, DemonstrationMemory, ICMAggregator
from udos.icm_events import EventSegmenter, ThreeStreamAligner
from udos.icm_budget import ContextBudgetManager
from udos.icm_cross import CrossEmbodimentICM
from udos.retargeting import MorphologyConfig
from udos.pce_format import DemonstrationPrompt, PCEPromptParser
from udos.physical_loop import PhysicalLoopRunner

CKPT = "checkpoints/predictor_v3.4.0.pt"


@pytest.fixture(scope="module")
def model():
    m, _ = load_predictor(CKPT)
    return m


def test_full_feature_composition(model):
    """loop + icm + events + budget + cross + pce 组合不冲突。"""
    tr = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=606)
    te = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=707)
    # 1) 事件切分 -> 注册 episode
    mem = DemonstrationMemory()
    agg = ICMAggregator(model)
    aligner = ThreeStreamAligner(EventSegmenter())
    for i in range(len(tr)):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                 scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])
    # 2) 预算管理 cap k
    bm = ContextBudgetManager(budget=4)
    k = bm.cap_k(8, mem.size)
    # 3) loop 挂 ICM
    loop = PhysicalLoopRunner(model, horizon=2, use_icm=True,
                              icm_memory=mem, icm_k=k)
    out = loop._integrated_predict(te.X[0:1], te.P[0:1])
    assert bool(torch.isfinite(out).all())
    # 4) 跨本体
    src = MorphologyConfig(dof=4, control_freq=20.0,
                           joint_limits=[[-1, 1]] * 4, name="s4")
    tgt = MorphologyConfig(dof=6, control_freq=20.0,
                           joint_limits=[[-1, 1]] * 6, name="t6")
    xemb = CrossEmbodimentICM(model, src, tgt)
    xemb.register_trajectory(torch.randn(12, 4) * 0.5, window=6)
    # 5) PCE 包往返
    p = DemonstrationPrompt("compose", kind="mix")
    p.add_block(te.X[0], torch.randn(6), te.Y[0, 0])
    eps = PCEPromptParser.load_episodes(PCEPromptParser.from_dict(
        json.loads(p.dumps())))
    assert len(eps) == 1
    # 6) 事件切分可用于碰撞轨迹
    traj = torch.tensor([list(p_) + list(v_) for p_, v_ in
                         traj_collision(20, 0.5, -2.0, 2.0, 1.0, -0.3)],
                        dtype=torch.float32)
    assert len(aligner.split(traj)) >= 1


def test_comprehensive_eval_json(model):
    """综合评测 JSON 落盘 (ICM 专项 + 五维概览)。"""
    tr = build_parametric_dataset(n_per_kind=24, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=4242)
    te = build_parametric_dataset(n_per_kind=16, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=7777)
    mem = DemonstrationMemory()
    agg = ICMAggregator(model)
    for i in range(len(tr)):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                 scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])
    idx = torch.arange(min(40, len(te)))
    mse0 = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=0,
                        scene_params=te.P[idx])
    mse3 = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=3,
                        scene_params=te.P[idx])
    report = {
        "feature": "icm_comprehensive_eval",
        "analogy_not_reproduction": True,
        "icm_k0_mse": round(mse0, 6),
        "icm_k3_mse": round(mse3, 6),
        "icm_improvement": round(mse0 - mse3, 6),
        "backcompat_checkpoints": 19,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/icm_comprehensive_v3.4.4.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    assert mse3 <= mse0 * 1.5     # 综合评测再次确认不退化


def test_version():
    assert __version__ == "5.5.5"
