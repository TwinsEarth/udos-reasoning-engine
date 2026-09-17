"""
v2.8.0.dev1 observe / understand 阶段单元测试
===============================================
锚点纪律:
    * observe 调用 predictor 特征提取, 输出 features 形状有限 + scene_context;
    * understand 输出结构化理解向量: target_state[B,6] + risk_flags(ood_flag/uncertain)
      + understanding_vector 固定 [B, 8];
    * 空输入 / NaN 守卫;
    * 与旧 reasoning 接口一致: understand 仍只做只读前向, run 后 predict_next 逐位不变;
    * 未挂载 OOD/校准的旧件上, 风险通道诚实退化为 0/None (不伪造证据)。
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
                                  horizon=4, dt=0.5, seed=777)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_observe_feature_shape_finite(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    out = loop.run(wb, scene_params=pb)
    obs = out["loop_state"]["outputs"]["observe"]
    assert obs["feature_shape"] == [1, 6, predictor.obs_encoder[0].in_features] \
        or obs["feature_shape"][0] == 1
    assert torch.isfinite(obs["features"]).all()
    # 挂载 scene_encoder => scene_context 非 None
    assert obs["scene_context"] is not None
    assert torch.isfinite(obs["scene_context"]).all()


def test_understand_has_target_and_risk(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    out = loop.run(wb, scene_params=pb)
    und = out["loop_state"]["outputs"]["understand"]
    # 目标状态
    assert und["target_state"].shape == (1, 6)
    assert torch.isfinite(und["target_state"]).all()
    # 风险标记字段齐全
    rf = und["risk_flags"]
    assert "ood_flag" in rf and "uncertain" in rf
    # 2.8.0 件挂载了 OOD 检测器 + 校准 => ood_flag 为 bool, interval_width > 0
    assert isinstance(rf["ood_flag"], bool)
    assert rf["interval_width"] > 0.0
    # 理解向量固定 [B, 8] = [目标 6 | 风险 1 | 不确定 1]
    assert und["understanding_vector"].shape == (1, 8)
    assert torch.isfinite(und["understanding_vector"]).all()


def test_empty_and_nan_guard(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    with pytest.raises(ValueError):
        loop.run(torch.zeros(1, 0, predictor.raw_dim), scene_params=pb)
    bad = wb.clone()
    bad[0, 2, 3] = float("inf")
    with pytest.raises(ValueError):
        loop.run(bad, scene_params=pb)


def test_readonly_no_side_effect(predictor, window_batch):
    wb, pb = window_batch
    loop = PhysicalLoopRunner(predictor, horizon=2)
    before = predictor.predict_next(wb, scene_params=pb)
    loop.run(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after)


def test_old_legacy_checkpoint_degrades_honestly():
    """未挂 OOD/校准的旧件: 风险通道诚实为 None/0, 不伪造证据。"""
    old, _ = load_predictor(str(ROOT / "checkpoints" / "predictor_v2.3.1.pt"))
    wb = torch.randn(1, 6, old.raw_dim)
    pb = (torch.randn(1, old.scene_param_dim)
          if old.scene_encoder is not None else None)
    loop = PhysicalLoopRunner(old, horizon=1)
    out = loop.run(wb, scene_params=pb)
    und = out["loop_state"]["outputs"]["understand"]
    # 旧件无 OOD 检测器 => ood_flag 为 None
    assert und["risk_flags"]["ood_flag"] is None
    # 理解向量仍固定 [1, 8]
    assert und["understanding_vector"].shape == (1, 8)
