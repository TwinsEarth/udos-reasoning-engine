"""
v2.6.0+dev2 场景参数辨识 / 归因单元测试
===========================================
- 真实 checkpoint: 匀速轨迹 v0 反演、Sobol 指数和≈1、同输入两次调用确定性。
- Sobol "spring 主导" 用解析弹簧 mock 预测器验证归因机制本身
  (真实 v2.6.0 模型对 omega 弱敏感——训练模型特性, 见 CHANGELOG 诚实记录)。
"""
import math
import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.identification import (SceneParameterIdentifier, sobol_attribution,
                                PARAM_RANGES)
from udos.dynamics import traj_uniform, _raw

CKPT = "checkpoints/predictor_v2.6.0.pt"


class _AnalyticSpringRollout:
    """解析弹簧 mock: 输出随 spring_omega 强烈变化, 其他参数仅微扰。
    用于验证 Sobol 归因机制能把主导参数排到第一。"""
    raw_dim = 6

    def eval(self):
        return self

    @torch.no_grad()
    def rollout(self, window, horizon, scene_params=None, **kw):
        n = scene_params.size(0)
        omega = scene_params[:, 2]                     # [n]
        out = torch.zeros(n, horizon, 6)
        for t in range(horizon):
            tt = (t + 1) * 0.5
            out[:, t, 0] = torch.cos(omega * tt)        # 强 omega 依赖
            out[:, t, 3] = -omega * torch.sin(omega * tt)
            # 其他参数仅小扰动
            out[:, t, 0] += 0.02 * scene_params[:, 0] + 0.01 * scene_params[:, 1]
        return out


@pytest.fixture(scope="module")
def model():
    m, _ = load_predictor(CKPT)
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_identify_uniform_v0(model):
    """匀速轨迹反演 v0 误差 < 0.5。"""
    raw = torch.tensor([_raw(p, v) for p, v in
                        traj_uniform(12, 0.5, v0=1.3, x0=0.0)], dtype=torch.float32)
    win = raw[:8].unsqueeze(0)
    r = SceneParameterIdentifier(model, grid_size=5).identify(win, horizon=2)
    assert r["param_names"] == ["v0", "accel_a", "spring_omega", "other_v2"]
    assert abs(float(r["identified_params"][0]) - 1.3) < 0.5


def test_sobol_sum_to_one(model):
    raw = torch.tensor([_raw(p, v) for p, v in
                        traj_uniform(12, 0.5, v0=1.0, x0=0.0)], dtype=torch.float32)
    win = raw[:6].unsqueeze(0)
    sp = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    s = sobol_attribution(model, win, sp, n_samples=32, seed=0)
    total = float(s["indices"].sum())
    assert 0.9 <= total <= 1.1


def test_sobol_spring_dominant():
    """解析弹簧上, spring_omega 的 Sobol 指数最大。"""
    torch.manual_seed(0)
    win = torch.zeros(1, 6, 6)
    sp = torch.tensor([[0.0, 0.0, 1.1, 0.0]])
    s = sobol_attribution(_AnalyticSpringRollout(), win, sp,
                          n_samples=32, seed=0)
    dom = int(torch.argmax(s["indices"]))
    assert dom == 2          # spring_omega
    assert float(s["indices"][2]) > 0.5


def test_identify_deterministic(model):
    raw = torch.tensor([_raw(p, v) for p, v in
                        traj_uniform(12, 0.5, v0=0.8, x0=0.0)], dtype=torch.float32)
    win = raw[:8].unsqueeze(0)
    ident = SceneParameterIdentifier(model, grid_size=5)
    r1 = ident.identify(win, horizon=2)
    r2 = ident.identify(win, horizon=2)
    assert torch.allclose(r1["identified_params"], r2["identified_params"])
    assert r1["loss_curve_min"] == r2["loss_curve_min"]
