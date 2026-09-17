"""v3.7.0.dev2 小脑边缘轨迹平滑/跟踪协调测试 (PID 代理 + 前馈 + 低通)。

覆盖:
    * 跟踪误差: 多步后命令朝目标逼近 (误差范数下降);
    * 平滑性: 低通使相邻 smoothed 增量有界 (阶跃不瞬移);
    * PID 参数: kp/ki/kd/kff/alpha 可配; alpha 越界显式 ValueError;
    * 阶跃响应/前馈: kff 对目标跳变给出前馈项;
    * 空目标守卫: ctx 无 target => ValueError;
    * 确定性: reset 后同序列命令逐位一致。
"""
import pytest
import torch

from udos import __version__
from udos.neural_control import CerebellumTracker


def test_version():
    assert __version__ == "5.5.5"


def test_alpha_bounds():
    with pytest.raises(ValueError):
        CerebellumTracker(alpha=1.5)
    with pytest.raises(ValueError):
        CerebellumTracker(alpha=-0.1)


def test_tracking_converges():
    """比例跟踪: 命令朝目标逼近, 误差范数随步下降。"""
    cb = CerebellumTracker(kp=0.4, ki=0.0, kd=0.0, alpha=0.5)
    state = torch.zeros(6)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    errs = []
    s = state.clone()
    for _ in range(20):
        out = cb.act(s.unsqueeze(0), {"target": target})
        errs.append(float(out["error"].norm()))
        s = out["command"]   # 命令回代到状态 (闭环模拟)
    # 低通存在相位滞后, 允许轻微超调; 稳态收敛: 终值误差远小于初值
    assert errs[-1] < errs[0] * 0.5
    assert errs[-1] < 0.1


def test_smoothness_lowpass():
    """低通平滑: alpha=1 时 smoothed 滞后 (相邻增量有界, 不瞬移)。"""
    cb = CerebellumTracker(kp=0.5, alpha=0.8)
    target = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s = torch.zeros(6)
    prev = None
    max_step = 0.0
    for _ in range(5):
        out = cb.act(s.unsqueeze(0), {"target": target})
        if prev is not None:
            max_step = max(max_step,
                           float((out["smoothed"] - prev).abs().max()))
        prev = out["smoothed"]
        s = out["command"]
    # 低通下每步 smoothed 增量受 kp*err*(1-alpha) 上界约束, 不瞬时跳满
    assert max_step < 2.0


def test_feedforward_reacts_to_target_jump():
    """kff>0 时, 目标跳变当步前馈项非零 (预判目标速度)。"""
    cb = CerebellumTracker(kp=0.0, kff=0.5, alpha=0.0)
    s = torch.zeros(6)
    t1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out1 = cb.act(s.unsqueeze(0), {"target": t1})   # 首步无 prev_target => ff=0
    assert float(out1["feedforward"].abs().sum()) == 0.0
    t2 = t1 + 0.5
    out2 = cb.act(s.unsqueeze(0), {"target": t2})
    assert float(out2["feedforward"].abs().sum()) > 0.0


def test_missing_target_guard():
    cb = CerebellumTracker()
    with pytest.raises(ValueError):
        cb.act(torch.zeros(1, 6), {})


def test_deterministic_after_reset():
    cb = CerebellumTracker(kp=0.3, ki=0.1, kd=0.05, kff=0.2, alpha=0.6)
    target = torch.tensor([0.5, -0.2, 0.1, 0.0, 0.0, 0.0])
    s = torch.zeros(6)
    seq = []
    for _ in range(4):
        out = cb.act(s.unsqueeze(0), {"target": target})
        seq.append(out["command"].clone())
        s = out["command"]
    cb.reset()
    s = torch.zeros(6)
    for c in seq:
        out = cb.act(s.unsqueeze(0), {"target": target})
        assert torch.equal(out["command"], c)
        s = out["command"]
