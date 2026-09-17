"""v3.8.0.dev2 规划-执行-反馈全域闭环: ClosedLoopOrchestrator。

覆盖:
    * 一步闭环: 大脑->小脑->脊髓->WM想象反馈->空间感知更新 各模块均被调用;
    * 反馈误差字段存在且有限; 感知窗正确滑窗;
    * 多步闭环状态机式推进 (step_index 递增, 窗更新);
    * 反射触发时 priority_winner=spinal;
    * 空/非法 window 守卫; 确定性; 零梯度 (主权重 md5 不变);
    * 与既有 HierarchicalController/LatentWorldModel 接口一致。
"""
import hashlib
from pathlib import Path

import torch

from udos import __version__, load_predictor
from udos.closed_loop import ClosedLoopOrchestrator

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_version():
    assert __version__ == "5.5.5"


def test_one_loop_step_wires_all_modules():
    model, _ = load_predictor(CKPT)
    orch = ClosedLoopOrchestrator(model, wm_horizon=2)
    w = torch.randn(1, 6, 6)
    out = orch.step(w)
    # 三层均被调用
    assert set(["cortex", "cerebellum", "spinal"]).issubset(out.keys())
    # WM 反馈存在且有限
    fb = out["wm_feedback"]
    assert fb["horizon"] == 2
    assert fb["feedback_l1_to_next"] >= 0
    # 感知窗滑窗: 形状 [1,6,6]
    assert out["perception"]["updated_window_shape"] == [1, 6, 6]
    assert out["perception"]["window_slid"] is True
    # command 6 维有限
    assert out["command"].shape == (6,)
    assert bool(torch.isfinite(out["command"]).all())


def test_state_machine_advances():
    model, _ = load_predictor(CKPT)
    orch = ClosedLoopOrchestrator(model, wm_horizon=2)
    w = torch.randn(1, 6, 6)
    o0 = orch.step(w)
    assert o0["step_index"] == 0
    # 下一步用上一步更新后的窗
    o1 = orch.step(orch.last_window)
    assert o1["step_index"] == 1
    assert len(orch.feedback_log) == 2


def test_reflex_winner_spinal():
    model, _ = load_predictor(CKPT)
    # 把一个障碍放在末帧位置附近 => 脊髓碰撞制动
    w = torch.zeros(1, 6, 6)
    w[:, -1, :3] = torch.tensor([0.0, 0.0, 0.0])
    orch = ClosedLoopOrchestrator(model, wm_horizon=1,
                                  collision_obstacles=[[0.0, 0.0, 0.0]],
                                  collision_radius=1.0)
    out = orch.step(w)
    assert out["reflex_triggered"] is True
    assert out["priority_winner"] == "spinal"


def test_deterministic_and_zero_grad():
    model, _ = load_predictor(CKPT)
    orch = ClosedLoopOrchestrator(model, wm_horizon=2)
    w = torch.randn(1, 6, 6)
    before = _md5(model)
    o1 = orch.step(w)
    o2 = orch.step(orch.last_window)
    after = _md5(model)
    assert before == after            # 零梯度
    # 第一步确定性 (同输入同输出)
    orch2 = ClosedLoopOrchestrator(model, wm_horizon=2)
    o1b = orch2.step(w)
    assert torch.allclose(o1["command"], o1b["command"])
    assert o1["wm_feedback"]["feedback_l1_to_next"] == \
        o1b["wm_feedback"]["feedback_l1_to_next"]


def test_bad_window_guard():
    model, _ = load_predictor(CKPT)
    orch = ClosedLoopOrchestrator(model)
    try:
        orch.step(torch.randn(2, 6, 6))     # batch=2
        assert False
    except ValueError:
        pass
    try:
        orch.step(torch.randn(1, 6, 6) * float("nan"))
        assert False
    except ValueError:
        pass


def test_wm_horizon_guard():
    model, _ = load_predictor(CKPT)
    try:
        ClosedLoopOrchestrator(model, wm_horizon=0)
        assert False
    except ValueError:
        pass


def test_reset_clears():
    model, _ = load_predictor(CKPT)
    orch = ClosedLoopOrchestrator(model)
    orch.step(torch.randn(1, 6, 6))
    orch.reset()
    assert orch.step_index == 0
    assert orch.last_window is None
    assert orch.feedback_log == []
