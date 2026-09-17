"""v3.6.0.dev2 接触/碰撞事件预测测试: ContactPredictor 几何代理。

覆盖:
    * 接近参考物 => 接触检测命中 + 事件序列标注;
    * 远离场景 => 无接触 (零事件);
    * 速度反转 => impact 代理步标注;
    * 与 udos.collision.CollisionDetector 球-球判定一致;
    * 空轨迹 / 非法半径 / 非有限输入守卫;
    * 纯解析: 不依赖主 predictor 权重。
"""
import numpy as np
import pytest
import torch

from udos import __version__
from udos.wm_events import ContactPredictor
from udos.collision import CollisionDetector
from udos.spatial import SpatialObject


def test_version():
    assert __version__ == "5.5.5"


def _state(pos, vel):
    return torch.tensor([pos[0], pos[1], pos[2],
                         vel[0], vel[1], vel[2]], dtype=torch.float32)


def test_contact_detected_when_approaching():
    # 主体从 x=3 匀速向原点移动, dt=0.5: 3,2.5,2,1.5,1,0.5... 半径和=1.0
    traj = torch.stack([_state([3.0 - 0.5 * t, 0, 0], [-1.0, 0, 0])
                        for t in range(8)]).unsqueeze(0)
    out = ContactPredictor().predict(traj, agent_radius=0.5,
                                     partner_radius=0.5, dt=0.5)
    r = out["batch"][0]
    # 距离 <=1.0 即接触: x 距离 |3-0.5t| <=1 => t>=4 (x=1.0)
    assert any(r["contact_flags"])
    # 事件步: 第一次进入接触的步
    assert r["contact_steps"][0] == 4
    assert r["n_events"] == 1
    # contact_prob 在接触步为 1
    assert r["contact_prob"][4] == 1.0


def test_no_contact_when_receding():
    # 主体从 x=2.0 (距离 2.0 > 半径和 1.0) 继续远离到 x=5.5, 全程不接触参考原点
    traj = torch.stack([_state([2.0 + 0.5 * t, 0, 0], [1.0, 0, 0])
                        for t in range(8)]).unsqueeze(0)
    out = ContactPredictor().predict(traj, agent_radius=0.5,
                                     partner_radius=0.5, dt=0.5)
    r = out["batch"][0]
    assert not any(r["contact_flags"])
    assert r["n_events"] == 0
    assert ContactPredictor().has_contact(traj, agent_radius=0.5,
                                          partner_radius=0.5) is False


def test_impact_proxy_on_velocity_reversal():
    # 主体 x 方向速度从 +1 反转为 -1 (碰撞反弹代理)
    states = [[1.0, 0, 0, 1.0, 0, 0], [1.5, 0, 0, -1.0, 0, 0],
              [1.0, 0, 0, -1.0, 0, 0]]
    traj = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    out = ContactPredictor().predict(traj, partner_position=[5, 0, 0])
    r = out["batch"][0]
    # 第 1 步速度符号反转
    assert 1 in r["impact_steps"]


def test_consistent_with_collision_module():
    """球-球接触判定与 udos.collision.CollisionDetector 一致。"""
    cd = CollisionDetector()
    cp = ContactPredictor()
    # 取一帧预测状态, 主体位置 x=0.8, 参考在原点, 半径各 0.5 => 距离 0.8 <=1 接触
    frame = torch.tensor([[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]])
    agent = SpatialObject("agent", [0.8, 0.0, 0.0], radius=0.5)
    partner = SpatialObject("partner", [0.0, 0.0, 0.0], radius=0.5)
    cc = cd.ball_ball(agent, partner)
    cp_out = cp.predict(frame, agent_radius=0.5, partner_radius=0.5)
    assert cp_out["batch"][0]["contact_flags"][0] == cc["contact"]
    # 距离数值一致
    assert abs(cp_out["batch"][0]["distances"][0] - cc["distance"]) < 1e-5


def test_empty_trajectory_guard():
    with pytest.raises(ValueError):
        ContactPredictor().predict(torch.zeros(1, 0, 6))


def test_bad_radius_guard():
    traj = torch.zeros(1, 4, 6)
    with pytest.raises(ValueError):
        ContactPredictor().predict(traj, agent_radius=0.0)
    with pytest.raises(ValueError):
        ContactPredictor().predict(traj, partner_radius=-1.0)


def test_non_finite_guard():
    bad = torch.tensor([[[float("nan"), 0, 0, 0, 0, 0]]])
    with pytest.raises(ValueError):
        ContactPredictor().predict(bad)


def test_batch_shape():
    traj = torch.zeros(3, 6, 6)
    out = ContactPredictor().predict(traj, partner_position=[5, 0, 0])
    assert out["n_batch"] == 3
    assert out["horizon"] == 6
    assert len(out["batch"]) == 3
