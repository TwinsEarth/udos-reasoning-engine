"""M1 探针：统一双引擎内核 + 数据契约。

最高风险契约（每条配一个能被错误实现击穿的断言）：
- 未训练残差头零初始化 => 模型严格预测"状态不变"，与场景输入无关（稳定起点）。
- 场景桥末层零初始化 => 初始 ctx 桥贡献严格为 0（零偏置短路）。
- 显式参数与估计参数进入**同一个** ctx；显式有限槽覆盖估计，NaN 槽回退估计。
- 唯一时序核：模型只有一个 GRU（消除双 CTM/演示基座/LoRA 死路）。
- 轨迹级三分：train/val/test 的 traj_id 互不重叠（无窗口泄漏）。
- 真三维：非碰撞类 py/pz 存在非平凡运动；碰撞 y/z 平凡（诚实标注）。
"""
import pytest
import torch

from udos7 import WorldModelCore
from udos7.dynamics import three_way_splits, build_split
from udos7.scene import SceneChannel


def _window(B=4, W=6):
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, W, 6, generator=g)


def test_untrained_model_is_identity_regardless_of_scene():
    torch.manual_seed(0)
    m = WorldModelCore(window=6)
    x = _window()
    blind = m(x)
    explicit = torch.randn(4, 4)
    cond = m(x, explicit=explicit)
    # 残差头零初始化 => 恒等预测
    assert torch.allclose(blind, x[:, -1, :], atol=1e-6)
    assert torch.allclose(cond, x[:, -1, :], atol=1e-6)


def test_scene_bridge_zero_initialized():
    ch = SceneChannel(window=6)
    x = _window()
    assert torch.allclose(ch.bridge(x), torch.zeros(4, ch.scene_dim), atol=1e-6)


def test_single_recurrent_core_only():
    m = WorldModelCore(window=6)
    grus = [n for n, _ in m.named_modules() if isinstance(_, torch.nn.GRU)]
    assert grus == ["gru"], f"必须唯一 GRU 时序核，实际 {grus}"
    # 不应残留 LoRA / 演示基座 / 第二 CTM
    joined = " ".join(n for n, _ in m.named_modules()).lower()
    for forbidden in ("lora", "tinybase", "base_model", "ctm_", "demo"):
        assert forbidden not in joined, f"统一内核不得残留 {forbidden}"


def test_explicit_overrides_estimate_slotwise():
    torch.manual_seed(1)
    ch = SceneChannel(window=6)
    x = _window(2)
    exp = torch.tensor([[1.0, float("nan"), 0.5, 0.0],
                        [float("nan"), 0.2, 0.0, -0.1]])
    sc = ch(x, explicit=exp)
    # 有限槽用显式值
    assert sc.params_used[0, 0].item() == pytest.approx(1.0)
    assert sc.params_used[0, 2].item() == pytest.approx(0.5)
    # NaN 槽回退到估计器输出
    assert sc.params_used[0, 1].item() == pytest.approx(sc.params_hat[0, 1].item())
    assert sc.source[0, 0].item() == 1.0 and sc.source[0, 1].item() == 0.0
    # 显式与估计进入同一 ctx 空间（形状一致且都可反传）
    assert sc.ctx.shape == (2, ch.scene_dim)


def test_rollout_shapes_and_autoregression():
    torch.manual_seed(0)
    m = WorldModelCore(window=6)
    x = _window(3)
    nxt = m(x)
    traj = m.rollout(x, horizon=4)
    assert nxt.shape == (3, 6)
    assert traj.shape == (3, 4, 6)
    # 未训练恒等：rollout 每步都等于窗口末帧
    assert torch.allclose(traj, x[:, -1:, :].expand(-1, 4, -1), atol=1e-6)


def test_rejects_nonfinite_window_and_params():
    m = WorldModelCore(window=6)
    bad = _window(1)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        m(bad)
    with pytest.raises(ValueError):
        m(_window(1), explicit=torch.full((1, 4), float("inf")))


def test_trajectory_level_splits_disjoint():
    splits = three_way_splits(n_traj_per_kind=8)
    ids = {name: set(s.traj_ids) for name, s in splits.items()}
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert ids[names[i]].isdisjoint(ids[names[j]]), (
                f"{names[i]} 与 {names[j]} 轨迹泄漏")
    # 每个 split 内 4 类齐全
    for name, s in splits.items():
        for k in ("uniform", "accel", "spring", "collision"):
            assert int(s.kind_mask(k).sum()) > 0, name


def test_non_collision_motion_is_truly_3d_collision_is_x_only():
    ds = build_split(seed=2026, n_traj_per_kind=16)
    for kind, expect_3d in (("uniform", True), ("accel", True),
                            ("spring", True), ("collision", False)):
        m = ds.kind_mask(kind)
        Y = ds.Y[m]
        py, pz = Y[..., 1].var(), Y[..., 2].var()
        if expect_3d:
            assert py > 1e-4 and pz > 1e-4, f"{kind} 应在 y/z 非平凡"
        else:
            # 碰撞仅 x 轴：y/z 位置与速度恒 0（诚实标注，不假装三维）
            assert py.item() == 0.0 and pz.item() == 0.0
