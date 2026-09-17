"""
v3.0.0.dev3 CrossModalAlignmentLoss 单元测试
==============================================
锚点纪律:
    * 一致性损失有限、非负; 在合成数据里用已知投影关系验证:
      构造相关的 rgb/depth/mask -> 损失低; 打乱 -> 损失高;
    * 少量训练步最小化该损失, 损失确有下降;
    * NaN 防护 (常数/零方差输入仍有限);
    * opt-in: 推理 (FutureMultimodalHead.forward) 不计算该损失。
"""
import torch

from udos import __version__
from udos.future_multimodal import (CrossModalAlignmentLoss,
                                    FutureMultimodalHead)


def test_version():
    assert __version__ == "5.5.5"


def test_loss_finite_nonneg():
    torch.manual_seed(0)
    loss_fn = CrossModalAlignmentLoss()
    rgb = torch.randn(4, 4, 8)
    depth = torch.randn(4, 4, 4)
    mask = torch.sigmoid(torch.randn(4, 4, 4))
    l = loss_fn(rgb, depth, mask)
    assert torch.isfinite(l)
    assert l.item() >= 0.0


def test_consistent_lower_than_shuffled():
    """已知投影关系: 共享信号 s 同时驱动 rgb/depth/mask -> 相关 -> 损失低;
    打乱 depth 时间步 -> 不相关 -> 损失高。"""
    torch.manual_seed(0)
    B, H = 8, 6
    s = torch.randn(B, H)
    # rgb = s 的多通道复制 + 小噪声; depth = 2s 复制; mask = sigmoid(s)
    rgb = s.unsqueeze(-1).repeat(1, 1, 8) + 0.01 * torch.randn(B, H, 8)
    depth = 2.0 * s.unsqueeze(-1).repeat(1, 1, 4) + 0.01 * torch.randn(B, H, 4)
    mask = torch.sigmoid(s.unsqueeze(-1).repeat(1, 1, 4))
    loss_fn = CrossModalAlignmentLoss()
    l_consistent = loss_fn(rgb, depth, mask)
    # 打乱 depth 时间步
    perm = torch.randperm(H)
    depth_shuf = depth[:, perm, :]
    l_shuffled = loss_fn(rgb, depth_shuf, mask)
    assert l_consistent < l_shuffled


def test_loss_decreases_with_training():
    """对一个小投影头做几步优化, 一致性损失确有下降。"""
    torch.manual_seed(0)
    B, H, lat = 8, 6, 16
    z = torch.randn(B, lat)
    # 初始随机 rgb/depth/mask 投影
    rgb_lin = torch.nn.Linear(lat, H * 8)
    dep_lin = torch.nn.Linear(lat, H * 4)
    msk_lin = torch.nn.Linear(lat, H * 4)
    opt = torch.optim.SGD(list(rgb_lin.parameters()) +
                          list(dep_lin.parameters()) +
                          list(msk_lin.parameters()), lr=0.2)
    loss_fn = CrossModalAlignmentLoss()
    first = None
    last = None
    for step in range(30):
        opt.zero_grad()
        rgb = rgb_lin(z).view(B, H, 8)
        depth = dep_lin(z).view(B, H, 4)
        mask = torch.sigmoid(msk_lin(z).view(B, H, 4))
        l = loss_fn(rgb, depth, mask)
        l.backward()
        opt.step()
        if step == 0:
            first = l.item()
        last = l.item()
    assert last < first, f"损失未下降: {first} -> {last}"
    assert last >= 0.0


def test_nan_protection():
    """常数/零方差输入不产生 NaN。"""
    loss_fn = CrossModalAlignmentLoss()
    rgb = torch.ones(4, 4, 8)
    depth = torch.ones(4, 4, 4)
    mask = torch.full((4, 4, 4), 0.5)
    l = loss_fn(rgb, depth, mask)
    assert torch.isfinite(l)
    assert not torch.isnan(l)


def test_inference_does_not_compute_loss():
    """FutureMultimodalHead.forward (推理) 不计算/返回对齐损失。"""
    torch.manual_seed(0)
    head = FutureMultimodalHead(16, horizon=4)
    z = torch.randn(3, 16)
    out = head(z)
    assert set(out.keys()) == {"rgb", "depth", "mask"}
    assert "alignment_loss" not in out
