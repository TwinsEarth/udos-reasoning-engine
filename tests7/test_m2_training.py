"""M2 探针：模型在轨迹级三分数据上真的把损失学下来（非恒等夹具）。

反例守卫：若残差头不训练/梯度断裂/场景通道未接线，oracle 路径无法显著低于
"状态不变"恒等基线。
"""
import torch

from udos7 import WorldModelCore
from udos7.dynamics import three_way_splits
from udos7.train import TrainConfig, fit
from udos7.metrics import mse


def test_training_beats_identity_baseline():
    torch.set_num_threads(2)
    splits = three_way_splits(n_traj_per_kind=12)
    torch.manual_seed(0)
    model = WorldModelCore(window=6, hidden=64, n_layers=1)
    H = splits["train"].horizon

    # 恒等基线（预测状态不变）
    with torch.no_grad():
        last = splits["val"].X[:, -1:, :].expand(-1, H, -1)
        identity_mse = mse(last, splits["val"].Y)

    cfg = TrainConfig(epochs=25, batch=64, lr=3e-3, patience=25,
                      scene_dropout=0.5, seed=0)
    hist = fit(model, splits["train"], splits["val"], cfg)

    model.eval()
    with torch.no_grad():
        oracle = mse(model.rollout(splits["val"].X, H, explicit=splits["val"].P),
                     splits["val"].Y)
        blind = mse(model.rollout(splits["val"].X, H, explicit=None),
                    splits["val"].Y)

    assert hist.train_loss[-1] < hist.train_loss[0], "训练损失应下降"
    # oracle 路径必须显著优于恒等基线（场景通道真的被用上）
    assert oracle < 0.6 * identity_mse, (
        f"oracle {oracle:.4f} 未显著优于恒等基线 {identity_mse:.4f}")
    # 盲路径至少不差于恒等（学到可观测量；不可观测量误差诚实保留）
    assert blind <= identity_mse * 1.05, (
        f"blind {blind:.4f} 异常劣于恒等 {identity_mse:.4f}")
