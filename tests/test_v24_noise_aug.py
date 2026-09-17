"""v2.4.3 回归: 数据增强/噪声鲁棒 (sigma=0 等价旧版 / sigma>0 展宽 / 确定性)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.dynamics import noise_augment, build_parametric_dataset  # noqa: E402
from udos.training import TrainConfig  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_sigma_zero_returns_same_tensor():
    x = torch.randn(4, 6, 6)
    out = noise_augment(x, 0.0)
    assert out is x                       # 零 sigma 原样返回, 不拷贝
    out_neg = noise_augment(x, -1.0)
    assert out_neg is x


def test_sigma_positive_widens_distribution():
    torch.manual_seed(0)
    x = torch.zeros(10000, 3)
    y = noise_augment(x, 0.1)
    assert not torch.allclose(y, x)
    # 零均值输入 => 输出近似 N(0, sigma^2)
    assert abs(float(y.std()) - 0.1) < 0.01
    # 展宽: 对有输入也增加方差
    x2 = torch.randn(1000, 4) * 0.5
    y2 = noise_augment(x2, 0.3)
    assert float(y2.var()) > float(x2.var())


def test_deterministic_with_generator():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(5, 6, 6)
    a = noise_augment(x, 0.2, generator=torch.Generator().manual_seed(7))
    b = noise_augment(x, 0.2, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    c = noise_augment(x, 0.2, generator=torch.Generator().manual_seed(8))
    assert not torch.equal(a, c)


def test_trainconfig_default_zero():
    assert TrainConfig().noise_sigma == 0.0


def test_training_with_noise_runs_and_loss_decreases():
    from udos.ctm_engine import CTMConfig
    from udos.training import PhysicsPredictor, CTMTrainer
    torch.manual_seed(0)
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=11)
    tr, te = ds.split(0.8)
    model = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)
    hist = CTMTrainer(model, TrainConfig(
        epochs=5, batch_size=32, noise_sigma=0.1)).train(tr, te)
    assert len(hist.train_loss) == 5
    assert hist.train_loss[-1] < hist.train_loss[0]
