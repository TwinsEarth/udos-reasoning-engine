"""v3.6.0.dev5 世界模型不确定性 / 置信度 / 回退测试。

覆盖:
    * 集成式不确定性: n_samples 次带噪想象的方差 (第 0 步锚定 std≈0, 后续步>0);
    * 置信度 ∈[0,1];
    * 高不确定回退: 超阈值步用主 predictor rollout 替换, used_fallback 记录;
    * 确定性 (同 seed 两次调用逐位一致);
    * 空窗口 / 非法 horizon / 非法 n_samples 守卫;
    * 零梯度: 调用后主 predictor 权重不变。
"""
import hashlib
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.world_model import LatentWorldModel
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.5.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def wm():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return LatentWorldModel(m)


@pytest.fixture(scope="module")
def batch():
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=17)
    return ds.X[:4], ds.P[:4]


def _md5(model):
    h = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_uncertainty_shapes(wm, batch):
    x, sp = batch
    out = wm.imagine_uncertain(x, 6, n_samples=5, noise_scale=0.1,
                                scene_params=sp)
    B, H = x.size(0), 6
    assert out["mean"].shape == (B, H, wm.raw_dim)
    assert out["std"].shape == (B, H, wm.raw_dim)
    assert out["uncertainty"].shape == (B, H)
    assert out["confidence"].shape == (B, H)


def test_step0_zero_variance_anchor(wm, batch):
    """第 0 步锚定真实 predict_next (无噪), 故集成方差≈0。"""
    x, sp = batch
    out = wm.imagine_uncertain(x, 4, n_samples=6, noise_scale=0.1,
                                scene_params=sp)
    assert float(out["uncertainty"][:, 0].max()) < 1e-6
    # 后续步随噪声放大, 不确定性应 > 0
    assert float(out["uncertainty"][:, -1].mean()) > 0.0


def test_confidence_range(wm, batch):
    x, sp = batch
    out = wm.imagine_uncertain(x, 4, n_samples=5, noise_scale=0.1,
                                scene_params=sp)
    assert bool((out["confidence"] >= 0).all()
                and (out["confidence"] <= 1.0).all())


def test_fallback_to_predictor(wm, batch):
    """高不确定步回退主 rollout: 回退步 states 等于真实 rollout。"""
    x, sp = batch
    out = wm.imagine_uncertain(x, 4, n_samples=6, noise_scale=0.5,
                                fallback_threshold=0.0, scene_params=sp)
    # threshold=0: 第 0 步锚定(方差恰为 0)不触发; 第 1..3 步有噪方差>0 全部回退
    assert set(out["used_fallback"]) == {1, 2, 3}
    real = wm.predictor.rollout(x, 4, scene_params=sp)
    # 回退步 (1..3) 直接取真实 rollout => 逐位相等
    assert torch.equal(out["states"][:, 1:, :], real[:, 1:, :])
    # 第 0 步本就是真实锚点 (mean 仅 K 次相同值平均, 数值级一致)
    assert torch.allclose(out["states"][:, 0, :], real[:, 0, :], atol=1e-6)


def test_no_fallback_when_none(wm, batch):
    x, sp = batch
    out = wm.imagine_uncertain(x, 4, n_samples=4, noise_scale=0.05,
                                fallback_threshold=None, scene_params=sp)
    assert out["used_fallback"] == []
    assert torch.equal(out["states"], out["mean"])


def test_deterministic(wm, batch):
    x, sp = batch
    a = wm.imagine_uncertain(x, 4, n_samples=5, noise_scale=0.1, scene_params=sp)
    b = wm.imagine_uncertain(x, 4, n_samples=5, noise_scale=0.1, scene_params=sp)
    assert torch.equal(a["mean"], b["mean"])
    assert torch.equal(a["std"], b["std"])


def test_zero_gradient(wm, batch):
    x, sp = batch
    before = _md5(wm.predictor)
    wm.imagine_uncertain(x, 4, n_samples=3, noise_scale=0.1,
                          fallback_threshold=0.5, scene_params=sp)
    assert _md5(wm.predictor) == before


def test_guards(wm, batch):
    x, sp = batch
    with pytest.raises(ValueError):
        wm.imagine_uncertain(x, 0, scene_params=sp)
    with pytest.raises(ValueError):
        wm.imagine_uncertain(x, 3, n_samples=0, scene_params=sp)
    with pytest.raises(ValueError):
        wm.imagine_uncertain(x, 3, noise_scale=-1.0, scene_params=sp)
    with pytest.raises(ValueError):
        wm.imagine_uncertain(torch.zeros(1, 0, 6), 3)
