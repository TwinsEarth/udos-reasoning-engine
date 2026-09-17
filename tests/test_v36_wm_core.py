"""v3.6.0 潜在空间前向世界模型核心测试 (PWM 阶段三·物理)。

覆盖:
    * 潜在状态 z_t 提取 (predictor 中间激活 -> [B, latent_dim]);
    * 外挂前向转移 z_{t+1}=f(z_t, action) 形状;
    * imagine 多步 rollout 形状;
    * H=1 与 predictor.rollout 逐位等价锚点;
    * 零梯度锚点: imagine/fit 后主 predictor state_dict md5 不变 (52191 参数);
    * 空输入 / 非法 horizon 显式 ValueError 守卫;
    * 外挂世界模型参数量显式 (不入主 state_dict)。
analogy, not reproduction —— 合成低维潜在代理, 非视频世界模型复现。
"""
import hashlib
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor
from udos.world_model import LatentWorldModel, LatentTransitionMLP
from udos.dynamics import build_parametric_dataset

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "predictor_v3.5.0.pt"


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def predictor():
    m, _ = load_predictor(str(CKPT))
    m.eval()
    return m


@pytest.fixture(scope="module")
def wm(predictor):
    return LatentWorldModel(predictor)


@pytest.fixture(scope="module")
def batch(predictor):
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=11)
    return ds.X[:4], ds.P[:4]


def _md5(model):
    h = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_main_param_count_unchanged(predictor):
    """主 predictor 52191 参数不被世界模型污染。"""
    assert sum(p.numel() for p in predictor.parameters()) == 52191


def test_latent_extraction_shape(predictor, wm, batch):
    x, sp = batch
    z = wm.encode_latent(x, scene_params=sp)
    assert z.shape == (x.size(0), wm.latent_dim)
    assert z.dtype == torch.float32
    assert bool(torch.isfinite(z).all())
    # 潜在维度与 obs_encoder 输出宽度一致 (= ctm.cfg.d_input)
    assert wm.latent_dim == predictor.ctm.cfg.d_input == 32


def test_transition_shape_and_identity_default(wm, batch):
    x, sp = batch
    z = wm.encode_latent(x, scene_params=sp)
    z2 = wm.transit_step(z)
    assert z2.shape == z.shape
    # 未拟合时末层零初始化 => 残差恒等: transit(z) == z
    assert torch.equal(z2, z)


def test_decoder_shape(wm, batch):
    x, sp = batch
    z = wm.encode_latent(x, scene_params=sp)
    s = wm.decode_latent(z)
    assert s.shape == (x.size(0), wm.raw_dim)


def test_imagine_shape(wm, batch):
    x, sp = batch
    out = wm.imagine(x, 4, scene_params=sp)
    assert out.shape == (x.size(0), 4, wm.raw_dim)
    assert bool(torch.isfinite(out).all())


def test_h1_bitwise_anchor(predictor, wm, batch):
    """H=1 imagine 与 predictor.rollout 逐位等价 (锚点)。"""
    x, sp = batch
    ref = predictor.rollout(x, 1, scene_params=sp)
    im = wm.imagine(x, 1, scene_params=sp)
    assert torch.equal(ref, im)


def test_zero_gradient_md5_anchor(predictor, wm, batch):
    """imagine 后 predictor state_dict md5 不变 (零梯度/只读)。"""
    x, sp = batch
    before = _md5(predictor)
    _ = wm.imagine(x, 4, scene_params=sp)
    assert _md5(predictor) == before
    assert sum(p.numel() for p in predictor.parameters()) == 52191


def test_fit_does_not_touch_predictor(predictor, wm, batch):
    """fit 后 predictor 权重 md5 不变 (优化器只含外挂世界模型参数)。"""
    x, sp = batch
    before = _md5(predictor)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=23)
    rep = wm.fit(ds, epochs=3)
    assert _md5(predictor) == before
    assert wm.fitted is True
    assert rep["predictor_grad_zero"] is True
    # 拟合 loss 下降
    assert rep["fit_last_loss"] < rep["fit_first_loss"]


def test_wm_params_reported_and_separate(predictor, wm):
    """外挂世界模型参数量显式, 且其参数不构成 predictor 子模块 (独立对象)。"""
    desc = wm.describe()
    assert desc["wm_params"] > 0
    assert desc["wm_params"] == wm.n_params
    # 世界模型持有独立 nn.Module (transit/decode), 未挂到 predictor 上
    assert isinstance(wm.transit, torch.nn.Module)
    assert isinstance(wm.decode, torch.nn.Module)
    # predictor 没有名为 transit/decode 的子模块
    assert not hasattr(predictor, "transit")
    assert not hasattr(predictor, "decode")
    # 主 state_dict 键全属主架构, 不含任何 transit./decode. 前缀
    main_keys = set(predictor.state_dict().keys())
    assert not any(k.startswith("transit.") for k in main_keys)
    assert not any(k.startswith("decode.") for k in main_keys)


def test_empty_window_guard(wm):
    with pytest.raises(ValueError):
        wm.imagine(torch.zeros(1, 0, 6), 2)
    with pytest.raises(ValueError):
        wm.encode_latent(torch.zeros(1, 0, 6))


def test_bad_horizon_guard(wm, batch):
    x, sp = batch
    for bad in (0, -1, 1.5, "3"):
        with pytest.raises(ValueError):
            wm.imagine(x, bad, scene_params=sp)


def test_transition_action_dim_consistency():
    """配置 action_dim>0 时未给 action 显式报错 (不静默退化)。"""
    tr = LatentTransitionMLP(latent_dim=8, action_dim=4, hidden=8)
    z = torch.zeros(2, 8)
    with pytest.raises(ValueError):
        tr(z, None)
    out = tr(z, torch.zeros(2, 4))
    assert out.shape == (2, 8)
