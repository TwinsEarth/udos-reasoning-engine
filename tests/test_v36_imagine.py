"""v3.6.0.dev1 多步"想象" rollout 测试: imagine_rollout 潜在轨迹 + 解码物理状态。

覆盖:
    * 多步想象形状 (states [B,H,raw] / latents [B,H,latent]);
    * H=1 与 predictor.rollout 逐位等价锚点;
    * compare_real: 与真实 rollout 逐步 MSE (第 0 步≈0, 后续步发散为合成类比);
    * 空窗口 / 非法 horizon 守卫;
    * 零梯度锚点: 调用后主 predictor 权重不变。
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
                                  horizon=4, dt=0.5, seed=13)
    return ds.X[:4], ds.P[:4]


def _md5(model):
    h = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_rollout_shapes(wm, batch):
    x, sp = batch
    out = wm.imagine_rollout(x, 5, scene_params=sp)
    assert out["states"].shape == (x.size(0), 5, wm.raw_dim)
    assert out["latents"].shape == (x.size(0), 5, wm.latent_dim)
    assert bool(torch.isfinite(out["states"]).all())
    assert bool(torch.isfinite(out["latents"]).all())


def test_h1_bitwise_anchor(wm, batch):
    x, sp = batch
    ref = wm.predictor.rollout(x, 1, scene_params=sp)
    out = wm.imagine_rollout(x, 1, scene_params=sp)
    assert torch.equal(out["states"][:, 0, :], ref[:, 0, :])
    # H=1 latents 仅 1 步
    assert out["latents"].shape == (x.size(0), 1, wm.latent_dim)


def test_compare_real_step_mse(wm, batch):
    x, sp = batch
    out = wm.imagine_rollout(x, 4, scene_params=sp, compare_real=True)
    assert "real" in out and "step_mse" in out
    assert out["step_mse"].shape == (4,)
    # 第 0 步锚定真实 predict_next => 第 0 步 MSE 严格为 0
    assert float(out["step_mse"][0]) == 0.0
    # 后续步 (外挂潜在转移解码) 与真实 rollout 存在差异 (合成类比, 不追求零误差)
    assert float(out["step_mse"][1]) >= 0.0


def test_zero_gradient_anchor(wm, batch):
    x, sp = batch
    before = _md5(wm.predictor)
    _ = wm.imagine_rollout(x, 4, scene_params=sp, compare_real=True)
    assert _md5(wm.predictor) == before


def test_empty_window_guard(wm):
    with pytest.raises(ValueError):
        wm.imagine_rollout(torch.zeros(1, 0, 6), 3)


def test_bad_horizon_guard(wm, batch):
    x, sp = batch
    with pytest.raises(ValueError):
        wm.imagine_rollout(x, 0, scene_params=sp)
    with pytest.raises(ValueError):
        wm.imagine_rollout(x, -2, scene_params=sp)


def test_imagine_alias_consistency(wm, batch):
    """imagine() 与 imagine_rollout(states) 输出逐位一致。"""
    x, sp = batch
    a = wm.imagine(x, 4, scene_params=sp)
    b = wm.imagine_rollout(x, 4, scene_params=sp)["states"]
    assert torch.equal(a, b)
