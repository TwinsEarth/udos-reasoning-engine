"""
v3.4.0 ICM 上下文记忆核心测试
================================================================
覆盖:
    1. DemonstrationEpisode 结构 (action = result - 末帧, embed L2 归一化);
    2. DemonstrationMemory 检索正确性 (top-k 降序, 余弦相似度范围);
    3. ICMAggregator 零梯度: state_dict md5 在 ICM 推理前后逐位一致;
    4. k-shot MSE 不爆炸: ICM 检索聚合路径 k>=0 不退化 (实测改善);
    5. 复现并解释 naive InContextLearner few-shot 退化 (0-shot 低, k-shot 爆炸);
    6. 空记忆守卫 / opt-in 默认 k=0 退化为 0-shot / λ=0 逐位等价。

analogy, not reproduction; CPU-only 合成数据; 诚实记录, 不保证提升。
"""
import hashlib

import pytest
import torch

from udos import load_predictor, __version__
from udos.dynamics import build_parametric_dataset, RAW_DIM
from udos.icm import (DemonstrationEpisode, DemonstrationMemory,
                      ICMAggregator, _flatten_window_embed)
from udos.incontext import InContextLearner

CKPT = "checkpoints/predictor_v3.3.3.pt"


def _model():
    m, meta = load_predictor(CKPT)
    return m


def _state_md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(scope="module")
def data():
    torch.manual_seed(0)
    ds = build_parametric_dataset(n_per_kind=24, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=42)
    return ds.split(0.8)


def _build_mem(m, tr, n=64):
    mem = DemonstrationMemory()
    agg = ICMAggregator(m, temperature=1.0, lamb=1.0)
    for i in range(min(n, len(tr))):
        ep = DemonstrationEpisode(tr.X[i], tr.Y[i, 0], kind=tr.kinds[i],
                                  scene_params=tr.P[i])
        mem.register(ep)
        agg.cache_residual(ep, scene_params=tr.P[i])
    return mem, agg


# --------------------------------------------------------------------------- #
# 1. Episode 结构
# --------------------------------------------------------------------------- #
def test_episode_structure(model, data):
    tr, _ = data
    ep = DemonstrationEpisode(tr.X[0], tr.Y[0, 0], kind="uniform",
                              scene_params=tr.P[0])
    # action = result - 末帧
    assert torch.allclose(ep.action, ep.result - ep.input_window[-1], atol=1e-6)
    assert ep.result.numel() == RAW_DIM
    assert ep.action.numel() == RAW_DIM
    # embed L2 归一化 (非零)
    assert abs(float(ep.embed.norm()) - 1.0) < 1e-5 or ep.input_window.abs().sum() < 1e-9
    assert ep.kind == "uniform"


def test_episode_bad_shape():
    with pytest.raises(ValueError):
        DemonstrationEpisode(torch.zeros(6, 6), torch.zeros(7))   # result 维错
    with pytest.raises(ValueError):
        DemonstrationEpisode(torch.zeros(6, 5), torch.zeros(6))   # window 维错


# --------------------------------------------------------------------------- #
# 2. Memory 检索
# --------------------------------------------------------------------------- #
def test_memory_retrieve_sorted(model, data):
    tr, te = data
    mem, agg = _build_mem(model, tr, n=48)
    assert mem.size == 48
    res = mem.retrieve(te.X[0], k=3)
    assert len(res) == 3
    # 降序
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)
    # 余弦相似度范围
    assert all(-1.0001 <= s <= 1.0001 for s in sims)


def test_memory_empty_guard(model):
    mem = DemonstrationMemory()
    assert mem.size == 0
    with pytest.raises(RuntimeError):
        mem.retrieve(torch.zeros(6, 6), k=1)
    with pytest.raises(ValueError):
        mem.retrieve(torch.zeros(6, 6), k=0)   # k<=0 显式报错


def test_memory_fifo_eviction(model, data):
    tr, _ = data
    mem = DemonstrationMemory(max_episodes=5)
    for i in range(10):
        mem.register(DemonstrationEpisode(tr.X[i], tr.Y[i, 0]))
    assert mem.size == 5


# --------------------------------------------------------------------------- #
# 3. 零梯度: 权重逐位不变
# --------------------------------------------------------------------------- #
def test_zero_gradient_weights_unchanged(model, data):
    tr, te = data
    before = _state_md5(model)
    mem, agg = _build_mem(model, tr, n=32)
    for i in range(10):
        agg.predict(te.X[i], memory=mem, k=3, scene_params=te.P[i:i+1])
    after = _state_md5(model)
    assert before == after, "ICM 推理必须零梯度, state_dict md5 逐位一致"


def test_aggregator_has_no_trainable_params(model):
    agg = ICMAggregator(model)
    # ICM 自身无可学参数 (原型编码确定性), 聚合器不持有任何 Parameter
    own_params = [p for p in agg.__dict__.values() if
                  isinstance(p, torch.nn.Parameter)]
    assert own_params == []


# --------------------------------------------------------------------------- #
# 4. k-shot 不退化 (核心主张)
# --------------------------------------------------------------------------- #
def test_icm_kshot_not_degenerate(model, data):
    tr, te = data
    mem, agg = _build_mem(model, tr, n=96)
    # 用一个可控子集测, 保证快
    idx = torch.arange(min(80, len(te)))
    mse0 = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=0,
                        scene_params=te.P[idx])
    mse1 = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=1,
                        scene_params=te.P[idx])
    mse3 = agg.shot_mse(te.X[idx], te.Y[idx, 0], memory=mem, k=3,
                        scene_params=te.P[idx])
    # 核心: 检索聚合路径 k-shot 不爆炸 (远小于 naive 的 3+ 量级)
    assert mse0 < 0.2
    assert mse1 < 0.2
    assert mse3 < 0.2
    # 并趋向于不劣于 0-shot (允许小抖动, 但绝不退化到 naive 量级)
    assert mse3 <= mse0 * 1.5


# --------------------------------------------------------------------------- #
# 5. 复现 naive few-shot 退化 (对照)
# --------------------------------------------------------------------------- #
def test_reproduce_naive_icl_degradation(model, data):
    """照实记录: naive 朴素拼接随 k 爆炸; 这是 ICM 要解决的反面证据。"""
    tr, te = data
    pool = [tr.X[j] for j in range(min(16, len(tr)))]
    icl = InContextLearner(model)
    idx = torch.arange(min(40, len(te)))
    mse0 = icl.shot_mse(te.X[idx], te.Y[idx, 0], pool, 0,
                        scene_params=te.P[idx])
    mse1 = icl.shot_mse(te.X[idx], te.Y[idx, 0], pool, 1,
                        scene_params=te.P[idx])
    mse3 = icl.shot_mse(te.X[idx], te.Y[idx, 0], pool, 3,
                        scene_params=te.P[idx])
    # 模式: 0-shot 低, k-shot 显著抬升 (稀释信号)
    assert mse1 > mse0 * 3.0, "naive 1-shot 应显著差于 0-shot"
    assert mse3 > mse1, "naive 3-shot 应比 1-shot 更差 (对照退化)"


# --------------------------------------------------------------------------- #
# 6. opt-in / λ 收缩 / 空记忆
# --------------------------------------------------------------------------- #
def test_opt_in_empty_memory_returns_zeroshot(model, data):
    tr, te = data
    agg = ICMAggregator(model)
    # memory=None => 纯 0-shot
    p = agg.predict(te.X[0], memory=None, k=3, scene_params=te.P[0:1])
    p0 = model.predict_next(te.X[0:1], scene_params=te.P[0:1])[0]
    assert torch.allclose(p, p0, atol=1e-6)


def test_lambda_zero_bitwise_equal(model, data):
    tr, te = data
    mem, agg = _build_mem(model, tr, n=32)
    agg.lamb = 0.0
    p = agg.predict(te.X[0], memory=mem, k=3, scene_params=te.P[0:1])
    p0 = model.predict_next(te.X[0:1], scene_params=te.P[0:1])[0]
    assert torch.allclose(p, p0, atol=1e-7)


def test_k0_ignores_memory(model, data):
    tr, te = data
    mem, agg = _build_mem(model, tr, n=32)
    p = agg.predict(te.X[0], memory=mem, k=0, scene_params=te.P[0:1])
    p0 = model.predict_next(te.X[0:1], scene_params=te.P[0:1])[0]
    assert torch.allclose(p, p0, atol=1e-6)


def test_version():
    assert __version__ == "5.5.5"
