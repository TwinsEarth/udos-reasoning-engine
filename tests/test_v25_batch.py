"""v2.5.0 回归: 批量推理引擎 (BatchPredictor)。

覆盖:
- 等长 [B,W,RAW] 批量与逐笔 predict_next 逐位一致 (atol=1e-5)
- 变长 List[Tensor(W_i,RAW)] 批量与逐笔一致
- padding/分组: 不同窗口长度混合批量, 每条结果与单独调用一致
- 大 batch 分片 (max_shard < N) 结果不分片一致
- 空 batch 守卫 (空列表 / 零维张量)
- 非法输入形状报错
- PhysicsPredictor.predict_batch 委托方法
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
from udos.batch import BatchPredictor, predict_batch  # noqa: E402


def _small_model(seed: int = 0) -> PhysicsPredictor:
    torch.manual_seed(seed)
    m = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), raw_dim=6, scene_param_dim=4)
    m.eval()
    return m


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_equal_length_batch_matches_individual():
    """等长 [B,W,RAW] 批量 == 逐笔 predict_next (atol=1e-5)。"""
    m = _small_model()
    torch.manual_seed(123)
    x = torch.randn(6, 6, 6)
    p = torch.randn(6, 4)
    batched = m.predict_batch(x, scene_params=p)
    individual = torch.stack([
        m.predict_next(x[i:i + 1], scene_params=p[i:i + 1])[0]
        for i in range(6)])
    assert batched.shape == (6, 6)
    assert torch.allclose(batched, individual, atol=1e-5), \
        f"max diff={(batched - individual).abs().max().item()}"


def test_variable_length_matches_individual():
    """变长 List[Tensor(W_i,RAW)] 批量 == 逐笔 (不同 W 混合)。"""
    m = _small_model()
    torch.manual_seed(456)
    seqs = [torch.randn(w, 6) for w in (4, 6, 5, 4, 6, 7, 5)]
    ps = torch.randn(7, 4)
    out = m.predict_batch(seqs, scene_params=ps)
    assert out.shape == (7, 6)
    for i in range(7):
        single = m.predict_batch([seqs[i]], scene_params=ps[i:i + 1])
        assert torch.allclose(out[i:i + 1], single, atol=1e-5), \
            f"item {i} (W={seqs[i].size(0)}) mismatch"


def test_variable_length_order_preserved():
    """变长批量输出顺序与输入顺序一致 (乱序 W 排列仍按原索引返回)。"""
    m = _small_model()
    torch.manual_seed(789)
    seqs = [torch.randn(5, 6), torch.randn(3, 6), torch.randn(5, 6),
            torch.randn(7, 6)]
    ps = torch.randn(4, 4)
    out = m.predict_batch(seqs, scene_params=ps)
    # 第 0 和第 2 都是 W=5 但内容不同, 必须各自对应
    assert not torch.allclose(out[0:1], out[2:3])
    # 顺序: out[0] 对应 seqs[0]
    ref0 = m.predict_batch([seqs[0]], scene_params=ps[0:1])
    assert torch.allclose(out[0:1], ref0, atol=1e-5)


def test_sharding_large_batch_consistent():
    """max_shard < N 时分片结果与不分片一致。"""
    m = _small_model()
    torch.manual_seed(321)
    x = torch.randn(10, 6, 6)
    p = torch.randn(10, 4)
    direct = m.predict_batch(x, scene_params=p, max_shard=100)
    sharded = m.predict_batch(x, scene_params=p, max_shard=3)
    assert torch.allclose(direct, sharded, atol=1e-5)


def test_empty_batch_guarded_list():
    """v2.6.1: 空列表 batch 返回 [0, RAW_DIM] 空张量 (不再报错)。"""
    m = _small_model()
    out = m.predict_batch([])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (0, 6), f"空列表应返回 [0,6], 实际 {out.shape}"


def test_empty_batch_guarded_tensor():
    """v2.6.1: 零维 batch 张量返回 [0, RAW_DIM] 空张量 (不再报错)。"""
    m = _small_model()
    empty = torch.empty(0, 6, 6)
    out = m.predict_batch(empty)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (0, 6), f"零维 batch 应返回 [0,6], 实际 {out.shape}"


def test_wrong_dim_rejected():
    """非 2D 单条 / 非 3D 张量 必须报错。"""
    m = _small_model()
    try:
        m.predict_batch([torch.randn(6)])          # 1D 单条
        assert False
    except ValueError:
        pass
    try:
        m.predict_batch(torch.randn(6, 6))          # 2D 张量
        assert False
    except ValueError:
        pass


def test_raw_dim_mismatch_rejected():
    """最后一维与模型 raw_dim 不一致必须报错。"""
    m = _small_model()
    try:
        m.predict_batch(torch.randn(2, 6, 7))       # RAW_DIM=6
        assert False
    except ValueError:
        pass


def test_scene_params_count_mismatch():
    """scene_params 数量与序列数不一致必须报错。"""
    m = _small_model()
    seqs = [torch.randn(6, 6), torch.randn(5, 6)]
    wrong_sp = torch.randn(3, 4)                   # 3 != 2
    try:
        m.predict_batch(seqs, scene_params=wrong_sp)
        assert False
    except ValueError:
        pass


def test_guard_false_transparent():
    """predict_batch(guard=False) 与 predict_next 逐位一致 (不挂守卫)。"""
    m = _small_model()
    torch.manual_seed(111)
    x = torch.randn(4, 6, 6)
    p = torch.randn(4, 4)
    a = m.predict_next(x, scene_params=p, guard=False)
    b = m.predict_batch(x, scene_params=p, guard=False)
    assert torch.equal(a, b)


def test_batch_predictor_class_direct():
    """BatchPredictor 类直接调用与便捷函数 predict_batch 一致。"""
    m = _small_model()
    torch.manual_seed(222)
    x = torch.randn(3, 6, 6)
    p = torch.randn(3, 4)
    a = BatchPredictor(m, max_shard=2).predict(x, scene_params=p)
    b = predict_batch(m, x, scene_params=p, max_shard=2)
    assert torch.equal(a, b)


def test_no_scene_params():
    """无 scene_params 时批量推理正常 (纯动力学模型)。"""
    torch.manual_seed(333)
    m = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), raw_dim=6).eval()
    x = torch.randn(4, 6, 6)
    out = m.predict_batch(x)
    assert out.shape == (4, 6)
    indiv = torch.stack([m.predict_next(x[i:i+1])[0] for i in range(4)])
    assert torch.allclose(out, indiv, atol=1e-5)
