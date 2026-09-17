"""v2.5.0 回归: 推理缓存 (InferenceCache)。

覆盖:
- 默认关闭 (enabled=False), get 始终返回 None (透传)
- 开启后 miss 计算并写回, 二次命中返回与未命中完全相同输出
- LRU 淘汰: 超 maxsize 弹出最久未用
- 模型参数哈希: 权重变化后旧 key 不命中
- 缓存关闭时 BatchPredictor 透传 (不影响数值)
- 线程安全计数: hits/misses/hit_rate
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
from udos.cache import InferenceCache  # noqa: E402


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


def test_cache_default_disabled():
    """默认 enabled=False, get 始终返回 None (透传)。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=8)
    assert c.enabled is False
    x = torch.randn(1, 6, 6)
    assert c.get(c.make_key(x)) is None
    # put 在关闭时不写
    c.put(c.make_key(x), torch.randn(1, 6))
    assert c.size == 0


def test_cache_hit_returns_identical_output():
    """开启后命中必须返回与未命中完全相同输出 (逐位)。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=16)
    c.enabled = True
    torch.manual_seed(99)
    x = torch.randn(4, 6, 6)
    p = torch.randn(4, 4)
    # 第一次: 全部 miss
    out1 = m.predict_batch(x, scene_params=p, cache=c)
    assert c.hits == 0
    assert c.misses == 4
    # 第二次: 全部 hit
    out2 = m.predict_batch(x, scene_params=p, cache=c)
    assert c.hits == 4
    assert c.misses == 4
    assert torch.equal(out1, out2), "命中输出与未命中不一致"


def test_cache_miss_computes_and_stores():
    """miss 时计算并写回, 下次同输入命中。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=16)
    c.enabled = True
    x1 = torch.randn(1, 6, 6)
    x2 = torch.randn(1, 6, 6)
    out1 = m.predict_batch(x1, cache=c)
    assert c.size == 1
    out2 = m.predict_batch(x2, cache=c)
    assert c.size == 2
    # 再次 x1 命中
    out1b = m.predict_batch(x1, cache=c)
    assert c.hits == 1
    assert torch.equal(out1, out1b)


def test_lru_eviction():
    """超 maxsize 时 LRU 淘汰最久未用项。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=3)
    c.enabled = True
    keys = []
    outs = []
    for i in range(5):
        x = torch.randn(1, 6, 6)
        out = m.predict_batch(x, cache=c)
        keys.append(x)
        outs.append(out)
    assert c.size == 3, f"LRU 后 size 应为 3, 实际 {c.size}"
    # 前 2 个应被淘汰 (最久未用), 后 3 个应在缓存
    # 重新请求第 0 个 -> miss (已淘汰)
    _ = m.predict_batch(keys[0], cache=c)
    assert c.hits == 0  # 第 0 个 miss
    # 第 3 个还在 -> hit
    _ = m.predict_batch(keys[3], cache=c)
    assert c.hits == 1


def test_cache_disabled_passthrough():
    """缓存关闭时 (enabled=False) 不影响数值, 与无缓存逐位一致。"""
    m = _small_model()
    torch.manual_seed(7)
    x = torch.randn(3, 6, 6)
    p = torch.randn(3, 4)
    out_no_cache = m.predict_batch(x, scene_params=p)
    out_with_disabled_cache = m.predict_batch(x, scene_params=p,
                                              cache=InferenceCache(m))
    assert torch.equal(out_no_cache, out_with_disabled_cache)


def test_model_weight_change_invalidates_cache():
    """模型参数哈希变化后 (权重更新), 旧 key 不命中。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=16)
    c.enabled = True
    x = torch.randn(1, 6, 6)
    out1 = m.predict_batch(x, cache=c)
    assert c.misses == 1
    # 随机扰动权重 (模拟重新训练/换件)
    with torch.no_grad():
        for param in m.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    m.eval()
    out2 = m.predict_batch(x, cache=c)
    # 模型参数哈希变了 -> key 不同 -> miss
    assert c.misses == 2, "权重变化后缓存未失效"
    assert not torch.equal(out1, out2)


def test_cache_stats():
    """hits/misses/hit_rate 计数正确。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=16)
    c.enabled = True
    x1 = torch.randn(1, 6, 6)
    x2 = torch.randn(1, 6, 6)
    m.predict_batch(x1, cache=c)       # miss
    m.predict_batch(x1, cache=c)       # hit
    m.predict_batch(x2, cache=c)       # miss
    assert c.hits == 1
    assert c.misses == 2
    assert abs(c.hit_rate - 1 / 3) < 1e-6


def test_cache_clear():
    """clear() 清空缓存。"""
    m = _small_model()
    c = InferenceCache(m, maxsize=8)
    c.enabled = True
    x = torch.randn(1, 6, 6)
    m.predict_batch(x, cache=c)
    assert c.size == 1
    c.clear()
    assert c.size == 0
