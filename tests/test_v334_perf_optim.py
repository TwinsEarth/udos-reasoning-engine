"""v3.3.4 hardening 性能优化单 delta 回归测试。

C1 (physical_loop 末尾冗余前向消除):
  - run() 最终 prediction 与 understand.target_state 逐位相等 (复用成立);
  - 且与直接 predictor.predict_next 逐位相等 (对外契约不变);
  - 自定义 understand_fn 时仍走 _integrated_predict 回退路径, 不破坏 hook 语义。

C2 (BatchPredictor 默认 max_shard 32->64):
  - bs=64 单片前向输出与旧两片[32,32]前向输出逐位相等 (批量独立性锚点);
  - 超过默认 max_shard 仍正确分片 (bs=128 -> 两片, 与显式 max_shard 一致结果)。
"""

from __future__ import annotations

import torch

from udos.ctm_engine import CTMConfig
from udos.dynamics import RAW_DIM, SCENE_PARAM_DIM
from udos.training import PhysicsPredictor
from udos.batch import BatchPredictor
from udos.physical_loop import PhysicalLoopRunner


def _tiny_predictor():
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, heads=2,
                    n_synch_out=8, n_synch_action=6, memory_length=4,
                    nlm_hidden=8, out_dims=16, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg, scene_param_dim=SCENE_PARAM_DIM)
    m.eval()
    return m


# ----------------------------- C1: loop 复用 target_state ----------------------------- #

def test_c1_loop_prediction_equals_understand_target():
    """默认 understand 路径下, 最终 prediction 与 understand.target_state 逐位相等。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    runner = PhysicalLoopRunner(m, horizon=2)
    w = torch.randn(1, 6, RAW_DIM)
    sp = torch.randn(1, SCENE_PARAM_DIM)
    with torch.no_grad():
        out = runner.run(w, scene_params=sp)
    pred = out["prediction"]
    target = out["loop_state"]["outputs"]["understand"]["target_state"]
    assert torch.equal(pred, target), \
        "C1: run() 最终 prediction 应复用 understand.target_state (bit-exact)"


def test_c1_loop_prediction_matches_direct_predict_next():
    """对外契约不变: run().prediction 与直接 predict_next 逐位相等。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    runner = PhysicalLoopRunner(m, horizon=2)
    w = torch.randn(1, 6, RAW_DIM)
    sp = torch.randn(1, SCENE_PARAM_DIM)
    with torch.no_grad():
        out = runner.run(w, scene_params=sp)
        direct = m.predict_next(w, scene_params=sp)
    assert torch.equal(out["prediction"], direct), \
        "C1: run().prediction 必须与直接 predict_next 逐位一致 (对外契约)"


def test_c1_custom_understand_fn_fallback():
    """挂载自定义 understand_fn 时, 回退 _integrated_predict, 不臆造 target_state。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    called = {}

    def fake_understand(ctx):
        # 故意不返回 target_state, 验证 run() 不会 KeyError/臆造, 而是自行预测
        called["hit"] = True
        return {"risk_flags": {}}

    runner = PhysicalLoopRunner(m, horizon=2, understand_fn=fake_understand)
    w = torch.randn(1, 6, RAW_DIM)
    sp = torch.randn(1, SCENE_PARAM_DIM)
    with torch.no_grad():
        out = runner.run(w, scene_params=sp)
        direct = m.predict_next(w, scene_params=sp)
    assert called.get("hit") is True
    assert torch.equal(out["prediction"], direct), \
        "C1: 自定义 understand_fn 时回退路径必须与 predict_next 一致"


# ----------------------------- C2: max_shard 默认 64 ----------------------------- #

def test_c2_bs64_single_shard_matches_two_shards():
    """bs=64 单片 (max_shard=64 默认) 与两片 (max_shard=32) 输出逐位相等。"""
    torch.manual_seed(0)
    m = _tiny_predictor()
    g = torch.Generator().manual_seed(7)
    batch = torch.randn(64, 6, RAW_DIM, generator=g)
    sp = torch.randn(64, SCENE_PARAM_DIM, generator=g)
    with torch.no_grad():
        two_shard = BatchPredictor(m, max_shard=32).predict(batch, scene_params=sp)
        one_shard_default = BatchPredictor(m).predict(batch, scene_params=sp)
    assert torch.equal(two_shard, one_shard_default), \
        "C2: bs=64 单片与两片输出应 bit-exact (批量独立性)"


def test_c2_default_max_shard_is_64():
    """默认 max_shard 提升到 64, 仍对超大片正确分片。"""
    m = _tiny_predictor()
    bp = BatchPredictor(m)
    assert bp.max_shard == 64
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(11)
    batch = torch.randn(128, 6, RAW_DIM, generator=g)   # >64 触发分片
    sp = torch.randn(128, SCENE_PARAM_DIM, generator=g)
    with torch.no_grad():
        sharded = bp.predict(batch, scene_params=sp)
        # 对照: 不分片 (足够大的 max_shard)
        direct = BatchPredictor(m, max_shard=1024).predict(batch, scene_params=sp)
    assert torch.equal(sharded, direct), \
        "C2: bs=128 分片结果应与不分片逐位一致"
