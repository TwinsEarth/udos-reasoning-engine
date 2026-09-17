"""
v2.8.0 PhysicalLoopRunner 骨架单元测试
=========================================
锚点纪律:
    * 空 window / 非有限 window 显式 ValueError (空循环守卫);
    * 五步顺序固定 observe->understand->predict_action->future_state->feedback;
    * 默认未挂载任何自定义 hook 时, runner.run 的 prediction 与直接
      predictor.predict_next 逐位一致 (torch.equal);
    * loop_state 完整: 每步 name/elapsed_ms/output_shape, 五次 run 累积;
    * 每步可插拔 (自定义 hook 覆盖内置默认);
    * 不破坏 checkpoint save/load, run 前后 predict_next 逐位不变 (外挂只读)。
"""
from pathlib import Path

import pytest
import torch

from udos import __version__  # noqa: E402
from udos.physical_loop import PhysicalLoopRunner, LOOP_STEPS  # noqa: E402
from udos.persistence import load_predictor, save_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v2.7.3.pt")


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=401)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_step_order_constant():
    assert LOOP_STEPS == ("observe", "understand", "predict_action",
                          "future_state", "feedback")


def test_empty_window_guard(predictor, window_batch):
    wb, pb = window_batch
    runner = PhysicalLoopRunner(predictor, horizon=2)
    # 空 batch
    with pytest.raises(ValueError):
        runner.run(torch.zeros(0, 6, predictor.raw_dim), scene_params=pb)
    # 空 window 长度
    with pytest.raises(ValueError):
        runner.run(torch.zeros(1, 0, predictor.raw_dim), scene_params=pb)
    # NaN 守卫
    bad = wb.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        runner.run(bad, scene_params=pb)


def test_default_bit_identical(predictor, window_batch):
    wb, pb = window_batch
    runner = PhysicalLoopRunner(predictor, horizon=2)
    out = runner.run(wb, scene_params=pb)
    direct = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(out["prediction"], direct), \
        "默认未挂载 hook 时, loop 预测必须与直接 predict_next 逐位一致"


def test_loop_state_complete(predictor, window_batch):
    wb, pb = window_batch
    runner = PhysicalLoopRunner(predictor, horizon=2)
    out = runner.run(wb, scene_params=pb)
    ls = out["loop_state"]
    names = [s["name"] for s in ls["steps"]]
    assert names == list(LOOP_STEPS)
    for s in ls["steps"]:
        assert "elapsed_ms" in s and s["elapsed_ms"] >= 0
        assert "output_shape" in s
    assert ls["run_count"] == 1
    # 再跑一次, run_count 累积
    runner.run(wb, scene_params=pb)
    assert ls["run_count"] == 2


def test_pluggable_hooks(predictor, window_batch):
    wb, pb = window_batch
    # 挂载自定义 understand hook, 验证其被调用且覆盖内置默认
    called = {}

    def fake_understand(ctx):
        called["hit"] = True
        return {"target_state": ctx["window"][:, -1, :], "custom": True}

    runner = PhysicalLoopRunner(predictor, horizon=2,
                                 understand_fn=fake_understand)
    out = runner.run(wb, scene_params=pb)
    assert called.get("hit") is True
    assert out["loop_state"]["outputs"]["understand"]["custom"] is True


def test_no_side_effect_checkpoint_roundtrip(predictor, window_batch, tmp_path):
    wb, pb = window_batch
    runner = PhysicalLoopRunner(predictor, horizon=2)
    before = predictor.predict_next(wb, scene_params=pb)
    runner.run(wb, scene_params=pb)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.equal(before, after), "跑 loop 不得改变 predictor 输出"
    # save/load 兼容: loop 不改权重, 重存重载后 predict 逐位一致
    p = tmp_path / "roundtrip.pt"
    save_predictor(predictor, str(p), metrics={"kind": "loop-test"})
    reloaded, meta = load_predictor(str(p))
    again = reloaded.predict_next(wb, scene_params=pb)
    assert torch.equal(after, again)
    assert meta["udos_version"] == __version__


def test_horizon_guard(predictor):
    with pytest.raises(ValueError):
        PhysicalLoopRunner(predictor, horizon=0)
