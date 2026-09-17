"""v2.2.x 回归: Scheduled Sampling 机制、早停/确定性、评估增强、模型管理服务。

原则: SS 单测只锁机制正确性, 不锁具体增益阈值(收益随种子变化, 见验证报告)。
"""
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import (PhysicsPredictor, CTMTrainer, TrainConfig,  # noqa: E402
                           set_seed)
from udos.evaluation import evaluate_predictor, _confidence_stratification  # noqa: E402


def _cfg():
    return CTMConfig(iterations=4, d_model=32, d_input=20, heads=2,
                     n_synch_out=10, n_synch_action=8, memory_length=6,
                     nlm_hidden=12, out_dims=20, certainty_threshold=0.0)


def _pred():
    return PhysicsPredictor(_cfg(), scene_param_dim=4)


def _ds(horizon=4, n=8):
    return build_parametric_dataset(n_per_kind=n, n_steps=6 + horizon + 4,
                                    window=6, horizon=horizon, dt=0.5, seed=1)


# ---------- F1 Scheduled Sampling 机制 ----------
def test_ss_prob_curriculum_monotone():
    cfg = TrainConfig(ss_max=0.4, ss_start=2, ss_warmup=4)
    probs = [cfg.ss_prob(e) for e in range(8)]
    assert probs[0] == probs[1] == 0.0           # start 前为 0
    assert probs[6] == 0.4 and probs[7] == 0.4   # 爬坡结束到上限
    assert all(b >= a for a, b in zip(probs, probs[1:]))  # 单调不减
    # 默认关闭恒为 0; warmup<=0 立即到上限
    assert TrainConfig(ss_max=0).ss_prob(99) == 0.0
    assert TrainConfig(ss_max=0.3, ss_warmup=0).ss_prob(0) == 0.3


def test_ss_zero_numerically_equals_teacher_forcing():
    """兼容锚点: ss_max=0 的滚动窗口必须与 2.1 的 cat-真值 teacher-forcing 逐位等价。"""
    ds = _ds()
    xb, pb, yb = ds.X[:6], ds.P[:6], ds.Y[:6]
    model = _pred(); model.eval()
    tr = CTMTrainer(model, TrainConfig(ss_max=0.0))
    torch.manual_seed(0)
    new_mse, _, _ = tr._parametric_loss(xb, pb, yb, 0.5, epoch=0)

    # 2.1 原始 teacher-forcing 实现作为 oracle (含相同的逐步权重)
    w = tr._step_weights(yb.size(1), xb.device)

    def old_tf(model, xb, pb, yb):
        ctx = model.scene_encoder(pb)
        total = xb.new_zeros(())
        for h in range(yb.size(1)):
            inp = torch.cat([xb[:, h:, :], yb[:, :h, :]], dim=1) if h > 0 else xb
            _, _, last, _ = model(inp, scene_context=ctx)
            total = total + w[h] * ((last - yb[:, h, :]) ** 2).mean()
        return total
    torch.manual_seed(0)
    old_mse = old_tf(model, xb, pb, yb)
    assert torch.allclose(new_mse, old_mse, atol=1e-7), \
        "ss_max=0 必须与 2.1 teacher-forcing 数值等价"


def test_ss_horizon1_is_noop():
    ds = _ds(horizon=1)
    xb, pb, yb = ds.X[:6], ds.P[:6], ds.Y[:6]
    m0, m1 = _pred(), _pred()
    m1.load_state_dict(m0.state_dict())
    t0 = CTMTrainer(m0, TrainConfig(ss_max=0.0))
    t1 = CTMTrainer(m1, TrainConfig(ss_max=0.9, ss_warmup=0))
    l0 = t0._parametric_loss(xb, pb, yb, 0.5, epoch=0)[0]
    l1 = t1._parametric_loss(xb, pb, yb, 0.5, epoch=0)[0]
    assert torch.allclose(l0, l1, atol=1e-7)  # 单步无未来可拼, SS 无操作


def test_ss_enabled_uses_autoregressive_and_backprops():
    """开启 SS(p=1) 走自回归喂入分支, loss 可正常反传。"""
    ds = _ds()
    xb, pb, yb = ds.X[:6], ds.P[:6], ds.Y[:6]
    model = _pred()
    tr = CTMTrainer(model, TrainConfig(ss_max=1.0, ss_warmup=0))
    mse, _, _ = tr._parametric_loss(xb, pb, yb, 0.5, epoch=0)
    mse.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ---------- F2 早停 / 确定性 ----------
def test_early_stop_triggers_and_records_best():
    ds = _ds(n=6); tr_ds, ev = ds.split(0.7)
    model = _pred()
    cfg = TrainConfig(epochs=30, patience=1, min_delta=1e9)  # 极大 min_delta => 无改善
    h = CTMTrainer(model, cfg).train(tr_ds, ev)
    assert h.stopped_early and len(h.train_loss) < 30
    assert h.best_epoch is not None and h.best_eval is not None


def test_early_stop_disabled_runs_full_by_default():
    ds = _ds(n=6); tr_ds, ev = ds.split(0.7)
    h = CTMTrainer(_pred(), TrainConfig(epochs=5)).train(tr_ds, ev)  # patience=None
    assert not h.stopped_early and len(h.train_loss) == 5
    assert h.best_epoch is None  # 未启早停不记录 best


def test_set_seed_deterministic_short_run():
    def run(seed):
        set_seed(seed)
        ds = _ds(n=6); tr_ds, _ = ds.split(0.7)
        m = _pred()
        h = CTMTrainer(m, TrainConfig(epochs=3, seed=seed)).train(tr_ds)
        return h.train_loss
    a, b = run(123), run(123)
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))


# ---------- F3 评估增强 ----------
def test_evaluation_new_keys_and_old_keys_preserved():
    ds = _ds(); _, te = ds.split(0.5)
    rep = evaluate_predictor(_pred(), te)
    for k in ("single_step_mse", "ablation", "per_kind_mse",
              "rollout_mse_curve", "kinematic_residual"):  # 2.1 旧键
        assert k in rep
    assert "rollout_growth_x" in rep and "confidence_stratification" in rep
    roll = rep["rollout_mse_curve"]
    assert abs(rep["rollout_growth_x"] - round(roll[-1] / roll[0], 3)) < 1e-9
    strat = rep["confidence_stratification"]
    assert len(strat["bins"]) == 3
    assert sum(b["n"] for b in strat["bins"]) == len(te)
    assert "high_over_low_mse_ratio" in strat


def test_confidence_stratification_handles_tiny_and_zerodivision():
    conf = torch.tensor([0.9, 0.1])
    yhat = torch.zeros(2, 3); y = torch.zeros(2, 3)  # 误差全 0, 不触发除零崩溃
    out = _confidence_stratification(conf, yhat, y, n_bins=3)
    assert len(out["bins"]) >= 1
    # 样本数少于 bin 数也安全
    out2 = _confidence_stratification(torch.tensor([0.5]),
                                      torch.zeros(1, 2), torch.zeros(1, 2), 3)
    assert out2["monotonic_decreasing"] in (True, False)


# ---------- F4 模型管理服务 ----------
def test_service_evaluate_save_flow_and_409(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService, ServiceNotReady
    s = UDOSService(preset="small",
                     checkpoints_dir=str(tmp_path / "checkpoints"))
    # 未训练 -> 409 语义
    for meth, body in (("evaluate", {}), ("save", {"name": "x.pt"})):
        try:
            getattr(s, meth)(body); assert False
        except ServiceNotReady:
            pass
    s.train({"epochs": 2, "n_per_kind": 6, "horizon": 4})
    ev = s.evaluate({"n_per_kind": 8, "horizon": 4, "seed": 777})
    assert ev["status"] == "evaluated" and "metrics" in ev
    assert ev["metrics"]["rollout_growth_x"] > 0
    sv = s.save({"name": "ck_v22"})
    assert os.path.exists(sv["path"]) and sv["bytes"] > 0
    from udos import load_predictor
    _, meta = load_predictor(sv["path"]); assert meta["kind"] == "PhysicsPredictor"
    # 危险名被无害化, 绝不逃出 checkpoints
    bad = s.save({"name": "../../escape.pt"})
    assert os.path.abspath(bad["path"]).startswith(
        os.path.abspath(str(tmp_path / "checkpoints")))


def test_service_invalid_args_400(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    s = UDOSService(preset="small")
    s.train({"epochs": 2, "n_per_kind": 6, "horizon": 4})
    for bad in ({"n_per_kind": 0}, {"horizon": 99}):
        try:
            s.evaluate(bad); assert False, bad
        except ValueError:
            pass
    try:
        s.save({"name": "   "}); assert False
    except ValueError:
        pass
    # /train 的 ss_max 越界也是 400 语义
    try:
        s.train({"epochs": 1, "ss_max": 1.5}); assert False
    except ValueError:
        pass


def test_verbose_training_runs_with_ss_curve():
    import io, contextlib
    ds = _ds(n=6); tr_ds, ev = ds.split(0.7)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        h = CTMTrainer(_pred(), TrainConfig(epochs=3, ss_max=0.4, ss_warmup=2)
                       ).train(tr_ds, ev, verbose=True)
    assert "ss=" in buf.getvalue() and len(h.ss_prob) == 3


def test_service_http_409_when_untrained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from udos.server import create_server
    httpd = create_server("127.0.0.1", 0, "small")
    port = httpd.server_address[1]
    import threading
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/evaluate",
            data=json.dumps({}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10); assert False
        except urllib.error.HTTPError as e:
            assert e.code == 409
    finally:
        httpd.shutdown(); httpd.server_close()
