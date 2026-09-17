"""v2.6.1 边缘加固回归
======================
覆盖迭代 9 的缺陷修复 / 边缘加固:
- 空 batch: predict_batch([]) 返回 [0, RAW_DIM] 空张量 (不再报错)
- 极端 scene_param: 1e6 不产生 inf; NaN/inf 显式 ValueError
- horizon=1 自适应 rollout 正常返回 horizon_used=1 (不除零)
- RiskGrader: OOD score 为 NaN 时降级为 0 并记录 ood_nan_degraded
- CounterfactualEngine: 干预值 NaN/inf 显式 ValueError
- hybrid + guard: guard 在 hybrid 之后清洗最终输出, 不崩
- calibrator 在 hybrid 挂载下 certainty 路径仍可 transform (hybrid 不改 certainty)
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
from udos.adaptive import adaptive_rollout  # noqa: E402
from udos.decision import RiskGrader  # noqa: E402
from udos.counterfactual import CounterfactualEngine  # noqa: E402
from udos.hybrid import HybridPhysicsCorrector  # noqa: E402


def _small_model(seed: int = 0) -> PhysicsPredictor:
    torch.manual_seed(seed)
    m = PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), raw_dim=6, scene_param_dim=4)
    m.eval()
    return m


def test_version():
    assert __version__ == "5.5.5"


def test_empty_batch():
    """predict_batch([]) 返回 shape [0, 6] 空张量 (不崩)。"""
    m = _small_model()
    out = predict_batch(m, [])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (0, 6), f"空 batch 应返回 [0,6], 实际 {tuple(out.shape)}"
    # BatchPredictor.predict 路径同样
    out2 = BatchPredictor(m).predict([])
    assert out2.shape == (0, 6)


def test_empty_batch_tensor():
    """零维 batch 张量 [0,W,R] 同样返回 [0,6]。"""
    m = _small_model()
    out = m.predict_batch(torch.empty(0, 6, 6))
    assert out.shape == (0, 6)


def test_extreme_scene_params():
    """scene_params=1e6 时 predict_next 不产生 inf/NaN。"""
    m = _small_model()
    x = torch.randn(2, 6, 6)
    p_ext = torch.full((2, 4), 1e6)
    out = m.predict_next(x, scene_params=p_ext)
    assert out.shape == (2, 6)
    assert torch.isfinite(out).all(), "1e6 scene_params 产生非有限输出"


def test_nan_scene_params_rejected():
    """scene_params 含 NaN 时显式 ValueError (不静默 inf)。"""
    m = _small_model()
    x = torch.randn(2, 6, 6)
    p_nan = torch.tensor([[float("nan"), 0, 0, 0], [0, 0, 0, 0]])
    try:
        m.predict_next(x, scene_params=p_nan)
        assert False, "NaN scene_params 应显式报错"
    except ValueError:
        pass


def test_adaptive_horizon_one():
    """adaptive_rollout(horizon=1) 正常返回 horizon_used=1, 不除零。"""
    m = _small_model()
    rw = torch.randn(1, 6, 6)
    p = torch.randn(1, 4)
    res = adaptive_rollout(m, rw, max_horizon=1, scene_params=p)
    assert res["horizon_used"] == 1
    assert res["trajectory"].shape == (1, 1, 6)
    assert torch.isfinite(res["trajectory"]).all()


def test_risk_nan_ood():
    """mock OOD 检测器返回 NaN 时, RiskGrader.grade 不崩且降级。"""
    m = _small_model()

    class _NaNood:
        fitted = True
        threshold_ = 1.0

        def score(self, w):
            return torch.full((w.size(0),), float("nan"))

    m.ood_detector = _NaNood()
    rw = torch.randn(1, 6, 6)
    out = RiskGrader().grade(m, rw, horizon=1)
    assert out["risk_score"] == out["risk_score"], "risk_score 不应为 NaN"
    assert out["components"]["ood_nan_degraded"] is True
    assert out["components"]["ood_score"] == 0.0
    assert out["risk_level"] in ("low", "medium", "high")


def test_counterfactual_nan_intervention():
    """干预值为 NaN 时 CounterfactualEngine 显式 ValueError。"""
    m = _small_model()
    ce = CounterfactualEngine(m)
    rw = torch.randn(1, 6, 6)
    p = torch.randn(1, 4)
    # scene_params 干预 NaN
    try:
        ce.counterfactual(rw, 2, scene_params=p,
                          intervention={"scene_params": {0: float("nan")}})
        assert False, "NaN 干预应报错"
    except ValueError:
        pass
    # velocity_override 干预 NaN
    try:
        ce.counterfactual(rw, 2, scene_params=p,
                          intervention={"velocity_override": float("nan")})
        assert False, "NaN velocity_override 应报错"
    except ValueError:
        pass


def test_hybrid_guard_compat():
    """hybrid=True + guard=True 不崩, guard 清洗 hybrid 修正后的最终输出。"""
    m = _small_model()
    m.attach_hybrid(HybridPhysicsCorrector())
    x = torch.randn(2, 6, 6)
    p = torch.randn(2, 4)
    out = m.predict_next(x, scene_params=p, hybrid=True, guard=True)
    assert out.shape == (2, 6)
    assert torch.isfinite(out).all()
    # guard 确已生效 (挂载后 self.guard 非空)
    assert m.guard is not None
    m.detach_hybrid()


def test_hybrid_does_not_touch_calibration_path():
    """hybrid 仅改预测中位数, 不改 certainty; 校准器路径仍可 transform。"""
    from udos.calibration import ConfidenceCalibrator
    m = _small_model()
    cal = ConfidenceCalibrator()
    # 构造伪校准输入 (raw_cert, correct) 做确定性拟合
    raw_cert = torch.linspace(0.1, 0.9, 50)
    correct = (torch.rand(50) < raw_cert).float()
    cal.fit(raw_cert, correct)
    m.attach_calibration(cal, residual_quantiles=None)
    assert m.is_calibrated
    # 挂载 hybrid 后 grade 仍走校准置信路径
    m.attach_hybrid(HybridPhysicsCorrector())
    rw = torch.randn(1, 6, 6)
    out = RiskGrader().grade(m, rw, horizon=1)
    assert out["risk_score"] == out["risk_score"]
    assert 0.0 <= out["components"]["confidence"] <= 1.0
    m.detach_hybrid()
