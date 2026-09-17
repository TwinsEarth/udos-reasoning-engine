"""v2.4.0 回归: OOD/分布漂移检测 (马氏距离 + KS)、外挂集成、持久化、版本断言。

单测锁机制正确性与向后兼容锚点, 不锁随种子波动的绝对阈值。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ood import (  # noqa: E402
    DistributionDriftDetector, ks_two_sample,
)
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


# ---------- 检测器基础机制 ----------
def test_detector_fit_score_threshold_and_ordering():
    torch.manual_seed(0)
    # 训练分布: 单位球内
    train = torch.randn(500, 6) * 0.5
    det = DistributionDriftDetector(ridge=1e-3, alpha=0.05).fit(train)
    assert det.fitted and det.dim == 6 and det.n_fit == 500
    assert det.threshold_ > 0
    # ID 样本得分低, 明显偏移样本得分高
    id_scores = det.score(torch.randn(200, 6) * 0.5)
    ood_scores = det.score(torch.randn(200, 6) * 5.0)
    assert float(ood_scores.mean()) > float(id_scores.mean()) * 2
    # 判异掩码: 偏移样本大部分超阈值, ID 样本小部分 (约 alpha=0.05) 误报
    ood_flags = det.is_ood(torch.randn(200, 6) * 5.0)
    assert float(ood_flags.float().mean()) > 0.8
    id_flags = det.is_ood(torch.randn(400, 6) * 0.5)
    assert float(id_flags.float().mean()) < 0.25


def test_detector_rejects_too_few_samples():
    try:
        DistributionDriftDetector().fit(torch.randn(1, 4))
        assert False
    except ValueError:
        pass


def test_ks_two_sample_statistic_and_same_dist_zero():
    torch.manual_seed(1)
    x = torch.randn(300)
    y = torch.randn(300)
    d_same = ks_two_sample(x, y)
    assert 0.0 <= d_same < 0.15          # 同分布: D 应小
    # 平移分布: D 应显著更大
    z = torch.randn(300) + 3.0
    d_shift = ks_two_sample(x, z)
    assert d_shift > d_same
    try:
        ks_two_sample(torch.zeros(0), torch.ones(3)); assert False
    except ValueError:
        pass


def test_detector_drift_summary_and_ks_vs_reference():
    torch.manual_seed(2)
    train = torch.randn(300, 4)
    det = DistributionDriftDetector().fit(train)
    rep = det.ks_drift(train + 0.1)
    assert rep["n_ref"] == 300 and "ood_rate" in rep and "threshold" in rep
    # 逐维严格 KS: 同分布 D 小, 平移 D 大
    ks_same = det.ks_two_sample_vs_reference(train, train + 0.05)
    ks_shift = det.ks_two_sample_vs_reference(train, train + 3.0)
    assert ks_shift["ks_d_max"] > ks_same["ks_d_max"]


def test_detector_roundtrip_state_dict():
    torch.manual_seed(3)
    train = torch.randn(200, 5)
    det = DistributionDriftDetector(ridge=1e-2, alpha=0.1).fit(train)
    probe = torch.randn(10, 5)
    s1 = det.score(probe)
    det2 = DistributionDriftDetector().load_state_dict(det.state_dict())
    assert torch.allclose(det2.score(probe), s1)
    assert abs(det2.threshold_ - det.threshold_) < 1e-9


# ---------- 与 PhysicsPredictor 外挂集成 ----------
def _pred():
    return PhysicsPredictor(CTMConfig(
        iterations=4, d_model=32, d_input=20, heads=2,
        n_synch_out=10, n_synch_action=8, memory_length=6,
        nlm_hidden=12, out_dims=20, certainty_threshold=0.0),
        scene_param_dim=4)


def test_predictor_ood_attachment_and_opt_in():
    model = _pred(); model.eval()
    # 未挂载 => 显式报错, 不静默退化
    try:
        model.ood_score(torch.randn(2, 6, 6)); assert False
    except RuntimeError:
        pass
    assert not model.has_ood_detector
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=7)
    det = DistributionDriftDetector().fit(ds.X)
    model.attach_ood_detector(det)
    assert model.has_ood_detector
    s = model.ood_score(ds.X[:4])
    assert s.shape == (4,) and bool((s >= 0).all())
    # 偏移样本得分更高
    s_shift = model.ood_score(ds.X[:4] * 6.0)
    assert float(s_shift.mean()) > float(s.mean())


def test_ood_detector_not_in_state_dict():
    model = _pred(); model.eval()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=8)
    model.attach_ood_detector(DistributionDriftDetector().fit(ds.X))
    keys = " ".join(model.state_dict().keys())
    assert "ood_detector" not in keys


def test_persistence_ood_roundtrip(tmp_path):
    from udos import save_predictor, load_predictor
    model = _pred(); model.eval()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=9)
    model.attach_ood_detector(DistributionDriftDetector().fit(ds.X))
    p = tmp_path / "p24.pt"
    save_predictor(model, p)
    loaded, meta = load_predictor(p)
    assert loaded.has_ood_detector
    probe = ds.X[:3]
    # 序列化往返经 tolist/重建, 内存 stride 不同触发 BLAS 归约顺序差 ~1e-4
    assert torch.allclose(loaded.ood_score(probe), model.ood_score(probe),
                          atol=1e-4)
    # 旧 checkpoint (无 ood 键) 不挂载检测器, 不崩
    legacy = ROOT / "checkpoints" / "predictor_v2.3.1.pt"
    if legacy.exists():
        old, _ = load_predictor(legacy)
        assert not old.has_ood_detector
