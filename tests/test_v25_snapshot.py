"""v2.5.1 回归: 无状态快照 (export_snapshot / import_snapshot)。

覆盖:
- export_snapshot 不含权重 (无 state_dict 键)
- export/import 后校准器/残差分位/OOD 一致
- import 架构不匹配拒绝
- 服务端点 export-snapshot / import-snapshot
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
from udos.persistence import export_snapshot, import_snapshot  # noqa: E402
from udos.calibration import ConfidenceCalibrator  # noqa: E402


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


def test_export_snapshot_no_weights():
    """导出的快照不含 state_dict / 权重。"""
    m = _small_model()
    snap = export_snapshot(m)
    assert "state_dict" not in snap, "快照不应包含权重"
    assert "ctm_config" in snap
    assert snap["raw_dim"] == 6
    assert snap["scene_param_dim"] == 4
    # 未校准则无 calibration 键
    assert "calibration" not in snap


def test_export_import_calibration_roundtrip():
    """导出快照 -> 新模型导入 -> 校准器/分位一致。"""
    m = _small_model(42)
    # 手动挂载校准器
    cal = ConfidenceCalibrator()
    # 拟合一个简单校准器
    scores = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    cal.fit(scores, labels)
    rq = [torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    m.attach_calibration(cal, rq)

    snap = export_snapshot(m)
    assert "calibration" in snap

    # 新建同架构模型, 导入快照
    m2 = _small_model(999)
    import_snapshot(m2, snap)
    assert m2.is_calibrated
    assert m2.residual_quantiles is not None
    assert torch.allclose(m2.residual_quantiles[0], rq[0])


def test_import_rejects_architecture_mismatch():
    """导入架构不匹配的快照必须拒绝。"""
    m = _small_model(42)
    snap = export_snapshot(m)
    # 不同架构的模型
    torch.manual_seed(0)
    m_wrong = PhysicsPredictor(CTMConfig(
        iterations=4, d_model=32, d_input=20, heads=2, n_synch_out=8,
        n_synch_action=8, memory_length=6, nlm_hidden=10, out_dims=16,
        certainty_threshold=0.0), raw_dim=8, scene_param_dim=4)
    m_wrong.eval()
    try:
        import_snapshot(m_wrong, snap)
        assert False, "架构不匹配应拒绝"
    except ValueError:
        pass


def test_service_export_import_snapshot(tmp_path, monkeypatch):
    """服务级: export-snapshot -> import-snapshot 往返一致。"""
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    from udos.dynamics import build_parametric_dataset
    s = UDOSService(preset="small")
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    out = s.export_snapshot_endpoint({})
    assert out["status"] == "ok"
    assert out["has_weights"] is False
    assert "state_dict" not in out["snapshot"]
    # 导入回同一个服务 (幂等)
    out2 = s.import_snapshot_endpoint({"snapshot": out["snapshot"]})
    assert out2["status"] == "ok"


def test_import_snapshot_missing_snapshot_field(tmp_path, monkeypatch):
    """缺少 snapshot 字段报错。"""
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    s = UDOSService(preset="small")
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    try:
        s.import_snapshot_endpoint({})
        assert False
    except ValueError:
        pass
