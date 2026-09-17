"""v2.5.2 回归: 旧 checkpoint 向后兼容加载。

覆盖 v2.1.0 / v2.2.1 / v2.3.1 / v2.4.0 / v2.5.0 五个 checkpoint:
- 均可正常 load_predictor
- 预测正常 (输出有限、形状正确)
- 版本元数据正确
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.persistence import load_predictor  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


# 旧 checkpoint 清单 (checkpoints/ 下应存在)
_OLD_CKPTS = [
    ("predictor_v2.1.0.pt", "2.1.0"),
    ("predictor_v2.2.1.pt", "2.2.1"),
    ("predictor_v2.3.1.pt", "2.3.1"),
    ("predictor_v2.4.0.pt", "2.4.0"),
    ("predictor_v2.5.0.pt", "2.5.0"),
]


def _ckpt_path(name: str) -> Path:
    return ROOT / "checkpoints" / name


def test_all_old_checkpoints_exist():
    for name, _ in _OLD_CKPTS:
        assert _ckpt_path(name).exists(), f"缺少 checkpoint: {name}"


def test_load_all_old_checkpoints():
    """所有旧 checkpoint 均可加载, 版本元数据正确。"""
    for name, expected_ver in _OLD_CKPTS:
        path = _ckpt_path(name)
        predictor, meta = load_predictor(str(path))
        assert meta["udos_version"] == expected_ver, \
            f"{name}: 版本 {meta['udos_version']} != {expected_ver}"
        assert meta["kind"] == "PhysicsPredictor"
        # 参数量 ~52191
        n_params = sum(p.numel() for p in predictor.parameters())
        assert 50000 < n_params < 55000, \
            f"{name}: 参数量 {n_params} 异常"


def test_old_checkpoints_predict():
    """所有旧 checkpoint 预测正常 (输出有限、形状正确)。"""
    torch.manual_seed(0)
    for name, _ in _OLD_CKPTS:
        path = _ckpt_path(name)
        predictor, meta = load_predictor(str(path))
        x = torch.randn(2, 6, predictor.raw_dim)
        if predictor.scene_encoder is not None:
            p = torch.randn(2, predictor.scene_param_dim)
            out = predictor.predict_next(x, scene_params=p)
        else:
            out = predictor.predict_next(x)
        assert out.shape == (2, predictor.raw_dim), \
            f"{name}: 输出形状 {out.shape} 异常"
        assert torch.isfinite(out).all(), \
            f"{name}: 输出含非有限值"


def test_v240_plus_has_ood():
    """v2.4.0 及以后的 checkpoint 应挂载 OOD 检测器。"""
    for name, _ in _OLD_CKPTS:
        if name < "predictor_v2.4.0.pt":
            continue
        path = _ckpt_path(name)
        predictor, meta = load_predictor(str(path))
        assert predictor.has_ood_detector, f"{name} 应挂载 OOD 检测器"


def test_v231_plus_calibrated():
    """v2.3.1 及以后的 checkpoint 应已校准。"""
    for name, _ in _OLD_CKPTS:
        if name < "predictor_v2.3.1.pt":
            continue
        path = _ckpt_path(name)
        predictor, meta = load_predictor(str(path))
        assert predictor.is_calibrated, f"{name} 应已校准"
