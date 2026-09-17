"""v2.7.3 后向兼容: 覆盖全部 10 个 checkpoint (v2.1.0 .. v2.7.3)。

每个 checkpoint:
    * load_predictor 成功、版本元数据正确;
    * predict_next 正常输出形状 [B, RAW_DIM=6]、有限;
    * v2.7 新模块属性 (online adapter / 主动 / lite / hierarchical / experiment)
      对旧件不报错, 且不污染旧件默认输出。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.persistence import load_predictor, export_snapshot  # noqa: E402
from udos.policy import MPCActionSelector  # noqa: E402
from udos.active_learning import UncertaintySampler  # noqa: E402
from udos.hierarchical import HierarchicalRollout  # noqa: E402

ALL_CKPTS = [
    ("predictor_v2.1.0.pt", "2.1.0"),
    ("predictor_v2.2.1.pt", "2.2.1"),
    ("predictor_v2.3.1.pt", "2.3.1"),
    ("predictor_v2.4.0.pt", "2.4.0"),
    ("predictor_v2.5.0.pt", "2.5.0"),
    ("predictor_v2.5.2.pt", "2.5.2"),
    ("predictor_v2.6.0.pt", "2.6.0"),
    ("predictor_v2.6.2.pt", "2.6.2"),
    ("predictor_v2.7.0.pt", "2.7.0"),
    ("predictor_v2.7.3.pt", "2.7.3"),
    ("predictor_v2.8.0.pt", "2.8.0"),
]


def test_version():
    assert __version__ == "5.5.5"


def test_all_checkpoints_exist():
    for name, _ in ALL_CKPTS:
        assert (ROOT / "checkpoints" / name).exists(), f"缺 {name}"


def test_all_load_predict_and_attributes():
    """9 件全部: 加载成功 / predict 形状 [B,6] / 有限 / 新模块对旧件不报错。"""
    torch.manual_seed(0)
    for name, expected_ver in ALL_CKPTS:
        predictor, meta = load_predictor(str(ROOT / "checkpoints" / name))
        assert meta["udos_version"] == expected_ver, \
            f"{name}: {meta['udos_version']} != {expected_ver}"
        x = torch.randn(2, 6, predictor.raw_dim)
        if predictor.scene_encoder is not None:
            p = torch.randn(2, predictor.scene_param_dim)
            out = predictor.predict_next(x, scene_params=p)
        else:
            out = predictor.predict_next(x)
        assert out.shape == (2, 6), f"{name}: 输出形状 {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: 输出非有限"


def test_v27_modules_work_on_all_checkpoints():
    """policy / active / hierarchical 对全部 9 件 (含无校准/无OOD旧件) 不报错。"""
    torch.manual_seed(1)
    for name, _ in ALL_CKPTS:
        predictor, _ = load_predictor(str(ROOT / "checkpoints" / name))
        x = torch.randn(1, 6, predictor.raw_dim)
        p = (torch.randn(1, predictor.scene_param_dim)
             if predictor.scene_encoder is not None else None)
        # policy
        sel = MPCActionSelector(predictor, horizon=1)
        r = sel.select(x, scene_params=p, candidate_actions=[{}])
        assert r["no_valid_action"] is False
        # active
        idx, sc = UncertaintySampler().select_top_k(
            predictor, x.repeat(3, 1, 1), 2,
            scene_params=(p.repeat(3, 1) if p is not None else None))
        assert idx.numel() == 2
        # hierarchical
        hr = HierarchicalRollout(predictor, coarse_factor=2).rollout(
            x, horizon=3, scene_params=p)
        assert hr["predictions"].shape == (1, 3, predictor.raw_dim)


def test_export_snapshot_all_checkpoints():
    for name, _ in ALL_CKPTS:
        predictor, _ = load_predictor(str(ROOT / "checkpoints" / name))
        snap = export_snapshot(predictor)
        assert snap["kind"] == "InferenceSnapshot"
