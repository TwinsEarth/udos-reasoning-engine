"""v2.4.9 回归: 噪声鲁棒性 A/B 脚本逻辑 (3x2 网格产出结构 / MSE 可比 / JSON 含三 train_sigma)。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ctm_engine import CTMConfig  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.training import PhysicsPredictor  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import ablation_noise_robustness as ab  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _tiny_model():
    return PhysicsPredictor(CTMConfig(
        iterations=3, d_model=24, d_input=16, heads=2, n_synch_out=8,
        n_synch_action=6, memory_length=4, nlm_hidden=8, out_dims=12,
        certainty_threshold=0.0), scene_param_dim=4)


def test_eval_mse_cells_structure_and_monotonic_noise():
    model = _tiny_model(); model.eval()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=31)
    X, P, Y = ds.X, ds.P, ds.Y
    Y0 = Y[:, 0, :]
    m0 = ab.eval_mse(model, X, P, Y0, 0.0, eval_seed=1)
    m1 = ab.eval_mse(model, X, P, Y0, 0.1, eval_seed=1)
    assert isinstance(m0, float) and isinstance(m1, float)
    assert m0 >= 0.0 and m1 >= 0.0
    # 加噪后 MSE 不低于干净输入 (期望关系; 噪声只增扰动)
    assert m1 >= m0 - 1e-6


def test_eval_mse_deterministic():
    model = _tiny_model(); model.eval()
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=32)
    X, P, Y = ds.X, ds.P, ds.Y
    Y0 = Y[:, 0, :]
    a = ab.eval_mse(model, X, P, Y0, 0.1, eval_seed=7)
    b = ab.eval_mse(model, X, P, Y0, 0.1, eval_seed=7)
    assert a == b  # 同 generator seed 逐位一致


def test_ablation_json_written():
    p = ROOT / "benchmarks" / "results" / "noise_robustness_v2.4.9.json"
    assert p.exists(), "请先运行 make noise-ablation 生成结果"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data["summary"].keys()) == {"0.0", "0.05", "0.1"}
    for ts in ("0.0", "0.05", "0.1"):
        assert "mse_test0.0" in data["summary"][ts]
        assert "robustness_gap" in data["summary"][ts]
    assert data["grid"]["test_sigma"] == [0.0, 0.1]
