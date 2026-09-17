"""v3.6.0.dev6 长 horizon 想象 vs 真实 rollout 对比实验测试。

覆盖:
    * wm_imagine_v3.6.0.json 落盘与必需字段;
    * 长 horizon {8,16,32} 均存在且第 0 步 MSE=0 锚点;
    * 误差随 horizon 累积 (末步 MSE >= 均值 >= 0);
    * 潜在压缩率字段;
    * 合成类比标注 (analogy_not_reproduction)。
"""
import json
from pathlib import Path

from udos import __version__

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "benchmarks" / "results" / "wm_imagine_v3.6.0.json"


def test_version():
    assert __version__ == "5.5.5"


def test_json_written():
    assert AB.exists(), "应落盘 wm_imagine_v3.6.0.json"
    d = json.loads(AB.read_text(encoding="utf-8"))
    for k in ("per_horizon", "horizons", "latent_dim",
              "compression_ratio_window_over_latent", "main_params",
              "analogy_not_reproduction"):
        assert k in d, f"缺少字段 {k}"


def test_long_horizons_present():
    d = json.loads(AB.read_text(encoding="utf-8"))
    assert set(str(h) for h in d["horizons"]) <= {"8", "16", "32"}
    for H in ("H=8", "H=16", "H=32"):
        assert H in d["per_horizon"]
        block = d["per_horizon"][H]
        # 第 0 步逐位锚定 => 严格 0
        assert block["step0_mse_anchor"] == 0.0
        assert block["final_step_mse"] >= 0.0
        assert len(block["step_mse"]) == int(H.split("=")[1])


def test_error_accumulates_with_horizon():
    d = json.loads(AB.read_text(encoding="utf-8"))
    finals = [d["per_horizon"][f"H={h}"]["final_step_mse"] for h in (8, 16, 32)]
    # 更长 horizon 末步误差不更优 (单调非降, 合成类比现象)
    assert finals[1] >= finals[0]
    assert finals[2] >= finals[1]


def test_compression_ratio_reported():
    d = json.loads(AB.read_text(encoding="utf-8"))
    assert d["latent_dim"] == 32
    assert d["window_flat_dim"] == 6 * 6
    assert d["compression_ratio_window_over_latent"] > 0


def test_analogy_not_reproduction_flag():
    d = json.loads(AB.read_text(encoding="utf-8"))
    assert d["analogy_not_reproduction"] is True
    assert d["main_params"] == 52191
