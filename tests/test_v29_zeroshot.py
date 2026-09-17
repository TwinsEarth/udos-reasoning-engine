"""
v2.9.0.dev3 零样本形态切换合成验证 + A/B 单测
================================================
锚点纪律:
    * zero_shot_transfer 输出动作有限、严格满足目标限位;
    * 限位违例率被诚实报告 (clamp 前);
    * A/B (重定向 vs 直接截断) JSON 落盘可复算, retarget mse <= truncate mse;
    * 未见/非法形态守卫; retargeting 默认不进旧预测路径 (opt-in)。
analogy, not reproduction: 合成数据上的形态切换代理。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from udos.retargeting import (MorphologyConfig, ActionRetargeter,
                              MorphologyLibrary)
from udos.persistence import load_predictor

ROOT = Path(__file__).resolve().parents[1]
AB_JSON = ROOT / "benchmarks" / "results" / "retarget_ab_v2.9.0.json"
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


@pytest.fixture(scope="module")
def rt():
    lib = MorphologyLibrary()
    return ActionRetargeter(lib.get("prime_u_60dof"), lib.get("gripper_4dof"))


def test_zero_shot_transfer_bounded(rt):
    X = torch.randn(64, 60) * 5.0      # 故意超限位
    out = rt.zero_shot_transfer(X)
    assert out["actions"].shape == (64, 4)
    assert torch.isfinite(out["actions"]).all()
    assert out["within_limits"] is True           # clamp 后全部在限位内
    assert 0.0 <= out["violation_rate"] <= 1.0     # 诚实报告 clamp 前违例率


def test_unseen_morphology_guard(rt):
    with pytest.raises(ValueError):
        rt.zero_shot_transfer(torch.randn(8, 7))   # 末维不符源 dof
    with pytest.raises(ValueError):
        rt.zero_shot_transfer(torch.empty(0, 60))  # 空动作
    # 真正未见的目标形态 (新建, 不在库里) 仍可重定向 -> 零样本成立
    unseen = MorphologyConfig(4, 999.0, [[-0.3, 0.3]] * 4, name="unseen")
    rt2 = ActionRetargeter(MorphologyLibrary().get("prime_u_60dof"), unseen)
    o = rt2.zero_shot_transfer(torch.randn(8, 60))
    assert o["within_limits"] is True


def test_ab_json_written_and_recomputable():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "retarget_ab_v29.py")],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(AB_JSON))
    assert d["src_dof"] == 60 and d["tgt_dof"] == 4
    assert d["retarget"]["final_within_limits"] is True
    assert d["truncate"]["final_within_limits"] is True
    # 重定向应不劣于朴素截断
    assert d["retarget"]["eval_mse_vs_ref"] <= d["truncate"]["eval_mse_vs_ref"]
    assert d["conclusion"] in ("synthetic_proxy_only", "no_gain_opt_in")


def test_retargeting_opt_in_default_path_unchanged():
    m, _ = load_predictor(CKPT)
    x = torch.randn(1, 6, 6)
    sp = torch.randn(1, 4)
    before = m.predict_next(x, scene_params=sp)
    # 使用 retargeter 不应改主模型权重
    rt = ActionRetargeter(MorphologyConfig(6, 60, [[-2, 2]] * 6),
                          MorphologyConfig(4, 120, [[-1, 1]] * 4))
    rt.retarget(torch.randn(3, 6))
    after = m.predict_next(x, scene_params=sp)
    assert torch.equal(before, after)
