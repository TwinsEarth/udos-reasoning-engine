"""
v2.9.1 重定向 A/B + affordance A/B 汇总 + MorphologyConfig 校验加固 单测
========================================================================
锚点纪律:
    * v29_feature_ab.json 落盘可复算 (retarget + affordance 两类);
    * affordance 引导成功率 > 无引导;
    * opt-in 默认关 (新特性不改旧 predict_next);
    * 被否决候选被记录;
    * MorphologyConfig 校验加固 (NaN/inf 限位、非法频率、非法 kinematics 守卫)。
analogy, not reproduction: 成功率为合成代理指标。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from udos.retargeting import MorphologyConfig, ActionRetargeter
from udos.persistence import load_predictor

ROOT = Path(__file__).resolve().parents[1]
AB_JSON = ROOT / "benchmarks" / "results" / "v29_feature_ab.json"
CKPT = str(ROOT / "checkpoints" / "predictor_v2.9.0.pt")


def test_ab_json_written_and_recomputable():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "v29_feature_ab.py")],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
    d = json.load(open(AB_JSON))
    assert d["version"] == "2.9.1"
    # affordance 引导应显著优于无引导
    aff = d["affordance_ab"]
    assert aff["guided_success"] > aff["unguided_success"]
    # retarget 不劣于截断
    assert d["retarget_ab"]["eval_mse_retarget"] <= \
        d["retarget_ab"]["eval_mse_truncate"]


def test_opt_in_default_off():
    m, _ = load_predictor(CKPT)
    x = torch.randn(1, 6, 6)
    sp = torch.randn(1, 4)
    before = m.predict_next(x, scene_params=sp)
    # 构造全部 2.9 外挂不应改主路径
    rt = ActionRetargeter(MorphologyConfig(6, 60, [[-2, 2]] * 6),
                          MorphologyConfig(4, 120, [[-1, 1]] * 4))
    rt.retarget(torch.randn(3, 6))
    after = m.predict_next(x, scene_params=sp)
    assert torch.equal(before, after)


def test_rejected_candidates_recorded():
    d = json.load(open(AB_JSON))
    assert d["opt_in_default"] is True
    assert len(d["rejected_candidates"]) >= 1


def test_morphology_validation_hardened():
    # NaN / inf 限位守卫
    with pytest.raises(ValueError):
        MorphologyConfig(2, 100.0, [[float("nan"), 1.0]])
    with pytest.raises(ValueError):
        MorphologyConfig(2, 100.0, [[float("inf"), 1.0]])
    # 非法频率
    with pytest.raises(ValueError):
        MorphologyConfig(2, float("nan"))
    with pytest.raises(ValueError):
        MorphologyConfig(2, -1.0)
    # 非法 kinematics 类型
    with pytest.raises(ValueError):
        MorphologyConfig(2, 100.0, [[-1, 1], [-1, 1]], kinematics="not-a-dict")
    # 合法配置仍可用
    c = MorphologyConfig(2, 100.0, [[-1, 1], [-1, 1]])
    assert c.dof == 2
