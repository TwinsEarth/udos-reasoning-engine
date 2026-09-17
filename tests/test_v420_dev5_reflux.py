"""
v4.2.0.dev5 数据回流三档配比代理测试
==========================================
纪律: 纯配比代理不训练主预测器; 空/非法 ValueError; 后训练档+退化 -> caution。
"""
import pytest

from udos import DataRefluxMixer


def test_pre_stage_pure_orig():
    m = DataRefluxMixer()
    rep = m.mix(n_orig=100, n_self=50, stage="pre")
    assert rep["orig_weight"] == 1.0
    assert rep["self_weight"] == 0.0
    assert rep["n_self_use"] == 0
    assert not rep["caution"]


def test_mid_stage_ratio():
    m = DataRefluxMixer(mid_orig=0.7)
    rep = m.mix(n_orig=100, n_self=50, stage="mid")
    assert rep["orig_weight"] == pytest.approx(0.7)
    assert rep["self_weight"] == pytest.approx(0.3)
    assert rep["n_orig_use"] == 70
    assert rep["n_self_use"] == 30


def test_post_stage_caution_when_degraded():
    m = DataRefluxMixer()
    rep = m.mix(n_orig=100, n_self=50, stage="post",
                student_verdict="degraded")
    assert rep["caution"] is True
    assert "崩塌" in rep["caution_note"]


def test_post_stage_no_caution_when_improved():
    m = DataRefluxMixer()
    rep = m.mix(n_orig=100, n_self=50, stage="post",
                student_verdict="improved")
    assert rep["caution"] is False


def test_self_use_clamped_to_available():
    m = DataRefluxMixer()
    rep = m.mix(n_orig=100, n_self=10, stage="post")
    assert rep["n_self_use"] <= 10   # 不超过自生成可用量


def test_bad_stage_rejected():
    m = DataRefluxMixer()
    with pytest.raises(ValueError):
        m.mix(n_orig=10, n_self=10, stage="final")


def test_bad_ratio_rejected():
    with pytest.raises(ValueError):
        DataRefluxMixer(mid_orig=1.5)
    with pytest.raises(ValueError):
        DataRefluxMixer(post_orig=0.0)


def test_bad_count_rejected():
    m = DataRefluxMixer()
    with pytest.raises(ValueError):
        m.mix(n_orig=-1, n_self=10, stage="mid")
