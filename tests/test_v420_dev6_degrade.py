"""
v4.2.0.dev6 退化检测与回滚测试
==================================
纪律: 纯解析/治理、空/非法 ValueError、不假设 RSI 必然改善。
"""
import pytest
import torch

from udos import DegradationDetector, RollbackManager


def test_improving_series():
    d = DegradationDetector()
    rep = d.analyze([0.10, 0.08, 0.06, 0.05])
    assert rep["verdict"] == "improving"
    assert rep["strictly_monotonic_decreasing"] is True
    assert rep["linear_slope"] < 0


def test_drifting_series():
    d = DegradationDetector()
    rep = d.analyze([0.10, 0.12, 0.11, 0.13])
    assert rep["verdict"] == "drifting"


def test_collapsed_series():
    d = DegradationDetector(collapse_ratio=2.0)
    rep = d.analyze([0.10, 0.09, 0.30])   # 0.30 > 0.09*2=0.18
    assert rep["verdict"] == "collapsed"
    assert rep["collapse_generation"] == 2


def test_too_short_series():
    d = DegradationDetector()
    with pytest.raises(ValueError):
        d.analyze([0.1])


def test_bad_ratio():
    with pytest.raises(ValueError):
        DegradationDetector(collapse_ratio=1.0)


def test_rollback_register_and_best():
    rm = RollbackManager(patience_ratio=1.1)
    rm.register(0, {"w": torch.zeros(2)}, 0.10)
    rm.register(1, {"w": torch.zeros(2)}, 0.08)
    rm.register(2, {"w": torch.zeros(2)}, 0.12)
    best = rm.best()
    assert best["gen"] == 1
    assert best["mse"] == 0.08


def test_rollback_trigger():
    rm = RollbackManager(patience_ratio=1.1)
    rm.register(0, {"w": torch.zeros(2)}, 0.10)
    # 当前 0.12 > 0.10*1.1=0.11 -> 触发回滚
    assert rm.should_rollback(0.12) is True
    # 0.105 <= 0.11 -> 不触发
    assert rm.should_rollback(0.105) is False


def test_rollback_duplicate_gen():
    rm = RollbackManager()
    rm.register(0, {"w": torch.zeros(2)}, 0.1)
    with pytest.raises(ValueError):
        rm.register(0, {"w": torch.zeros(2)}, 0.2)


def test_rollback_empty():
    rm = RollbackManager()
    with pytest.raises(ValueError):
        rm.best()
