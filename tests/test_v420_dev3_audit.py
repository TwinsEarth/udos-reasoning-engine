"""
v4.2.0.dev3 人工抽检 hook + 数据质量账本测试
==============================================
纪律: 纯治理工具不改数据、空/非法显式 ValueError、确定性。
"""
import json
import os

import pytest
import torch

from udos import (CTMConfig, PhysicsPredictor, TransitionTripletGenerator,
                  HumanAuditHook, DataQualityLedger)
from udos.world_model import LatentWorldModel
from udos.dynamics import build_parametric_dataset


def _setup(seed=7):
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    m = PhysicsPredictor(cfg, scene_param_dim=4)
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=seed)
    wm = LatentWorldModel(m, action_dim=0)
    wm.fit(ds, epochs=10)
    gen = TransitionTripletGenerator(m, wm, action_dim=1)
    trip = gen.generate(ds, n=20, seed=0)
    return m, trip


def test_audit_sample_deterministic():
    m, trip = _setup()
    hook = HumanAuditHook(sample_rate=0.2, seed=42)
    idx1, rep1 = hook.sample(trip)
    idx2, rep2 = hook.sample(trip)
    assert torch.equal(idx1, idx2)
    assert rep1["n_total"] == 20
    assert rep1["n_audited"] == 4   # 20*0.2
    assert rep1["audit_indices"] == idx1.tolist()


def test_audit_sample_clamps_to_one():
    m, trip = _setup()
    hook = HumanAuditHook(sample_rate=1.0, seed=1)
    idx, rep = hook.sample(trip)
    assert rep["n_audited"] == 20


def test_record_decision_counts():
    hook = HumanAuditHook()
    counts = hook.record_decision([0, 1, 2], ["approve", "reject", "approve"])
    assert counts == {"approve": 2, "reject": 1, "pending": 0}


def test_record_decision_bad():
    hook = HumanAuditHook()
    with pytest.raises(ValueError):
        hook.record_decision([0], ["maybe"])
    with pytest.raises(ValueError):
        hook.record_decision([0, 1], ["approve"])


def test_audit_empty_triplet_rejected():
    empty = HumanAuditHook
    hook = HumanAuditHook()
    # 空 triplet 由构造器已拦; 这里只验空输入
    with pytest.raises(ValueError):
        hook.sample(type("T", (), {"__len__": lambda self: 0})())


def test_audit_bad_rate():
    with pytest.raises(ValueError):
        HumanAuditHook(sample_rate=0.0)
    with pytest.raises(ValueError):
        HumanAuditHook(sample_rate=1.5)


def test_ledger_log_and_summary():
    led = DataQualityLedger()
    led.log("b1", n=20, triplet_quality={"action_spread": 0.8},
            execution={"pass_rate": 0.9})
    led.log("b2", n=30, review={"keep_rate": 0.5})
    s = led.summary()
    assert s["n_batches"] == 2
    assert s["total_samples"] == 50
    assert s["batch_ids"] == ["b1", "b2"]


def test_ledger_to_json(tmp_path):
    led = DataQualityLedger()
    led.log("b1", n=10, note="test")
    p = tmp_path / "ledger.json"
    led.to_json(str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["summary"]["n_batches"] == 1
    assert data["entries"][0]["batch_id"] == "b1"


def test_ledger_bad_batch_id():
    led = DataQualityLedger()
    with pytest.raises(ValueError):
        led.log("", n=10)
