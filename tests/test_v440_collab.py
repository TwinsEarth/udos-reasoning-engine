"""v4.4.0 多智能体协作线 —— 框架/决策树/Bundle/治理/四拓扑 单测。

纯前向、确定性、CPU 合成机制类比; opt-in; 不改主权重。
"""
import pytest

from udos import (TransferBundle, OwnerLedger, TraceChain, StopGuard, ClaimLock,
                  Topology, TopologySelector, ToolSpec, ToolResult,
                  CapabilityRegistry, build_default_registry,
                  StarOrchestrator, ChainHandoff, MeshSwarm)
from udos.transfer_bundle import REQUIRED_KEYS


@pytest.fixture(scope="module")
def reg():
    return build_default_registry()


# ---- Transfer Bundle 五要素 / 完整性校验 ---- #
class TestTransferBundle:
    def test_complete_bundle_roundtrip(self):
        b = TransferBundle(goal="达成 X", trace={"trace_id": "t1", "owner": "a"},
                           context="预算有限", done=["调研"], todo=["执行"])
        assert b.is_complete()
        d = b.to_dict()
        b2 = TransferBundle.from_dict(d)
        assert b2.goal == "达成 X" and b2.trace["owner"] == "a"
        assert set(REQUIRED_KEYS) == {"goal", "trace"}

    def test_missing_trace_id_rejected(self):
        b = TransferBundle(goal="g", trace={"owner": "a"})
        with pytest.raises(ValueError):
            b.validate()

    def test_missing_owner_rejected(self):
        b = TransferBundle(goal="g", trace={"trace_id": "t"})
        with pytest.raises(ValueError):
            b.validate()

    def test_goal_empty_rejected(self):
        with pytest.raises(ValueError):
            TransferBundle(goal="", trace={"trace_id": "t", "owner": "a"})

    def test_reassign_owner_leaves_history(self):
        b = TransferBundle(goal="g", trace={"trace_id": "t", "owner": "a"})
        b.reassign_owner("b", reason="handoff")
        assert b.trace["owner"] == "b"
        assert b.trace["owner_history"][0]["from"] == "a"

    def test_record_done(self):
        b = TransferBundle(goal="g", trace={"trace_id": "t", "owner": "a"})
        b.record_done("step1")
        assert b.done == ["step1"]


# ---- 治理三件套 ---- #
class TestGovernance:
    def test_owner_unique_and_transfer(self):
        led = OwnerLedger()
        led.assign("task", "alice")
        assert led.current("task") == "alice"
        led.assign("task", "bob", reason="handoff")
        assert led.current("task") == "bob"
        assert led.history("task")[-1]["to"] == "bob"

    def test_require_owner_raises(self):
        led = OwnerLedger()
        with pytest.raises(ValueError):
            led.require_owner("nope")

    def test_trace_chain_rebuild_and_break_detect(self):
        tc = TraceChain()
        e1 = tc.append("t", "a", "start")
        e2 = tc.append("t", "b", "handoff", parent_event=e1)
        tc.append("t", "a", "close", parent_event=e2)
        rep = tc.integrity_report("t")
        assert rep["chain_intact"] and rep["closed"]
        assert rep["closer"] == "a"
        # 悬空父事件
        with pytest.raises(ValueError):
            tc.append("t", "b", "handoff", parent_event="nope:e999")

    def test_stop_guard_max_hops_and_loop(self):
        sg = StopGuard(max_hops=2)
        assert not sg.tick("k", "s1")["stop"]
        assert not sg.tick("k", "s2")["stop"]
        assert sg.tick("k", "s3")["over_hops"]  # 第3跳超限
        sg2 = StopGuard(max_hops=10)
        assert not sg2.tick("k2", "x")["loop_detected"]
        assert sg2.tick("k2", "x")["loop_detected"]  # 同指纹循环

    def test_claim_lock_dedup(self):
        lk = ClaimLock()
        assert lk.try_claim("task:a", "w1") is True
        assert lk.try_claim("task:a", "w2") is False  # 去重
        assert lk.owner("task:a") == "w1"


# ---- 拓扑选择器决策树 ---- #
class TestSelector:
    def test_default_star(self):
        s = TopologySelector().select(True, False, False, False)
        assert s.topology is Topology.STAR and s.accepted

    def test_chain_expert(self):
        s = TopologySelector().select(False, True, False, False)
        assert s.topology is Topology.CHAIN

    def test_tool(self):
        s = TopologySelector().select(False, False, True, False)
        assert s.topology is Topology.TOOL

    def test_autonomy_without_optin_refused(self):
        # mesh 默认关: 要自治但没 opt-in -> 退回 star, 不擅自 swarm
        s = TopologySelector().select(False, False, False, True)
        assert s.topology is Topology.STAR

    def test_autonomy_optin_mesh(self):
        s = TopologySelector(allow_mesh=True).select(False, False, False, True)
        assert s.topology is Topology.MESH and s.accepted

    def test_insufficient_control_rejected(self):
        s = TopologySelector().select(False, False, False, False)
        assert s.accepted is False  # 控制需求不足, 拒绝自治

    def test_bool_validation(self):
        with pytest.raises(ValueError):
            TopologySelector().select(1, 0, 0, 0)


# ---- 能力注册表 Agent-as-Tool ---- #
class TestRegistry:
    def test_default_registry_has_nine_tools(self, reg):
        assert len(reg.names()) == 9

    def test_call_ok(self, reg):
        r = reg.call("ctm_reasoning", {"query": "hi", "horizon": 2})
        assert r["error_type"] == "none" and 0 <= r["confidence"] <= 1

    def test_missing_input_schema(self, reg):
        r = reg.call("ctm_reasoning", {})
        assert r["error_type"] == "invalid_input"

    def test_unknown_tool(self, reg):
        with pytest.raises(ValueError):
            reg.get("not_a_tool")

    def test_discover_by_tag(self, reg):
        found = reg.discover(["reasoning"])
        names = {t.name for t in found}
        assert "ctm_reasoning" in names

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            ToolSpec(name="x", description="d", input_schema={"a": "int"},
                     output_schema={"b": "int"}, confidence=1.5)


# ---- 星型 Orchestrator ---- #
class TestStar:
    def test_run_collects_and_closes(self, reg):
        star = StarOrchestrator(reg)
        r = star.run("star1", "复合目标", [
            {"key": "a", "tool": "ctm_reasoning", "payload": {"query": "q", "horizon": 2}},
            {"key": "b", "tool": "gpm_scene", "payload": {"scene_id": "s1"}},
        ])
        assert r["topology"] == "star"
        assert r["collector"]["n_unique"] == 2
        assert r["trace_integrity"]["closed"]
        assert r["trace_integrity"]["closer"] == "orchestrator"
        assert r["main_params_untouched"] is True

    def test_dedup_same_key(self, reg):
        star = StarOrchestrator(reg)
        r = star.run("star2", "g", [
            {"key": "a", "tool": "ctm_reasoning", "payload": {"query": "q", "horizon": 2}},
            {"key": "a", "tool": "ctm_reasoning", "payload": {"query": "q", "horizon": 2}},
        ])
        assert "a" in r["skipped_duplicates"]
        assert r["collector"]["n_unique"] == 1

    def test_no_stop_refused(self, reg):
        star = StarOrchestrator(reg)
        with pytest.raises(ValueError):
            star.run("star3", "g", [{"key": "a", "tool": "ctm_reasoning",
                                     "payload": {"query": "q", "horizon": 2}}],
                     has_stop=False)

    def test_empty_subtasks_refused(self, reg):
        star = StarOrchestrator(reg)
        with pytest.raises(ValueError):
            star.run("star4", "g", [])


# ---- 链式 Handoff ---- #
class TestChain:
    def _ch(self, reg):
        return ChainHandoff(reg, {"inq": "sfm_space", "bill": "wla_action"})

    def test_chain_handles_and_returns(self, reg):
        r = self._ch(reg).run("c1", "g", ["inq", "bill"],
                              {"query_region": "r", "body_part": "arm", "dim": 7})
        assert r["topology"] == "chain"
        assert r["recovered"] is True
        assert r["final_owner"] == "triage"
        assert r["trace_integrity"]["closed"]

    def test_unknown_specialty_refused(self, reg):
        ch = self._ch(reg)
        with pytest.raises(ValueError):
            ch.run("c2", "g", ["ghost"], {})

    def test_chain_bundle_carries_done(self, reg):
        r = self._ch(reg).run("c3", "g", ["inq"], {"query_region": "r"})
        assert any("inq" in d for d in r["bundle"]["done"])


# ---- 网状 Swarm (默认关) ---- #
class TestSwarm:
    def test_swarm_default_off(self, reg):
        with pytest.raises(ValueError):
            MeshSwarm(reg)  # 不传 enabled

    def test_swarm_optin_runs(self, reg):
        sw = MeshSwarm(reg, enabled=True)
        r = sw.run("m1", "g", ["reasoning", "planning"], {"query": "q", "horizon": 2})
        assert r["topology"] == "mesh"
        assert r["n_peers"] >= 1
        assert "governance_overhead" in r
        assert r["trace_integrity"]["closed"]

    def test_swarm_no_peers_refused(self, reg):
        sw = MeshSwarm(reg, enabled=True)
        with pytest.raises(ValueError):
            sw.run("m2", "g", ["no_such_tag"], {})
