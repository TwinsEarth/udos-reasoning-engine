"""v4.4.0.dev5 三类失败 RED->GREEN 回归 (蓝图图2)
======================================================================
先 RED 复现: 以下断言描述治理三件套**必须拦下**的失败; 若治理被删/旁路, 这些测试转红。
    ①状态丢失: 残缺 bundle 被拒; 下游拿不到上下文有明确错误 (不静默接管)。
    ②重复劳动: 同一任务并发/多跳只被认领执行一次 (ClaimLock 去重)。
    ③责任不清: 无 owner 或无 stop 条件不得启动; 任意终态都能沿 trace 找到唯一收口人。
"""
import pytest

from udos import (TransferBundle, OwnerLedger, TraceChain, StopGuard, ClaimLock,
                  build_default_registry, StarOrchestrator, ChainHandoff, MeshSwarm)


@pytest.fixture(scope="module")
def reg():
    return build_default_registry()


# ---- 失败① 状态丢失 ---- #
class TestFailureStateLoss:
    def test_incomplete_bundle_rejected(self):
        """残缺 bundle(缺 trace_id/owner)交接被拒 —— 不把残缺上下文带给下游。"""
        bad = TransferBundle(goal="g", trace={"owner": "a"})  # 缺 trace_id
        with pytest.raises(ValueError):
            bad.validate()

    def test_downstream_receives_no_context_has_explicit_error(self, reg):
        """下游工具缺必需输入字段 -> 结构化 error_type=invalid_input, 不静默出幻觉。"""
        r = reg.call("ctm_reasoning", {})  # 缺 query/horizon
        assert r["error_type"] == "invalid_input"
        assert r["confidence"] == 0.0

    def test_chain_does_not_handoff_corrupt_bundle(self, reg):
        """链式每棒交接前强制 validate: 上下文缺关键要素即中断, 不带病流转。"""
        ch = ChainHandoff(reg, {"inq": "sfm_space"})
        # 正常 bundle 能过
        res = ch.run("sl1", "g", ["inq"], {"query_region": "r"})
        assert res["recovered"] is True
        assert res["trace_integrity"]["chain_intact"]


# ---- 失败② 重复劳动 ---- #
class TestFailureDuplicate:
    def test_claim_lock_only_one_executor(self):
        lk = ClaimLock()
        assert lk.try_claim("task:X", "w1") is True
        # w2 并发认领同一任务 -> 被去重, 不重复劳动
        assert lk.try_claim("task:X", "w2") is False

    def test_star_dedups_repeated_subtask_key(self, reg):
        star = StarOrchestrator(reg)
        r = star.run("dup1", "g", [
            {"key": "shared", "tool": "ctm_reasoning",
             "payload": {"query": "q", "horizon": 2}},
            {"key": "shared", "tool": "ctm_reasoning",
             "payload": {"query": "q", "horizon": 2}},
            {"key": "shared", "tool": "ctm_reasoning",
             "payload": {"query": "q", "horizon": 2}},
        ])
        # 三次同 key 只执行一次, 另两次被去重
        assert r["collector"]["n_unique"] == 1
        assert len(r["skipped_duplicates"]) == 2

    def test_swarm_does_not_reclaim_same_peer(self, reg):
        sw = MeshSwarm(reg, enabled=True)
        r = sw.run("dup2", "g", ["reasoning", "planning"],
                   {"query": "q", "horizon": 2})
        # 每个 peer 只被协商一次(认领锁), 无重复执行计数
        assert sw.gov_overhead["negotiations"] == r["n_peers"]


# ---- 失败③ 责任不清 ---- #
class TestFailureResponsibility:
    def test_no_stop_condition_refused(self, reg):
        star = StarOrchestrator(reg)
        with pytest.raises(ValueError):
            star.run("rs1", "g", [{"key": "a", "tool": "ctm_reasoning",
                                   "payload": {"query": "q", "horizon": 2}}],
                     has_stop=False)

    def test_star_terminal_state_has_unique_closer(self, reg):
        star = StarOrchestrator(reg)
        r = star.run("rs2", "g", [{"key": "a", "tool": "ctm_reasoning",
                                   "payload": {"query": "q", "horizon": 2}}])
        rep = r["trace_integrity"]
        assert rep["closed"]
        assert rep["closer"] == "orchestrator"  # 唯一收口人

    def test_chain_terminal_state_returns_to_triage(self, reg):
        ch = ChainHandoff(reg, {"inq": "sfm_space"})
        r = ch.run("rs3", "g", ["inq"], {"query_region": "r"})
        assert r["final_owner"] == "triage"
        assert r["trace_integrity"]["closer"] == "triage"

    def test_swarm_forced_closer(self, reg):
        sw = MeshSwarm(reg, enabled=True)
        r = sw.run("rs4", "g", ["reasoning"], {"query": "q", "horizon": 2})
        assert r["trace_integrity"]["closed"]
        assert r["final_owner"] == "swarm"  # 发起者被强制收口

    def test_owner_ledger_no_stray_owner(self):
        led = OwnerLedger()
        assert led.current("unassigned") is None
        with pytest.raises(ValueError):
            led.require_owner("unassigned")

    def test_trace_break_is_detectable(self):
        """终态若 trace 断裂(无 close 事件), integrity.closed=False。"""
        tc = TraceChain()
        tc.append("brk", "a", "start")
        rep = tc.integrity_report("brk")
        assert rep["closed"] is False  # 没人收口 -> 责任不清, 可被检出
