"""v7.4.2 Agent-as-Tool 契约测试。"""
import pytest

from udos7.topology.agent_tool import (AgentTool, MainAgent, ToolEnvelope,
                                       ToolInput)


def review_expert(tin: ToolInput):
    code = tin.payload["code"]
    issues = [f"line {i}" for i, ln in enumerate(code.splitlines())
              if "eval(" in ln]
    return {"issues": issues, "verdict": "fail" if issues else "pass"}, \
        0.9 if code else 0.3, "trace://expert/1"


def boom(tin):
    raise RuntimeError("kaboom")


def slow(tin):
    raise TimeoutError()


@pytest.fixture
def main():
    m = MainAgent()
    m.register(AgentTool("code_review", ("code",), review_expert))
    m.register(AgentTool("boom", ("x",), boom))
    m.register(AgentTool("slow", ("x",), slow))
    m.register(AgentTool("strict", ("code",), review_expert,
                         confidence_floor=0.5))
    return m


def test_success_envelope_fields(main):
    env = main.call("code_review", ToolInput({"code": "a = 1\nprint(a)\n"}))
    assert env.error_type is None
    assert env.report["verdict"] == "pass"
    assert env.confidence == 0.9
    assert env.trace_ref == "trace://expert/1"


def test_missing_required_field_is_typed_error(main):
    env = main.call("code_review", ToolInput({}))
    assert env.error_type == "bad_input"
    assert env.report["missing"] == ["code"]


def test_unknown_tool_capability_gap(main):
    env = main.call("nonexistent", ToolInput({"x": 1}))
    assert env.error_type == "capability_gap"


def test_internal_exception_does_not_escape(main):
    env = main.call("boom", ToolInput({"x": 1}))
    assert env.error_type == "internal_error"
    assert env.report["exc"] == "RuntimeError"


def test_timeout_is_typed(main):
    env = main.call("slow", ToolInput({"x": 1}))
    assert env.error_type == "timeout"


def test_low_confidence_floor(main):
    env = main.call("strict", ToolInput({"code": ""}))
    assert env.error_type == "low_confidence"
    assert env.confidence == 0.3


def test_main_context_only_holds_envelopes(main):
    main.call("code_review", ToolInput({"code": "eval(1)\n"}))
    main.call("code_review", ToolInput({"code": "ok\n"}))
    assert len(main.context_log) == 2
    # 专家内部 trace 全文不进入主上下文，只有引用
    assert all("issues" in e["report"] for e in main.context_log)
    assert all(isinstance(e["trace_ref"], str) for e in main.context_log)


def test_tool_independent_iteration_keeps_contract(main):
    # 专家内部实现替换，只要信封契约不变，主 Agent 无感知
    tool = main.tools["code_review"]
    old = tool.calls
    tool._expert = lambda tin: ({"issues": [], "verdict": "pass"}, 0.95, "t2")
    env = main.call("code_review", ToolInput({"code": "x=1\n"}))
    assert env.confidence == 0.95 and env.trace_ref == "t2"
    assert tool.calls == old + 1
