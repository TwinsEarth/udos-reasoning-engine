"""v7.3 AI 可观测性端到端演示（零依赖、CPU 可跑）。

跑通：流水线追踪 -> AI 指标 -> 尾采样 -> 结构化日志(JSONL) -> 在线质量评估
-> Prometheus 文本。产物：reports7/observability_demo.{json,prom,jsonl}
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from udos7.observability import (MetricsRegistry, OnlineEvaluator, AgentRegistry,
                                 Tracer, render_prometheus, resource_usage,
                                 run_pipeline, tail_sample_keep,
                                 FormatJudge, GroundingJudge, ToxicityJudge)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "reports7")


def main():
    random.seed(7)
    os.makedirs(OUT, exist_ok=True)
    tr = Tracer(redact_pii=True)
    reg = MetricsRegistry()
    agents = AgentRegistry()
    evaluator = OnlineEvaluator(
        judges=[FormatJudge(), GroundingJudge(), ToxicityJudge()], sample_rate=1.0)

    # 模拟 20 条请求走完整流水线
    kept = 0
    quality = []
    for i in range(20):
        stages = {
            "intent_classify": lambda p: (p.update(intent="qa") or p),
            "query_rewrite": lambda p: (p.update(q=p["q"] + " 精简") or p),
            "retrieve": lambda p: (p.update(ctx=["苹果是水果", "香蕉是水果"]) or p),
            "rerank": lambda p: p,
            "context_compress": lambda p: p,
            "llm_generate": _make_generate(p_reg=reg, i=i),
            "fact_check": lambda p: p,
        }
        payload = {"q": "苹果和香蕉是什么？", "log": []}
        data, root = run_pipeline(tr, stages, payload, root_name=f"request-{i}")
        spans = tr.traces()[root.trace_id]
        if tail_sample_keep(spans, base_rate=0.10, latency_ms_threshold=50.0):
            kept += 1
        q = evaluator.evaluate(data["eval_record"], force=True)
        quality.append(q["overall"])

    reg.set_queue_depth(3)
    logs_n = tr.export_jsonl(os.path.join(OUT, "observability_demo.jsonl"))

    summary = {
        "schema": "udos7.observability_demo/v1",
        "evidence": "verified",
        "requests": 20,
        "traces_kept_by_tail_sampling": kept,
        "structured_log_lines": logs_n,
        "quality_mean": round(sum(quality) / len(quality), 4),
        "quality_regression": evaluator.quality_regression(),
        "resource": resource_usage(),
        "semconv": "gen-ai-semconv-v1-udos1",
        "agents_registered": agents.list(),
        "notes": [
            "引擎内零依赖：追踪/指标/脱敏/尾采样/确定性评估均为 verified。",
            "Prometheus/Grafana/OTel Collector/eBPF/GPU/LLM-as-judge 为部署/资源闸门。",
        ],
    }
    with open(os.path.join(OUT, "observability_demo.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "observability_demo.prom"), "w", encoding="utf-8") as f:
        f.write(render_prometheus(reg))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _make_generate(p_reg, i):
    def llm_generate(p):
        t0 = time.perf_counter()
        # 模拟 TTFT/ITL/Token/工具
        p_reg.record_ttft(round(300 + random.random() * 900, 2), model="udos-cpu")
        p_reg.record_itl(round(30 + random.random() * 40, 2), model="udos-cpu")
        p_reg.record_tokens(80, 120, model="udos-cpu")
        p_reg.record_tool_call(i % 13 != 0, round(5 + random.random() * 20, 2), tool="kb_search")
        if i % 13 == 0:
            p_reg.record_error("rate_limit_429")
        time.sleep(0.002)
        good = i % 7 != 0
        out = json.dumps({"answer": "苹果和香蕉都是水果"}, ensure_ascii=False) if good \
            else "@@非法非JSON输出"
        p["output"] = out
        p["eval_record"] = {
            "request_id": f"req-{i}", "output": out,
            "context": " ".join(p.get("ctx", [])),
            "claims": ["苹果", "香蕉"] if good else ["火箭"],
        }
        return p
    return llm_generate


if __name__ == "__main__":
    main()
