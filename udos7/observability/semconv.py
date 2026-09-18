"""OpenTelemetry GenAI 语义约定（版本化契约）。

字段命名对齐 OpenTelemetry GenAI semantic conventions；UDOS 自有扩展以
``udos.`` 前缀显式区分，避免与上游标准混淆。该契约视为版本化对象，上游变更时
bump SEMCONV_VERSION，防止字段静默断裂。
"""

SEMCONV_VERSION = "gen-ai-semconv-v1-udos1"

# ---- 标准 GenAI 属性 ----
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"

# ---- UDOS 扩展属性（非上游标准）----
EXT_AGENT_NAME = "udos.agent.name"
EXT_AGENT_PURPOSE = "udos.agent.purpose"
EXT_AGENT_STATUS = "udos.agent.status"
EXT_STAGE = "udos.pipeline.stage"
EXT_EVIDENCE = "udos.evidence"

GEN_AI_OPERATIONS = (
    "chat", "embeddings", "tool_call", "retrieve", "rerank",
    "agent.run", "evaluate",
)

# RAG / Agent 工作流标准流水线阶段
PIPELINE_STAGES = (
    "intent_classify",   # 意图分类
    "query_rewrite",     # 查询重写
    "retrieve",          # 检索
    "rerank",            # 重排序
    "context_compress",  # 上下文压缩
    "llm_generate",      # LLM 生成
    "fact_check",        # 事实校验
)
