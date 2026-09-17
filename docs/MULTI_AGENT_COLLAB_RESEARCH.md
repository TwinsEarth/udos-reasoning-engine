# 多智能体协作&协同&协调 — 调研与设计映射（UDOS v4.4 线）

> 调研日期 2026-09-16。前置：v4.3.9（1359 passed / 33 checkpoint / 主参数 52191）。
> 图意来源：用户提供的公众号"派派又森森"科普长图（**二手方法论**）。本文对关键模式联网补一手/权威来源，分层标注：
> **【官方/论文】**= 厂商工程博客/标准组织/论文；**【媒体科普】**= 第三方博客/教程转述；**【分析推断】**= 本文据 UDOS 现状的设计取舍；图中无出处的断言标 **[UNVERIFIED]**。
> 性质：**CPU 合成数据上的机制类比（analogy, not reproduction）**，不宣称复刻真实多机器人集群，也不实现 A2A/MCP 商业协议的线上字节兼容。

---

## 1. 图意结构化复述

### 1.1 四拓扑（图1/图5/图6/图7/图8）
| 拓扑 | 图意主张 | 触发场景（图1四问） |
|---|---|---|
| **star 星型 / Orchestrator-Worker** | 中心调度：拆任务→分 Worker→结果收集器合并收口；统一状态、责任边界清晰、易审计 | ①流程清晰、可事前规划 |
| **chain 链式 / Handoff** | Triage→Specialist→Return，按专业路由，上下文完整交接，可回退/补救 | ②需不同专业分工、按专业移交 |
| **tool 工具化 / Agent-as-Tool** | 主 agent 只看能力接口；Tool Schema 明确 Input/Output + confidence + error_type | ③能力可封装为稳定接口、可复用降耦 |
| **mesh 网状 / Swarm** | Peer 各自声明能力、无中心、动态组队、局部协商委派、结果可被再委派；灵活但治理成本高 | ④需自主协作、动态探索 |
| **拒绝自治分支** | 四条都不满足 = **控制需求不足**，放任自治=不可控风险；**默认从 Orchestrator 起步再按需放开** | — |

### 1.2 三类典型失败 + 修复三件套（图2）
- **状态丢失**：交接少传上下文 → 目标偏移/信息遗漏。
- **重复劳动**：多 agent 做同一件事。
- **责任不清**：没人收口、Trace 断裂。
- **治理三件套**：**Owner**（唯一负责人/任务归属）、**Trace**（可追溯执行链、上下文完整传递）、**Stop Condition**（清晰完成标准，防无限循环与重复劳动）。

### 1.3 Transfer Bundle 五要素（图3/图7）
交接必须带五要素，缺一即状态丢失：
1. **Goal**：最终目标 + 成功标准。
2. **Context**：用户约束、历史、关键事实。
3. **Done**：已完成步骤与结果。
4. **Todo**：下一步与风险。
5. **Trace**：trace id 与责任人。

### 1.4 核心论断
- "**协作 = 分工 + 交接 + 责任**；**拓扑比数量更重要**"（图5）。
- "**star 最可控**"、"**swarm 治理成本高**"（图5/图4）。

---

## 2. 联网一手/权威佐证（分层）

### 2.1 Orchestrator-Worker 编排范式 — 【官方】
- Anthropic, *Building Effective Agents*（2024-12-19）：在 **workflow / orchestrator-workers** 一节明确——中心 LLM 动态拆解任务、委派 worker、**合成（synthesize）结果**；适用于"无法预先预测子任务"的复杂任务（如改代码涉及多少文件取决于任务本身）。这是 star 拓扑的一手定义。
  https://www.anthropic.com/engineering/building-effective-agents
- Anthropic, *Building multi-agent systems*（2026-01）：orchestrator 路由到 specialist、每个 worker 独立上下文、lead 做最终综合；并强调"不是所有场景都需要多智能体"的选型观（与图1"没有银弹只有匹配"一致）。
  https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them
- LangGraph 多智能体文档：**supervisor（主管）模式**——单个中央主管协调所有通信流与委派，按上下文决定调用哪个智能体；与 star 同构。
  https://langchain.com.cn/agents/multi-agent.1.html

### 2.2 Handoff / 责任转移 — 【官方】+【媒体科普】
- LangGraph 官方概念：**handoff = 控制权移交**，节点可返回 `Command` 指定**目标 agent + 载荷（payload，状态更新）**；即"目标+要传给对方的信息"。这直接支撑"Transfer Bundle = 目标+上下文"。
  https://langchain.com.cn/concepts/multi_agent.1.html
- langgraph-supervisor（官方库）：handoff 是**机制**而非转发消息——supervisor 委派给 specialist，specialist 经 handoff-back 把控制权交还；记录 current owner / previous owner、交接原因、输出契约。
  https://deepwiki.com/langchain-ai/langgraph-supervisor-py/2.2-handoff-mechanism
- 【媒体科普】*Multi-Agent Systems Don't Fail at Reasoning. They Fail at Handoff*（Chanl, 2026-04）：建议"离场 agent 发出结构化摘要，接手方不从原始消息历史重建上下文"；并建议**限制顺序深度**（>5 跳改 supervisor 并行）。与图3"五要素完整交接"一致；其"15–20% token"等具体数字属【分析推断/媒体口径】，UDOS 不直接采用。
  https://www.channel.tel/blog/handoff-is-the-new-prompt

### 2.3 Agent-as-Tool / 工具化封装 — 【官方】+【官方标准】
- 【官方标准】MCP（Model Context Protocol，Anthropic 2024-11）：标准化 agent 如何连接**工具/资源**；用**结构化 schema 描述工具能力**（类似 function calling），传输入、收结构化输出。对应图8"Tool Schema 明确 Input/Output"。
  https://a2a-protocol.org/latest/topics/a2a-and-mcp/
- Mastra（2026-07）：orchestrator 把每个 worker **包成 tool**，tool call 即把子问题交给另一 agent；主 agent 只看接口。对应图8 Main Agent 只看能力接口。
  https://mastra.ai/blog/multi-agent-orchestration
- UDOS 的 confidence∈[0,1] + error_type 枚举：工程上对标 LLM function-calling 的结构化返回 + 置信度，但 **UDOS 不实现 MCP 线上协议**，只在内存里复用"name/input schema/output schema/confidence/error_type"这套契约思想。

### 2.4 去中心化 Swarm 与可观测性治理 — 【官方】+【媒体科普】
- 【官方】OpenAI Swarm（教学/轻量实验库）：以 **handoff 函数返回目标 agent 引用、上下文（对话历史）随控制移交** 为核心；强调"实验性、非生产支撑"。CrewAI 则以 **role + task + process（sequential/hierarchical）** 自主委派为核心，并已对接 MCP（工具）与 A2A（agent 协作发现）。
  https://docs.crewai.com/  ; OpenAI Swarm 教学库（业界广泛引用）
- 【媒体科普/官方云】可观测性：业界已收敛到 **OpenTelemetry GenAI 语义约定（gen_ai.*）**，单请求扇出成十余个内部 span，需追踪模型调用/工具调用/**agent handoff**/token/延迟；并警示 **guardrail 不随 handoff 自动继承**、trace 断裂。这支撑"Trace 事件链可重建责任链且检测断裂"。
  https://learn.microsoft.com/en-us/azure/foundry/observability/concepts/trace-agent-concept  ; https://www.fiddler.ai/blog/trace-agent-handoffs-multi-agent-llm-systems

### 2.5 A2A / MCP 协议分层 — 【官方标准】
- 【官方标准】A2A（Agent2Agent Protocol，Google 2025-04 起社区驱动）：解决 **agent↔agent** 协作（发现、协商、任务委派、多轮有状态）；核心构件 **Agent Card / Task / Message / Part / Artifact**，JSON-RPC 2.0 over HTTP+SSE。
- 【官方标准】MCP：解决 **agent↔tool**（单函数调用、通常无状态）。
- 两者**互补不竞争**：agent 用 A2A 委派给 specialist，specialist 再用 MCP 调工具。
  https://a2a-protocol.org/latest/  ; https://arxiv.org/pdf/2505.03864.pdf
- **UDOS 边界**：只在概念层借鉴"能力发现/Agent Card=能力声明"，**不做 A2A/MCP 字节级兼容**（无跨网络进程、无 JSON-RPC）。

### 2.6 图意中需标注 [UNVERIFIED] 的断言
- "拓扑比数量更重要"、"star 最可控"、"swarm 治理成本高"——**方向性**与 Anthropic/LangGraph 选型观一致（【官方】支持"按需选、默认中心"），但图中未给定量指标，UDOS 用**本机合成 A/B 实测**自证，不引用图中数字。
- "涌现可能"（swarm）——在 UDOS 确定性 CPU 合成机制下**不承诺涌现**，仅做"局部协商/再委派"的可测骨架。
- 任何具体 token 节省百分比、成功率提升倍数——**[UNVERIFIED]**，本线一律不采用，改由 dev6 A/B 落真实 JSON。

---

## 3. 图意 → UDOS 模块 / 端点 / 测试 映射表

| 图意 | UDOS 模块（新增/复用） | HTTP 端点 | 测试 |
|---|---|---|---|
| 四拓扑统一接口 star/chain/tool/mesh | `udos/collab_topology.py`（统一基类 + 枚举） | — | `tests/test_v440_collab_topology.py` |
| 图1 决策树四问 + "控制需求不足拒绝自治" | `collab_topology.py::TopologySelector` | `POST /collab/select` | 决策树各分支 + 拒绝分支 |
| Transfer Bundle 五要素 + 完整性校验 | `udos/transfer_bundle.py`（`TransferBundle` dataclass + `validate`） | `POST /collab/handoff` | 缺关键要素被拒 |
| 治理三件套 Owner/Trace/Stop | `udos/collab_governance.py`（`OwnerLedger`/`TraceChain`/`StopGuard`/`ClaimLock`） | `GET /collab/trace/{id}` | 责任链重建/断裂/认领去重 |
| 能力注册表 Agent-as-Tool（图8） | `udos/collab_agents.py`（`ToolSpec` + `CapabilityRegistry`） | （被 select/run 内部用） | Tool Schema 校验 |
| 星型 Orchestrator + 结果收集器（图6） | `udos/collab_orchestrator.py`（复用/升级 `closed_loop`/`wm_scheduler` 思想） | `POST /collab/run{topology=star}` | 拆分→分配→去重→收口 |
| 链式 Handoff Triage→Specialist→Return（图7） | `udos/collab_handoff.py` | `POST /collab/run{topology=chain}` | 路由+回退+Bundle 交接 |
| 网状 Swarm（图4，默认关 opt-in） | `udos/collab_swarm.py`（能力发现/局部协商/再委派/冲突检测） | `POST /collab/run{topology=mesh, opt_in}` | 冲突-未对齐/治理开销统计 |
| 三类失败回归（图2） | 上述三件套机制 | — | `tests/test_v440_collab_failures.py`（RED→GREEN） |
| 拓扑 A/B（图5 论断自证） | `scripts/collab_ab_v44.py` + `benchmarks/results/collab_ab_v44.json` | — | Makefile `collab-ab` |
| 全量回归 / 错误语义 / 版本 | `udos/server.py` 增 4 路由 | `/collab/*` | `tests/test_v441_collab_service.py` |

---

## 4. CPU 类比边界（诚实声明）
- **类比对象**：上述业界范式是"多 LLM agent 通过网络/消息协作"。UDOS 是**单进程、纯前向、确定性**的合成机制：所谓"agent"是带 Tool Schema 的**纯函数能力包装**（CTM 推演/GPM 场景/SFM 空间/PWM 想象/分层神经控制/curriculum/self_train/self_evolution/wla），"消息"是内存 dict，"trace"是事件列表。
- **不做**：不跑真实 LLM、不开网络进程间通信、不实现 A2A/MCP 字节协议、不承诺 swarm 涌现、不修改主 predictor 52191 参数。
- **价值**：在零梯度外挂内，用确定性合成任务把"拓扑选择/Bundle 完整性/Owner-Trace-Stop 治理/失败三病/拓扑 A/B"这几件事**可复算、可回归、可审计**地跑通，给后续真多智能体线提供骨架与账本。
