# 多智能体拓扑与百万级 Agent 矩阵（v7.4.1–v7.5.0）

证据等级：**cpu-proto**。本层所有 Agent 都是带能力声明与可注入故障模式的
确定性处理器，不是 LLM 进程；规模数字是消息/tick 级模拟与闭式计数，
不是真实分布式部署结果。真实网络、拜占庭网络分区、跨机通信与安全边界
属未达资源闸门（unverified）。

## 1. 三种基础拓扑（v7.4.1，`topology/base.py`）

同一套三阶段处理器（triage → specialist → qa，真值内置于工单，验收机械
判定），三种控制权形态：

| 拓扑 | 控制结构 | 消息路径 | 适用 |
|---|---|---|---|
| Orchestrator 星型 | 中心派发/合并 | 每阶段 2 条（派发+回传） | 流程明确、强审计 |
| Handoff 链式 | 责任接力 | 每阶段 1 条移交 | 多领域专家接力 |
| Swarm 网状 | contract-net 广播-投标-授标 | 广播+投标+授标+公告 | 开放探索、负载均衡 |

可注入故障：`drop_context / byzantine / duplicate / crash`。

## 2. 协作机制（v7.4.2–v7.4.5）

- **Agent-as-Tool（7.4.2，agent_tool.py）**：固定信封
  `{report, confidence, error_type, trace_ref}`，主 Agent 不收专家内部
  trace；错误五分类 bad_input / capability_gap / internal_error /
  timeout / low_confidence；专家实现可独立迭代。
- **Transfer Bundle（7.4.3，transfer.py）**：Goal/Context/Done/Todo/
  Trace/Owner 六字段；交接前机械校验（缺字段、空 todo、done∩todo 重叠），
  不完整则责任不转移；replay 仅凭包重建状态。
- **Owner/Trace/Stop 治理（7.4.4，governance.py）**：哈希链追加式
  TraceLedger（篡改/断链可检出）；StopCondition 完成谓词+跳数预算；
  audit_run 机械识别状态丢失、重复劳动、无人收口、提前终止。
- **拓扑决策树（7.4.5，decision.py）**：流程明确→星型；专家接力→链式；
  能力可封装→工具化；开放探索→网状；高风险自治一律 escalate。

## 3. 规模治理（v7.4.6–v7.4.10）

- **分层混合 b 叉聚合树（7.4.6，hierarchy.py）**：顶层 Orchestrator /
  中层 Handoff / 底层 Swarm。小规模逐边显式模拟并与闭式计数互验，
  大规模用已验证公式外推（explicit/analytical 分级）。
  实测（b=8）：星型中心扇入 O(N)、串行轮次=单元数；分层每节点最大扇入
  恒为 b+1=9，并行轮次 2·log_b N；100 万 Agent 档星型中心承接
  20,971,520 条消息/8,388,608 轮，分层 14 轮、顶层只见 8 条领域汇总。
- **BFT-lite 停止共识（7.4.7，consensus.py）**：n≥3f+1、2f+1 法定人数；
  拜占庭谎报无法对抗诚实多数；同轮 equivocation 检出并整轮作废；
  超 f 缺席触发 view_change，不提前终止、不无限等待。
- **多层熔断（7.4.8，circuit_breaker.py）**：Agent 级 closed/open/
  half_open 错误率断路器、子矩阵隔离比例熔断、全局 kill-switch；
  TraceLedger 快照/回滚截断半截状态，在途工单改派健康节点。
- **内部市场（7.4.9，market.py）**：质量/成本性价比投标；仅验收通过的
  唯一完成者获付，重复劳动不付，拜占庭结果可罚没；守恒断言
  总支出≤总预算、余额和=已付−罚没。
- **Stigmergy（7.4.10，stigmergy.py）**：共享黑板+原子认领+信息素引导；
  30 任务/5 Agent 实测 60 条痕迹 vs contract-net 240 条协商消息，
  负载差 ≤1，Agent 间零直接通信。

## 4. 矩阵集成（v7.5.0，`matrix.py`）

闭环：Stigmergy 认领 → 分层 specialist 执行（熔断节点被排除）→
失败回滚 trace 并释放改派 → BFT-lite QA 委员会 2f+1 验收 →
内部市场按验收结算 → 哈希链 Trace + 治理审计 → 批次停止。

实测（24 工单，7 人 QA 委员会含 2 拜占庭，3 个故障 specialist：
crash/drop_context/byzantine 各一）：24/24 全部验收收口，3 个故障节点
全部被隔离，5 次返工改派，治理审计 ok，市场预算守恒，Trace 哈希链完整。

运行：

```bash
python scripts7/matrix_demo.py
# 或
bin/udos demo matrix
```

产物：`reports7/matrix_scale_demo.json(.traces.jsonl)`。

## 5. 与真实百万级系统的差距（诚实边界）

- 本层验证的是**协议性质与消息复杂度**（扇入上界、法定人数、守恒、
  无重复认领），不是吞吐/延迟；没有真实网络、进程调度与时钟。
- BFT-lite 只覆盖二进制停止决定的委员会投票，不是通用 BFT 复制状态机，
  签名为哈希占位、非密码学安全。
- 内部市场为内部记账单位，无真实定价博弈与 Sybil 防御。
- 真正的百万进程部署、跨机 Trace 存储、eBPF/OTel 指标接入需要
  GPU/HPC 与长期在线集群（资源闸门，约 ¥35,400/月档），本仓不冒充。

## 6. 测试索引

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| test_v741_topologies.py | 8 | 三拓扑/故障注入/负载均衡 |
| test_v742_agent_tool.py | 8 | 信封契约/五类错误 |
| test_v743_transfer.py | 9 | 交接校验/责任转移/replay |
| test_v744_governance.py | 9 | 三类失败/哈希链/停止条件 |
| test_v745_decision.py | 12 | 决策树全分支/高风险护栏 |
| test_v746_hierarchy.py | 9 | 显式=闭式/扇入上界/百万外推 |
| test_v747_consensus.py | 10 | 法定人数/equivocation/view change |
| test_v748_circuit_breaker.py | 11 | 熔断/隔离/改派/回滚 |
| test_v749_market.py | 10 | 授标/拒付/罚没/守恒 |
| test_v7410_stigmergy.py | 8 | 原子认领/消息节省/负载均衡 |
| test_v750_matrix.py | 9 | 端到端集成/故障收口/规模基准 |
