# UDOS v7.2 多 Agent 协同层与 Agent 军团

状态：随 v7.2.2 发布；预测内核沿用 v7.0.3（checkpoint 不变）。证据分级：
**verified**（本机 CPU 固定 seed 实测/测试断言）、**cpu-proto**（CPU 原型，机制可跑但非目标规模/真实后端）、**unverified**（需外部资源，门禁拦截）。

## 1. 它解决什么

v7.0.x 只有一个预测内核。v7.2 在其上加一层**可审计的多 Agent 协同**：一个不写核心
代码的协调器接收高阶目标，分类、拆解成 DAG，异步派发给隔离工作区里的专家 Agent，
再经“验证→讨论→提名→投票→碰撞裁决→选优→合并”产出当时最优解。所有过程写入
共享记忆（只追加事件 + 版本化 KV + 资源声明），形成血缘。

## 2. 模块

| 文件 | 职责 |
|---|---|
| `agents/types.py` | Task / WorkProduct / Vote / Claim / Collision / Decision / Event |
| `agents/memory.py` | `SharedMemory`：CAS 版本化 KV、私有草稿、事件日志、写-写碰撞、快照/合并 |
| `agents/workers.py` | 专家：预测(blind/explicit/analytic)、动作比较、验证、批评、分类、拆解、受约束代码改进 |
| `agents/protocols.py` | 讨论/提名/投票/裁决/选优/PR 式合并/加权集成 |
| `agents/coordinator.py` | `CoordinatorAgent`：分类→拆解 DAG→异步隔离派发→选优→上下文同步 |
| `agents/legion.py` | 组织树、分层投票、吞吐/质量 Scaling Law 实测、I/O 边界曲线 |
| `agents/automation.py` | Epoch AI 口径 AL0–AL5 分级与门禁 `GateError` |
| `agents/cloud.py` | `LocalTransport`（可用）/ `CloudTransport`（未配置即 AL4 门禁） |

## 3. 关键设计与实测结论（均为实测，非宣称）

- **三种预测专家互为独立信息源**：学习模型盲预测、显式参数预测、纯运动学解析积分。
  在 uniform/accel 上解析基线近乎零误差，spring/collision 上学习模型更稳；协调器按
  held-out 客观 MSE 选优（有真值）或校准加权集成（无真值）。
- **Agent Scaling Law（`reports7/agent_scaling.json`，cpu-proto/verified）**：
  - *CPU 密集型*：2 线程盒子上并发 1→2 加速 1.59×，4 并发 1.03×，8 并发 **0.70×**
    （超订反降）。本地同质 CPU 推理的扩展受物理核/GIL 硬约束。
  - *远程 I/O 型*（asyncio.sleep 模拟远程 LLM/工具等待，simulation 时延）：并发
    1/2/4/8 加速 1×/2.0×/4.0×/7.99×，近线性直到并发上限——这才是云端数千子 Agent
    的真实受益区间。
  - *质量*：按**独立校准集** MSE softmax 加权，集成 held-out MSE≈0.0092；朴素等权
    平均把差专家也算进来，MSE 恶化到 0.213（约 23 倍）。超过独立信息源数后扩招同质
    Agent，质量在 0.0092 附近**饱和**（拟合 mse(k)=a+b/k，gain_b≈−4e-5）。
    结论：**增益来自多样性 + 验证加权，而非堆人头。**
- **组织编制精确、对象有界**：`build_org` 用公式给出每层精确人数，亿级 headcount
  构建约 0.001s、仅约 2k 个预览节点；虚拟子树以整数计数表示，不实例化亿级对象，
  更不代表亿个在线 Agent 在运行。

## 4. 自动化分级（诚实口径，`baseline_assessment`）

| 能力 | AL | 证据 |
|---|---|---|
| 确定性单任务预测 | 1 | verified |
| 多 Agent 预测集成/讨论/投票选优 | 2 | verified |
| 协调器高阶目标→拆解→并行验证→选优合并（窄域） | 3 | verified |
| 受约束代码/配置改进（封闭候选集跑分选优） | 2 | cpu-proto |
| 自由形式云端数千 LLM 子 Agent 写码/改架构 | 4 | unverified（需 LLM key+云） |
| 递归自我改进 RSI 自治闭环 | 5 | unverified（本仓库不提供） |

## 5. 闸门（CPU 沙箱不可达，需用户资源）

- 多供应商 LLM API key：自由任务拆解/写码/多 LLM 交叉打分（AL4）。
- 云端共享上下文与数千在线子 Agent：`CloudTransport` 真实后端（对象存储/数据库/消息）。
- GPU/HPC：vLLM KV-offload、NEURON/CoreNEURON DHS、MuJoCo-MJX、0.5B/5B 端到端。

未配置时相关入口直接抛 `GateError`，不用 CPU 原型冒充 AL4/AL5。

## 6. 复现

```bash
PYTHONPATH=. python3 -m pytest tests7/test_v71_agents.py tests7/test_v72_legion.py -q
PYTHONPATH=. python3 scripts7/legion_demo.py
PYTHONPATH=. python3 scripts7/agent_scaling_bench.py \
  --ckpt checkpoints7/worldmodel_v7.0.3.pt --out reports7/agent_scaling.json
```
