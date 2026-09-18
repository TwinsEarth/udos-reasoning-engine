# v7.3.7 具身混合控制：语义层 × 动作先验（CPU 原型）

## 1. 对标对象
公开仿真报告（匿名 GitHub，银河通用团队作者，媒体转述）用 GPT-6 Astra 与
机器人 VLA 模型 π0.5 做双臂闭环对比：

- **Direct**：大模型直接看三路相机 + 机器人状态，输出末端位姿/夹爪；
- **Hybrid**：每个决策时刻 π0.5 先生成 50 步候选动作，大模型看同一画面、
  执行历史与候选轨迹，二选一——**采纳先验的前 1–15 步，或自己出 1–5 步
  末端修正**；走完一段再重新观察。
- 报告口径（RoboDojo，**UNVERIFIED，非本仓实测**）：direct 26%/37.81 分，
  hybrid 48%/62.60 分、仅修改 14.4% 控制步、direct 累计约 11.3 亿 Token、
  hybrid 约 6.25 亿；接触类任务（搭塔/杠牌）direct 近乎失败，π0.5 显著补足。

本版把这套**架构机制**在引擎 6 维状态契约 `[px,py,pz,vx,vy,vz]` 上做成可
复跑原型，而不是复刻其仿真器或数字。

## 2. 模块
`udos7/embodied/`
- `env.py`：三维点质量“末端”双积分环境（已知模型），支持**有序目标**、
  **球形障碍接触**（碰撞可设为终止失败）、**中途扰动**；任务套件
  reach_free / ordered_sort / contact_gate / disturb_recover。
- `hybrid.py`：
  - `MotionPrior`（π0.5 类比）：对当前目标提 K=32 条短动作段（PD 机动 +
    横向避让采样），段长 6 步；
  - `SemanticCritic`（Astra 类比，**确定性规则，非 LLM**）：对每条候选在
    克隆环境上 rollout，按顺序进展/到达/碰撞/努力度打分；hybrid 额外加入
    1 条“语义接管”短程序（3 步直奔目标、无避让先验），argmax 决定采纳或接管；
  - `run_episode`：闭环“观察→提候选→裁决→执行一段→重新观察”，输出成功率、
    分数、碰撞、**接管率**、Token 代理量，并接可观测 Tracer
    （`motion.propose` / `semantic.critic` span 与接管事件）。
- `benchmark.py`：三模式 × 四任务 × N 回合配对对比，证据等级 `cpu-proto`，
  外部报告数字单列 `external_reference`（`unverified`）。

## 3. 实测结果（本机 CPU，seed 1000–1009，每任务 10 回合，可复现）

| 任务 | direct 成功率/分 | motion 成功率/分 | hybrid 成功率/分 |
|---|---|---|---|
| reach_free 自由到达 | 100% / 100 | 100% / 100 | 100% / 100 |
| ordered_sort 有序分拣（语义顺序） | 100% / 100 | **0% / 22.1** | 100% / 100（接管 17%） |
| contact_gate 接触过门（精细控制） | **0% / 7.9** | 100% / 100 | 100% / 100（接管 0%） |
| disturb_recover 扰动恢复 | 100% / 100 | 10% / 56.2 | 100% / 100（接管 2%） |
| **汇总** | **75% / 77.0**，碰撞 0.25 | **53% / 69.6**，碰撞 0 | **100% / 100**，碰撞 0，接管 **5%** |

Token 代理总量（**proxy，非真实计费**）：direct in/out 616k/182k，
hybrid 331k/40k，motion 0。方向上与外部报告一致：**hybrid 用更少语言模型
调用、绝大多数步骤交给动作先验，仅在关键节点语义接管**（本原型 5%，
外部报告 14.4%，量级同阶）。

结论（仅在本合成环境成立）：语义层强在任务顺序/状态判断/扰动恢复，动作
先验强在障碍接触；二者互补，hybrid 拿到两边能力，验证了报告的核心机制
主张，而非其绝对数值。

## 4. 证据边界与闸门
- 本版为**已知模型的点质量控制基准**：无学习型 VLA、无真实接触动力学、
  无图像；语义裁判是确定性规则，不是 LLM。等级 **cpu-proto**。
- 闸门（代码留口，不以 CPU 数字冒充）：
  1. LLM 语义裁判：需多供应商 API key，届时以真实 usage 替换 token proxy，
     并做来源脱敏与多模型交叉；
  2. MuJoCo / MuJoCo-MJX 接触、双臂与夹爪、真机迁移：需 GPU/HPC；
  3. 学习型动作先验（π 类）与世界模型打分：需训练管线与 GPU。
- 外部报告数字属第三方口径，仿真器/任务/初始条件不同，**禁止与本原型数字
  直接比较或相互背书**。

## 5. 运行
```bash
python scripts7/embodied_hybrid_demo.py        # 默认每任务 10 回合
udos demo embodied                              # 或用 CLI 菜单/命令
pytest tests7/test_v737_hybrid.py -q           # 9 条契约/互补性测试
```
产物：`reports7/embodied_hybrid_demo.json`（含 raw_rows）、
`reports7/embodied_hybrid_demo.traces.jsonl`。
