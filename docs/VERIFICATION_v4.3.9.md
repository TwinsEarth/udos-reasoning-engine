# UDOS v4.3.9 验证报告 — 完全自进化线（自进化飞轮 + 基础设施自优化缩微类比）

> 线：v4.2.9 → v4.3.9（10 节点，全局终点）　日期：2026-09-16
> 性质：analogy, not reproduction。收口智谱"完全自训练"三维之**基础设施自我优化**
> （数据自产=4.2 / 环境自造=4.1 / 基础设施自优化=4.3）。全部 CPU 合成、外挂零梯度、默认 opt-in。

---

## 1. 硬性验收结果

| 项 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 全量测试 | 只增不删，线末全量回归报真实总数/覆盖率 | **1359 passed / 0 failed / 0 error**（基线 1336 → +23），覆盖率 **93%**（9738 stmts / 686 miss） | ✅ |
| 主参 | 外挂不入主 state_dict，主参恒 52191 | 52191（所有 checkpoint 加载后校验） | ✅ |
| 正式训练 eval_mse | 2 次正式训练 eval_mse=0.045556 | 4.3.0=0.045556、4.3.9=0.045556（与锚点逐位一致） | ✅ |
| checkpoint 代际 | 31 旧代全可加载，新增 2 件共 33 代向后兼容 | **33 代**（v2.1.0..v4.3.9）逐件可加载、主参恒 52191 | ✅ |
| 旧锚点逐位未变 | 锚点权重 md5 不变 | v3.8.6=7351250…、v3.4.5=52993ca7…、v3.3.3=f993bcb…（与 v4.2.9 基线逐位相同） | ✅ |
| 2 次正式训练 | 4.3.0、4.3.9 | predictor_v4.3.0.pt（第 32 代）、predictor_v4.3.9.pt（第 33 代） | ✅ |
| 新模块日志 | logging_config stderr，不污染 stdout/HTTP/metrics | 沿用 `logging.getLogger("udos.self_evolution")`；端点响应不含日志；`/metrics` 纯文本 | ✅ |
| HTTP 语义 | 400/404/409/500 不崩进程，新端点逐路径打正常+异常 | `/self-evolution/{search,ab,long-horizon}` 各 200/400/404 全绿 | ✅ |
| 版本号全来源同步 | __init__/pyproject/Makefile/Dockerfile/compose + 测试断言 | bump 脚本统一 4.2.9→4.3.0→4.3.1→4.3.2→4.3.9 | ✅ |
| CHANGELOG | 增补 10 条（累计 42 条：12+10+10+10） | v4.3 节新增 10 条 | ✅ |

## 2. 关键机制实测（全部来自 benchmarks/results/*.json，可复算）

### 2.1 配置自优化收益（4.3.0 / dev2）
- 固定基准负载（24 窗口 × repeats=3），36 个网格候选**全部过"输出保真"硬门**（max_abs_diff ≤ 1e-4）。
- 默认成本代理 = 8.0；搜索选中配置 = {max_shard=64, cache_enabled=True, cortex_hz=1.0, codebook_size=4}，成本 = 3.5。
- **A/B：A_wins=True，降本 56.25%，预测输出逐位/容差内保真**（缓存开→重复请求命中；cortex_hz↓→规划调用↓；codebook↓→存储成本↓）。
- 证据：`benchmarks/results/training_v4.3.0.json`、`training_v4.3.9.json`、`self_evolution_v43.json`。

### 2.2 自进化多代曲线 / 真改进 vs 退化（dev4 / dev5）
- 2 代成本曲线 = [3.5, 3.5]（确定性网格下 best 逐代稳定），honest_verdict = `efficiency_improving`。
- 全局停止/纠正判据：成本无进一步下降 → 推荐 `stop_self_evolution`（收益递减）。
- **诚实标注**：不宣称飞轮必然持续上升；曲线照实记录，若代际失真会判 `collapsed` 并回滚。

### 2.3 长程任务闭环（dev6）
- horizon=4 拆 2 子目标，主预测器 rollout 4 步，轨迹全有限、verified=True、错误恢复 0 次。
- 证据：`self_evolution_v43.json` 的 `dev6_long_horizon`。

## 3. 被否决 / 降级 / 候选账本
- 本线无"靠降质换速度"的候选被采纳：所有候选先过输出保真硬门，未通过即 REJECT（本负载下 36 候选全过，故无 REJECT 记录）。
- 沿自上一代（4.2）的诚实纪律：teacher→student 崩塌、自产数据 A/B 落败均维持 opt-in，不在本线翻案。
- 多代曲线无新改进时全局判 `stop_self_evolution`（见 §2.2）。

## 4. 调研覆盖度（docs/SELF_TRAINING_EVOLUTION_RESEARCH.md）
- 智谱完全自训练三维（数据自产/环境自造/基础设施自优化）RSI 定义：**已对齐**，引 HKEX 2026-09-13 公告原文释义。
- 公告事实：配售 714 港元 + RMB201.4 亿零息可转债、60/15/25 用途切分——**已核**。
- **"1GW 国产算力""收购中科加禾"在本可核附件中查无实据 → 标 [UNVERIFIED]**，未采信。
- AI-Scientist（arXiv 2408.06292）/ShinkaEvolve（arXiv 2509.19349）/世界模型自产数据：已做机制类比映射，未编造作者/DOI/指标。

## 5. 仍存缺口
- 配置成本为**确定性代理**（前向当量/规划调用/码本规模），非壁钟 ms；CPU 壁钟有噪声，仅记录不参与门控。
- 集成权重（ensemble_weights）为 knob 维度，单预测器下为均匀 no-op；多成员集成 A/B 未在终件展开（避免构造多成员小模型的额外开销）。
- 自进化为有界玩具类比（冻结主权重），不构成端到端 RSI 复现。
