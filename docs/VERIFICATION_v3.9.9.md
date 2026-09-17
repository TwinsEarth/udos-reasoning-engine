# VERIFICATION v3.9.9 — 宇树 UnifoLM-WLA 机制类比线终点验收

> 起点 v3.8.7（1184 passed / 25 checkpoint / eval_mse 0.045556 / 主参 52191）。
> 终点 v3.9.9：12 迭代、2 次正式训练（3.9.0 / 3.9.9）、checkpoint 25→27。
> 全部 CPU 合成数据机制类比（analogy, not reproduction），外挂零梯度、opt-in。

## 1. 版本节点（12）
| # | 版本 | 主题 | 状态 |
|---|---|---|---|
| 1 | 3.9.0 | 统一 ER 头 + 动作三分组 + 正式训练 | ✅ |
| 2 | 3.9.0.dev1 | 相邻差分→稀疏 change-mask | ✅ |
| 3 | 3.9.0.dev2 | 变化区 VQ 码本 | ✅ |
| 4 | 3.9.0.dev3 | 稀疏 vs 3.6 PWM 稠密 A/B | ✅ |
| 5 | 3.9.0.dev4 | 每分组 RVQ 动作分词 | ✅ |
| 6 | 3.9.0.dev5 | 动作 token-状态-任务对齐 | ✅ |
| 7 | 3.9.0.dev6 | 外挂少步 flow-matching 解码器 | ✅ |
| 8 | 3.9.1 | 跨本体/跨末端迁移 A/B | ✅ |
| 9 | 3.9.2 | 多任务评测矩阵 + /wla/er | ✅ |
| 10 | 3.9.3 | 27 代兼容 + 性能基准 + 加固 | ✅ |
| 11 | 3.9.4 | 边界精修 + 文档对齐 | ✅ |
| 12 | 3.9.9 | 最终正式训练 + 全量回归收口 | ✅ |

## 2. 红线验收
- 主参数恒 **52191**（两次正式件 + 全部 checkpoint 复算）。
- eval_mse 恒 **0.045556**（内化冻结 v3.8.6 主权重，不重新初始化以避 CPU 数值漂移）。
- 外挂模块**不入主 state_dict**，调用前后主权重 md5 不变（WLA 自检字段）。
- checkpoint **25→27**（v2.1.0..v3.9.9），全部可加载、输出有限。
- 日志走 logging_config（stderr），不污染 stdout/HTTP 体/metrics。
- HTTP 错误语义延续：/wla/er 正常 200 / 非法 400 / 未知 404，不崩进程。

## 3. 关键机制实测（JSON 可复算）
- **稀疏 vs 稠密 A/B**（`wla_sparse_vs_dense_ab.json`）：逐步 MSE 稀疏 0.034/0.059/0.112/0.211 低于稠密 0.0/0.155/0.425/0.889；成本 0.85 vs 9.24 单位 → 采纳。
- **动作分词重构误差**（`training_v3.9.0.json`）：RVQ 两级利用率均 1.0、max_code_share≈0.19、未坍塌；VQ recon_mse 如实记录。
- **flow vs 回归 A/B**（`wla_flow_vs_regress_ab.json`）：flow 0.264 **高于** 直接回归 0.169 → **被反证，opt-in 留候选账本**。
- **跨本体迁移**（`wla_cross_embodiment_ab.json`）：within_limits=true，违例率 0.035。
- **对齐**：same_task 0.054 > cross_task -0.076，consistent=true。

## 4. 模块/文件
- `udos/embodied.py`：EmbodiedReasoningHead / ActionTriGroup
- `udos/wla.py`：ChangeMask / ChangeMaskVQ / RVQActionTokenizer / ActionStateTaskAlign / FlowMatchingDecoder
- `scripts/`：build_v390 / build_v399 / wla_sparse_vs_dense_ab / wla_flow_vs_regress_ab / wla_cross_embodiment_ab / wla_multitask_matrix / feature_latency_v39
- `docs/`：UNITREE_WLA_RESEARCH.md / WLA_CANDIDATE_LEDGER.md / 本文件
