# ACT (Action Chunking with Transformers) — 架构卡片

## 1. 身份与证据等级
- 对象：ACT（ALOHA 双臂遥操作，chunked action prediction，MIT）。
- 证据等级：**inferred**（公开实现 `tonyzhaozh/act` 与 LeRobot 移植版；本次未逐文件复核）。
- UDOS 分级：L1（仅动作分块格式适配，不跑 Transformer 权重）。

## 2. 入口与调用链（inferred）
- 训练/推理：policy 一次性预测 `chunk_size` 步动作（action chunking）+ 可选 temporal ensemble。
- 输入：观测（图像 + 本体 state/proprio）；输出：`[chunk_len, action_dim]` 动作块。

## 3. 数据 / 动作格式
- 关键表示：动作块 `[H, action_dim]`；UDOS 适配器把 chunk 归一化为 canonical 轨迹。

## 4. 运行时装配
- L1：`ActionChunkingConnector.to_udos({chunk/actions, proprio})` → `normalize_trajectory`。无权重。

## 5. 模块责任（UDOS 映射）
- → `udos/connectors/specialized.py::ActionChunkingConnector`；对照 wla 动作分块设计。

## 6. 最小反例 / 不可证项
- 反例：payload 缺 `chunk` → ValueError。
- 不可证：未下载权重/数据集，不复刻 ACT 前向。
