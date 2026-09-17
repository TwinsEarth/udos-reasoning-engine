# LeRobot (lerobot datasets) — 架构卡片

## 1. 身份与证据等级
- 对象：HuggingFace LeRobot（机器人框架 + 社区数据集，Apache-2.0）。
- 证据等级：**inferred**（公开 `huggingface/lerobot`；parquet-based dataset v2/v3 schema，本次未逐文件复核）。
- UDOS 分级：L1（episode/step schema 适配，不下载 parquet 数据集）。

## 2. 入口与调用链（inferred）
- `LeRobotDataset`：每个 episode 由 parquet step 表组成，列如 `observation.images.<cam>`、`observation.state`、`action`、`timestamp`。
- step → trajectory：按时间戳聚合 state/action。

## 3. 数据 / 动作格式
- step 列表 `[{state, action, timestamp}, ...]`。

## 4. 运行时装配
- L1：`EpisodeSchemaConnector.to_udos({steps:[...]})` 解析点分列名 → canonical 轨迹。

## 5. 模块责任（UDOS 映射）
- → `EpisodeSchemaConnector`；Open-X/DROID/Bridge/RH20T/LIBERO/CALVIN 复用同一 schema。

## 6. 最小反例 / 不可证项
- 反例：缺 `steps` → ValueError。
- 不可证：未读真实 parquet；列名以 fixture 代表。
