# RoboMimic / MimicGen — 架构卡片

## 1. 身份与证据等级
- 对象：RoboMimic（数据集/算法库）+ MimicGen（自动数据生成），MIT。
- 证据等级：**inferred**（公开 `ARISE-Initiative/robomimic`；HDF5 demo 结构，本次未逐文件复核）。
- UDOS 分级：L1（HDF5 episode schema 适配，不读大 HDF5）。

## 2. 入口与调用链（inferred）
- 数据：`hdf5_dataset.hdf5`，顶层 `data/<demo_id>` 含 `obs/`、`actions`、`dones`、`model`（XML）。
- 读取：`h5py.File(...) → demo['actions'][:]`。

## 3. 数据 / 动作格式
- 每 demo：observations dict + actions `[T, action_dim]`。

## 4. 运行时装配
- L1：`EpisodeSchemaConnector`（用最小 step fixture 代表 obs/action schema）。

## 5. 模块责任（UDOS 映射）
- → `EpisodeSchemaConnector`。

## 6. 最小反例 / 不可证项
- 不可证：未下载 HDF5；不实现 h5py 真实读取（仅 schema 适配）。
