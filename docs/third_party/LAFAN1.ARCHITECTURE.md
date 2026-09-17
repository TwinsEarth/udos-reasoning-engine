# LAFAN1 / Unitree LAFAN1_Retargeting — 架构卡片

## 1. 身份与证据等级
- 对象：LAFAN1（动画 BVH 运动捕捉数据集）；Unitree LAFAN1_Retargeting（重定向到宇树人形）。
- 证据等级：**inferred**（LAFAN1 标准 BVH；Unitree 重定向为公开数据集；本次未逐文件复核 BVH）。
- UDOS 分级：L1（BVH 片段格式适配，不下载大 BVH）。

## 2. 入口与调用链（inferred）
- BVH：`HIERARCHY`（骨骼树 + channel）+ `MOTION`（每帧各关节旋转数值行）。
- 重定向：源骨骼 → 目标人形本体关节映射。

## 3. 数据 / 动作格式
- `frames: List[List[float]]`（每帧通道）+ `frame_time`。

## 4. 运行时装配
- L1：`BVHClipConnector.to_udos({frames, joints, frame_time})` → canonical clip。

## 5. 模块责任（UDOS 映射）
- → `BVHClipConnector`。

## 6. 最小反例 / 不可证项
- 反例：缺 `frames` → ValueError。
- 不可证：未下载真实 BVH；骨骼/通道数以 fixture 代表；重定向映射未真跑。
