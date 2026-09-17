# AMASS / SMPL-X — 架构卡片

## 1. 身份与证据等级
- 对象：AMASS（多数据集 SMPL/SMPL-X 参数化人体语料；同线 GRAB/HumanPlus/OmniH2O/ExBody）。
- 证据等级：**inferred**（SMPL 标准 axis-angle 参数；本次未逐文件复核）。
- UDOS 分级：L1（SMPL 姿态表示适配，不下载 SMPL 模型/数据）。

## 2. 入口与调用链（inferred）
- 姿态：`body_pose` axis-angle `[N*3]` + `global_orient`；SMPL 前向蒙皮依赖 skinned model。
- 表示转换：axis-angle → 旋转 6D（retargeting 友好）。

## 3. 数据 / 动作格式
- `body_pose: List[float]`（长度 3 的倍数）→ rot6d。

## 4. 运行时装配
- L1：`SMPLPoseConnector.to_udos`：Rodrigues 把每轴角转旋转矩阵前两列（rot6d），零角返单位旋转 6D。

## 5. 模块责任（UDOS 映射）
- → `SMPLPoseConnector`；覆盖 AMASS/GRAB/HumanPlus/OmniH2O/ExBody。

## 6. 最小反例 / 不可证项
- 反例：body_pose 长度非 3 倍数 → ValueError。
- 不可证：未跑 SMPL 蒙皮；rot6d 仅数学转换，不渲染网格。
