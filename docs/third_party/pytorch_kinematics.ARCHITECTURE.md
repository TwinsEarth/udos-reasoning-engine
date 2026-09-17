# pytorch_kinematics — 架构卡片

## 1. 身份与证据等级
- 对象：`pytorch_kinematics`（基于 PyTorch 的 URDF/SDF 运动学，支持 batch FK/IK，BSD/MIT 类宽松许可）。
- 证据等级：**inferred（公开仓库 `readdy/pytorch_kinematics`；本环境未 pip 安装，未本次逐文件复核源码）**。
- 本环境状态：**absent**（`pip install` 未执行，惰性 import 探测未命中；不破坏核心）。

## 2. 入口与调用链（inferred）
- 典型入口：`pytorch_kinematics.build_robot_from_urdf(urdf_path)` → `Robot` 对象。
- FK：`robot.forward_kinematics(joint_angles)` → 各 link 的 `Transform3d`（batch 张量）。
- IK：`robot.inverse_kinematics(...)`（基于优化）。
- 主链（inferred）：`build_robot_from_urdf → forward_kinematics(joints) → link poses (torch.Tensor)`。

## 3. 数据 / 动作格式
- 输入：URDF + batch 关节角张量 `[B, n_dof]`。
- 输出：刚体变换张量（se3/4x4）。

## 4. 运行时装配
- L2：UDOS `L2_PKG` 声明 requires_pkg=pytorch_kinematics；未装 → absent。装后可与 yourdfpy 互为 FK/IK 后端。

## 5. 模块责任（UDOS 映射）
- → `udos/connectors/specialized.py::URDFFKConnector`（候选 L2 FK/IK 后端）。

## 6. 最小反例 / 不可证项
- 不可证：本环境未安装，实际 API 签名/IK 是否可在 CPU 跑通未验证；装后需补 smoke。
