# yourdfpy — 架构卡片

## 1. 身份与证据等级
- 对象：`yourdfpy`（Python URDF 解析 + 正运动学库，Apache-2.0）。
- 证据等级：**confirmed**（本次直接读取 `~/.local/lib/python3.12/site-packages/yourdfpy/` 已安装真实源码 `urdf.py` / `__init__.py`，并真实跑通 FK）。
- 作用域：仅确认 UDOS L2 connector `URDFFKConnector` 实际调用到的 API 面。

## 2. 入口与调用链（confirmed）
- 包导出（`yourdfpy/__init__.py`）：`from .urdf import URDF, Joint, Link, ...`。
- 加载：`yourdfpy.URDF.load(fname_or_file, build_scene_graph=True, load_meshes=True)`，入参可为路径或 file-like 对象（`urdf.py:936`）。
- 零位：`URDF.zero_cfg`（property，返回 `np.zeros(num_dofs)`，`urdf.py:772`）。
- 正运动学：`URDF.update_cfg(configuration)`（"Update joint configuration of URDF; does forward kinematics."，`urdf.py:1087`；configuration 接受 dict/list/tuple/ndarray，维度不符抛 ValueError）。
- 取变换：`URDF.get_transform(frame_to, frame_from=None)`（返回 **(4,4) 齐次矩阵**，`urdf.py:1141`；scene graph 未建时抛 ValueError）。

主链：`URDF.load(urdf_bytes) → robot.update_cfg(robot.zero_cfg) → robot.get_transform("tip") → (4,4) ndarray`。

## 3. 数据 / 动作格式
- 输入：URDF XML（link/joint/origin/axis/limit）。
- 输出：关节配置 `[num_dofs]` ndarray；末端位姿 `(4,4)` 齐次矩阵。

## 4. 运行时装配
- L2：UDOS 惰性 `importlib.import_module("yourdfpy")`；`_smoke` 对内联单关节 URDF 跑 `load→update_cfg→get_transform`，成功则 status=available。
- 缺包：probe → absent（不崩核心 import）。

## 5. 模块责任（UDOS 映射）
- → `udos/connectors/specialized.py::URDFFKConnector`。
- 负责把 URDF 关节角归一化为 canonical joint states（L1 面无包也可用），并在装包后提供真实 FK。

## 6. 最小反例 / 不可证项
- 反例：configuration 维度 ≠ num_dofs 时 update_cfg 抛 ValueError（已读签名确认）。
- 不可证：**未**验证多自由度/碰撞体 scene graph、mesh 加载路径；未验证 IK（yourdfpy 本身不提供解析 IK）。
