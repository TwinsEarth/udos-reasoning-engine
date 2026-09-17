# Legacy（udos/ v5.5.5）冻结与迁移说明

## 关系

- `udos/`（v5.5.5）：**冻结 legacy**，保留可运行、可对照、全部既有测试通过，但不再演进。
- `udos7/`（v7.0.2）：**主线**，单一统一推理图。发行版 7.0.2 同时包含两者：
  - 分发包版本（pyproject）= 7.0.2；
  - `udos.__version__` 保持 5.5.5（不动，避免破坏 legacy 版本断言）；
  - `udos7.__version__` = 7.0.2。

## checkpoint 不兼容（有意为之）与 v7 内部兼容

v7 模型结构（WorldModelCore：观测编码 + 场景通道 + 唯一 GRU + 残差解码）
与 legacy `PhysicsPredictor`（CTM 多 tick + 独立 GPM/LoRA 演示链）不同，
**旧 legacy checkpoint 不能、也不应被 v7 加载**。v7 checkpoint：

```
checkpoints7/worldmodel_v7.0.2.pt   # hidden256, 967,182 参数, use_kinematics=True（默认）
checkpoints7/worldmodel_v7.0.1.pt   # hidden256, 966,830 参数, 无运动学通道（保留对照）
```

**v7.0.1 → v7.0.2 向后兼容**：v7.0.2 新增的运动学编码器由 config 的
`use_kinematics` 控制；加载 v7.0.1 旧 checkpoint（config 无该字段）时
`load_worldmodel` 默认 `use_kinematics=False`，不构建 `kin_encoder`，
state_dict 不错位（有 `test_persistence_roundtrip_and_legacy_config_compat` 守护）。

加载方式：

```python
from udos7.persistence import load_worldmodel
model, ckpt = load_worldmodel("checkpoints7/worldmodel_v7.0.2.pt")
```

旧 `checkpoints/predictor_v4.3.9.pt`（52,191 可学习参数 + 4 个 buffer）仅作为
**被击败的基线**保留，其 held-out 指标与“conformal α 失效（三档覆盖全 0.8988）”
证据见 `scripts/v7/baseline_v4.3.9_heldout_seed2026.json`。

## 从 v5.5.5 迁移调用方

| v5.5.5（legacy） | v7.0.2 |
|---|---|
| `PhysicsPredictor.predict(window, scene_params)` | `WorldModelCore.forward(window, explicit=P)` |
| `rollout(window, H, ...)` | `WorldModelCore.rollout(window, H, explicit=P)` |
| GPM 生成 LoRA 注入 TinyBaseModel（死路） | **删除**；场景经 SceneChannel 真实求和进前向 |
| 盲估计“带符号标量 accel_a”（skill 为负） | 确定性三维加速度向量 `kinematic_features`（skill=1.0）+ 学习估计器并存 |
| 演示 CTM + 训练 CTM 两套 | **唯一 GRU 时序核** |
| 旧 conformal（α 失效） | split conformal，按 α 真分水平 + 扇形包络 |
| 无 HTTP 版本前缀 | `/api/v7/*`（校准器按 `(use_explicit, horizon)` 缓存） |

## 弃用清单（v7 已删除，测试断言不得复活）

LoRA 注入链、TinyBaseModel、未训练演示 CTM、base_model 前向补丁、
多套并行预测器、与名义 α 不挂钩的区间宽度。
