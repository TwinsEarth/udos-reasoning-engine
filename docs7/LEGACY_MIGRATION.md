# Legacy（udos/ v5.5.5）冻结与迁移说明

## 关系

- `udos/`（v5.5.5）：**冻结 legacy**，保留可运行、可对照、全部既有测试通过，但不再演进。
- `udos7/`（v7.0.3）：**主线**，单一统一推理图。发行版 7.0.3 同时包含两者：
  - 分发包版本（pyproject）= 7.0.3；
  - `udos.__version__` 保持 5.5.5（不动，避免破坏 legacy 版本断言）；
  - `udos7.__version__` = 7.0.3。

## v7.0.2 → v7.0.3 权重兼容（加性迁移，自动）

v7.0.3 把运动学特征 `KIN_DIM` 10→11（新增 `ca_conf`）并新增解析积分门控小头
`kin_gate`（Sequential 11→32→6，末层零初始化）。`persistence.load_worldmodel`
对 v7.0.2 权重自动做两处加性迁移，加载后预测与 v7.0.2 逐位一致（有测试守护）：

1. `scene.kin_encoder.weight` 由 [32,10] **末列补零**到 [32,11]（新 ca_conf 列贡献 0）；
2. `kin_gate.*` 缺失 → 保留零初始化（门输出 g≡0，等价纯学习残差）。

任何 unexpected 键、或非白名单 missing 键直接 RuntimeError，不静默吞错。

## checkpoint 不兼容（有意为之）与 v7 内部兼容

v7 模型结构（WorldModelCore：观测编码 + 场景通道 + 唯一 GRU + 残差/解析积分混合头）
与 legacy `PhysicsPredictor`（CTM 多 tick + 独立 GPM/LoRA 演示链）不同，
**旧 legacy checkpoint 不能、也不应被 v7 加载**。v7 checkpoint：

```
checkpoints7/worldmodel_v7.0.3.pt   # hidden256, 967,796 参数, use_kinematics=True（默认）
checkpoints7/worldmodel_v7.0.2.pt   # hidden256, 967,182 参数, 加载时零列填充 KIN_DIM 10→11
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
