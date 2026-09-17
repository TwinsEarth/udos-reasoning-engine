# UDOS 推演引擎 v3.4.5 验证报告（ICM 上下文记忆大版本收尾）

> 终点版本：**v3.4.5**（20 代 checkpoint 谱系 v2.1.0..v3.4.5）
> 正式件：`checkpoints/predictor_v3.4.5.pt`（52191 参数，与 v3.4.0/v3.3.3 同配方，eval_mse 逐位一致 0.045556）
> 性质：12 迭代节点全部真实落地代码+测试+CHANGELOG；analogy, not reproduction；CPU-only 2 线程。

## 1. 测试与覆盖率

- 全量 pytest：**846 passed / 0 failed / 0 errors / 0 skipped**（基线 v3.3.4 = 766，净增 80 个用例）；覆盖率 **93%**（ICM 各模块：icm.py 87% / icm_events.py 86% / icm_cross.py 98% / icm_budget.py 95%）。
- 新测试文件与条数：
  - test_v34_icm_core.py（13）
  - test_v34_events.py（9）
  - test_v34_cross_embodiment.py（7）
  - test_v34_pce_prompt.py（8）
  - test_v34_budget.py（8）
  - test_v34_shot_scaling.py（5）
  - test_v34_three_route.py（3）
  - test_v341_integration.py（8）
  - test_v342_service.py（3）
  - test_v343_edge.py（10）
  - test_v344_integration.py（3）
  - test_v345_final.py（3）
- 兼容铁律：旧 766 测试只增不删，全绿；默认输出/数值逐位等价有锚点测试守护。

## 2. 核心主张的正反证据

### 2.1 ICM 检索+聚合 vs naive 朴素拼接（k-shot 不退化）
- **naive（反面，被 REJECT）**：`naive_k0=0.093 → k1=2.30 → k3=4.19`，随 k 爆炸。根因=朴素拼接 k·W 帧原始窗口稀释 52k 小模型注意力。
- **ICM（正面，ACCEPT）**：`icm_k0=0.045 → k1=0.023 → k3=0.015 → k5=0.014`，单调改善、~5-shot 饱和。
- 三路线对照中，**上下文 Scaling（ICM）单位成本收益最高**：mse 0.057→0.019 而延迟几乎不增。

### 2.2 零梯度原则（权重逐位不变）
- ICM 全程 `@torch.no_grad`，聚合器可训参数=0，不入主 state_dict。
- `zero_grad_state_dict_md5_unchanged=true`：ICM 推理前后主 `state_dict` md5 逐位一致。
- 主件重训仅 3.4.0 / 3.4.5 两次，同配方 eval_mse 逐位一致（0.045556），n_params 恒为 52191。

## 3. 被回退/否决候选清单（诚实账本）

| 候选 | 证据 | 处置 |
|---|---|---|
| naive 朴素拼接 few-shot ICL | k0=0.093 → k3=4.19，退化 | **REJECT**，改残差空间检索+聚合 |
| 结构化剪枝不重训（历代） | mse 2.84 | REJECT，opt-in |
| 蒸馏学生（历代） | mse 2.63 | REJECT，opt-in |
| budget>8 继续加大 | k 上限后 mse 不再降、延迟不增 | 收益边际递减，记录不强行扩容 |

## 4. Checkpoint 兼容与权重变动说明

- **20 代** checkpoint（v2.1.0..v3.4.5）全部可 `load_predictor`，预测输出有限。
- save/load 逐位一致：v3.4.5 经临时文件 round-trip 后 state_dict md5 不变。
- 权重变动：主架构/参数量（52191）与 v3.3.3 完全一致；ICM 为外挂零参数模块，不进入 state_dict。

## 5. HTTP 端点矩阵（真实起端口验证）

| 端点 | 场景 | 期望 | 实测 |
|---|---|---|---|
| POST /icm/predict | 未注册演示 | 409 | ✓ |
| POST /icm/predict | 注册后 k=2 | 200 | ✓ |
| POST /icm/demo/register | 缺 result | 400 | ✓ |
| POST /icm/predict | k<0 | 400 | ✓ |
| POST /icm/nonexistent | 未知路由 | 404 | ✓ |

## 6. 性能（2 线程 CPU）

- 纯 0-shot predict_next：3.34 ms；ICM k=3：3.72 ms（+0.4 ms）。
- 事件切分 0.08 ms / 三流对齐 0.10 ms / PCE 解析 0.21 ms。

## 7. 已知限制

- 合成数据类比，非复现视频 VLA / 大规模预训练；跨本体用 DOF 数差异代理，非真实 URDF。
- ICM 残差收益依赖同分布演示库；跨域检索收益不稳时已 opt-in（λ 收缩最坏退化为 0-shot）。
- CPU-only 2 线程，未测 GPU/边缘实时吞吐上限。
