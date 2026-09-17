# UDOS v3.8.6 验证报告 —— 全域调度 / 数字孪生 / 多体协同与收口（终点）

> 起点 v3.7.3，终点 **v3.8.6**（对外发布）。主 predictor 重训但主参恒 **52191**，
> `eval_mse = 0.045556`（与 v3.8.0 / v3.7.0 逐位同口径）。多体为合成参数化代理、
> 数字孪生为合成场景（analogy, not reproduction）。

## 1. 全量测试与覆盖率

- **全量 pytest：1160 passed，EXIT=0**（v3.7.3 基线 1085 + 3.8 线新增 75）。
- **覆盖率：93%**（TOTAL 8002 语句 / 566 未覆盖）。
- 3.8 线新模块覆盖率：
  - `udos/multi_agent.py` 92%
  - `udos/wm_scheduler.py` 98%
  - `udos/closed_loop.py` 98%
  - `udos/digital_twin.py` 100%

## 2. 10 迭代交付与各线正反证据

| # | 版本 | 交付 | 关键实测 |
|---|------|------|----------|
| 1 | 3.8.0 | `multi_agent.py` + 训练 v3.8.0.pt | eval_mse 0.045556；3 体冲突 1 对，高优先保持/低优先减速 |
| 2 | 3.8.0.dev1 | `wm_scheduler.py` | 5 体低预算 5/5 全回退真实；高预算 used=15≤cap |
| 3 | 3.8.0.dev2 | `closed_loop.py` | 反射触发 winner=spinal；反馈 L1≥0；零梯度 md5 不变 |
| 4 | 3.8.0.dev3 | `digital_twin.py` | 同 seed 逐位可复现；快照/回放状态逐位一致 |
| 5 | 3.8.1 | 多体 A/B | 对穿碰撞事件 1/6/28 → 0（降 100%）；让行减速致到达率 0（诚实权衡，opt-in） |
| 6 | 3.8.2 | HTTP /twin/step、/twin/scene | 200/400/409/404 全矩阵 |
| 7 | 3.8.3 | 24 代兼容 + 延迟基准 | 见 §6 |
| 8 | 3.8.4 | 全特性综合评测 | 7 特性同 predictor 叠加不冲突、零梯度 |
| 9 | 3.8.5 | 边界加固 + 文档对齐 | 8 边界用例全绿 |
| 10 | 3.8.6 | 最终重建 v3.8.6.pt + 25 代兼容 | eval_mse 0.045556；25 件 |

### 被否决 / 回退候选（诚实记录）
- **协调器默认开启**：被否决。A/B 显示让行减速使对穿场景到达率由 1.0 降至 0.0（安全换速度），收益不稳 → 保持 **opt-in**，不显式调用即无让行。
- **WM 调度轮转式分配**：被否决。初版轮转分配把预算均摊，高优先体未获更多想象；改为**集中式贪婪**（高优先优先占满 max_horizon），语义更清晰。
- **dev 版本号进 checkpoint 文件名**：被否决。dev 节点仅改 `__version__`，不产生 dev checkpoint；正式件只有 v3.8.0 / v3.8.6 两件，backcompat 列表干净。
- **既有 backcompat 精确计数 `==23`**：在新增 checkpoint 前改为下界 `>=23`，避免误报；最终由 v3.8.3/v3.8.4 用 `==24/==25` 重新锚定真值。

## 3. 兼容性（25 代 checkpoint）

- `checkpoints/predictor_v*.pt` 共 **25 件**（v2.1.0 .. v3.8.6）。
- `test_v383_service.py` / `test_v384_eval.py`：25 件全部 `load_predictor` 成功，`predict_next` 输出有限，主参恒 **52191**。
- 正式件 `predictor_v3.8.6.pt` 元数据 `udos_version == "3.8.6"`；`save/load` 重载 `eval_mse` 与训练时逐位一致（`reload_consistent=true`，差值 < 1e-9）。

## 4. HTTP 端点韧性矩阵（真实起服务，v3.8.6.pt 挂载）

| 请求 | 预期 | 实测 |
|------|------|------|
| GET /health | 200 | 200 |
| POST /twin/step（未建场景） | 409 | 409 |
| POST /twin/scene（查询，未建） | 409 | 409 |
| POST /twin/scene（create） | 200 | 200 |
| POST /twin/scene（query） | 200 | 200 |
| POST /twin/step | 200 | 200 |
| POST /twin/step（带 window，附跑闭环） | 200 | 200 |
| POST /twin/scene {n_agents:0} | 400 | 400 |
| POST /twin/step {dt:-1} | 400 | 400 |
| POST 非法 JSON | 400 | 400 |
| POST /twin/nope（未知路由） | 404 | 404 |
| 未捕获异常 | 500 | 框架兜底不崩进程 |

错误语义沿用全工程：ServiceNotReady→409、ValueError→400、未知路由→404、异常→500。

## 5. 零梯度 / 外挂可证

- 全部 3.8 模块为纯推理外挂：不进主 state_dict、不改主 52191 参数。
- `build_v386_checkpoint.py` 离线自检：ICM / PWM / 分层控制 / 多体 / 闭环 调用前后主权重 md5 全部不变（`zero_grad_state_dict_md5_unchanged=true`）。
- 多体协调器、WM 调度、孪生生成均为**确定性算法**，无可训参数（`learned=false`）。
- 默认输出逐位等价：不构造新模块时，旧推理路径与全部旧端点不变（opt-in）。

## 6. 分层控制 / 多体延迟预算（合成可测，CPU 2 线程）

`benchmarks/results/feature_latency_v3.8.0.json`：

| 组件 | 单次延迟 |
|------|----------|
| baseline predict_next | ≈3.05 ms |
| 闭环一步（含 WM 想象） | ≈3.91 ms |
| 数字孪生一步（8 体） | ≈0.61 ms |
| 多体冲突消解（8 体） | ≈0.22 ms |
| WM 调度分配 | ≈0.003 ms |

分层控制（3.7）延迟预算沿用：脊髓 ≪ 小脑 ≪ 大脑。

## 7. 产物清单

- 正式件：`checkpoints/predictor_v3.8.0.pt`、`predictor_v3.8.6.pt`（共 25 代）。
- 基准 JSON：`training_v3.8.0.json`、`training_v3.8.6.json`、`multi_agent_ab_v3.8.0.json`、
  `wm_budget_ab_v3.8.0.json`、`feature_latency_v3.8.0.json`、`comprehensive_eval_v3.8.0.json`。
- 新模块：`udos/multi_agent.py`、`udos/wm_scheduler.py`、`udos/closed_loop.py`、`udos/digital_twin.py`。
- CHANGELOG：3.5–3.8 线累计 **40 条**（3.8 线 10 条）。

## 8. 已知限制

- 多体为**合成参数化代理**，非真机多机器人；冲突消解为确定性几何规则，不含学习。
- 数字孪生为**合成场景**（seed 确定性），非真机物理/实时总线。
- WM 想象 H>1 步为潜在空间外挂转移，精度随 horizon 累积偏差（H=1 逐位锚定真实）。
- 协调器在对穿场景降低碰撞但牺牲到达速度，收益场景依赖 → 保持 opt-in。
- CPU-only（torch 2 线程）；延迟为合成 perf_counter 实测，非真机总线。
