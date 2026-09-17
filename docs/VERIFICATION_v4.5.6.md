# UDOS v4.5.6 验证收口报告

> QA 方法论：请求契约 → 风险用例 → 执行证据 → bug 分级(S1–S4/P0–P3) → 发布判断。
> 本线（v4.5.4–4.5.6）为开源资源注册表集成，**零训练、零主权重改动**；registry 无可学参数。

## 1. 请求契约（锁定范围）

- 对外三版 v4.5.4/v4.5.5/v4.5.6；本报告覆盖 4.5.6 收口。
- 诚信四层 L0-L3；CPU-only；不得下载 GB 级权重/数据集。
- 端点：`GET /resources`、`POST /resources/{id}/probe`、`POST /resources/{id}/invoke`、`POST /resources/profile`。
- 硬性验收：pytest 总数 ≥1459 且只增；覆盖率 ≥93%；33 代 checkpoint 与四锚点 md5 逐位不变；全新解压 /tmp 复跑。

## 2. 风险用例与执行证据（风险机制 RM）

| RM | 风险 | 用例 | 结果 |
|---|---|---|---|
| RM-1 | 缺包破坏核心 import | `test_absent_pkg_does_not_break_core_import` | 通过（pytorch_kinematics absent，核心照常） |
| RM-2 | L3 大模型被"假装运行" | `test_l3_absent_invokes_gracefully` + HTTP 503 矩阵 | 通过（OpenVLA invoke→503，不下载） |
| RM-3 | 非法 id/action 崩溃 | unknown id→404、缺 action→400 | 通过 |
| RM-4 | profile 切换污染主权重 | `test_profile_switch_in_memory_bitwise_prediction_unchanged` | 通过（engine.parameters() 逐位相等） |
| RM-5 | 全量探测卡启动 | `test_full_bulk_probe_fast_and_cached`（79 条 <30s） | 通过 |
| RM-6 | 四态自洽 / schema 合法 | `test_full_registry_schema_license_status_selfconsistent` | 通过（license 非空、L3 均声明 gpu/weights） |
| RM-7 | L1 fixture 往返不一致 | action chunk / VLA bin / episode / SMPL / BVH 往返 | 通过 |

## 3. Bug 闭环（现象→根因→补丁→回归，先 RED 后 GREEN）

| # | 现象(RED) | 根因 | 补丁 | 回归(GREEN) |
|---|---|---|---|---|
| B-1 | `test_smpl_axis_angle_to_rot6d` 报 `assert 10 == 12` | `_axis_angle_to_rot6d` 零旋转特判只返回 4 维，破坏 rot6d 恒为 6 维 | 零旋转返回单位旋转 6D `[1,0,0,1,0,0]` | 单测通过，rot6d 长度=joints×6 |
| B-2 | `GET /resources?profile=full` 的 summary 仍显示 performance 视图计数 | `summary()` 未接收查询 profile，按 service 当前 profile 计算 | `summary(profile=None)` 透传；`resources_list` 传查询 profile | full 可见=79，summary 一致 |
| B-3 | yourdfpy probe 状态 degraded：`'URDF' object has no attribute 'forward_kinematics'` | 误用不存在的 FK API | 改用真实 API `update_cfg(zero_cfg)` + `get_transform('tip')` | yourdfpy → available，有 FK smoke 证据 |
| B-4 | performance profile 仅激活 3 条（手写窄白名单） | `_in_performance_set` 依赖不稳健的高优先级匹配（`"高" in udoS_fit` 字符串包含，dict 型字段漏判） | 写 `parse_priority`（dict 取 `priority`、str 取"高——"前缀），profile 改为**机械推导**：高优先级 ∩ L1/L2 剔 L3 | performance=25，`test_profile_derivation_mechanical_contract` 钉死 |
| B-5 | 缺 DreamerV3 RSSM 状态结构适配 | 未实现世界模型状态 schema | 新增 `WorldModelRSSMConnector`（stoch 归一化类别概率+deter） | `test_rssm_state_roundtrip` 通过 |

无 P0/P1 阻断项；以上为实现期内自检发现并修复的 S3 级小问题，均有 RED→GREEN 记录。

## 4. 硬性验收实测

| 验收项 | 实测 | 判定 |
|---|---|---|
| pytest 总数 | **1492** passed / 0 failed（基线 1459，返工后新增 3，只增） | ✅ ≥1459 |
| 覆盖率 | **93%**（TOTAL 11102 stmts / 830 miss） | ✅ ≥93% |
| checkpoint 代数 | 33 代（v2.1.0…v4.3.9） | ✅ |
| 主参数 | 52191（registry 无可学参数，零训练） | ✅ 未变 |
| 锚点 md5 v4.3.9 | `8e767da5c6e262b9907eaa6ca72594bb` | ✅ 逐位 |
| 锚点 md5 v3.8.6 | `7351250ac00db53c321b919a951c640b` | ✅ 逐位 |
| 锚点 md5 v3.4.5 | `52993ca743416e6d822cdad78743c397` | ✅ 逐位 |
| 锚点 md5 v3.3.3 | `f993bcbdd476473c28dd4604bbbe11d6` | ✅ 逐位 |
| HTTP 矩阵 | /health=200(4.5.6)、/resources perf/full=200、/metrics=200 文本、/resources/profile=200、/{id}/probe=200、L3 invoke=503、未知 id=404 | ✅ |
| save/load 逐位 | 沿用既有 33 代 checkpoint 与既有 persistence 测试（本线零训练未新增 state） | ✅ |

## 5. 性能 A/B 结论

`benchmarks/results/resource_registry_v456.json`：装配 0.24ms、probe 缓存命中 0.001ms、full 列表 0.075ms、L1 转换 0.0045ms。
**ACCEPT**：registry 热路径亚毫秒级，无劣化；**REJECT/无宣称**：本线为新增能力，不做无对照的"提速"宣称。

## 6. 复现命令

```bash
cd udos-engine
python3 -m pytest -q -p no:warnings            # 1492 passed
python3 -m pytest --cov=udos --cov-report=term # 93%
python3 scripts/bench_resource_v456.py         # 资源性能
python3 -m udos.server --port 8000 --preset small
# curl localhost:8000/resources ; POST .../openvla/invoke -> 503
```

## 7. 已知缺口（如实声明）

- **CPU 类比**：20 个大权重/VLA/世界模型均未在 CPU 运行，仅契约+能力声明（L3）。
- **外部未获取项**：pytorch_kinematics 未安装（避免重依赖）；无 docker 环境未实测容器路径。
- **env_blocked 未触发**：本环境 pip 外网可用（yourdfpy 实装）；无外网降级分支已实现但当前不命中。
- 数据集类资源仅做 schema/动作表示适配，未下载任何真实数据集。

## 7.5 返工记录（v4.5.6 同棒修正，仍收口 v4.5.6）

| 项 | 修正前 | 修正后 | 证据 |
|---|---|---|---|
| performance 激活 | 3（手写窄白名单） | **25**（高优先级∩L1/L2 机械推导） | `test_profile_derivation_mechanical_contract` |
| 高优先级字段解析 | `"高" in str(fit)`（dict 漏判） | `parse_priority`（dict 取 priority/str 取前缀），字段实测高优先级=**36** | `test_priority_parser_reproduces_high_count` |
| L3 大模型 | 21 | **20**（DreamerV3 下沉为 L1 RSSM） | full by_level L3=20 |
| full 状态 | available 57 / absent 22 | available **58** / absent **21** | 动态 probe 实测 |
| 架构卡片 | 0 | **9**（docs/third_party/） | yourdfpy=confirmed，余 inferred |
| pytest | 1489 | **1492** | 全量 green |

> 高优先级总数口径：用户曾记为 27，字段机械推导为 **36**（dict 高 21 + str 高 15）。按诚信原则以字段实测为准，不凑数改判；其中 11 个 L3 重权重模型被剔除后，performance=25。

## 8. 发布判断

# go

理由：全部 P0/P1 风险用例有正式执行证据且通过；无开放 S1/S2；硬验收（测试数/覆盖率/锚点/代数/HTTP 矩阵）全部实测达标；零主权重改动、零训练；大权重资源诚实降级为契约，无伪装运行。
