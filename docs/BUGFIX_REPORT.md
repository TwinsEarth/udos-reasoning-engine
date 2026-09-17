# UDOS 推演引擎 v0.1.0 — Bug 闭环报告

记录从初版示意 Demo 到可运行工程过程中发现的全部缺陷、根因、修复与验证。

| # | 缺陷 | 根因 | 修复 | 验证 |
|---|------|------|------|------|
| 1 | `infer_dims_from_model` 抛 `not enough values to unpack (expected 3, got 2)` | 误把 `named_modules()` 当三元组遍历, 其实际只 yield `(name, module)` | 改为二元组遍历 | `test_gpm.py` 全部通过 |
| 2 | CTM 前向 `mat1 and mat2 shapes cannot be multiplied` | 注意力输出宽度为 `d_input`, 神经群体宽度为 `d_model`, 直接拼接得到 `d_input+d_model ≠ 2*d_model` (上游用 LazyLinear 自适应, 初版写死宽度) | 新增 `attn_to_state: Linear(d_input→d_model)`, 拼接宽度恒为 `2*d_model` | `test_ctm_output_shapes` |
| 3 | GPM 生成的 B 矩阵出现 `[n_layer,r,0]` 空张量, 注入时报 `size 128 must match 96` | head 槽位宽度误取 `max(d_in,d_out)`, 而 A/B 拼接需要 `d_in+d_out` (对齐 D2L `_to_lora_dict`) | 改为 `max(d_in+d_out)` | `test_lora_shapes` / `test_inject_and_lossless_reset` |
| 4 | 初版 Demo 直接改写 `module.weight.data` 再做减法回滚 | 与 D2L 真实实现不符, 且浮点减法存在累积误差、无法保证无损 | 改为 D2L 同款**前向补丁** `y=Linear(x)+B(Ax)·s`, 原始权重永不修改, `reset()` 恢复原 forward | `test_inject_and_lossless_reset` 断言 `torch.equal(y0,y2)`, 实测最大还原误差 `0.000e+00` |
| 5 | 未训练超网络注入后输出无变化 | D2L 中 `scaler_B=0` 是训练稳定起点, 训练后才非零; 未训练演示时 B 恒为 0 | `GPMConfig.init_scaler_b_zero` 开关: 默认 `True` 忠于上游, 演示/推理置 `False` | `test_zero_scaler_b_init_means_no_delta` + `test_inject_and_lossless_reset` 双向覆盖 |
| 6 | 初版场景向量维度 (3 位置+属性) 与超网络输入宽度写死不匹配 | 缺少统一物理量投影层 | `PhysicsSceneEncoder` 对任意属性维度自适应投影到 `d_model` | `test_encoder_adapts_attr_dims` |
| 7 | PCE 序列化测试取错元素 / 测试辅助重复关键字传参 | 时间排序后索引变化; helper 显式参数与 `**kw` 冲突 | 按 object_id+timestamp 定位; helper 统一 `setdefault` | 全量测试通过 |

## 机制对齐校验 (与上游真实代码库)

- `tests/test_upstream_ctm.py` 直接加载 `third_party/ctm` 的 `ContinuousThoughtMachine`
  完成真实前向, 并断言内部实现与上游输出张量结构一致
  `predictions[B,out,T]` / `certainties[B,2,T]` / `sync[B,n_synch]`。
- 结果: 上游真实 CTM 加载成功 (197,634 参数), 8 tick 前向通过。

## 最终验证

```
18 passed (含 2 项上游真实代码库交叉验证)
Demo1/2/3 端到端全部跑通
```
