# UDOS v5.4.3 精细生物物理内核 — 验证报告
> release decision: **go**

## 端点(UDOS_FINESIM=on, off→503; /intel/finesim 始终 200)
POST /finesim/{cable,hh,hines_dhs,nmda_inhibition,payeur,robustness}。

## curl 实测
- hh: I=10 -> spikes=4, peak=42.72mV; f-I 2→0Hz/5→1Hz/10→4Hz/20→5Hz。
- hines_dhs: serial=5, layers=4, numerical_consistent=true。
- nmda: with_mg_range=0.107(敏感), no_mg_range=0(移除不敏感, 可证伪)。
- /intel/finesim: gpu=false, neural_simulator=ENV_BLOCKED。
- score 回归 0.09。

## 测试/覆盖率
**1572 passed / 0 failed**(1565+7 只增)。TOTAL = **12537 stmts / 874 miss = 93%**。

## 文献核实
16线程/10×/100–1000×/5万神经元: 未逐条证实条件 → [UNVERIFIED], 仅引用。

## 零训练硬验收
主参 52191、eval_mse 0.045556、33 代 checkpoint、四锚点逐位不变; grep 口令/姓名=0。
