# UDOS v5.4.3 精细生物物理数值内核

> 纯 CPU、numpy 参考/类比内核, 不运行 NEURON; 真模拟器 ENV_BLOCKED。

## 模块 (udos/finesim/)
- cable.py：被动电缆 λ=√(d·R_m/4R_a), τ=R_m·C_m; 衰减拟合 vs 解析。
- hh.py：Hodgkin-Huxley 1952 门控积分, 阶跃电流产生动作电位 + f-I 曲线。
- nmda.py：NMDA Mg²⁺ 阻滞(Jahr-Stevens) + 时序性抑制可证伪对照。
- payeur.py：四类信息处理 + 突触位置鲁棒性 + NGRAD 假设(外挂)。
- hines_dhs.py：Hines 串行 vs DHS 层级并行, 数值一致+计数。

## 文献引用(DOI)
- Rall 1959 电缆理论; Hodgkin-Huxley 1952。
- Beniaguev/Segev/London, Neuron 109(17):2727–2739, 2021。
- DeepDendrite, Nat Commun 14:5798, DOI 10.1038/s41467-023-41553-7, PMC10507119。
- Payeur 2019 Curr Opin Neurobiol 58:78; Du 2017 PubMed 28827326;
  Doron 2017 Cell Reports 21 / ModelDB 152901。

## 关键可证伪
NMDA 镁阻滞: 有镁→时序敏感(range>0); 移除→不敏感(range=0)。

## 文献数字核实
"最多16线程/约10×/GPU 100–1000×/8 GPU 5万神经元": 未在本机读 PMC 原文逐条证实条件
→ **[UNVERIFIED]**, 仅作引用; 本引擎自测只报本机 CPU 步数/层数。

## 端点(UDOS_FINESIM=on, off→503)
POST /finesim/{cable,hh,hines_dhs,nmda_inhibition,payeur,robustness};
GET /intel/finesim 始终 200。
