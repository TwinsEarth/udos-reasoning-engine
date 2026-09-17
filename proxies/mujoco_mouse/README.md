# 最小虚拟小鼠因果代理（cpu-proxy）

证据分级：**cpu-proxy**。这是 MuJoCo CPU 上的最小平面铰接代理，用于验证世界模型的
因果主张，**不是** DeepMind 的 virtual rodent：没有强化学习训练的运动策略、
没有神经活动对齐、没有 38 自由度全身模型。

## 运行

```bash
pip install mujoco          # CPU wheel，本机实测 3.13.0
python3 proxies/mujoco_mouse/run_proxy.py
# 写 reports7/mujoco_mouse_proxy.json
```

## 模型

`minimal_mouse.xml`：free-joint 躯干（胶囊）+ 前后两条铰链腿，位置伺服，
重力 + 地面接触；开环半周期相位差正弦步态。矢状面（xz）内运动。

## 因果实验（同初始状态隔离干预）

- A 基线步态；
- B 动作干预：仅把步态相位偏移 π/2；
- C 身体干预：仅把躯干质量加倍（对应论文 mass-doubling）。

实测（MuJoCo 3.13.0 CPU，400 步×5ms=2s）：三组初始状态一致；
B、C 相对 A 的状态轨迹从第 1 个仿真步即发散，净 x 位移不同
（A −0.384、B −0.298、C −0.415），证明该仿真是可干预的**因果模型**
（改变动作/身体参数会物理地改变后续状态），而非对观测序列的描述性拟合。

## 与 v7 内核的关系 / 边界

- 它提供“物理复现行为 + 干预”的因果环境，可作为 v7 `WorldModelCore` 的
  外部真值数据源（未来：在其 rollout 上训练预测器并验证干预响应）。
- 未做：RL 步态策略、神经活动预测、具身图灵测试、MuJoCo-MJX GPU 批量仿真
  （后者属 M5 GPU 闸门）。
