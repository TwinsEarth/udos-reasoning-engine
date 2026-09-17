# UDOS v0.2.0 — 测试 / 验证 / 部署报告

三线协同：**测试线**（契约+反例）、**验证线**（性能基准+回归守卫+上游对照）、
**部署线**（HTTP 服务+容器+冒烟）。运行环境：Ubuntu、Python 3.12、torch 2.14.0+cpu、2 线程。

---

## 一、测试线

### 1.1 环境故障闭环（diagnose）
| 项 | 内容 |
|----|------|
| 现象 | 新会话基线命令报 `/opt/python3.12 No module named pytest/torch` |
| 根因 | 沙箱运行时被重置，旧虚拟环境消失、解释器切换为 3.12，**非代码缺陷** |
| 修复 | 当前解释器重装 CPU torch + pytest/pytest-cov/huggingface_hub |
| 验证 | 依赖导入成功，全部测试可运行 |

### 1.2 契约→反例加固（新增 `tests/test_contracts.py`、`tests/test_service.py`）
每条用例对应一个能让错误实现变红的反例：

| 契约 | 反例（错误实现） | 结果 |
|------|-----------------|------|
| CTM 两次前向逐元素一致 | trace 跨调用残留 | PASS |
| CTM 拼 batch == 逐条单跑 | 跨样本共享隐状态 | PASS |
| CTM 同步递推 == α=rα+a·b, β=rβ+1, α/√β | 用普通均值替代衰减递推 | PASS |
| CTM 损失可回传到核心模块 | 某参数脱离计算图 | PASS |
| CTM 非法 2D 输入快速失败 | 静默产出错误结果 | PASS |
| GPM 补丁输出 == Wx+s·B(Ax) 逐项 | einsum 转置/缩放错误 | PASS |
| GPM 重复 inject 不叠加 | forward_orig 被套两层 | PASS（并据此修复生产缺陷，见 1.3） |
| GPM reset 三轮幂等零误差 | 补丁/浮点累积 | PASS |
| GPM 只改变目标层 | 污染无关线性层 | PASS |
| GPM 权重和≠1 正确归一 | 直接相加尺度漂移 | PASS |
| 协同重复内化不叠加、不伪造因果边 | 补丁翻倍 / 虚构 pce-explicit | PASS |
| HTTP 四接口 + 400/404 + /demo | 错误输入打崩进程 / 路由缺失 | PASS（真实起端口） |

### 1.3 测试驱动出的真实生产修复
- **缺陷**：`PhysicsHypernetwork` 每次前向新建随机初始化的场景编码器，
  同场景重复内化结果不确定，服务化后会漂移（被 G3/确定性测试击穿）。
- **修复**：编码器持久化为子模块 `ctx_encoder`，首次按属性键构建、之后复用、
  随 `state_dict` 保存与训练，仅在出现新属性键时重建。
- **证据**：修复后重复内化输出 `allclose(atol=1e-6)`，全量测试转绿。

### 1.4 结果
- **36 passed**（v0.1 的 18 项 + 新增 18 项契约/服务测试）。
- 覆盖率 `pytest --cov`：ctm_engine **96%**、gpm_engine **99%**、reasoning 93%、
  server 81%、总 **87%**；未覆盖部分为 debug 面板与服务 main 启动块。

---

## 二、验证线

### 2.1 性能基准（`benchmarks/benchmark.py`，warmup 后采样，结果落
`benchmarks/results/baseline_v0.2.0.json`）

| workload | median | min | max | 参数量/产物 |
|----------|-------:|----:|----:|-------------|
| ctm_small_forward | 7.138 ms | 7.051 | 8.667 | 216,192 |
| ctm_medium_forward | 17.816 ms | 17.130 | 22.151 | 691,424 |
| gpm_small_internalize | 2.671 ms | 2.604 | 2.861 | LoRA 30 KB |
| gpm_medium_internalize | 3.719 ms | 3.598 | 3.828 | LoRA 240 KB |
| e2e_small（内化后 reason） | 7.991 ms | 7.934 | 8.094 | — |
| 上游真实 CTM 对照 | 9.113 ms | 8.915 | 10.107 | 134,146 |
| 进程峰值 RSS | 288 MB | — | — | — |

内部机制对齐版与上游真实 CTM 同量级（7.1 vs 9.1 ms），无数量级偏差。
数字随 CPU 波动，仅作基线，不做性能宣传。

### 2.2 回归守卫
`--guard` 以工程自定宽松门槛（基线中位数约 2 倍）判定显著回归，本次 **通过**。

### 2.3 上游交叉验证
`tests/test_upstream_ctm.py` 真实加载 `third_party/ctm` 的
`ContinuousThoughtMachine` 前向，并断言内部实现与上游输出结构一致，PASS。

---

## 三、部署线

### 3.1 交付物
- `udos/server.py`：标准库零额外依赖 HTTP 服务（ThreadingHTTPServer + 推理锁），
  路由 `/health /internalize /reason /reset /demo`，输入错误 400、未知路由 404、单请求异常不拖垮进程。
- `Dockerfile`（python:3.12-slim、CPU torch、HEALTHCHECK）、`docker-compose.yml`
  （CPU/内存 limit、restart、healthcheck）、`Makefile`（test/cov/bench/guard/serve/smoke/docker-*）。
- `scripts/smoke_test.py`：自动起服务→轮询健康→逐接口断言→回收进程。
- `docs/DEPLOYMENT.md`：API 契约、curl、裸机/Docker/compose/systemd、资源、排障。

### 3.2 冒烟结果（真实起进程打 HTTP）
```
[PASS] GET /health
[PASS] POST /internalize
[PASS] POST /reason
[PASS] POST /reset
[PASS] POST /reason 缺字段返回400
[PASS] GET /demo 全链路
全部 6 项冒烟通过
```

### 3.3 容器镜像验证边界（如实说明）
沙箱**无 docker/podman/buildah daemon**，未实际执行 `docker build`。
替代证据：按 Dockerfile 的 COPY 清单构造**等价文件集**（仅 udos + demos + third_party/ctm），
在其中运行镜像 CMD `python -m udos.server`，服务成功启动并返回
`{"status":"ok","version":"0.2.0"}`，无 error/traceback，证明 COPY 清单完整、启动命令可运行。
真实镜像构建需在具备 Docker 的环境执行 `make docker-build`（命令已随工程交付）。

---

## 四、剩余缺口
1. 容器镜像未在本沙箱实建（无 daemon），仅完成等价文件集验证。
2. 权重仍为随机初始化，基准衡量的是机制/数据流性能，非训练后精度；精度需训练。
3. 服务为单副本、推理串行（加锁）；高并发需多副本水平扩展（见 DEPLOYMENT §7）。

## 五、复现实验命令
```bash
make test      # 36 项
make cov       # 覆盖率
make bench     # 基准, 落 JSON
make guard     # 性能守卫
make smoke     # HTTP 6 项冒烟
```
