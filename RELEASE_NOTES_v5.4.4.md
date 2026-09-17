# UDOS Reasoning Engine v5.4.4

开源修订版：修复 v5.4.3 的 Docker 打包硬伤与版本/看板一致性问题，并精确化能力边界。**不改动任何模型、算法或 checkpoint**，从 v5.4.3 升级无需迁移。

## 修复

- **Docker 构建修复（关键）**：v5.4.3 的 `Dockerfile` 复制了仓库中并不存在的 `third_party/ctm`，导致 `docker build` 必然失败。已删除该 COPY，`.dockerignore` 整体忽略 `third_party/`；镜像不内置上游源码，CTM/D2L 适配器在显式启用时经 `huggingface_hub` 运行时拉取。
- **默认 checkpoint 更新**：容器与 `docker compose` 默认挂载由过时的 `predictor_v3.3.3.pt` 更新为最新正式件 `predictor_v4.3.9.pt`，开箱即带当前多步推演能力。
- **版本号一致性**：统一到 v5.4.4（`udos/__init__.py`、`pyproject.toml`、Docker OCI LABEL、compose image、Web 控制台 title/角标/总览）；修正 `__init__` docstring 中过时的 third_party 表述。
- **Web 控制台诚实化**：总览版本改取 `udos.__version__`、测试数改为构建时实时 `pytest --collect-only`（兼容 pytest 9），构建脚本同时回写内联 HTML，消除 JSON 与 HTML 脱节；明确标注注册表/性能图表为 v4.5.6 **静态快照**，仅 `/health`、`/resources`、`/reason/latent` 为实时。
- **能力边界精确化**：README 新增"双引擎耦合范围"——GPM 生成的 LoRA 仅注入演示用 `TinyBaseModel`，`reason()` 不调用其前向；GPM 场景嵌入只进内部轻量 CTM 支路；对外物理轨迹来自独立 `PhysicsPredictor.rollout` 且不消费场景嵌入；自然语言 `query` 仅回显、不参与计算；多模态为低维代理头；"自进化"是冻结主模型的配置搜索；资源注册表"列出"不等于权重已运行。

## 验证

- 全量 **1570 passed + 2 skipped（共收集 1572）/ 0 failed**（CPU，torch 2.14.0，pytest 9.1.1），覆盖率约 **93%**。
- 以容器同款默认命令实启动服务：`/health` 返回 **200**、`version=5.4.4`、`predictor_trained=true`（默认 `predictor_v4.3.9.pt` 加载成功）。
- 主模型锚点不变：**52191** 可学习参数、`eval_mse = 0.045556`、33 个自训练 checkpoint 全部可加载。
- Dockerfile 逐条 COPY 源路径与默认 checkpoint 路径静态核对通过；沙箱无 Docker daemon，镜像实际构建/启动待有 Docker 环境复验（见 `docs/VERIFICATION_v5.4.4.md`）。

## 兼容性

完全向后兼容：纯打包/文档/默认配置补丁，无 API、权重或数据格式变更。

## 许可与归属

Apache License 2.0。仓库不包含上游或第三方源代码/预训练权重，第三方元数据与唯一内置资产（Apache ECharts）归属见 [NOTICE](NOTICE)。
