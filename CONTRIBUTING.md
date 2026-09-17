# 贡献指南（Contributing）

感谢关注 UDOS Reasoning Engine。本项目的工程纪律围绕**可复现、可证伪、可追溯**，提交代码前请先阅读本文件。

## 1. 开发环境

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

依赖刻意保持轻量：核心仅需 `torch`、`numpy`；`huggingface_hub` 仅在显式启用真实上游 CTM 适配器时需要。

## 2. 必须通过的检查

```bash
python -m pytest -q -p no:warnings
```

- 所有提交必须保持全量测试通过；测试用例数**只增不删**（修复时以新的回归测试替代，而非删除断言）。
- 不要在仓库中放入任何真实密钥、口令、令牌或个人信息；`tests/test_security_guard_v501.py` 会对此做守卫扫描。
- 依赖外部硬件/凭证（GPU、QAT、NEURON、在线 LLM、FIDO2）的能力必须在不可用时显式返回 `ENV_BLOCKED` / 503，**不得伪造结果**。

## 3. 新能力的接入纪律

1. 受前沿工作启发时，只实现 CPU 可跑的**机制对齐（analogy, not reproduction）**，不声称复现论文效应量。
2. 新能力默认**外挂、零梯度、opt-in**：不进主模型 `state_dict`、默认不改变默认输出；关闭开关时行为与旧版逐位一致。
3. 任何"更好/更快/更省"的结论都要有同 seed、同合同的 baseline/candidate 证据，结果（含负面结果）落到 `benchmarks/results/`。
4. 关键机制尽量配一个**可证伪对照**（例如移除某条件后效应应消失）。
5. 未逐条证实的文献微观数字统一标注 `[UNVERIFIED]`，自测数据与厂商/论文口径严格分开。

## 4. 提交与文档

- 一个提交只做一件事，写清动机、改动点、验证命令与结果。
- 行为变更请同步更新对应 `docs/VERIFICATION_*.md` / `CHANGELOG.md` 与必要的专题文档。
- 不提交生成物、缓存、`.env`、checkpoint 临时文件（见 `.gitignore`）。

## 5. 安全问题

请勿在公开 issue 中提交可利用细节。安全相关问题请先联系维护者（在仓库 About 中预留的联系方式）。
