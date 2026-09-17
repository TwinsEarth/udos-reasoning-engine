# UDOS v5.4.4 开源修订 — 验证报告
> release decision: **go**（打包 / 文档 / 默认配置一致性补丁；不动模型、算法、checkpoint）

## 起因
外部对 v5.4.3 的一份**只读代码评审**指出若干打包与一致性缺陷。本次逐条回到源码复核，确认属实后修复；评审中"README 首页仍写 4.5.3"一条经核对**不成立**（开源发布版 README 顶部已是 v5.4.3）。

## 修复项与证据
| # | 问题 | 修复前证据 | 修复 |
|---|------|-----------|------|
| 1 | `docker build` 必然失败 | `Dockerfile` 含 `COPY third_party/ctm ./third_party/ctm`，而 `git ls-files` 中无 `third_party/` | 删除该 COPY；`.dockerignore` 改为整体忽略 `third_party/`；注释改为"镜像不内置上游源码、适配器运行时按需拉取" |
| 2 | 容器默认挂旧 checkpoint | `Dockerfile` CMD 与 `docker-compose.yml` command 均用 `predictor_v3.3.3.pt` | 统一改为最新正式件 `predictor_v4.3.9.pt`（第 33 代，eval_mse 同为 0.045556） |
| 3 | Web 控制台版本停在 v4.5.6、测试数 1492 | `web/udos_console.html` 的 title/角标/内嵌 `DATA.overview`、`web/dashboard_data.json` | 版本→5.4.4、测试数→1572（实测）；构建脚本动态化并回写 HTML |
| 4 | 看板静态数据与实时数据未区分 | HTML 脚注仅称"构建时内联" | 明确标注注册表/性能图表为 v4.5.6 历史静态快照，仅 `/health`、`/resources`、`reason/latent` 为实时 |
| 5 | "双引擎"耦合范围被名字夸大 | `reasoning.py`：LoRA 仅注入演示用 `TinyBaseModel`（`gpm_engine.py:423`，注入在 `reasoning.py:139`），`reason()` 全程不调用该基座前向；`scene_embedding` 只进内部轻量 CTM（`reasoning.py:162-165`）；对外物理轨迹来自 `predictor.rollout`（`reasoning.py:172-181`），其输入只有位置/速度、不接场景嵌入；`query` 仅在结果中回显（`reasoning.py:187`） | 在 README"能力边界"新增"双引擎耦合范围"及 query/多模态/自进化/注册表边界 |
| 6 | `__init__` docstring 过时 | `udos/__init__.py` 写"上游开源代码库（已随工程克隆到 third_party/）"，与 NOTICE"不内置上游"矛盾 | 改为"开源包不内置其源码，适配器显式启用时经 huggingface_hub 运行时拉取" |

## 验证（本机 CPU：torch 2.14.0+cpu，Python 3，pytest 9.1.1）
- 全量 `python3 -m pytest -p no:warnings`：**1570 passed, 2 skipped（共收集 1572）/ 0 failed**，进程退出码 0；2 个 skipped 为既有条件跳过，非本次新增。
- 用例计数交叉核对：`pytest --collect-only` 末尾报 `1572 tests collected`；pytest 9 quiet 模式按文件输出 `path: N`，逐文件求和同为 **1572**。
- 版本一致性：`python3 -c "import udos; print(udos.__version__)"` 输出 `5.4.4`；HTTP `/health` 等响应的 `version` 字段取自 `__version__`；当前版本源（`udos/__init__.py`、`pyproject.toml`、Dockerfile OCI LABEL、docker-compose image、HTML title/角标/总览、JSON 总览）均为 5.4.4。版本号 bump 同步更新了 132 处测试中的版本断言（`assert __version__ == "5.4.4"` 等）。
- **服务实启动验证**：以容器同款命令 `python -m udos.server --preset small --checkpoint checkpoints/predictor_v4.3.9.pt` 启动，`GET /health` 返回 **HTTP 200**，body 为 `status=ok / version=5.4.4 / predictor_trained=true`，证明更新后的默认 checkpoint 路径在应用层可成功加载。
- Dockerfile 逐条 `COPY` 源路径在仓库中均存在：`requirements.txt`、`udos/`、`demos/`、`checkpoints/`；`checkpoints/predictor_v4.3.9.pt` 存在。
- 看板一致性：HTML 内联 `DATA` 与 `web/dashboard_data.json` 解析后**完全一致**；`registry` 仍为 **79** 条；`resource_bench.version` 刻意保留 `"4.5.6"`（该基准的历史时间戳），全站仅此一处与 JSON 一处保留 4.5.6。
- 零训练硬验收：主模型 `PhysicsPredictor` 仍为 **52191** 可学习参数、`eval_mse = 0.045556`、`checkpoints/` 仍 33 代且可加载。
- 工作区卫生：全量测试会按本机性能覆写若干 `benchmarks/results/*latency*`、`*ab*` 类 JSON（延迟/MSE 末位随机器漂移），发布提交前已用 `git checkout` 全部还原，历史基准快照保持字节不变。

## 未执行 / 限制
- **[本环境未执行]** 沙箱无 Docker daemon，未实际运行 `docker build` 与 `docker compose up`；已对 Dockerfile 做逐行 COPY 源路径与默认 checkpoint 路径的静态核对，镜像内真实启动需在具备 Docker 的环境复验。
- 本次为打包/文档/默认配置补丁，**未重训模型、未改任何算法与权重**；`finesim`/`dendrite`/`kvcache` 相关 `[UNVERIFIED]` 口径维持不变。
- 在线多 LLM 交叉打分、GPU + vLLM KV offload 实测、NEURON/DeepDendrite 对拍、MuJoCo 因果虚拟小鼠仍未实现/未启动，沿用 v5.4.3 能力边界。
