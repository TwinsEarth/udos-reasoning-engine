# macOS 一键部署（小白直接看这里）

- 双击入口（第一次请「右键 → 打开」）：仓库根目录 **`UDOS-Mac一键启动.command`**
- 图文步骤与常见问题：**`docs7/Mac部署_小白教程.md`**

## 文件说明
| 文件 | 作用 |
|---|---|
| `UDOS-Mac一键启动.command`（仓库根目录） | 中文图形菜单：安装环境 / 启动 v7.0.3 / 启动 v5.5.5 / 停止 / 自检 / 开机自启 |
| `deploy/mac/common.sh` | 公共逻辑：架构检测、找 Python 3.10+、建虚拟环境、装 Mac 原生 torch、启动/健康检查/停止、自检 |
| `deploy/mac/autostart.sh` | launchd 开机自启的安装/卸载/状态（`install v7|v5`、`uninstall`、`status`） |
| `deploy/mac/requirements-mac.txt` | macOS 依赖（默认 PyPI，**不要**用 Linux 的 cpu wheel 索引） |
| `deploy/mac/logs/`、`deploy/mac/run/` | 运行后自动生成：日志、PID（可随时删除，不影响源码） |

## 版本与端口
- v7.0.3（主线，推荐）：`python -m udos7.server --host 127.0.0.1 --port 8777`，健康检查 `/api/v7/health`
- v5.5.5（legacy 冻结）：`python -m udos.server --host 127.0.0.1 --port 8000 --preset small --checkpoint checkpoints/predictor_v4.3.9.pt`，健康检查 `/health`

## 命令行等价操作（进阶）
```bash
# 环境（在仓库根目录）
python3 -m venv .venv-udos
.venv-udos/bin/python -m pip install -r deploy/mac/requirements-mac.txt
# 启动 v7.0.3
.venv-udos/bin/python -m udos7.server --host 127.0.0.1 --port 8777
# 启动 v5.5.5
.venv-udos/bin/python -m udos.server --host 127.0.0.1 --port 8000 --preset small --checkpoint checkpoints/predictor_v4.3.9.pt
```

> 验证状态见教程第七节：启动/健康检查/停止/plist 已在等价 Linux+CPU 环境实测；
> Finder 双击、Gatekeeper、`open`、真实 `launchctl`、Mac 版 torch 联网安装为 unverified-on-mac。
