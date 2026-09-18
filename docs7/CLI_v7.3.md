# UDOS Engine 跨平台命令行工具（v7.3.4+）

## 最快用法（不用记命令）
- **Windows**：双击仓库根目录的 `UDOS-命令行菜单.bat`，按数字选择（信息/测试/启动/健康检查/演示…）。
- **macOS**：双击 `UDOS-命令行菜单.command`（首次被 Gatekeeper 拦截时，右键→打开，或
  `xattr -dr com.apple.quarantine UDOS-命令行菜单.command`）。
- 根目录便捷命令：Windows `udos.bat info`；macOS/Linux `./udos.sh info`。
- 习惯完整 CLI：Windows 用 `bin\udos.bat <命令>`，macOS/Linux 用 `./bin/udos <命令>`。
- 只有执行过 `pip install -e .` 之后，才能在任意目录直接敲 `udos <命令>`（否则请用上面的相对路径）。

---

一个统一入口 `udos`，在 **macOS / Linux / Windows** 上用法完全一致。
实现为纯标准库的 Python 模块 `udos7.cli`，并提供原生包装脚本与 pip 入口。

## 三种调用方式（等价）
| 平台 | 命令 |
|---|---|
| macOS / Linux | `./bin/udos <command>`（仓库根目录下） |
| Windows | `bin\udos.bat <command>`（cmd / PowerShell） |
| 任意（已 `pip install -e .`） | `udos <command>` |
| 任意（不装也行） | `python -m udos7.cli <command>` |

包装脚本会自动选用一键部署创建的虚拟环境（mac 为 `.venv-udos`、Windows 为 `.venv-win`），
无需手动 activate；找不到虚拟环境则回退系统 Python。

## 命令一览
| 命令 | 作用 |
|---|---|
| `udos version` | 打印版本号；`udos --version` 同义 |
| `udos info` | 打印环境/平台/Python/torch/CUDA·MPS/资源/默认检查点（JSON） |
| `udos serve [--host 127.0.0.1] [--port 8777] [--checkpoint P] [--auto-port]` | 启动 v7 HTTP 引擎；`--auto-port` 端口被占时自动换空闲端口 |
| `udos test [--suite tests7\|tests\|all] [-- <pytest 参数>]` | 跑回归测试，默认 tests7 |
| `udos train [透传参数]` | 训练 v7 预测器（scripts7/train_v7.py，较慢） |
| `udos verify` | 生成 v7 验证与证据报告（scripts7/verify_v7.py） |
| `udos demo --list` / `udos demo obs\|obs-pro\|legion\|scaling` | 运行内置演示 |
| `udos health [--port 8777] [--timeout 5]` | 探测运行中引擎健康检查，连不上返回退出码 1 |
| `udos metrics [--pretty]` | 拉取引擎指标 JSON |
| `udos predict <请求体.json> [--pretty]` | 向运行中引擎提交历史窗口做预测；`-` 表示标准输入 |

## 典型用法

### macOS
```bash
cd udos-engine
./bin/udos version
./bin/udos info
./bin/udos test                 # 跑 tests7
./bin/udos serve                # 起服务，Ctrl+C 停止
# 另开一个终端：
./bin/udos health
./bin/udos demo obs-pro
```

### Windows（cmd）
```bat
cd /d D:\udos-engine
bin\udos.bat version
bin\udos.bat info
bin\udos.bat test
bin\udos.bat serve
:: 另开一个 cmd 窗口：
bin\udos.bat health
bin\udos.bat demo obs-pro
```
PowerShell 同样用 `.\bin\udos.bat ...`。

### 预测请求示例
`window.json`（字段以 v7 API 契约为准，window 形状 [W,6]）：
```json
{ "window": [[0,0,0,1,0,0],[0.1,0,0,1,0,0],[0.2,0,0,1,0,0]] }
```
```bash
udos serve                      # 终端 1
udos predict window.json --pretty   # 终端 2
```

## 退出码约定
- `0` 成功；`1` 运行期/连接失败（如引擎未启动）；`2` 参数错误或脚本/依赖缺失。
  便于在 shell、批处理、CI 与定时任务中串联。

## 说明
- CLI 本身零第三方强依赖；`serve/train/test/demo` 仍需安装引擎依赖（torch/numpy/pytest）。
- Windows 首次使用若执行 `.bat` 被 SmartScreen 拦截，点“更多信息→仍要运行”。
- 卸载或换 Python 后，删除仓库内 `.venv-win`（Windows）/`.venv-udos`（mac）后重跑一键脚本即可重建。

## 一次安装、多版本共享（v7.3.5+）
torch 等重依赖**只安装一次**：所有 UDOS 版本共用主目录下的一个虚拟环境，升级/换版本目录默认零安装。
- 共享环境位置：
  - Windows：`%USERPROFILE%\.udos\venv`
  - macOS / Linux：`~/.udos/venv`
- 想换位置：设置环境变量 `UDOS_VENV`（Windows：`setx UDOS_VENV D:\udos-venv`；macOS：`export UDOS_VENV=~/udos-venv`）。
- 何时会再安装：仅当依赖清单变化（标记文件 `udos_provision.json` 内哈希不匹配）或导入自检失败时，才增量安装；torch 锁定 `2.14.0` CPU/macOS 官方 wheel。
- 旧版本在仓库内建过 `.venv-udos` / `.venv-win` 的，会被自动沿用，不重复下载。
- 手动一次性安装（可选）：
  - Linux/Windows CPU：`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`
  - macOS：`pip install torch==2.14.0`
  - 其余：`pip install -r deploy/requirements-cpu.txt`
- 离线/内网：可预先在一台联网机执行一次，再把整个共享环境目录或 pip 缓存（`pip cache dir`）拷贝到同平台机器。
