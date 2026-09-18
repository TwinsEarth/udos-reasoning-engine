# UDOS Engine v7.3.3 · Windows 一键部署（小白教程）

无需 WSL、无需 Docker、无需管理员权限。macOS 见 `deploy/mac/`，本目录专用于 Windows。

## 一、准备（仅第一次）
1. 安装 Python 3.10–3.12：到 https://www.python.org/downloads/ 下载安装，
   **安装第一屏务必勾选 `Add Python to PATH`**，再点 Install Now。
2. 解压 `udos-engine-v7.3.3-full-source.zip`，得到 `udos-engine` 文件夹，
   放到**纯英文路径**（例如 `D:\udos-engine`，避免“桌面/下载”中文与空格引发的小概率问题）。

## 二、启动
双击 `deploy\windows\UDOS-Windows一键启动.bat`。脚本会自动：
1. 在仓库内创建虚拟环境 `.venv-win`（不污染系统 Python）；
2. 安装 CPU 版 `torch / numpy / huggingface_hub / pytest`（默认 PyPI，失败自动切清华镜像）；
3. 跑 `tests7` 回归测试；
4. 启动引擎服务。

看到下面两行即成功：
- 健康检查：浏览器打开 http://127.0.0.1:8777/api/v7/health
- 指标：http://127.0.0.1:8777/api/v7/metrics

停止：在命令行窗口按 `Ctrl+C`。

## 三、常见问题
| 现象 | 处理 |
|---|---|
| 双击闪退 | 在文件夹地址栏输入 `cmd` 回车，再输入 `deploy\windows\UDOS-Windows一键启动.bat` 运行，可看到报错 |
| “Windows 已保护你的电脑” SmartScreen | 点“更多信息” → “仍要运行”（本地脚本，未做代码签名） |
| `python 不是内部或外部命令` | 重装 Python 并勾选 Add to PATH，或手动 `py -m venv .venv-win` |
| pip 下载慢/超时 | 脚本已内置清华镜像回退；也可手动 `pip install ... -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 8777 端口被占用 | 编辑 bat 末尾端口号，例如改成 `--port 8787` |
| 想跑可观测演示 | 激活虚拟环境后 `set PYTHONPATH=.` 再 `python scripts7\observability_pro_demo.py` |

## 四、跨平台说明
- 引擎纯 Python，Linux/macOS/Windows 同一套代码；v7.3.3 的资源采集
  （CPU 核数、RSS 内存、网络收发字节）已做三平台回退：
  Linux 读 `/proc`，macOS 走 `resource`/`psutil`，Windows 走 `ctypes`(psapi)/`psutil`。
- GPU 利用率/显存、eBPF、OTel Collector gRPC 直推属于部署/资源闸门：
  Windows 上同样需要 NVIDIA 驱动 + GPU 或独立 Collector，未配置时相关字段诚实留 `null`。

## 五、命令行工具 udos（v7.3.4+）
除双击一键启动外，还提供跨平台命令行（cmd / PowerShell，仓库根目录下）：
```
bin\udos.bat version      查看版本
bin\udos.bat info         查看环境/资源
bin\udos.bat test         跑回归测试
bin\udos.bat serve        启动引擎（--auto-port 端口占用自动换）
bin\udos.bat health       健康检查
bin\udos.bat demo --list  列出演示（obs-pro 为可观测六层演示）
bin\udos.bat predict 请求.json --pretty   向运行中引擎提交预测
```
完整说明见 docs7/CLI_v7.3.md。
