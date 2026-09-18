# UDOS 推演引擎 · Mac mini（M 系列）小白一键部署教程

适用：**Mac mini M4 / M1 / M2 / M3（Apple Silicon）**，也兼容 Intel Mac；macOS 13 及以上。
覆盖两个版本，可同时运行、互不冲突：

| 版本 | 角色 | 本机地址 | 说明 |
|---|---|---|---|
| **v7.0.3**（推荐） | 主线新版，统一世界模型内核 | http://127.0.0.1:8777/api/v7/health | API 服务 |
| **v5.5.5** | legacy 冻结版（对照用） | http://127.0.0.1:8000/health | API + 静态控制台页 |

你**不需要懂命令行、不需要装 Docker**。全程双击 + 看菜单选数字即可。
首次安装会下载 PyTorch（体积较大），请预留约 **3GB** 磁盘空间、保持网络畅通。

> 文件清单：双击入口在仓库根目录 `UDOS-Mac一键启动.command`；
> 底层脚本在 `deploy/mac/`；虚拟环境装在仓库内 `.venv-udos/`（不污染系统）。

---

## 一、第一次使用（3 步）

### 第 1 步：解压到固定位置
1. 下载 `udos-engine-v7.0.3-full-source.zip`，双击解压，得到 `udos-engine/` 文件夹。
2. 建议放到「用户目录」或「应用程序」，**不要放在 iCloud 同步盘、U 盘或会被清理的下载临时目录**。
3. 记住这个文件夹的位置（设置开机自启后路径不能随意移动；移动了需重新设置，见 FAQ）。

### 第 2 步：右键打开启动器（仅第一次需要）
1. 进入 `udos-engine/` 文件夹，找到 **`UDOS-Mac一键启动.command`**。
2. **第一次请用「右键（或双指点按）→ 打开」**，在弹窗里再点一次「打开」。
   - 直接双击可能提示“无法打开，因为无法验证开发者/来自身份不明的开发者”，这是因为脚本未做苹果公证，**不是病毒**。
   - 若仍被拦截：打开「系统设置 → 隐私与安全性」，到底部会出现“已阻止使用 UDOS-Mac一键启动.command”，点「仍要打开」。
3. 系统会询问是否允许打开“终端”，点「允许」。随后出现中文菜单即成功。

### 第 3 步：选「1」安装环境
菜单里输入 `1` 回车，脚本会自动：
- 检测芯片（Apple Silicon / Intel）；
- 查找 Python 3.10 及以上；没有则引导你安装（见下方“关于 Python”）；
- 在仓库内创建独立虚拟环境 `.venv-udos`；
- 安装 **Mac 原生版 torch/numpy**（首次约几分钟到十几分钟）。
  - 默认源太慢会**自动切换清华镜像**重试一次。

看到 `[ 成功 ] 环境安装完成` 并打印出 Python / torch / numpy 版本号，即完成。以后无需再装。

#### 关于 Python（只有提示缺失时才看）
- 新版 macOS 自带的 `/usr/bin/python3` 可能是 3.9，**版本不够**，脚本会提示。
- 最简单：浏览器打开 https://www.python.org/downloads/macos/ ，下载 **Python 3.12** 的 `.pkg`，双击一路「继续」装完，回到启动器按回车重新检测即可。
- 如果你装了 Homebrew，也可在终端执行 `brew install python@3.12`。

---

## 二、启动与验证

在菜单中选择：

- 输入 `2`：启动 **v7.0.3**。成功后会自动用浏览器打开健康检查页，看到类似
  ```json
  {"status": "ok", "version": "7.0.3", "model_loaded": false, "checkpoint": "worldmodel_v7.0.3.pt", "evidence_grade": "verified", "api": "v7"}
  ```
  - `model_loaded` 首次为 `false` 是**正常的懒加载**：第一次调用预测时才载入模型，之后变 `true`。
  - 指标报告页：http://127.0.0.1:8777/api/v7/metrics
- 输入 `3`：启动 **v5.5.5**。打开 http://127.0.0.1:8000/health ，应看到
  `"status": "ok", "version": "5.5.5", "predictor_trained": true`；同时打开静态可视化控制台
  `web/udos_console.html`。
- 输入 `4`：两个版本一起启动。
- 输入 `5`：停止全部后台服务。
- 输入 `6`：自检（v7.0.3 端到端冒烟 + `tests7` 测试，约一两分钟；全绿即环境正常）。

> 关闭终端窗口**不会**停止后台服务（已用 nohup 托管）。要停止请用菜单 `5`。

---

## 三、开机自动启动（可选）

- 菜单 `7`：选择让 **7（新版）** 或 **5（旧版）** 开机/登录后自动运行（基于 macOS 原生
  `launchd`，写入 `~/Library/LaunchAgents/com.udos.engine7.plist` 或 `com.udos.engine5.plist`，
  **仅对当前用户生效，不需要管理员密码，开机即监听 127.0.0.1**）。
- 菜单 `8`：取消指定版本的开机自启。
- 菜单 `9`：查看当前自启安装状态。

设置后，下次登录系统服务会自动在后台就绪，直接用浏览器访问上面的地址即可。

---

## 四、日常使用
以后只需双击 `UDOS-Mac一键启动.command`（第一次右键打开过，之后可直接双击），选 `2` 或 `3`。

- 运行日志：`deploy/mac/logs/udos7.log`、`udos5.log`（自启日志为 `*.launch.log`）。
- 进程号记录：`deploy/mac/run/`。
- 想让同一局域网的手机/其他电脑访问？默认只监听本机 `127.0.0.1`（更安全）。进阶做法是用
  `--host 0.0.0.0` 启动（可自行修改 `deploy/mac/common.sh` 中对应命令），并在 Mac「系统设置 →
  网络 → 防火墙」放行；**请仅在可信网络开启**，不要直接暴露到公网。

---

## 五、常见问题（FAQ）

1. **提示“无法打开，因为来自身份不明的开发者”**
   用「右键 → 打开」；或「系统设置 → 隐私与安全性 → 仍要打开」。
   也可在终端执行一次（把路径换成你的实际路径）：
   `xattr -dr com.apple.quarantine /路径/udos-engine`

2. **双击 .command 一闪而过 / 没反应**
   打开「终端」，把 `UDOS-Mac一键启动.command` 直接拖进终端窗口后回车，可看到完整报错；
   或给它加执行权限：`chmod +x "/路径/udos-engine/UDOS-Mac一键启动.command"`。

3. **安装 torch 很慢或超时**
   脚本会自动切清华源重试；仍失败请检查网络/代理后重跑菜单 `1`（已装好的部分会复用，可反复执行）。

4. **提示 Python 版本不够 / 找不到 Python 3.10+**
   按上文“关于 Python”安装 python.org 的 3.12 `.pkg`，回菜单按回车重检。

5. **端口被占用（8777 或 8000）**
   先选菜单 `5` 停止；若仍占用，终端执行 `lsof -ti tcp:8777 | xargs kill -9`
   （8000 同理）后再启动。

6. **需要 Rosetta 吗？** 不需要。Apple Silicon 安装的是原生 arm64 版 torch；引擎只用 CPU 线程。

7. **移动了 `udos-engine` 文件夹后自启失效**
   自启配置里写的是绝对路径。移动后重新执行菜单 `8` 取消、再 `7` 安装即可。

8. **v7 健康页 model_loaded 一直是 false？**
   这是懒加载，调用一次预测（或跑菜单 `6` 自检）后即变 true，不影响使用。

---

## 六、卸载
1. 菜单 `8` 取消两个版本的开机自启（或终端 `launchctl unload ~/Library/LaunchAgents/com.udos.engine*.plist` 后删除这些 plist）。
2. 菜单 `5` 停止服务。
3. 删除仓库内 `.venv-udos/`、`deploy/mac/run/`、`deploy/mac/logs/`（这些都是本机生成的）。
4. 整个 `udos-engine/` 文件夹删除即彻底移除，不会在系统目录残留。

---

## 七、验证情况与能力边界（诚实说明）
- 以下已在等价的 **Linux + Python 3.12 + CPU torch** 环境**实测通过**（命令与仓库真实入口逐字
  对齐，跨平台通用）：两版服务启动命令与 checkpoint 路径、`/health` 健康检查返回、
  启动→健康检查→**停止后端口确实释放**、`tests7` 33 项测试、开机自启 **plist 的 XML 合法性与
  install/status/uninstall 流程**。
- 以下受条件限制**未在 Mac 实机验证（unverified-on-mac）**：Finder 双击 `.command`、
  Gatekeeper/隔离属性弹窗、浏览器自动 `open`、真实 `launchctl load`、以及 Mac 版 torch 的联网
  下载安装（本环境为 Linux、无 Mac）。如在 Mac 上遇到报错，请把 `deploy/mac/logs/` 下日志发回。
- 引擎能力边界同 `docs7/ARCHITECTURE.md`：当前为 **CPU 可跑的真实训练/预测原型（verified）**；
  vLLM KV-offload、NEURON/CoreNEURON、MuJoCo-MJX、0.5B/5B 端到端等 **GPU/HPC 项未包含**，
  需另行提供 GPU 资源，切勿用 CPU 数字代替 GPU 结论。
