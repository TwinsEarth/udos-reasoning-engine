#!/bin/bash
# UDOS 推演引擎 —— macOS 一键部署公共函数库
# 被「UDOS-Mac一键启动.command」及 deploy/mac 下脚本 source 使用。
# 兼容 macOS 自带 bash 3.2，不使用 bash 4 专有语法。

set -o pipefail

# ---------- 路径（自动定位仓库根，不依赖双击时的当前目录）----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../deploy/mac
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"                        # 仓库根
# 共享环境：默认放在用户主目录 ~/.udos/venv，多个 UDOS 版本只装一次 torch；
# 可用 UDOS_VENV 自定义；若旧版本在仓库内建过 .venv-udos 则继续沿用。
if [ -n "${UDOS_VENV:-}" ]; then
  VENV="$UDOS_VENV"
elif [ -x "$ROOT/.venv-udos/bin/python" ]; then
  VENV="$ROOT/.venv-udos"
else
  VENV="$HOME/.udos/venv"
fi
PYBIN="$VENV/bin/python"
REQ_MAC="$SCRIPT_DIR/requirements-mac.txt"
LOG_DIR="$SCRIPT_DIR/logs"
RUN_DIR="$SCRIPT_DIR/run"
mkdir -p "$LOG_DIR" "$RUN_DIR"

# ---------- 版本与端口（与仓库真实入口逐字对齐）----------
V7_VERSION="7.0.3"
V5_VERSION="5.5.5"
V7_PORT="8777"
V5_PORT="8000"
V7_PID="$RUN_DIR/udos7.pid"
V5_PID="$RUN_DIR/udos5.pid"
V7_LOG="$LOG_DIR/udos7.log"
V5_LOG="$LOG_DIR/udos5.log"
V7_CKPT="checkpoints7/worldmodel_v7.0.3.pt"
V5_CKPT="checkpoints/predictor_v4.3.9.pt"
LABEL_V7="com.udos.engine7"
LABEL_V5="com.udos.engine5"

# ---------- 颜色（双击 .command 的终端里生效）----------
if [ -t 1 ]; then
  C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_B=$'\033[36m'; C_0=$'\033[0m'
else
  C_R=""; C_G=""; C_Y=""; C_B=""; C_0=""
fi
say()  { echo "${C_B}[UDOS]${C_0} $*"; }
ok()   { echo "${C_G}[ 成功 ]${C_0} $*"; }
warn() { echo "${C_Y}[ 提示 ]${C_0} $*"; }
err()  { echo "${C_R}[ 出错 ]${C_0} $*" 1>&2; }

# ---------- 架构检测 ----------
detect_arch() {
  local m; m="$(uname -m)"
  if [ "$m" = "arm64" ]; then
    echo "apple-silicon"
  else
    echo "intel"
  fi
}

# ---------- 找到满足 >=3.10 的系统 python3（不依赖虚拟环境）----------
# 输出可用解释器路径；找不到输出空串。
find_system_python() {
  local candidates=""
  # Apple Silicon Homebrew 优先，其次 Intel Homebrew，再 python.org 与系统
  [ -x /opt/homebrew/bin/python3.12 ] && candidates="$candidates /opt/homebrew/bin/python3.12"
  [ -x /opt/homebrew/bin/python3.11 ] && candidates="$candidates /opt/homebrew/bin/python3.11"
  [ -x /opt/homebrew/bin/python3.13 ] && candidates="$candidates /opt/homebrew/bin/python3.13"
  [ -x /usr/local/bin/python3.12 ] && candidates="$candidates /usr/local/bin/python3.12"
  [ -x /usr/local/bin/python3.11 ] && candidates="$candidates /usr/local/bin/python3.11"
  ls /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 2>/dev/null | while read -r p; do echo "$p"; done > "$RUN_DIR/pyorg.txt"
  candidates="$candidates $(cat "$RUN_DIR/pyorg.txt" 2>/dev/null | tr '\n' ' ')"
  candidates="$candidates /usr/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3"

  for p in $candidates; do
    command -v "$p" >/dev/null 2>&1 || [ -x "$p" ] || continue
    if "$p" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]>=(3,10) else 1)' >/dev/null 2>&1; then
      echo "$p"; return 0
    fi
  done
  echo ""
}

# ---------- 安装/修复环境（建虚拟环境 + 装 Mac 原生 torch）----------
ensure_env() {
  local arch arch_name; arch="$(detect_arch)"
  if [ "$arch" = "apple-silicon" ]; then arch_name="Apple Silicon（M1/M2/M3/M4）"; else arch_name="Intel"; fi
  say "检测到芯片架构：$arch（$arch_name）"

  if [ -x "$PYBIN" ] && "$PYBIN" -c 'import torch, numpy' >/dev/null 2>&1; then
    ok "虚拟环境已就绪：$VENV"
    echo
    "$PYBIN" -c 'import sys,torch,numpy;print("  Python",sys.version.split()[0],"| torch",torch.__version__,"| numpy",numpy.__version__)'
    return 0
  fi

  local SP; SP="$(find_system_python)"
  if [ -z "$SP" ]; then
    err "未找到 Python 3.10 及以上版本。"
    echo
    echo "  请二选一安装（推荐第 1 种，最简单）："
    echo "  1) 打开官网下载安装包：https://www.python.org/downloads/macos/"
    echo "     下载最新 Python 3.12，双击 .pkg 一路继续即可。"
    echo "  2) 若已装 Homebrew，在终端执行：brew install python@3.12"
    echo
    read -r -p "  装好后按回车重新检测（或按 Ctrl+C 退出）：" _
    SP="$(find_system_python)"
    if [ -z "$SP" ]; then
      err "仍未检测到 Python 3.10+，请确认安装成功后重跑本程序。"
      return 1
    fi
  fi
  say "使用基础 Python：$SP（$("$SP" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')）"

  if [ ! -d "$VENV" ]; then
    say "正在创建虚拟环境 .venv-udos（仅放在本文件夹内，不污染系统）..."
    "$SP" -m venv "$VENV" || { err "创建虚拟环境失败"; return 1; }
  fi
  "$PYBIN" -m pip install --upgrade pip >/dev/null 2>&1

  say "正在安装依赖（torch 较大，首次约需几分钟，请耐心等待）..."
  # 默认 PyPI 即提供 macOS arm64/x86_64 原生 wheel；不要加 Linux 的 cpu index。
  if ! "$PYBIN" -m pip install -r "$REQ_MAC"; then
    warn "默认源安装失败/过慢，尝试切换清华镜像源重试..."
    if ! "$PYBIN" -m pip install -r "$REQ_MAC" \
          -i https://pypi.tuna.tsinghua.edu.cn/simple; then
      err "依赖安装失败。请把上方日志、以及 $LOG_DIR 内容反馈。"
      return 1
    fi
  fi

  echo
  if "$PYBIN" -c 'import torch, numpy' >/dev/null 2>&1; then
    ok "环境安装完成。"
    "$PYBIN" -c 'import sys,torch,numpy;print("  Python",sys.version.split()[0],"| torch",torch.__version__,"| numpy",numpy.__version__)'
  else
    err "依赖导入校验失败，请检查网络后重试。"
    return 1
  fi
}

# ---------- 健康探测（纯标准库，跨平台）----------
# wait_health URL [期望子串] [最多等待秒]
wait_health() {
  local url="$1" want="${2:-}" secs="${3:-40}" i body
  for ((i=0;i<secs;i++)); do
    body="$(curl -fsS --max-time 2 "$url" 2>/dev/null)" && {
      if [ -z "$want" ] || echo "$body" | grep -q "$want"; then
        echo "$body"; return 0
      fi
    }
    sleep 1
  done
  return 1
}

pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }

open_url() {
  local url="$1"
  if command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1; fi
  return 0
}

# ---------- 启动 v7.0.3 ----------
start_v7() {
  ensure_env || return 1
  if pid_alive "$V7_PID"; then
    warn "v$V7_VERSION 已在运行（端口 $V7_PORT），直接打开页面。"
  else
    say "启动 UDOS v$V7_VERSION（主线新版），端口 $V7_PORT ..."
    # exec 让后台子 shell 直接替换为 python，使 $! 即真正的监听进程 PID
    ( cd "$ROOT" && exec nohup "$PYBIN" -m udos7.server --host 127.0.0.1 --port "$V7_PORT" \
        >"$V7_LOG" 2>&1 </dev/null ) &
    echo $! > "$V7_PID"; disown 2>/dev/null || true
    sleep 1
    local body
    if ! body="$(wait_health "http://127.0.0.1:$V7_PORT/api/v7/health" '"status": "ok"' 45)"; then
      err "v$V7_VERSION 启动未通过健康检查，请查看日志：$V7_LOG"; tail -n 15 "$V7_LOG" 2>/dev/null; return 1
    fi
    ok "v$V7_VERSION 已启动：$body"
  fi
  say "首次推理会加载模型（约几秒）。正在打开健康检查页..."
  open_url "http://127.0.0.1:$V7_PORT/api/v7/health"
  echo "  健康检查 : http://127.0.0.1:$V7_PORT/api/v7/health"
  echo "  指标报告 : http://127.0.0.1:$V7_PORT/api/v7/metrics"
  echo "  运行日志 : $V7_LOG"
}

# ---------- 启动 v5.5.5（legacy）----------
start_v5() {
  ensure_env || return 1
  if pid_alive "$V5_PID"; then
    warn "v$V5_VERSION 已在运行（端口 $V5_PORT），直接打开页面。"
  else
    if [ ! -f "$ROOT/$V5_CKPT" ]; then err "缺少 legacy checkpoint：$V5_CKPT"; return 1; fi
    say "启动 UDOS v$V5_VERSION（legacy 冻结版），端口 $V5_PORT ..."
    ( cd "$ROOT" && exec nohup "$PYBIN" -m udos.server --host 127.0.0.1 --port "$V5_PORT" \
        --preset small --checkpoint "$V5_CKPT" >"$V5_LOG" 2>&1 </dev/null ) &
    echo $! > "$V5_PID"; disown 2>/dev/null || true
    sleep 1
    local body
    if ! body="$(wait_health "http://127.0.0.1:$V5_PORT/health" '"status": "ok"' 45)"; then
      err "v$V5_VERSION 启动未通过健康检查，请查看日志：$V5_LOG"; tail -n 15 "$V5_LOG" 2>/dev/null; return 1
    fi
    ok "v$V5_VERSION 已启动：$body"
  fi
  say "正在打开健康检查页与可视化控制台..."
  open_url "http://127.0.0.1:$V5_PORT/health"
  # legacy 控制台为静态页（file:// 打开，内含内嵌数据并尝试请求 /health）
  [ -f "$ROOT/web/udos_console.html" ] && open_url "file://$ROOT/web/udos_console.html"
  echo "  健康检查 : http://127.0.0.1:$V5_PORT/health"
  echo "  控制台   : $ROOT/web/udos_console.html"
  echo "  运行日志 : $V5_LOG"
}

# ---------- 停止 ----------
stop_one() {
  local pidfile="$1" port="$2" name="$3"
  if pid_alive "$pidfile"; then
    kill "$(cat "$pidfile")" >/dev/null 2>&1
    local i; for ((i=0;i<8;i++)); do pid_alive "$pidfile" || break; sleep 1; done
    pid_alive "$pidfile" && kill -9 "$(cat "$pidfile")" >/dev/null 2>&1
    rm -f "$pidfile"; ok "$name 已停止。"
  else
    # 兜底：按端口清理（可能由开机自启等方式拉起）
    if command -v lsof >/dev/null 2>&1; then
      local pids; pids="$(lsof -ti tcp:"$port" 2>/dev/null | tr '\n' ' ')"
      if [ -n "$pids" ]; then kill $pids >/dev/null 2>&1; ok "$name（端口 $port）已停止。"; fi
    fi
    rm -f "$pidfile"; warn "$name 未在运行。"
  fi
}
stop_all() {
  stop_one "$V7_PID" "$V7_PORT" "v$V7_VERSION"
  stop_one "$V5_PID" "$V5_PORT" "v$V5_VERSION"
}

# ---------- 自检 ----------
self_check() {
  ensure_env || return 1
  cd "$ROOT"
  say "运行 v$V7_VERSION 端到端冒烟（临时端口，含 predict/interval/metrics）..."
  "$PYBIN" scripts7/service_smoke.py || { err "v$V7_VERSION 冒烟未通过"; return 1; }
  echo
  say "运行 v$V7_VERSION 测试集（tests7）..."
  "$PYBIN" -m pytest tests7 -p no:warnings -q || { err "tests7 未全部通过"; return 1; }
  echo
  say "v$V5_VERSION 仅做导入与健康检查（legacy 冻结，不重跑其全量测试）..."
  "$PYBIN" -c "import udos; print('  udos version', udos.__version__)" || return 1
  ok "自检完成。"
}
