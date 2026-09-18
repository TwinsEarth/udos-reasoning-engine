#!/bin/bash
# =====================================================================
# UDOS 推演引擎 —— macOS 小白一键启动器（双击运行）
# 覆盖：v7.0.3（主线新版，端口 8777）与 v5.5.5（legacy 冻结版，端口 8000）
# 首次使用：在 Finder 里「右键 → 打开」一次（绕过未公证提示），之后可双击。
# =====================================================================
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPLOY="$HERE/deploy/mac"
# shellcheck source=deploy/mac/common.sh
source "$DEPLOY/common.sh"

pause() { echo; read -r -p "按回车返回菜单..." _; }

while true; do
  clear 2>/dev/null || true
  echo "============================================================"
  echo "   UDOS 推演引擎  ·  macOS 一键启动器"
  echo "   新版 v$V7_VERSION（推荐，端口 $V7_PORT）  |  旧版 v$V5_VERSION（端口 $V5_PORT）"
  echo "============================================================"
  echo "   1) 首次安装 / 修复环境（建虚拟环境、装 torch，约几分钟）"
  echo "   2) 启动新版 v$V7_VERSION 并打开页面"
  echo "   3) 启动旧版 v$V5_VERSION 并打开页面"
  echo "   4) 两个版本一起启动（端口互不冲突）"
  echo "   5) 停止全部服务"
  echo "   6) 自检（v$V7_VERSION 冒烟 + tests7 测试）"
  echo "   7) 设置开机自启（登录后自动运行）"
  echo "   8) 取消开机自启"
  echo "   9) 查看自启状态"
  echo "   0) 退出（退出本窗口不会停止已在后台运行的服务）"
  echo "------------------------------------------------------------"
  read -r -p "请输入选项数字后回车: " choice
  case "$choice" in
    1) ensure_env; pause ;;
    2) start_v7; pause ;;
    3) start_v5; pause ;;
    4) start_v7; echo; start_v5; pause ;;
    5) stop_all; pause ;;
    6) self_check; pause ;;
    7)
       echo "要让哪个版本开机自启？输入 7=新版v$V7_VERSION，5=旧版v$V5_VERSION："
       read -r -p "版本 [7/5]: " v
       [ "$v" = "7" ] && bash "$DEPLOY/autostart.sh" install v7
       [ "$v" = "5" ] && bash "$DEPLOY/autostart.sh" install v5
       pause ;;
    8)
       read -r -p "取消哪个版本自启 [7/5]: " v
       [ "$v" = "7" ] && bash "$DEPLOY/autostart.sh" uninstall v7
       [ "$v" = "5" ] && bash "$DEPLOY/autostart.sh" uninstall v5
       pause ;;
    9) bash "$DEPLOY/autostart.sh" status; pause ;;
    0) echo "再见。"; exit 0 ;;
    *) warn "无效选项"; sleep 1 ;;
  esac
done
