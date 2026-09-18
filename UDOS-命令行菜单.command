#!/bin/bash
# UDOS Engine 命令行菜单（macOS / Linux），双击或在终端运行。
cd "$(dirname "$0")" || exit 1
CLI="./bin/udos"

pause() { printf '\n按回车键返回菜单…'; read -r _; }

while true; do
  clear
  cat <<'MENU'
============================================================
           UDOS Engine 命令行菜单（macOS / Linux）
============================================================
  1  环境信息（版本 / Python / torch / 共享环境路径）
  2  运行回归测试 tests7
  3  启动引擎服务（启动后用浏览器开健康检查，Ctrl+C 停止）
  4  健康检查（需已用第 3 项启动服务）
  5  可观测六层演示（护栏/成本/异常/OTLP）
  6  Agent 军团演示
  7  查看完整命令帮助
  8  首次安装/修复依赖（手动重建共享环境）
  0  退出
============================================================
MENU
  read -r -p "请输入数字后回车: " choice
  case "$choice" in
    1) "$CLI" info; pause ;;
    2) "$CLI" test; pause ;;
    3) "$CLI" serve ;;
    4) "$CLI" health; pause ;;
    5) "$CLI" demo obs-pro; pause ;;
    6) "$CLI" demo legion; pause ;;
    7) "$CLI" --help; pause ;;
    8) python3 bin/udos-env.py ensure; pause ;;
    0) exit 0 ;;
    *) echo "无效选项"; pause ;;
  esac
done
