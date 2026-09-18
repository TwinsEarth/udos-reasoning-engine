#!/bin/bash
# UDOS —— macOS 开机自启（launchd）安装/卸载
# 用法:
#   autostart.sh install v7
#   autostart.sh install v5
#   autostart.sh uninstall v7
#   autostart.sh uninstall v5
#   autostart.sh status
# 说明: 在 ~/Library/LaunchAgents 生成 plist，登录后自动在前台托管服务；
#       不使用 sudo，仅对当前用户生效。

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

PLIST_DIR="$HOME/Library/LaunchAgents"

plist_path() { echo "$PLIST_DIR/$1.plist"; }

write_plist() {
  local label="$1" port="$2" module="$3" extra="$4" plist="$5"
  local args="
      <string>$PYBIN</string>
      <string>-m</string>
      <string>$module</string>
      <string>--host</string><string>127.0.0.1</string>
      <string>--port</string><string>$port</string>"
  if [ -n "$extra" ]; then
    args="$args
      <string>--preset</string><string>small</string>
      <string>--checkpoint</string><string>$extra</string>"
  fi
  mkdir -p "$PLIST_DIR"
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>$args
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>$LOG_DIR/${label}.launch.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/${label}.launch.err.log</string>
</dict>
</plist>
EOF
}

install_one() {
  local which="$1" label port module ckpt plist
  if [ "$which" = "v7" ]; then
    label="$LABEL_V7"; port="$V7_PORT"; module="udos7.server"; ckpt=""
  elif [ "$which" = "v5" ]; then
    label="$LABEL_V5"; port="$V5_PORT"; module="udos.server"; ckpt="$V5_CKPT"
  else
    err "未知版本：$which（应为 v7 或 v5）"; return 1
  fi
  ensure_env || return 1
  plist="$(plist_path "$label")"
  write_plist "$label" "$port" "$module" "$ckpt" "$plist"
  launchctl unload "$plist" >/dev/null 2>&1
  launchctl load "$plist" || { err "launchctl load 失败"; return 1; }
  ok "已设置开机自启并立即启动：$which（标签 $label，端口 $port）"
  echo "  plist: $plist"
  echo "  卸载 : $0 uninstall $which"
}

uninstall_one() {
  local which="$1" label plist
  if [ "$which" = "v7" ]; then label="$LABEL_V7"; else label="$LABEL_V5"; fi
  plist="$(plist_path "$label")"
  launchctl unload "$plist" >/dev/null 2>&1
  rm -f "$plist"
  ok "已取消开机自启：$which（$label）"
}

status_all() {
  for l in "$LABEL_V7" "$LABEL_V5"; do
    if [ -f "$(plist_path "$l")" ]; then echo "  $l : 已安装自启"; else echo "  $l : 未安装"; fi
  done
}

case "${1:-}" in
  install)   install_one "${2:-}" ;;
  uninstall) uninstall_one "${2:-}" ;;
  status)    status_all ;;
  *) echo "用法: $0 {install|uninstall} {v7|v5} ; $0 status"; exit 1 ;;
esac
