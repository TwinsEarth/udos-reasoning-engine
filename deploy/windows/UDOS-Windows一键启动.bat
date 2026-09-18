@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
rem ============================================================
rem  UDOS Engine  Windows 一键启动（无需 WSL / Docker）
rem  双击运行；torch 等重依赖只在“共享环境”装一次，升级版本不重复安装
rem ============================================================
cd /d "%~dp0..\.."
echo ============================================================
echo   UDOS Engine  Windows 一键启动
echo   仓库根目录: %CD%
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python，请先到 https://www.python.org/downloads/ 安装 Python 3.10-3.12，
  echo        安装时务必勾选 "Add Python to PATH"，然后重新双击本脚本。
  pause
  exit /b 1
)

echo [1/3] 准备共享虚拟环境（首次会下载 torch，较慢；之后各版本直接复用）...
set "PY="
for /f "usebackq delims=" %%i in (`python "%CD%\bin\udos-env.py" ensure`) do set "PY=%%i"
if not exist "%PY%" (
  echo [错误] 共享环境准备失败，请把上方日志反馈。
  pause
  exit /b 1
)
echo 使用 Python: %PY%

set PYTHONPATH=.
echo [2/3] 运行 tests7 回归测试 ...
"%PY%" -m pytest tests7 -q
if errorlevel 1 (
  echo [警告] 回归测试存在失败，可先把日志反馈；仍继续启动服务。
)

echo [3/3] 启动引擎服务 ...
echo   健康检查: http://127.0.0.1:8777/api/v7/health
echo   指标:     http://127.0.0.1:8777/api/v7/metrics
echo   命令行:   bin\udos.bat （另开窗口可用，如 bin\udos.bat health）
echo   按 Ctrl+C 停止。
"%PY%" -m udos7.server --host 127.0.0.1 --port 8777

pause
