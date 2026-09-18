@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
rem ============================================================
rem  UDOS Engine v7.3.3  Windows 一键启动（无需 WSL / Docker）
rem  双击运行；自动建虚拟环境、装 CPU 依赖、跑回归、起引擎服务
rem ============================================================
cd /d "%~dp0..\.."
echo ============================================================
echo   UDOS Engine v7.3.3  Windows 一键启动
echo   仓库根目录: %CD%
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python，请先到 https://www.python.org/downloads/ 安装 Python 3.10-3.12，
  echo        安装时务必勾选 "Add Python to PATH"，然后重新双击本脚本。
  pause
  exit /b 1
)

if not exist ".venv-win" (
  echo [1/4] 创建虚拟环境 .venv-win ...
  python -m venv .venv-win
)
call ".venv-win\Scripts\activate.bat"

echo [2/4] 升级 pip 并安装 CPU 依赖（首次较慢，请耐心等待）...
python -m pip install --upgrade pip
python -m pip install torch numpy huggingface_hub pytest
if errorlevel 1 (
  echo [提示] 默认源失败，尝试清华镜像 ...
  python -m pip install torch numpy huggingface_hub pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
)

set PYTHONPATH=.
echo [3/4] 运行 tests7 回归测试 ...
python -m pytest tests7 -q
if errorlevel 1 (
  echo [警告] 回归测试存在失败，可先把日志反馈；仍继续启动服务。
)

echo [4/4] 启动引擎服务 ...
echo   健康检查: http://127.0.0.1:8777/api/v7/health
echo   指标:     http://127.0.0.1:8777/api/v7/metrics
echo   按 Ctrl+C 停止。
python -m udos7.server --host 127.0.0.1 --port 8777

pause
