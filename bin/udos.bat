@echo off
rem UDOS Engine 命令行包装器（Windows）。放在仓库 bin\ 下。
rem 自动定位仓库根，优先使用一键部署创建的 .venv-win，无需手动 activate。
setlocal
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."

if exist "%ROOT%\.venv-win\Scripts\python.exe" (
  set "PY=%ROOT%\.venv-win\Scripts\python.exe"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [UDOS] 未找到 Python，请先安装 Python 3.10-3.12 并勾选 Add Python to PATH。 1>&2
    exit /b 1
  )
  set "PY=python"
)

cd /d "%ROOT%"
"%PY%" -m udos7.cli %*
