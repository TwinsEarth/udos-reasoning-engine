@echo off
rem UDOS Engine 命令行包装器（Windows）。放在仓库 bin\ 下。
rem 通过 bin\udos-env.py 复用“用户主目录下的共享虚拟环境”，多个 UDOS 版本只装一次 torch。
rem 可用环境变量 UDOS_VENV 指定自定义环境路径。
setlocal
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."

where python >nul 2>nul
if errorlevel 1 (
  echo [UDOS] 未找到 Python，请先安装 Python 3.10-3.12 并勾选 Add Python to PATH。 1>&2
  exit /b 1
)

rem ensure 输出最后一行是共享环境的 python.exe 路径
set "PY="
for /f "usebackq delims=" %%i in (`python "%ROOT%\bin\udos-env.py" ensure`) do set "PY=%%i"
if not exist "%PY%" (
  echo [UDOS] 共享环境引导失败，回退系统 Python。 1>&2
  set "PY=python"
)

cd /d "%ROOT%"
"%PY%" -m udos7.cli %*
