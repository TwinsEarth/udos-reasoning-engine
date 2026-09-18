@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "CLI=%~dp0bin\udos.bat"

:menu
cls
echo ============================================================
echo            UDOS Engine 命令行菜单（Windows）
echo ============================================================
echo   1  环境信息（版本 / Python / torch / 共享环境路径）
echo   2  运行回归测试 tests7
echo   3  启动引擎服务（启动后用浏览器开健康检查，Ctrl+C 停止）
echo   4  健康检查（需已用第 3 项启动服务）
echo   5  可观测六层演示（护栏/成本/异常/OTLP）
echo   6  Agent 军团演示
echo   7  查看完整命令帮助
echo   8  首次安装/修复依赖（手动重建共享环境）
echo   0  退出
echo ============================================================
set /p choice=请输入数字后回车:

if "%choice%"=="1" ( call "%CLI%" info & pause & goto menu )
if "%choice%"=="2" ( call "%CLI%" test & pause & goto menu )
if "%choice%"=="3" ( call "%CLI%" serve & pause & goto menu )
if "%choice%"=="4" ( call "%CLI%" health & pause & goto menu )
if "%choice%"=="5" ( call "%CLI%" demo obs-pro & pause & goto menu )
if "%choice%"=="6" ( call "%CLI%" demo legion & pause & goto menu )
if "%choice%"=="7" ( call "%CLI%" --help & pause & goto menu )
if "%choice%"=="8" ( python "%~dp0bin\udos-env.py" ensure & pause & goto menu )
if "%choice%"=="0" exit /b 0
echo 无效选项，请重新输入。
pause
goto menu
