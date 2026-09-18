@echo off
rem 根目录便捷入口：在仓库根直接输入  udos.bat <命令>
rem 例如：udos.bat info  / udos.bat serve / udos.bat health
"%~dp0bin\udos.bat" %*
