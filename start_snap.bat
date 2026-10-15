@echo off
setlocal
rem 用脚本自身所在目录，避免硬编码路径写错
cd /d "%~dp0"

set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" src\main.py snap %*
echo.
echo Exit code %ERRORLEVEL%
pause
endlocal
