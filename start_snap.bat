@echo off
setlocal
cd /d E:\Users\Admin\SNOOKER

set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" main.py snap %*
echo.
echo Exit code %ERRORLEVEL%
pause
endlocal
