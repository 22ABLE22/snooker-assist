@echo off
setlocal
cd /d E:\Users\Admin\SNOOKER

set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
if not exist "%PY%" set "PY=python"

echo Working dir: %CD%
echo Python: %PY%
echo Starting overlay...
echo.

"%PY%" overlay.py
set "RC=%ERRORLEVEL%"

echo.
echo ============================================
echo overlay exited with code %RC%
echo ============================================
if not "%RC%"=="0" (
    echo If the window flashed, please copy the error above.
)
pause
endlocal
