@echo off
setlocal
cd /d E:\Users\Admin\SNOOKER

set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo  Snooker Assist Overlay
echo ============================================
echo  Hotkeys:
echo    F6  Clear felt lock (auto detect)
echo    F7  Calibrate felt: drag edges or buttons, F7 to lock
echo    F8  Analyze (pot first, then escape)
echo    F9  Pot lines only
echo    F10 Cycle target color
echo    F11 Show / hide overlay
echo    Esc Quit
echo  After F7 lock, F8/F9 will not resize the table
echo ============================================
echo.
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
