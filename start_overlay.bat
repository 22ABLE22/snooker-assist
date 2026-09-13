@echo off
setlocal
cd /d E:\Users\Admin\SNOOKER

set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo  腾讯桌球解球辅助 - Overlay
echo ============================================
echo  热键:
echo    F6  清除台面锁定（恢复自动检测）
echo    F7  校准台面：拖四边或底部按钮，再按 F7 锁定
echo    F8  截图分析（优先进球，其次解球）
echo    F9  只分析进球
echo    F10 切换目标球颜色
echo    F11 显示/隐藏标注
echo    Esc 退出
echo  说明: 校准确认后，本盘内 F8/F9 不再改台面大小
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
