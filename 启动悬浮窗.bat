@echo off
chcp 65001 >nul
title 腾讯桌球解球辅助 - Overlay

cd /d "E:\Users\Admin\SNOOKER"

rem 优先用 PEMAE 环境里的 python
set "PY="
if exist "D:\ProgramData\miniconda3\envs\PEMAE\python.exe" (
    set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
) else (
    where conda >nul 2>nul
    if not errorlevel 1 (
        call conda activate PEMAE 2>nul
        set "PY=python"
    )
)

if not defined PY (
    echo [错误] 未找到 PEMAE 环境，请先安装/激活 conda PEMAE
    pause
    exit /b 1
)

echo 启动悬浮窗: %PY% overlay.py
"%PY%" overlay.py
if errorlevel 1 (
    echo.
    echo [异常] 程序退出码 %errorlevel%
    pause
)
