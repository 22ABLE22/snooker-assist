@echo off
chcp 65001 >nul
title 腾讯桌球解球辅助 - 命令行截图

cd /d "E:\Users\Admin\SNOOKER"

set "PY=D:\ProgramData\miniconda3\envs\PEMAE\python.exe"
if not exist "%PY%" (
    where conda >nul 2>nul
    if not errorlevel 1 (
        call conda activate PEMAE 2>nul
        set "PY=python"
    )
)

"%PY%" main.py snap %*
pause
