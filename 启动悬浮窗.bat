@echo off
REM 中文名入口：转发到 start_overlay.bat（避免编码/路径问题）
cd /d "%~dp0"
call "%~dp0start_overlay.bat"
