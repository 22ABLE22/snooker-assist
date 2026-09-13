@echo off
REM 中文名入口：转发到 start_overlay.bat（避免编码/路径问题）
REM 热键说明见 start_overlay.bat 启动时的提示，或 README.md
cd /d "%~dp0"
echo 提示: F7 校准锁定台面 | F6 清锁定 | F8 解球 | F9 进球 | Esc 退出
call "%~dp0start_overlay.bat"
