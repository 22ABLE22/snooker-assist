@echo off
REM Launcher: calls start_overlay.bat (ASCII-only for cmd encoding)
cd /d "%~dp0"
echo Hotkeys: F7 calibrate/lock felt | F6 clear lock | F8 analyze | F9 pot | Esc quit
call "%~dp0start_overlay.bat"
