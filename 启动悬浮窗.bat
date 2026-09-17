@echo off
REM Launcher: calls start_overlay.bat (ASCII-only for cmd encoding)
cd /d "%~dp0"
echo Hotkeys: F9 pot+aim-line | F8 analyze | F7 calibrate | F6 clear lock | F5 toggle line | Esc quit
call "%~dp0start_overlay.bat"
